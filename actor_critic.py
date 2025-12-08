"""
Delete Ratio Actor-Critic Network for Durotaxis Environment

This module implements a decoupled actor-critic architecture that handles:
1. Single global continuous action: [delete_ratio]
   - delete_ratio: fraction of leftmost nodes to delete (0.0 to 0.5)
   - Spawn parameters (gamma, alpha, noise) are fixed constants from config.yaml
2. Multi-component value estimation for different reward components
3. Graph neural network integration via GraphInputEncoder
4. Pre-trained ResNet backbone for enhanced feature extraction and stability

The architecture is designed to work with the durotaxis environment's
reward component dictionary structure for flexible learning updates.

Architecture Strategy:
- Processes all nodes through ResNet backbone
- Aggregates node features via mean pooling
- Outputs single global action vector (not per-node actions)
- Delete ratio determines which nodes to delete based on x-position sorting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Tuple, Optional, List
from torchvision.models import resnet18, ResNet18_Weights
from encoder import GraphInputEncoder
from config_loader import ConfigLoader


def _safe_masked_logits(logits: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Make logits numerically stable and mask invalid actions.
    
    Args:
        logits: Raw action logits [N, num_actions]
        action_mask: Boolean mask [N, num_actions] where True = valid action
        
    Returns:
        Stable masked logits safe for Categorical(logits=...)
        
    Key features:
    - Aggressive NaN/Inf cleaning at entry
    - Standard max-subtraction after masking for softmax stability
    - Handles all-invalid rows (fallback to uniform distribution)
    - Final safety clamping before return
    """
    # 1. Clean any NaN or Inf values immediately at entry
    masked = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
    
    # 2. Apply action mask if provided (fill invalid actions with -inf)
    if action_mask is not None:
        mask = action_mask.to(device=masked.device, dtype=torch.bool)
        
        # Ensure mask has same shape as logits (handle broadcasting)
        if mask.dim() == 1 and masked.dim() == 2:
            mask = mask.unsqueeze(1).expand_as(masked)
        
        # Mask invalid actions with -inf (will have ~0 probability after softmax)
        masked = masked.masked_fill(~mask, float('-inf'))
        
    # 3. Check for rows where all actions are invalid (all -inf)
    all_invalid = torch.isinf(masked).all(dim=1)
    if all_invalid.any():
        # Fallback to uniform distribution (zeros -> equal probs after softmax)
        masked[all_invalid] = 0.0
        
    # 4. Apply max-subtraction for numerical stability
    # This is the standard approach to prevent overflow in softmax's exp()
    # Keeps the largest logit at 0, preventing exp(large_number) = inf
    row_max = masked.max(dim=1, keepdim=True).values
    
    # Normalize: subtract the max logit from all logits
    # For all_invalid rows where max=0.0, other entries are also 0.0, so subtraction is safe
    masked = masked - torch.nan_to_num(row_max, nan=0.0)
    
    # 5. Final clamp to prevent any unexpected extreme values before use
    # After max-subtraction, all values should be <= 0
    masked = torch.clamp(masked, -30.0, 0.0)
    
    return masked


class Actor(nn.Module):
    """
    The Actor network for the Delete Ratio Actor-Critic agent.
    Outputs a SINGLE GLOBAL continuous action: [delete_ratio]
    
    Architecture:
    - Processes all nodes through ResNet backbone
    - Aggregates node features (mean pooling)
    - Outputs single continuous action (delete_ratio) for the entire graph
    
    Delete Ratio Strategy:
    - delete_ratio ∈ [0.0, 0.5]: fraction of leftmost nodes to delete
    - Spawn parameters (gamma, alpha, noise) are fixed constants from config
    - All remaining nodes spawn with the same fixed parameters
    """
    def __init__(self, encoder_out_dim, hidden_dim, continuous_dim, dropout_rate,
                 pretrained_weights='imagenet',
                 backbone_cfg: Optional[dict] = None):
        super().__init__()
        backbone_cfg = backbone_cfg or {}
        self.input_adapter = backbone_cfg.get('input_adapter', 'repeat3')  # 'repeat3' | '1ch_conv'
        self.freeze_mode = backbone_cfg.get('freeze_mode', 'none')         # 'none'|'all'|'until_layer3'|'last_block'

        self.feature_proj = nn.Linear(encoder_out_dim * 2, 512)

        if pretrained_weights == 'imagenet':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("  🔧 Actor using ResNet18 with ImageNet weights")
        else:
            resnet = resnet18(weights=None)
            print(f"  🔧 Actor using ResNet18 with random initialization")
        self.resnet_body = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_body.eval()

        # Input adaptation: prefer repeat3 to preserve pretrained conv1
        if self.input_adapter == '1ch_conv':
            # Replace first conv to accept 1 channel
            self.resnet_body[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # else: keep original 3-channel conv1 and repeat input later

        # Optional freezing strategy
        self._apply_freeze(self.resnet_body, self.freeze_mode)

        # Global action MLP: aggregates per-node features into single graph-level action
        self.action_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Single continuous action head for global parameters
        # Output: [delete_ratio] (single continuous action)
        self.continuous_mu_head = nn.Linear(hidden_dim, continuous_dim)
        self.continuous_logstd_head = nn.Linear(hidden_dim, continuous_dim)
        
        # Initialize log_std to -0.5 (std ≈ 0.6) to maintain exploration
        # Default initialization often leads to std collapse
        nn.init.constant_(self.continuous_logstd_head.bias, -0.5)
        nn.init.normal_(self.continuous_logstd_head.weight, mean=0.0, std=0.01)
        print(f"  ✅ Initialized log_std head: bias={self.continuous_logstd_head.bias.mean().item():.3f}, weight_std={self.continuous_logstd_head.weight.std().item():.3f}")

    def _apply_freeze(self, backbone: nn.Module, mode: str):
        # mode: 'all' | 'until_layer3' | 'last_block' | 'none'
        if mode == 'none':
            return
        def set_requires(m, flag):
            for p in m.parameters():
                p.requires_grad = flag
        if mode == 'all':
            set_requires(backbone, False)
        elif mode == 'until_layer3':
            # Freeze conv1, bn1, layer1, layer2, layer3
            modules_to_freeze = ['0', '1', '4', '5', '6']  # conv1, bn1, layer1, layer2, layer3 in resnet_body indices
            for name, m in backbone._modules.items():
                if name in modules_to_freeze:
                    set_requires(m, False)
        elif mode == 'last_block':
            # Freeze everything except layer4
            for name, m in backbone._modules.items():
                if name != '7':  # '7' is layer4 in the body sequence
                    set_requires(m, False)

    def forward(self, node_tokens, graph_token):
        num_nodes = node_tokens.shape[0]
        device = node_tokens.device
        
        # For very large graphs on CUDA, use gradient checkpointing to save memory
        # instead of moving to CPU (which causes device mismatch in backward pass)
        use_checkpointing = (num_nodes > 200 and device.type == 'cuda')
        
        graph_context = graph_token.unsqueeze(0).repeat(num_nodes, 1)
        combined_features = torch.cat([node_tokens, graph_context], dim=-1)

        # Project features and reshape to be "image-like" for ResNet
        # [num_nodes, 512] -> [num_nodes, 1, H, W]
        # We need to find H, W such that H*W is close to 512. Let's use sqrt.
        # A 22x23 image is close. Let's pad to 24x24=576 for simplicity with conv strides.
        projected_features = self.feature_proj(torch.cat([node_tokens, graph_token.unsqueeze(0).repeat(node_tokens.size(0), 1)], dim=-1))
        padded_features = F.pad(projected_features, (0, 576 - 512))
        image_like_features = padded_features.view(-1, 1, 24, 24)

        # Adapt channels for ResNet input
        if self.input_adapter == 'repeat3':
            image_like_features = image_like_features.repeat(1, 3, 1, 1)
        # elif '1ch_conv': keep as 1 channel

        # Pass through ResNet body in batches to avoid OOM with large graphs
        batch_size = 64  # Small batch size for 4GB GPU
        if num_nodes <= batch_size:
            # Small enough, process all at once
            resnet_out = self.resnet_body(image_like_features)
            shared_features = resnet_out.view(num_nodes, -1)
        else:
            # Process in batches to reduce peak memory usage
            resnet_outputs = []
            for i in range(0, num_nodes, batch_size):
                batch_end = min(i + batch_size, num_nodes)
                batch_features = image_like_features[i:batch_end]
                
                # Use gradient checkpointing for large graphs to save memory
                if use_checkpointing and torch.is_grad_enabled():
                    batch_out = torch.utils.checkpoint.checkpoint(
                        self.resnet_body, batch_features, use_reentrant=False
                    )
                else:
                    batch_out = self.resnet_body(batch_features)
                
                resnet_outputs.append(batch_out.view(batch_end - i, -1))
            
            shared_features = torch.cat(resnet_outputs, dim=0)

        # Aggregate per-node features into single graph-level representation
        # Use mean pooling to get a global context from all nodes
        aggregated_features = shared_features.mean(dim=0, keepdim=True)  # [1, 512]
        
        # Pass through final MLP
        global_features = self.action_mlp(aggregated_features)  # [1, hidden_dim]
        
        # Aggressive NaN/Inf check on global features (critical failure point)
        if torch.isnan(global_features).any() or torch.isinf(global_features).any():
            print("⚠️  WARNING: NaN/Inf detected in Actor global_features! Sanitizing...")
            global_features = torch.nan_to_num(global_features, nan=0.0, posinf=10.0, neginf=-10.0)

        # --- Single Global Continuous Action ---
        # Output: [delete_ratio] only (spawn parameters now fixed in config)
        continuous_mu = self.continuous_mu_head(global_features).squeeze(0)  # [1]
        continuous_logstd = self.continuous_logstd_head(global_features).squeeze(0)  # [1]
        
        # Tighter clamping for numerical stability (prevent exp overflow in distributions)
        continuous_mu = torch.clamp(continuous_mu, -10.0, 10.0)
        continuous_logstd = torch.clamp(continuous_logstd, -10.0, 5.0)
        
        # Final NaN check before returning (defensive programming)
        continuous_mu = torch.nan_to_num(continuous_mu, nan=0.0)
        continuous_logstd = torch.nan_to_num(continuous_logstd, nan=0.0)

        return continuous_mu, continuous_logstd


class Critic(nn.Module):
    """
    The Critic network for the Delete Ratio Actor-Critic agent.
    
    Processes graph-level features through ResNet backbone to estimate state values
    for different reward components. Works with the delete ratio architecture where
    actions are global (not per-node).
    """
    def __init__(self, encoder_out_dim, hidden_dim, value_components: List[str], dropout_rate,
                 pretrained_weights='imagenet', backbone_cfg: Optional[dict] = None):
        super().__init__()
        backbone_cfg = backbone_cfg or {}
        self.input_adapter = backbone_cfg.get('input_adapter', 'repeat3')
        self.freeze_mode = backbone_cfg.get('freeze_mode', 'none')

        self.value_components = value_components
        self.feature_proj = nn.Linear(encoder_out_dim, 512)

        if pretrained_weights == 'imagenet':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("  🔧 Critic using ResNet18 with ImageNet weights")
        else:
            resnet = resnet18(weights=None)
            print(f"  🔧 Critic using ResNet18 with random initialization")
        self.resnet_body = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_body.eval()

        if self.input_adapter == '1ch_conv':
            self.resnet_body[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self._apply_freeze(self.resnet_body, self.freeze_mode)

        self.value_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.value_heads = nn.ModuleDict({component: nn.Linear(hidden_dim, 1) for component in self.value_components})

    def _apply_freeze(self, backbone: nn.Module, mode: str):
        # Same as in Actor
        if mode == 'none': return
        def set_requires(m, flag):
            for p in m.parameters():
                p.requires_grad = flag
        if mode == 'all':
            set_requires(backbone, False)
        elif mode == 'until_layer3':
            modules_to_freeze = ['0', '1', '4', '5', '6']
            for name, m in backbone._modules.items():
                if name in modules_to_freeze:
                    set_requires(m, False)
        elif mode == 'last_block':
            for name, m in backbone._modules.items():
                if name != '7':
                    set_requires(m, False)

    def forward(self, graph_token):
        device = graph_token.device
        
        # Project and reshape to be "image-like" for ResNet
        padded_features = F.pad(self.feature_proj(graph_token).unsqueeze(0), (0, 576 - 512))
        image_like_features = padded_features.view(-1, 1, 24, 24)
        if self.input_adapter == 'repeat3':
            image_like_features = image_like_features.repeat(1, 3, 1, 1)
        # Pass through ResNet body
        resnet_out = self.resnet_body(image_like_features)
        shared_features = resnet_out.view(1, -1)  # [1, 512]

        # Pass through value MLP
        value_features = self.value_mlp(shared_features)

        # --- Value Heads ---
        value_predictions = {
            component: self.value_heads[component](value_features).squeeze(-1)
            for component in self.value_components
        }

        return value_predictions, value_features


class HybridActorCritic(nn.Module):
    """
    Delete Ratio Actor-Critic network for the durotaxis environment.
    
    This class orchestrates the GraphInputEncoder, Actor, and Critic modules
    to produce a single global continuous action vector: [delete_ratio].
    Spawn parameters are fixed constants from config.yaml.
    
    Delete Ratio Strategy:
    - Actor outputs one action for the entire graph (not per-node)
    - delete_ratio determines fraction of leftmost nodes to delete
    - Remaining nodes spawn with global parameters (gamma, alpha, noise)
    - Critic evaluates graph-level state values
    """
    
    def __init__(self, 
                 encoder: GraphInputEncoder,
                 config_path: str = "config.yaml",
                 **overrides):
        """
        Initialize HybridActorCritic with configuration from YAML file.
        
        Args:
            encoder: GraphInputEncoder for processing graph state
            config_path: Path to config.yaml
            **overrides: Configuration overrides
        """
        super().__init__()
        
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_actor_critic_config()
        
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        self.encoder = encoder
        self.hidden_dim = config.get('hidden_dim', 128)
        self.continuous_dim = 1  # Only delete_ratio (spawn params fixed in config)
        self.dropout_rate = config.get('dropout_rate', 0.1)

        # Value components configuration
        value_components = config.get('value_components', ['total_value'])
        if value_components is None:
            value_components = ['total_value']
        self.value_components = value_components
        
        # Pretrained weights configuration
        pretrained_weights = config.get('pretrained_weights', 'imagenet')
        print(f"\n📦 Loading Actor-Critic with pretrained weights: {pretrained_weights}")
        
        # Backbone configuration for freezing and input adaptation
        backbone_cfg = config.get('backbone', {})
        
        print("\n🔧 Using standard Actor with delete_ratio action space")
        self.actor = Actor(
            encoder_out_dim=self.encoder.out_dim,
            hidden_dim=self.hidden_dim,
            continuous_dim=self.continuous_dim,
            dropout_rate=self.dropout_rate,
            pretrained_weights=pretrained_weights,
            backbone_cfg=backbone_cfg
        )
        self.critic = Critic(
            encoder_out_dim=self.encoder.out_dim,
            hidden_dim=self.hidden_dim,
            value_components=self.value_components,
            dropout_rate=self.dropout_rate,
            pretrained_weights=pretrained_weights,
            backbone_cfg=backbone_cfg
        )
        
        # Validate value heads match config (defensive check)
        self._validate_value_heads()
        
        # Parameter bounds for continuous actions from config
        # New action space: [delete_ratio] only
        # Spawn parameters (gamma, alpha, noise) are fixed in config.yaml
        action_bounds_config = config.get('action_parameter_bounds', {
            'delete_ratio': [0.0, 0.5],
            'gamma': [0.5, 15.0],
            'alpha': [0.5, 4.0],
            'noise': [0.05, 0.5],
            'theta': [-0.5236, 0.5236]
        })
        
        # Create action_bounds with explicit float32 dtype for device consistency
        bounds_list = [
            action_bounds_config.get('delete_ratio', [0.0, 0.5]),
            action_bounds_config.get('gamma', [0.5, 15.0]),
            action_bounds_config.get('alpha', [0.5, 4.0]),
            action_bounds_config.get('noise', [0.05, 0.5]),
            action_bounds_config.get('theta', [-0.5236, 0.5236])
        ]
        self.register_buffer('action_bounds', torch.tensor(bounds_list, dtype=torch.float32))
        
        # Load fixed spawn parameters from config (used in deployment)
        env_config = config_loader.get_environment_config()
        spawn_params = env_config.get('spawn_parameters', {})
        self.fixed_spawn_params = (
            float(spawn_params.get('gamma', 5.0)),
            float(spawn_params.get('alpha', 2.0)),
            float(spawn_params.get('noise', 0.5))
        )
        
        self.apply(self._init_weights)
        
        # Print parameter information for verification
        self._print_parameter_info(backbone_cfg)
    
    def _validate_value_heads(self):
        """
        Validate that critic value heads exactly match configured value_components.
        
        This defensive check catches config/architecture mismatches at initialization
        rather than failing silently or crashing during training.
        """
        expected_components = set(self.value_components)
        actual_heads = set(self.critic.value_heads.keys())
        
        if expected_components != actual_heads:
            missing_heads = expected_components - actual_heads
            extra_heads = actual_heads - expected_components
            
            error_msg = "❌ Value head configuration mismatch!\n"
            if missing_heads:
                error_msg += f"  Missing heads: {missing_heads}\n"
            if extra_heads:
                error_msg += f"  Extra heads: {extra_heads}\n"
            error_msg += f"  Expected: {sorted(expected_components)}\n"
            error_msg += f"  Actual: {sorted(actual_heads)}"
            
            raise ValueError(error_msg)
        
        # Success message
        print(f"✅ Value head validation passed: {len(expected_components)} components")
        for component in sorted(self.value_components):
            print(f"   - {component}")
    
    def validate_component_weights(self, trainer_component_weights: Dict[str, float]) -> None:
        """
        Validate that trainer component weights are compatible with value heads.
        
        This method should be called by the trainer after initialization to ensure
        that the critic can provide values for all weighted components.
        
        Args:
            trainer_component_weights: Dictionary of component weights from trainer config
            
        Raises:
            ValueError: If there are critical mismatches between components
        """
        vc_set = set(self.value_components)
        tcw_set = set(trainer_component_weights.keys())
        
        # Check for components in trainer weights but not in value heads
        missing_in_critic = tcw_set - vc_set
        if missing_in_critic:
            raise ValueError(
                f"❌ Trainer component_weights has keys not present in critic value_components:\n"
                f"  Missing in critic: {missing_in_critic}\n"
                f"  Critic components: {sorted(vc_set)}\n"
                f"  Trainer weights: {sorted(tcw_set)}"
            )
        
        # Warn about components in value heads but not in trainer weights
        missing_in_trainer = vc_set - tcw_set
        if missing_in_trainer:
            print(f"⚠️  WARNING: Value components not in trainer.component_weights (will use weight=0):")
            for component in sorted(missing_in_trainer):
                print(f"   - {component}")
        else:
            print(f"✅ Component weights validation passed: All {len(tcw_set)} components aligned")
    
    def _print_parameter_info(self, backbone_cfg: dict):
        """Print detailed parameter information for debugging and verification."""
        print("\n" + "="*70)
        print("📊 NETWORK PARAMETER INFORMATION")
        print("="*70)
        
        # Input adapter information
        input_adapter = backbone_cfg.get('input_adapter', 'repeat3')
        freeze_mode = backbone_cfg.get('freeze_mode', 'none')
        
        if input_adapter == 'repeat3':
            print("  🖼️  Input Adapter: repeat3 (conv1 PRESERVED from pretrained)")
        else:
            print("  🖼️  Input Adapter: 1ch_conv (conv1 REPLACED for single channel)")
        
        print(f"  ❄️  Freeze Mode: {freeze_mode}")
        if freeze_mode == 'none':
            print("      → All backbone layers are trainable")
        elif freeze_mode == 'all':
            print("      → All backbone layers are frozen")
        elif freeze_mode == 'until_layer3':
            print("      → Frozen: conv1, bn1, layer1, layer2, layer3")
            print("      → Trainable: layer4")
        elif freeze_mode == 'last_block':
            print("      → Frozen: conv1, bn1, layer1, layer2, layer3")
            print("      → Trainable: layer4 only")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n  📈 Total Parameters: {total_params:,}")
        print(f"  ✅ Trainable Parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print(f"  ❄️  Frozen Parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
        
        # Break down by component (standard Actor-Critic only, WSA removed)
        actor_backbone_params = sum(p.numel() for p in self.actor.resnet_body.parameters())
        actor_backbone_trainable = sum(p.numel() for p in self.actor.resnet_body.parameters() if p.requires_grad)
        
        critic_backbone_params = sum(p.numel() for p in self.critic.resnet_body.parameters())
        critic_backbone_trainable = sum(p.numel() for p in self.critic.resnet_body.parameters() if p.requires_grad)
        
        actor_head_params = sum(p.numel() for p in self.actor.parameters()) - actor_backbone_params
        critic_head_params = sum(p.numel() for p in self.critic.parameters()) - critic_backbone_params
        
        print(f"\n  🎭 Actor Breakdown:")
        print(f"      Backbone: {actor_backbone_params:,} ({actor_backbone_trainable:,} trainable)")
        print(f"      Heads: {actor_head_params:,} (all trainable)")
        
        print(f"  🎯 Critic Breakdown:")
        print(f"      Backbone: {critic_backbone_params:,} ({critic_backbone_trainable:,} trainable)")
        print(f"      Heads: {critic_head_params:,} (all trainable)")
        
        # Learning rate information
        backbone_lr = backbone_cfg.get('backbone_lr', 1e-4)
        head_lr = backbone_cfg.get('head_lr', 3e-4)
        
        print(f"\n  🎓 Learning Rate Configuration:")
        print(f"      Backbone LR: {backbone_lr:.6f}")
        print(f"      Head LR: {head_lr:.6f}")
        print(f"      LR Ratio (head/backbone): {head_lr/backbone_lr:.1f}x")
        
        print("="*70 + "\n")
    
    def train(self, mode: bool = True):
        """
        Override the default train method to keep ResNet in eval mode.
        The rest of the network will be in the specified mode.
        """
        super().train(mode)
        # Ensure ResNet bodies remain in evaluation mode
        if self.actor and hasattr(self.actor, 'resnet_body'):
            self.actor.resnet_body.eval()
        if self.critic and hasattr(self.critic, 'resnet_body'):
            self.critic.resnet_body.eval()
        return self

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, state_dict: Dict[str, torch.Tensor], 
                deterministic: bool = False,
                action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the delete ratio actor-critic network.
        
        Args:
            state_dict: Graph state containing node_features, graph_features, edge_attr, edge_index
            deterministic: If True, return mean actions; if False, sample from distributions
            action_mask: Unused in delete ratio architecture (kept for API compatibility)
        
        Returns:
            Dictionary containing:
            - continuous_mu: Mean of continuous action distribution [5]
            - continuous_std: Std of continuous action distribution [5]
            - continuous_actions: Sampled or deterministic actions [delete_ratio]
            - value_predictions: Dict of value estimates for each component
            - encoder_out: Graph and node embeddings
        """
        # Extract components from state
        node_features = state_dict['node_features']
        graph_features = state_dict['graph_features']
        edge_features = state_dict['edge_attr']
        edge_index = state_dict['edge_index']

        # Ensure all inputs are on the same device as the network
        device = self.action_bounds.device
        node_features = node_features.to(device)
        graph_features = graph_features.to(device)
        edge_features = edge_features.to(device)

        if isinstance(edge_index, tuple):
            src, dst = edge_index
            src = src.to(device)
            dst = dst.to(device)
            edge_index_tensor = torch.stack([src, dst], dim=0)
        else:
            edge_index_tensor = edge_index.to(device)
        
        num_nodes = node_features.shape[0]
        if num_nodes == 0:
            return self._empty_output()
        
        # Get embeddings from encoder
        encoder_out = self.encoder(
            graph_features=graph_features,
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index_tensor
        )
        
        graph_token = encoder_out[0]
        node_tokens = encoder_out[1:]
        
        # === ACTOR and CRITIC FORWARD PASS ===
        # Actor now outputs SINGLE GLOBAL continuous action: [delete_ratio]
        # Spawn parameters are fixed constants from config
        continuous_mu, continuous_logstd = self.actor(node_tokens, graph_token)
        value_predictions, graph_value_features = self.critic(graph_token)

        # --- Post-processing and Sanitization ---
        # Clean continuous parameters
        continuous_mu = torch.nan_to_num(continuous_mu, nan=0.0)
        continuous_logstd = torch.nan_to_num(continuous_logstd, nan=0.0)
        continuous_logstd = torch.clamp(continuous_logstd, min=-10.0, max=2.0)
        continuous_std = torch.exp(continuous_logstd)
        # Ensure minimum std to prevent policy collapse (0.3 gives positive entropy)
        # Entropy = 0.5 + 0.5*log(2π) + log(σ), with σ=0.3 → entropy ≈ 0.3 per dim
        continuous_std = torch.clamp(continuous_std, min=0.3, max=2.0)
        
        continuous_mu_bounded = self._apply_bounds(continuous_mu)
        
        # === OUTPUT PREPARATION ===
        output = {
            'encoder_out': encoder_out,
            'continuous_mu': continuous_mu_bounded,
            'continuous_std': continuous_std,
            'value_predictions': value_predictions,
            'graph_features': graph_value_features,
        }
        
        if not deterministic:
            continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)
            continuous_actions_raw = continuous_dist.sample()
            continuous_log_probs = continuous_dist.log_prob(continuous_actions_raw).sum(dim=-1)
            
            continuous_actions = self._apply_bounds(continuous_actions_raw)
            
            output.update({
                'continuous_actions': continuous_actions,
                'continuous_log_probs': continuous_log_probs,
                'total_log_probs': continuous_log_probs  # Only continuous actions now
            })
        else:
            output.update({
                'continuous_actions': continuous_mu_bounded
            })
        
        return output
    
    def _apply_bounds(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply parameter bounds to continuous actions.
        Action space: [delete_ratio] (spawn params fixed in config)
        """
        bounded_actions = actions.clone()
        
        for i, (min_val, max_val) in enumerate(self.action_bounds):
            if i < actions.shape[-1]:
                if i == 4:  # theta (index 4): use tanh for circular bounds [-π/4, π/4]
                    bounded_actions[..., i] = torch.tanh(actions[..., i]) * (max_val - min_val) / 2.0 + (max_val + min_val) / 2.0
                else:  # delete_ratio, gamma, alpha, noise: use sigmoid scaling
                    bounded_actions[..., i] = torch.sigmoid(actions[..., i]) * (max_val - min_val) + min_val
        
        return bounded_actions
    
    def _empty_output(self) -> Dict[str, torch.Tensor]:
        """Return empty output for graphs with no nodes."""
        # Get device from action_bounds (a registered buffer)
        device = self.action_bounds.device
        
        value_predictions = {component: torch.tensor(0.0, device=device) for component in self.value_components}
        
        return {
            'encoder_out': torch.empty(1, self.encoder.out_dim, device=device),
            'continuous_mu': torch.zeros(self.continuous_dim, device=device),
            'continuous_std': torch.ones(self.continuous_dim, device=device),
            'value_predictions': value_predictions,
            'continuous_actions': torch.zeros(self.continuous_dim, device=device),
            'continuous_log_probs': torch.tensor(0.0, device=device),
            'total_log_probs': torch.tensor(0.0, device=device)
        }
    
    def evaluate_actions(self, state_dict: Dict[str, torch.Tensor], 
                        continuous_actions: torch.Tensor,
                        cached_output: Optional[Dict[str, torch.Tensor]] = None,
                        action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate given continuous actions (used for policy updates).
        Delete ratio architecture: Only continuous action [delete_ratio]
        Spawn parameters (gamma, alpha, noise) are fixed constants from config
        """
        if cached_output is not None:
            output = cached_output
        else:
            output = self.forward(state_dict, deterministic=True, action_mask=action_mask)
        
        # Check if we have valid output
        num_nodes = state_dict['node_features'].shape[0]
        if num_nodes == 0:
            device = self.action_bounds.device
            return {
                'continuous_log_probs': torch.empty(0, device=device),
                'total_log_probs': torch.empty(0, device=device),
                'value_predictions': output['value_predictions'],
                'entropy': torch.tensor(0.0, device=device)
            }
        
        # Re-run actor forward pass to get distributions for entropy calculation
        node_tokens = output['encoder_out'][1:]
        graph_token = output['encoder_out'][0]
        continuous_mu, continuous_logstd = self.actor(node_tokens, graph_token)
        
        # Clamp to prevent overflow/underflow
        continuous_mu = torch.clamp(continuous_mu, -10.0, 10.0)
        continuous_logstd = torch.clamp(continuous_logstd, -10.0, 2.0)
        continuous_std = torch.exp(continuous_logstd)
        
        # Ensure std is not too small (avoid division by zero or extreme log-probs)
        # Minimum of 0.3 ensures positive entropy (~0.3 per dim, ~1.5 total for 5 dims)
        continuous_std = torch.clamp(continuous_std, min=0.3, max=2.0)
        
        # Evaluate continuous actions with safety checks
        continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)
        
        # Clamp continuous_actions to reasonable range before log_prob
        continuous_actions_clamped = torch.clamp(continuous_actions, -10.0, 10.0)
        continuous_log_probs = continuous_dist.log_prob(continuous_actions_clamped).sum(dim=-1)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)
        
        # Debug: Check std and entropy values (including negative entropy)
        raw_entropy_mean = continuous_entropy.mean().item()
        if abs(raw_entropy_mean) < 1.0:  # Very low entropy (positive or negative)
            print(f"⚠️  WARNING: Entropy is critically low!")
            print(f"   Raw entropy: {raw_entropy_mean:.6f}")
            print(f"   std: min={continuous_std.min().item():.6f}, max={continuous_std.max().item():.6f}, mean={continuous_std.mean().item():.6f}")
            print(f"   logstd: min={continuous_logstd.min().item():.6f}, max={continuous_logstd.max().item():.6f}")
            print(f"   Expected entropy with std=0.01: {(0.5 + 0.5*np.log(2*np.pi) + np.log(0.01))*5:.6f}")
        
        # Safety: Clamp log_probs to prevent extreme values
        continuous_log_probs = torch.clamp(continuous_log_probs, -20.0, 20.0)
        
        # Final NaN check
        continuous_log_probs = torch.nan_to_num(continuous_log_probs, nan=0.0)
        continuous_entropy = torch.nan_to_num(continuous_entropy, nan=0.0)
        
        return {
            'continuous_log_probs': continuous_log_probs,
            'total_log_probs': continuous_log_probs,  # Only continuous actions
            'value_predictions': output['value_predictions'],
            'entropy': continuous_entropy.mean(),
            'continuous_entropy': continuous_entropy.mean(),
            'continuous_mu': continuous_mu,  # For proper KL divergence
            'continuous_std': continuous_std  # For proper KL divergence
        }
    
    def get_topology_actions(self, output: Dict[str, torch.Tensor], node_positions: List[Tuple[int, float]]) -> Dict[int, str]:
        """
        Convert network output to topology-compatible action format using delete ratio strategy.
        
        Args:
            output: Network output containing continuous_actions [delete_ratio] (single action)
            node_positions: List of (node_id, is_leftmost_marker) tuples where:
                - is_leftmost_marker = 1.0 means node should be deleted (is in leftmost k positions)
                - is_leftmost_marker = 0.0 means node should spawn (not in leftmost k positions)
                This is a binary flag used for efficient O(n) selection, NOT the delete_ratio value itself.
        
        Returns:
            Dict mapping node_id to action ('spawn' or 'delete')
        """
        if 'continuous_actions' not in output:
            return {}
        
        continuous_actions = output['continuous_actions']
        delete_ratio = continuous_actions[0].item()  # First element is delete_ratio
        
        num_nodes = len(node_positions)
        if num_nodes == 0:
            return {}
        
        # Calculate number of nodes to delete (leftmost nodes)
        num_to_delete = int(delete_ratio * num_nodes)
        num_to_delete = min(num_to_delete, num_nodes)  # Clamp to valid range
        
        # OPTIMIZATION 1: node_positions now contains pre-computed deletion info
        # Format: (node_id, is_leftmost) where is_leftmost is a binary marker:
        #   - is_leftmost = 1.0 means node should be deleted
        #   - is_leftmost = 0.0 means node should spawn
        # The threshold 0.5 below is just checking which value it's closer to (NOT related to delete_ratio bounds)
        actions = {}
        if num_to_delete == 0:
            # No deletion
            for node_id, _ in node_positions:
                actions[node_id] = 'spawn'
        elif num_to_delete >= num_nodes:
            # Delete all
            for node_id, _ in node_positions:
                actions[node_id] = 'delete'
        else:
            # Use pre-computed leftmost marker (threshold 0.5 = midpoint between 0.0 and 1.0)
            for node_id, is_leftmost in node_positions:
                actions[node_id] = 'delete' if is_leftmost > 0.5 else 'spawn'
        
        return actions
    
    def get_spawn_parameters(self, output: Dict[str, torch.Tensor]) -> Tuple[float, float, float]:
        """
        Get spawn parameters from configuration (no longer from network output).
        
        Returns:
            Tuple of (gamma, alpha, noise) - fixed constants from config
            Note: theta is removed as it's not used in the refactored architecture
        """
        # Return fixed spawn parameters from config
        # These are loaded in HybridPolicyAgent __init__
        return self.fixed_spawn_params


class HybridPolicyAgent:
    """
    Agent wrapper for the Delete Ratio Actor-Critic network.
    
    Converts network output (single global action: delete_ratio) into topology operations:
    - Sorts nodes by x-position
    - Deletes leftmost nodes based on delete_ratio
    - Spawns from remaining nodes using FIXED spawn parameters from config
    """
    
    def __init__(self, topology, state_extractor, hybrid_network: HybridActorCritic, config_loader=None):
        self.topology = topology
        self.state_extractor = state_extractor
        self.network = hybrid_network
        
        # Load fixed spawn parameters from config
        if config_loader is not None:
            env_config = config_loader.get_environment_config()
            spawn_params = env_config.get('spawn_parameters', {})
            self.fixed_spawn_params = (
                float(spawn_params.get('gamma', 5.0)),
                float(spawn_params.get('alpha', 2.0)),
                float(spawn_params.get('noise', 0.5))
            )
        else:
            # Fallback defaults if no config provided
            self.fixed_spawn_params = (5.0, 2.0, 0.5)
        
        print(f"  ✅ Fixed spawn parameters: gamma={self.fixed_spawn_params[0]:.2f}, "
              f"alpha={self.fixed_spawn_params[1]:.2f}, noise={self.fixed_spawn_params[2]:.2f}")
        
        # Store last action parameters for logging/analysis
        self.last_continuous_actions = None
        self.last_spawn_params = self.fixed_spawn_params  # Always use fixed params
        
    @torch.no_grad()
    def get_actions_and_values(self, deterministic: bool = False,
                              action_mask: Optional[torch.Tensor] = None) -> Tuple[Dict[int, str], 
                                                                         Tuple[float, float, float], 
                                                                         Dict[str, torch.Tensor]]:
        """
        Get actions, spawn parameters, and value predictions using delete ratio strategy.
        
        Returns:
            - actions: Dict mapping node_id to action ('spawn' or 'delete')
            - spawn_params: Fixed tuple (gamma, alpha, noise) from config (used for ALL spawns)
            - value_predictions: Dict of value predictions
        """
        state = self.state_extractor.get_state_features(include_substrate=True)
        
        if state['num_nodes'] == 0:
            # Get device from network's action_bounds
            device = self.network.action_bounds.device
            empty_values = {component: torch.tensor(0.0, device=device) for component in self.network.value_components}
            return {}, self.fixed_spawn_params, empty_values
        
        output = self.network(state, deterministic=deterministic, action_mask=action_mask)
        
        # Store continuous actions for logging (only delete_ratio now)
        self.last_continuous_actions = output['continuous_actions']  # [1] tensor (delete_ratio)
        
        # OPTIMIZATION 1: Use argpartition for O(n) selection instead of O(n log n) sort
        # Get node x-positions directly from node features
        node_features = state['node_features']
        num_nodes = state['num_nodes']
        
        # Extract delete_ratio to determine k (number of nodes to delete)
        delete_ratio = output['continuous_actions'][0].item()
        num_to_delete = int(delete_ratio * num_nodes)
        num_to_delete = min(max(num_to_delete, 0), num_nodes)  # Clamp to [0, num_nodes]
        
        # Build node_positions efficiently based on num_to_delete
        if num_to_delete == 0:
            # No deletion needed - all nodes spawn
            node_positions = [(i, 0.0) for i in range(num_nodes)]  # x-position irrelevant
        elif num_to_delete >= num_nodes:
            # Delete all - positions don't matter, mark all for deletion
            node_positions = [(i, 0.0) for i in range(num_nodes)]
        else:
            # Use argpartition to find k leftmost nodes in O(n) time
            import numpy as np
            x_positions = np.array([node_features[i][0].item() for i in range(num_nodes)], dtype=np.float32)
            
            # Partition: indices of k smallest x-positions
            partition_indices = np.argpartition(x_positions, num_to_delete - 1)
            leftmost_k_indices = set(partition_indices[:num_to_delete].tolist())
            
            # Build node_positions with deletion info embedded in the tuple structure
            # Format: (node_id, is_leftmost_marker) where:
            #   - marker = 1.0 means node is in leftmost k positions (delete)
            #   - marker = 0.0 means node is not in leftmost k (spawn)
            # This is a binary flag, NOT related to delete_ratio parameter value
            node_positions = [(i, 1.0 if i in leftmost_k_indices else 0.0) for i in range(num_nodes)]
        
        actions = self.network.get_topology_actions(output, node_positions)
        
        # Get single global spawn parameters from config (fixed constants)
        spawn_params = self.fixed_spawn_params
        
        return actions, spawn_params, output['value_predictions']
    
    @torch.no_grad()
    def act_with_policy(self, deterministic: bool = False,
                       action_mask: Optional[torch.Tensor] = None) -> Dict[int, str]:
        """
        Execute actions using the hybrid network with delete ratio strategy.
        
        CRITICAL: Uses persistent_id to track nodes across spawn/delete operations.
        Since spawning shifts node IDs, we must convert current node_ids to persistent_ids
        before executing actions, then look them up after spawning completes.
        """
        actions, spawn_params_global, value_predictions = self.get_actions_and_values(deterministic, action_mask)
        
        # Store spawn parameters for logging
        self.last_spawn_params = spawn_params_global
        
        spawn_actions = {node_id: action for node_id, action in actions.items() if action == 'spawn'}
        delete_actions = {node_id: action for node_id, action in actions.items() if action == 'delete'}
        
        # CRITICAL FIX: Convert current node_ids to persistent_ids BEFORE any modifications
        # This prevents index shifting bugs when spawns execute before deletes
        spawn_persistent_ids = {}
        for node_id in spawn_actions.keys():
            persistent_id = self.topology.node_id_to_persistent_id(node_id)
            if persistent_id is not None:
                spawn_persistent_ids[persistent_id] = node_id  # Map persistent_id -> original_node_id for logging
        
        delete_persistent_ids = {}
        for node_id in delete_actions.keys():
            persistent_id = self.topology.node_id_to_persistent_id(node_id)
            if persistent_id is not None:
                delete_persistent_ids[persistent_id] = node_id  # Map persistent_id -> original_node_id for logging
        
        # First, mark nodes for deletion (set to_delete flag) using CURRENT node IDs (before spawning)
        if 'to_delete' in self.topology.graph.ndata and len(delete_actions) > 0:
            # Reset all to_delete flags to 0
            self.topology.graph.ndata['to_delete'] = torch.zeros(
                self.topology.graph.num_nodes(), dtype=torch.float32, device=self.topology.device
            )
            # Set to_delete=1 for nodes that should be deleted (use current node_id)
            for node_id in delete_actions.keys():
                if node_id < self.topology.graph.num_nodes():
                    self.topology.graph.ndata['to_delete'][node_id] = 1.0
        
        # Execute spawn actions (ALL use the same global spawn parameters from config)
        # Use persistent_ids to handle index shifting via topology helper
        gamma, alpha, noise = spawn_params_global  # Fixed params from config (no theta)
        for persistent_id, original_node_id in spawn_persistent_ids.items():
            try:
                # theta=0.0 is default in spawn method signature
                new_id = self.topology.spawn_by_persistent_id(persistent_id, gamma=gamma, alpha=alpha, noise=noise, theta=0.0)
                if new_id is None:
                    print(f"⚠️  Spawn skipped: persistent_id={persistent_id} no longer exists")
            except Exception as e:
                print(f"Failed to spawn (persistent_id={persistent_id}): {e}")
        
        # Execute delete actions using persistent_ids (handles index shifting from spawns)
        # Sort by CURRENT node_id in reverse order to prevent shifting issues during deletion
        delete_items = []
        for persistent_id, original_node_id in delete_persistent_ids.items():
            current_node_id = self.topology.persistent_id_to_node_id(persistent_id)
            if current_node_id is not None:
                # Validate current_node_id is a valid integer in range
                if isinstance(current_node_id, (int, np.integer)) and 0 <= current_node_id < self.topology.graph.num_nodes():
                    delete_items.append((int(current_node_id), persistent_id))
                else:
                    print(f"⚠️  Delete skipped: persistent_id={persistent_id} maps to invalid node_id={current_node_id} (type={type(current_node_id)})")
        
        # Sort by current_node_id in reverse order (highest first)
        delete_items.sort(reverse=True, key=lambda x: x[0])
        
        for current_node_id, persistent_id in delete_items:
            try:
                ok = self.topology.delete_by_persistent_id(persistent_id)
                if not ok:
                    print(f"⚠️  Delete skipped: persistent_id={persistent_id} no longer exists")
            except Exception as e:
                print(f"Failed to delete (persistent_id={persistent_id}): {e}")
        
        return actions


if __name__ == '__main__':
    # Test the HybridActorCritic network
    print("🧪 Testing Decoupled HybridActorCritic Network with ResNet")
    print("=" * 60)
    
    try:
        encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=2)
        
        reward_components = ['total_reward', 'delete_reward', 'distance_reward', 'termination_reward']
        network = HybridActorCritic(
            encoder=encoder,
            hidden_dim=128,
            value_components=reward_components
        )
        
        num_nodes = 5
        num_edges = 4
        
        state_dict = {
            'node_features': torch.randn(num_nodes, 9),
            'graph_features': torch.randn(14),
            'edge_attr': torch.randn(num_edges, 3),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }
        
        print(f"\nTest state: {num_nodes} nodes, {num_edges} edges")
        
        output_stochastic = network(state_dict, deterministic=False)
        print(f"Stochastic forward pass works: ✅")
        print(f"Continuous actions shape: {output_stochastic['continuous_actions'].shape}")
        print(f"Value predictions: {list(output_stochastic['value_predictions'].keys())}")
        
        output_deterministic = network(state_dict, deterministic=True)
        print(f"Deterministic forward pass works: ✅")
        
        eval_output = network.evaluate_actions(
            state_dict,
            output_stochastic['continuous_actions']
        )
        print(f"Action evaluation works: ✅")
        print(f"Entropy: {eval_output['entropy']:.4f}")
        
        # Delete ratio architecture: Single global continuous action
        print(f"Delete ratio action: {output_stochastic['continuous_actions']}")
        print(f"  - delete_ratio: {output_stochastic['continuous_actions'][0]:.3f}")
        print(f"  - gamma: {output_stochastic['continuous_actions'][1]:.3f}")
        print(f"  - alpha: {output_stochastic['continuous_actions'][2]:.3f}")
        print(f"  - noise: {output_stochastic['continuous_actions'][3]:.3f}")
        print(f"  - theta: {output_stochastic['continuous_actions'][4]:.3f}")
        
        print("\n✅ All tests passed! Delete ratio architecture HybridActorCritic is ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
