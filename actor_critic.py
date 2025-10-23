"""
Hybrid Actor-Critic Network for Durotaxis Environment

This module implements a decoupled actor-critic architecture that handles:
1. Discrete actions: spawn/delete decisions per node
2. Continuous actions: spawn parameters (gamma, alpha, noise, theta)
3. Multi-component value estimation for different reward components
4. Graph neural network integration via GraphInputEncoder
5. Pre-trained ResNet backbone for enhanced feature extraction and stability.

The architecture is designed to work with the durotaxis environment's
reward component dictionary structure for flexible learning updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
    - If an entire row is invalid (all masked), fallback to zeros (uniform after softmax)
    - Only normalize when logits are dangerously large (>20) to preserve training dynamics
    - Clean any NaN/Inf values
    """
    # Apply action mask if provided
    if action_mask is not None:
        mask = action_mask.to(device=logits.device, dtype=torch.bool)
        
        # Ensure mask has same shape as logits
        if mask.dim() == 1 and logits.dim() == 2 and mask.size(0) == logits.size(0):
            # Broadcast mask per row over action dimension
            mask = mask.unsqueeze(1).expand_as(logits)
        
        # Mask invalid actions with -inf (will have ~0 probability after softmax)
        masked = logits.masked_fill(~mask, float('-inf'))
        
        # Check for rows where all actions are invalid
        all_invalid = (~mask).all(dim=1)
        if all_invalid.any():
            # Fallback to uniform distribution (zeros -> equal probs after softmax)
            masked[all_invalid] = 0.0
    else:
        masked = logits
    
    # Clean any NaN or Inf values
    masked = torch.nan_to_num(masked, nan=0.0, posinf=20.0, neginf=-20.0)
    
    # Only apply max-subtraction if logits are dangerously large (>20)
    # This preserves the natural scale of logits during training
    row_max = masked.max(dim=1, keepdim=True).values
    needs_normalization = (row_max > 20.0) | (row_max < -20.0)
    
    if needs_normalization.any():
        # Only normalize rows that need it
        normalized = masked - torch.nan_to_num(row_max, nan=0.0)
        masked = torch.where(needs_normalization, normalized, masked)
    
    return masked


class Actor(nn.Module):
    """
    The Actor network for the Hybrid Actor-Critic agent.
    It takes node and graph features and outputs action distributions.
    """
    def __init__(self, encoder_out_dim, hidden_dim, num_discrete_actions, continuous_dim, dropout_rate, 
                 pretrained_weights='imagenet', spawn_bias_init: float = 0.0):
        """
        Args:
            pretrained_weights: 'imagenet', 'random', or None
                - 'imagenet': Use ImageNet pre-trained weights
                - 'random': Random initialization
                - None: Same as 'random'
        """
        super().__init__()
        
        # Initial projection from GNN output to ResNet input size
        self.feature_proj = nn.Linear(encoder_out_dim * 2, 512)

        # Use a ResNet18 as a powerful feature extractor
        # Configure weights based on pretrained_weights parameter
        if pretrained_weights == 'imagenet':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("  🔧 Actor using ResNet18 with ImageNet weights")
        else:
            resnet = resnet18(weights=None)
            print(f"  🔧 Actor using ResNet18 with random initialization")
        
        # We'll use the body of the ResNet, excluding the final classification layer
        self.resnet_body = nn.Sequential(*list(resnet.children())[:-1])
        
        # Put ResNet in evaluation mode to use pre-trained batch norm stats
        self.resnet_body.eval()

        # Adapt the first conv layer for our "image"
        # ResNet18's first conv layer is conv1: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # We will replace it to accept a single channel input, representing our feature map
        self.resnet_body[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Final MLP heads for actions, taking ResNet's output
        self.action_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim), # ResNet18 output is 512
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # Discrete action head
        self.discrete_head = nn.Linear(hidden_dim, num_discrete_actions)
        
        # Continuous action heads
        self.continuous_mu_head = nn.Linear(hidden_dim, continuous_dim)
        self.continuous_logstd_head = nn.Linear(hidden_dim, continuous_dim)

        # Optional learnable bias to gently encourage spawning early in training
        # Shape [2]: [spawn_bias, delete_bias]
        bias_tensor = torch.tensor([float(spawn_bias_init), 0.0], dtype=torch.float32)
        self.discrete_bias = nn.Parameter(bias_tensor, requires_grad=True)

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
        projected_features = self.feature_proj(combined_features) # [num_nodes, 512]
        
        # Pad to 576 and reshape
        padded_features = F.pad(projected_features, (0, 576 - 512)) # [num_nodes, 576]
        image_like_features = padded_features.view(-1, 1, 24, 24) # [num_nodes, 1, 24, 24]

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

        # Pass through final MLP (stays on original device)
        shared_features = self.action_mlp(shared_features)
        
        # Check for NaN in shared features
        if torch.isnan(shared_features).any():
            print("⚠️  WARNING: NaN detected in Actor shared_features!")
            shared_features = torch.nan_to_num(shared_features, nan=0.0)

        # --- Action Heads ---
        discrete_logits = self.discrete_head(shared_features)
        # Apply optional spawn bias to encourage growth; broadcast to all nodes
        if self.discrete_bias is not None:
            discrete_logits = discrete_logits + self.discrete_bias.unsqueeze(0).expand_as(discrete_logits)
        continuous_mu = self.continuous_mu_head(shared_features)
        continuous_logstd = self.continuous_logstd_head(shared_features)
        
        # Clamp outputs for numerical stability
        discrete_logits = torch.clamp(discrete_logits, -20.0, 20.0)
        continuous_mu = torch.clamp(continuous_mu, -10.0, 10.0)
        continuous_logstd = torch.clamp(continuous_logstd, -10.0, 5.0)

        return discrete_logits, continuous_mu, continuous_logstd


class Critic(nn.Module):
    """
    The Critic network for the Hybrid Actor-Critic agent.
    It takes node and graph features and outputs state values for different reward components.
    """
    def __init__(self, encoder_out_dim, hidden_dim, value_components: List[str], dropout_rate,
                 pretrained_weights='imagenet'):
        """
        Args:
            pretrained_weights: 'imagenet', 'random', or None
                - 'imagenet': Use ImageNet pre-trained weights
                - 'random': Random initialization
                - None: Same as 'random'
        """
        super().__init__()
        self.value_components = value_components

        # Initial projection from GNN output to ResNet input size
        self.feature_proj = nn.Linear(encoder_out_dim, 512)

        # Use a ResNet18 as a powerful feature extractor
        # Configure weights based on pretrained_weights parameter
        if pretrained_weights == 'imagenet':
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("  🔧 Critic using ResNet18 with ImageNet weights")
        else:
            resnet = resnet18(weights=None)
            print(f"  🔧 Critic using ResNet18 with random initialization")
        
        # We'll use the body of the ResNet, excluding the final classification layer
        self.resnet_body = nn.Sequential(*list(resnet.children())[:-1])
        
        # Put ResNet in evaluation mode to use pre-trained batch norm stats
        self.resnet_body.eval()

        # Adapt the first conv layer for our "image"
        self.resnet_body[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Final MLP heads for value prediction, taking ResNet's output
        self.value_mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),  # ResNet18 output is 512
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # Value heads - one per component
        self.value_heads = nn.ModuleDict({
            component: nn.Linear(hidden_dim, 1) for component in value_components
        })

    def forward(self, graph_token):
        device = graph_token.device
        
        # Project and reshape to be "image-like" for ResNet
        projected_features = self.feature_proj(graph_token)  # [batch_size, 512]
        
        # Pad to 576 and reshape
        padded_features = F.pad(projected_features.unsqueeze(0), (0, 576 - 512))  # [1, 576]
        image_like_features = padded_features.view(-1, 1, 24, 24)  # [1, 1, 24, 24]

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
    Decoupled Hybrid Actor-Critic network for the durotaxis environment.
    
    This class orchestrates the GraphInputEncoder, Actor, and Critic modules.
    """
    
    def __init__(self, 
                 encoder: GraphInputEncoder,
                 config_path: str = "config.yaml",
                 **overrides):
        """
        Initialize HybridActorCritic with configuration from YAML file
        """
        super().__init__()
        
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_actor_critic_config()
        
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        self.encoder = encoder
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_discrete_actions = config.get('num_discrete_actions', 2)
        self.continuous_dim = config.get('continuous_dim', 4)
        self.dropout_rate = config.get('dropout_rate', 0.1)

        # Value components configuration
        value_components = config.get('value_components', ['total_value'])
        if value_components is None:
            value_components = ['total_value']
        self.value_components = value_components
        
        # Pretrained weights configuration
        pretrained_weights = config.get('pretrained_weights', 'imagenet')
        print(f"\n📦 Loading Actor-Critic with pretrained weights: {pretrained_weights}")
        
        # Check if WSA (Weight Sharing Attention) is enabled
        wsa_config = config.get('wsa', {})
        self.use_wsa = wsa_config.get('enabled', False)

        # Spawn bias configuration (optional, low-risk exploration boost)
        self.spawn_bias_init = float(config.get('spawn_bias_init', 0.0))

        # Decoupled Actor and Critic
        if self.use_wsa:
            print("\n" + "="*60)
            print("🔄 WSA (Weight Sharing Attention) ENABLED")
            print("="*60)
            
            # Import WSA-enhanced actor
            try:
                from pretrained_fusion import WSAEnhancedActor
                
                self.actor = WSAEnhancedActor(
                    encoder_out_dim=self.encoder.out_dim,
                    hidden_dim=self.hidden_dim,
                    num_discrete_actions=self.num_discrete_actions,
                    continuous_dim=self.continuous_dim,
                    dropout_rate=self.dropout_rate,
                    wsa_config=wsa_config,
                    use_wsa=True
                )
                
                print("✅ WSA-Enhanced Actor initialized successfully")
                print("="*60 + "\n")
                
            except ImportError as e:
                print(f"❌ Failed to import WSA module: {e}")
                print("⚠️  Falling back to standard Actor")
                self.use_wsa = False
                
                self.actor = Actor(
                    encoder_out_dim=self.encoder.out_dim,
                    hidden_dim=self.hidden_dim,
                    num_discrete_actions=self.num_discrete_actions,
                    continuous_dim=self.continuous_dim,
                    dropout_rate=self.dropout_rate,
                    pretrained_weights=pretrained_weights,
                    spawn_bias_init=self.spawn_bias_init
                )
        else:
            print("\n🔧 Using standard Actor (WSA disabled)")
            self.actor = Actor(
                encoder_out_dim=self.encoder.out_dim,
                hidden_dim=self.hidden_dim,
                num_discrete_actions=self.num_discrete_actions,
                continuous_dim=self.continuous_dim,
                dropout_rate=self.dropout_rate,
                pretrained_weights=pretrained_weights,
                spawn_bias_init=self.spawn_bias_init
            )
        
        self.critic = Critic(
            encoder_out_dim=self.encoder.out_dim,
            hidden_dim=self.hidden_dim,
            value_components=self.value_components,
            dropout_rate=self.dropout_rate,
            pretrained_weights=pretrained_weights
        )
        
        # Parameter bounds for continuous actions from config
        spawn_bounds = config.get('spawn_parameter_bounds', {
            'gamma': [0.1, 10.0],
            'alpha': [0.1, 5.0],
            'noise': [0.0, 2.0],
            'theta': [-math.pi, math.pi]
        })
        
        self.register_buffer('action_bounds', torch.tensor([
            spawn_bounds.get('gamma', [0.1, 10.0]),
            spawn_bounds.get('alpha', [0.1, 5.0]),
            spawn_bounds.get('noise', [0.0, 2.0]),
            spawn_bounds.get('theta', [-math.pi, math.pi])
        ]))
        
        self.apply(self._init_weights)
    
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
        # Also handle WSA feature extractors
        if self.use_wsa and hasattr(self.actor, 'feature_extractor'):
            for ptm in self.actor.feature_extractor.ptms:
                if hasattr(ptm, 'backbone'):
                    ptm.backbone.eval()
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
        Forward pass through the hybrid actor-critic network.
        """
        # Extract components from state
        node_features = state_dict['node_features']
        graph_features = state_dict['graph_features']
        edge_features = state_dict['edge_attr']
        edge_index = state_dict['edge_index']
        
        if isinstance(edge_index, tuple):
            src, dst = edge_index
            edge_index_tensor = torch.stack([src, dst], dim=0)
        else:
            edge_index_tensor = edge_index
        
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
        discrete_logits, continuous_mu, continuous_logstd = self.actor(node_tokens, graph_token)
        value_predictions, graph_value_features = self.critic(graph_token)

        # --- Post-processing and Sanitization ---
        # Clamp logits before masking to prevent extreme values
        discrete_logits = torch.clamp(discrete_logits, min=-30.0, max=30.0)
        
        # Apply numerically stable masking and get safe logits
        stable_logits = _safe_masked_logits(discrete_logits, action_mask)
        
        # Clean continuous parameters
        continuous_mu = torch.nan_to_num(continuous_mu, nan=0.0)
        continuous_logstd = torch.nan_to_num(continuous_logstd, nan=0.0)
        continuous_logstd = torch.clamp(continuous_logstd, min=-10.0, max=5.0)
        continuous_std = torch.exp(continuous_logstd)
        
        continuous_mu_bounded = self._apply_bounds(continuous_mu)
        
        # === OUTPUT PREPARATION ===
        output = {
            'encoder_out': encoder_out,
            'discrete_logits': stable_logits,  # Return stabilized logits
            'continuous_mu': continuous_mu_bounded,
            'continuous_std': continuous_std,
            'value_predictions': value_predictions,
            'graph_features': graph_value_features,
        }
        
        if not deterministic:
            # Use Categorical(logits=...) for numerical stability (PyTorch handles softmax internally)
            discrete_dist = torch.distributions.Categorical(logits=stable_logits)
            discrete_actions = discrete_dist.sample()
            discrete_log_probs = discrete_dist.log_prob(discrete_actions)
            
            continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)
            continuous_actions_raw = continuous_dist.sample()
            continuous_log_probs = continuous_dist.log_prob(continuous_actions_raw).sum(dim=-1)
            
            continuous_actions = self._apply_bounds(continuous_actions_raw)
            
            output.update({
                'discrete_actions': discrete_actions,
                'continuous_actions': continuous_actions,
                'discrete_log_probs': discrete_log_probs,
                'continuous_log_probs': continuous_log_probs,
                'total_log_probs': discrete_log_probs + continuous_log_probs
            })
        else:
            output.update({
                'discrete_actions': torch.argmax(stable_logits, dim=-1),
                'continuous_actions': continuous_mu_bounded
            })
        
        return output
    
    def _apply_bounds(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply parameter bounds to continuous actions."""
        bounded_actions = actions.clone()
        
        for i, (min_val, max_val) in enumerate(self.action_bounds):
            if i < actions.shape[-1]:
                if i == 3:  # theta: use tanh for circular bounds
                    bounded_actions[..., i] = torch.tanh(actions[..., i]) * math.pi
                else:  # gamma, alpha, noise: use sigmoid scaling
                    bounded_actions[..., i] = torch.sigmoid(actions[..., i]) * (max_val - min_val) + min_val
        
        return bounded_actions
    
    def _empty_output(self) -> Dict[str, torch.Tensor]:
        """Return empty output for graphs with no nodes."""
        # Get device from action_bounds (a registered buffer)
        device = self.action_bounds.device
        
        value_predictions = {component: torch.tensor(0.0, device=device) for component in self.value_components}
        
        return {
            'encoder_out': torch.empty(1, self.encoder.out_dim, device=device),
            'discrete_logits': torch.empty(0, self.num_discrete_actions, device=device),
            'discrete_probs': torch.empty(0, self.num_discrete_actions, device=device),
            'continuous_mu': torch.empty(0, self.continuous_dim, device=device),
            'continuous_std': torch.empty(0, self.continuous_dim, device=device),
            'value_predictions': value_predictions,
            'discrete_actions': torch.empty(0, dtype=torch.long, device=device),
            'continuous_actions': torch.empty(0, self.continuous_dim, device=device),
            'discrete_log_probs': torch.empty(0, device=device),
            'continuous_log_probs': torch.empty(0, device=device),
            'total_log_probs': torch.empty(0, device=device)
        }
    
    def evaluate_actions(self, state_dict: Dict[str, torch.Tensor], 
                        discrete_actions: torch.Tensor,
                        continuous_actions: torch.Tensor,
                        cached_output: Optional[Dict[str, torch.Tensor]] = None,
                        action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate given actions (used for policy updates).
        """
        if cached_output is not None:
            output = cached_output
        else:
            output = self.forward(state_dict, deterministic=True, action_mask=action_mask)
        
        if output['discrete_logits'].shape[0] == 0:
            # Get device from action_bounds
            device = self.action_bounds.device
            return {
                'discrete_log_probs': torch.empty(0, device=device),
                'continuous_log_probs': torch.empty(0, device=device),
                'total_log_probs': torch.empty(0, device=device),
                'value_predictions': output['value_predictions'],
                'entropy': torch.tensor(0.0, device=device)
            }
        
        # Re-run actor forward pass to get distributions for entropy calculation
        node_tokens = output['encoder_out'][1:]
        graph_token = output['encoder_out'][0]
        discrete_logits, continuous_mu, continuous_logstd = self.actor(node_tokens, graph_token)
        
        # Clamp logits to prevent overflow/underflow
        discrete_logits = torch.clamp(discrete_logits, min=-30.0, max=30.0)
        continuous_logstd = torch.clamp(continuous_logstd, -10, 5)
        continuous_std = torch.exp(continuous_logstd)
        
        # Apply numerically stable masking and get safe logits
        stable_logits = _safe_masked_logits(discrete_logits, action_mask)
        
        # Use Categorical(logits=...) for numerical stability (PyTorch handles softmax internally)
        discrete_dist = torch.distributions.Categorical(logits=stable_logits)
        discrete_log_probs = discrete_dist.log_prob(discrete_actions)
        discrete_entropy = discrete_dist.entropy()
        
        # Evaluate continuous actions
        # Note: We need to invert the bounding function to get the raw action for log_prob
        # This is complex, so for now we approximate by using the unbounded mu.
        # A more accurate implementation would store the raw actions.
        continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)
        # This is an approximation, as continuous_actions are bounded.
        continuous_log_probs = continuous_dist.log_prob(continuous_actions).sum(dim=-1)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1)
        
        total_log_probs = discrete_log_probs + continuous_log_probs
        total_entropy = discrete_entropy + continuous_entropy
        
        return {
            'discrete_log_probs': discrete_log_probs,
            'continuous_log_probs': continuous_log_probs,
            'total_log_probs': total_log_probs,
            'value_predictions': output['value_predictions'],
            'entropy': total_entropy.mean()
        }
    
    def get_topology_actions(self, output: Dict[str, torch.Tensor]) -> Dict[int, str]:
        """
        Convert network output to topology-compatible action format.
        """
        if 'discrete_actions' not in output or output['discrete_actions'].shape[0] == 0:
            return {}
        
        actions = {}
        for node_id, action in enumerate(output['discrete_actions']):
            actions[node_id] = 'spawn' if action.item() == 0 else 'delete'
        
        return actions
    
    def get_spawn_parameters(self, output: Dict[str, torch.Tensor], 
                           node_id: int) -> Tuple[float, float, float, float]:
        """
        Get spawn parameters for a specific node.
        """
        if 'continuous_actions' not in output or node_id >= output['continuous_actions'].shape[0]:
            return (1.0, 1.0, 0.5, 0.0)
        
        params = output['continuous_actions'][node_id]
        return (params[0].item(), params[1].item(), params[2].item(), params[3].item())


class HybridPolicyAgent:
    """
    Agent wrapper for the HybridActorCritic network.
    """
    
    def __init__(self, topology, state_extractor, hybrid_network: HybridActorCritic):
        self.topology = topology
        self.state_extractor = state_extractor
        self.network = hybrid_network
        
    @torch.no_grad()
    def get_actions_and_values(self, deterministic: bool = False,
                              action_mask: Optional[torch.Tensor] = None) -> Tuple[Dict[int, str], 
                                                                         Dict[int, Tuple[float, float, float, float]], 
                                                                         Dict[str, torch.Tensor]]:
        """
        Get actions, spawn parameters, and value predictions.
        """
        state = self.state_extractor.get_state_features(include_substrate=True)
        
        if state['num_nodes'] == 0:
            # Get device from network's action_bounds
            device = self.network.action_bounds.device
            empty_values = {component: torch.tensor(0.0, device=device) for component in self.network.value_components}
            return {}, {}, empty_values
        
        output = self.network(state, deterministic=deterministic, action_mask=action_mask)
        
        actions = self.network.get_topology_actions(output)
        
        spawn_params = {}
        for node_id in range(state['num_nodes']):
            spawn_params[node_id] = self.network.get_spawn_parameters(output, node_id)
        
        return actions, spawn_params, output['value_predictions']
    
    @torch.no_grad()
    def act_with_policy(self, deterministic: bool = False,
                       action_mask: Optional[torch.Tensor] = None) -> Dict[int, str]:
        """
        Execute actions using the hybrid network.
        """
        actions, spawn_params, _ = self.get_actions_and_values(deterministic, action_mask)
        
        spawn_actions = {node_id: action for node_id, action in actions.items() if action == 'spawn'}
        delete_actions = {node_id: action for node_id, action in actions.items() if action == 'delete'}
        
        # First, mark nodes for deletion (set to_delete flag)
        if 'to_delete' in self.topology.graph.ndata and len(delete_actions) > 0:
            # Reset all to_delete flags to 0
            self.topology.graph.ndata['to_delete'] = torch.zeros(
                self.topology.graph.num_nodes(), dtype=torch.float32
            )
            # Set to_delete=1 for nodes that should be deleted
            for node_id in delete_actions.keys():
                if node_id < self.topology.graph.num_nodes():
                    self.topology.graph.ndata['to_delete'][node_id] = 1.0
        
        # Execute spawn actions
        for node_id in spawn_actions:
            gamma, alpha, noise, theta = spawn_params[node_id]
            try:
                self.topology.spawn(node_id, gamma=gamma, alpha=alpha, noise=noise, theta=theta)
            except Exception as e:
                print(f"Failed to spawn from node {node_id}: {e}")
        
        # Execute delete actions (now that to_delete flags are set)
        delete_node_ids = sorted(delete_actions.keys(), reverse=True)
        for node_id in delete_node_ids:
            try:
                self.topology.delete(node_id)
            except Exception as e:
                print(f"Failed to delete node {node_id}: {e}")
        
        return actions


if __name__ == '__main__':
    # Test the HybridActorCritic network
    print("🧪 Testing Decoupled HybridActorCritic Network with ResNet")
    print("=" * 60)
    
    try:
        encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=2)
        
        reward_components = ['total_value', 'graph_value', 'spawn_value', 'node_value', 'edge_value']
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
        print(f"Discrete actions shape: {output_stochastic['discrete_actions'].shape}")
        print(f"Continuous actions shape: {output_stochastic['continuous_actions'].shape}")
        print(f"Value predictions: {list(output_stochastic['value_predictions'].keys())}")
        
        output_deterministic = network(state_dict, deterministic=True)
        print(f"Deterministic forward pass works: ✅")
        
        eval_output = network.evaluate_actions(
            state_dict,
            output_stochastic['discrete_actions'],
            output_stochastic['continuous_actions']
        )
        print(f"Action evaluation works: ✅")
        print(f"Entropy: {eval_output['entropy']:.4f}")
        
        topology_actions = network.get_topology_actions(output_stochastic)
        print(f"Topology actions: {topology_actions}")
        
        if len(topology_actions) > 0:
            node_id = list(topology_actions.keys())[0]
            spawn_params = network.get_spawn_parameters(output_stochastic, node_id)
            print(f"Spawn params for node {node_id}: gamma={spawn_params[0]:.3f}, alpha={spawn_params[1]:.3f}, noise={spawn_params[2]:.3f}, theta={spawn_params[3]:.3f}")
        
        print("\n✅ All tests passed! Decoupled ResNet-based HybridActorCritic is ready.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
