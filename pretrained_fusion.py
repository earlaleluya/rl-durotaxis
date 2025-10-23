"""
Weight Sharing Attention (WSA) Module for Combining Pre-Trained Models

Based on "Combining Pre-Trained Models for Enhanced Feature Representation in RL"
This module integrates multiple pre-trained feature extractors using dynamic attention
to create rich, adaptive state representations for the durotaxis environment.

Key Components:
1. Multiple Pre-Trained Models (PTMs): Different perspectives on the graph state
2. Weight Sharing Attention (WSA): Dynamically weighs PTM contributions
3. State Encoder: Computes context for attention weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from torchvision.models import resnet18, ResNet18_Weights
import math


class StateEncoder(nn.Module):
    """
    Encodes the current state into a context vector used for attention weighting.
    This helps WSA understand what aspects of the state are most relevant.
    """
    def __init__(self, input_dim: int, context_dim: int):
        super().__init__()
        self.context_dim = context_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, context_dim * 2),
            nn.LayerNorm(context_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(context_dim * 2, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU()
        )
    
    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_features: [batch_size, input_dim] - global state features
        Returns:
            context: [batch_size, context_dim] - context vector for attention
        """
        return self.encoder(state_features)


class WeightSharingAttention(nn.Module):
    """
    Weight Sharing Attention (WSA) module that dynamically combines
    multiple pre-trained model embeddings based on current state context.
    
    Key Innovation: Uses a SHARED MLP to compute attention weights for all PTMs,
    conditioned on both the context and each individual embedding.
    """
    def __init__(self, embedding_dim: int, context_dim: int, num_models: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.num_models = num_models
        
        # Shared MLP: Takes [context; embedding] and outputs weight
        # This weight-sharing makes the model efficient and forces consistent weighting logic
        self.shared_mlp = nn.Sequential(
            nn.Linear(context_dim + embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)  # Output: single weight per model
        )
        
        # Optional: Learnable temperature for softmax sharpness
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, context: torch.Tensor, embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: [batch_size, context_dim] - state context
            embeddings: List of [batch_size, embedding_dim] - PTM embeddings
        
        Returns:
            weighted_representation: [batch_size, embedding_dim] - fused representation
            attention_weights: [batch_size, num_models] - weights assigned to each PTM
        """
        batch_size = context.shape[0]
        
        # Compute attention weight for each PTM embedding
        weights = []
        for emb in embeddings:
            # Concatenate context with embedding
            combined = torch.cat([context, emb], dim=-1)  # [batch, context_dim + emb_dim]
            
            # Shared MLP predicts weight
            weight = self.shared_mlp(combined)  # [batch, 1]
            weights.append(weight)
        
        # Stack weights and apply softmax
        weights = torch.cat(weights, dim=-1)  # [batch, num_models]
        attention_weights = F.softmax(weights / self.temperature, dim=-1)  # [batch, num_models]
        
        # Weighted sum of embeddings
        stacked_embeddings = torch.stack(embeddings, dim=1)  # [batch, num_models, emb_dim]
        weighted_representation = torch.sum(
            stacked_embeddings * attention_weights.unsqueeze(-1),  # [batch, num_models, 1]
            dim=1
        )  # [batch, emb_dim]
        
        return weighted_representation, attention_weights


class PreTrainedModelWrapper(nn.Module):
    """
    Wrapper for pre-trained models to extract features from graph states.
    Each PTM provides a different perspective on the environment.
    
    Supported models:
    - resnet18: ResNet18 architecture
    - resnet34: ResNet34 architecture (deeper)
    - resnet50: ResNet50 architecture (wider)
    - efficientnet_b0: EfficientNet-B0 (efficient)
    - graph_cnn: Custom CNN for graph patterns
    
    Supported weights:
    - 'imagenet': Pre-trained on ImageNet
    - 'random': Random initialization
    - None/null: Random initialization
    """
    def __init__(
        self, 
        model_type: str, 
        weights: Optional[str], 
        output_dim: int, 
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.model_type = model_type
        self.weights = weights
        self.output_dim = output_dim
        
        # Import additional models as needed
        from torchvision.models import (
            resnet18, resnet34, resnet50,
            ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
        )
        
        # Initialize backbone based on model type and weights
        if model_type == 'resnet18':
            if weights == 'imagenet':
                resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:  # random or None
                resnet = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            backbone_out_dim = 512
            
        elif model_type == 'resnet34':
            if weights == 'imagenet':
                resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet34(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            backbone_out_dim = 512
            
        elif model_type == 'resnet50':
            if weights == 'imagenet':
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.backbone[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            backbone_out_dim = 2048
            
        elif model_type == 'graph_cnn':
            # Simple CNN for graph structure patterns
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            backbone_out_dim = 256
            freeze_backbone = False  # Always train graph_cnn
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported: resnet18, resnet34, resnet50, graph_cnn")
        
        # Freeze backbone if requested (only for pre-trained models)
        if freeze_backbone and model_type.startswith('resnet') and weights == 'imagenet':
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  🔒 Frozen {model_type} backbone (ImageNet weights)")
        elif model_type.startswith('resnet'):
            print(f"  🔓 Trainable {model_type} backbone ({weights or 'random'} weights)")
        
        # Projection head to common embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, height, width] - image-like features
        Returns:
            embedding: [batch_size, output_dim] - extracted features
        """
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        embedding = self.projection(features)
        return embedding


class MultiPTMFeatureExtractor(nn.Module):
    """
    Combines multiple pre-trained models using Weight Sharing Attention
    to create enhanced feature representations for the durotaxis RL agent.
    
    Architecture:
    1. Graph state → Multiple PTMs → Diverse embeddings
    2. Graph features → State Encoder → Context vector
    3. WSA(context, embeddings) → Weighted fusion → Rich representation
    
    Can be configured via YAML or direct parameters.
    """
    def __init__(
        self,
        input_channels: int = 1,
        input_height: int = 24,
        input_width: int = 24,
        graph_feature_dim: int = 14,
        embedding_dim: int = 256,
        context_dim: int = 128,
        ptm_configs: Optional[List[Dict]] = None,
        config_dict: Optional[Dict] = None  # Load from YAML config
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        
        # Load from config_dict if provided (YAML configuration)
        if config_dict is not None:
            embedding_dim = config_dict.get('embedding_dim', 256)
            context_dim = config_dict.get('context_dim', 128)
            
            # Parse pretrained_models from config
            ptm_config_list = config_dict.get('pretrained_models', [])
            ptm_configs = []
            
            for ptm_cfg in ptm_config_list:
                ptm_configs.append({
                    'name': ptm_cfg.get('name', 'unknown'),
                    'model_type': ptm_cfg.get('model_type'),
                    'weights': ptm_cfg.get('weights'),
                    'freeze': ptm_cfg.get('freeze_backbone', True)
                })
            
            print(f"📦 Loading WSA with {len(ptm_configs)} pre-trained models from config")
        
        # Default PTM configuration if none provided
        if ptm_configs is None or len(ptm_configs) == 0:
            ptm_configs = [
                {
                    'name': 'resnet18_imagenet',
                    'model_type': 'resnet18',
                    'weights': 'imagenet',
                    'freeze': True
                },
                {
                    'name': 'graph_cnn',
                    'model_type': 'graph_cnn',
                    'weights': None,
                    'freeze': False
                },
                {
                    'name': 'resnet18_random',
                    'model_type': 'resnet18',
                    'weights': 'random',
                    'freeze': False
                }
            ]
            print(f"📦 Using default WSA configuration with {len(ptm_configs)} models")
        
        # Initialize Pre-Trained Models
        print("🔧 Initializing Pre-Trained Models:")
        self.ptms = nn.ModuleList()
        self.ptm_names = []
        
        for config in ptm_configs:
            name = config.get('name', config['model_type'])
            self.ptm_names.append(name)
            
            print(f"  • {name}: {config['model_type']} ({config.get('weights', 'random')})")
            
            ptm = PreTrainedModelWrapper(
                model_type=config['model_type'],
                weights=config.get('weights'),
                output_dim=embedding_dim,
                freeze_backbone=config.get('freeze', True)
            )
            self.ptms.append(ptm)
        
        self.num_models = len(self.ptms)
        print(f"✅ Initialized {self.num_models} pre-trained models")
        
        # State Encoder: Converts graph features to context
        self.state_encoder = StateEncoder(
            input_dim=graph_feature_dim,
            context_dim=context_dim
        )
        
        # Weight Sharing Attention: Dynamically combines PTM outputs
        self.wsa = WeightSharingAttention(
            embedding_dim=embedding_dim,
            context_dim=context_dim,
            num_models=self.num_models
        )
        
        # Optional: Additional refinement layer
        self.refinement = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        graph_features: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            image_features: [batch_size, channels, height, width] - visual state
            graph_features: [batch_size, graph_feature_dim] - graph-level features
            return_attention: If True, return (representation, attention_weights)
        
        Returns:
            representation: [batch_size, embedding_dim] - fused representation
            or (representation, attention_weights) if return_attention=True
        """
        batch_size = image_features.shape[0]
        
        # Step 1: Extract embeddings from all PTMs
        embeddings = []
        for ptm in self.ptms:
            emb = ptm(image_features)  # [batch, embedding_dim]
            embeddings.append(emb)
        
        # Step 2: Compute state context
        context = self.state_encoder(graph_features)  # [batch, context_dim]
        
        # Step 3: Apply Weight Sharing Attention
        fused_representation, attention_weights = self.wsa(context, embeddings)
        
        # Step 4: Optional refinement
        refined_representation = self.refinement(fused_representation)
        
        if return_attention:
            return refined_representation, attention_weights
        return refined_representation


class WSAEnhancedActor(nn.Module):
    """
    Enhanced Actor using WSA-based multi-PTM feature extraction.
    Replaces the simple ResNet backbone with the WSA architecture.
    
    Can be configured via YAML or direct parameters for ablation studies.
    """
    def __init__(
        self,
        encoder_out_dim: int,
        hidden_dim: int,
        num_discrete_actions: int,
        continuous_dim: int,
        dropout_rate: float,
        wsa_config: Optional[Dict] = None,
        embedding_dim: int = 256,
        use_wsa: bool = True
    ):
        super().__init__()
        self.use_wsa = use_wsa
        self.encoder_out_dim = encoder_out_dim
        
        # Initial projection from GNN output to input size
        # Combined features = node_tokens + graph_token = encoder_out_dim * 2
        self.feature_proj = nn.Linear(encoder_out_dim * 2, 512)
        
        if use_wsa:
            # Load embedding_dim from config if provided
            if wsa_config is not None:
                embedding_dim = wsa_config.get('embedding_dim', 256)
            
            # WSA-based multi-PTM feature extraction
            self.feature_extractor = MultiPTMFeatureExtractor(
                input_channels=1,
                input_height=24,
                input_width=24,
                graph_feature_dim=encoder_out_dim,  # Will use graph token
                embedding_dim=embedding_dim,
                context_dim=wsa_config.get('context_dim', 128) if wsa_config else 128,
                config_dict=wsa_config  # Pass full config for PTM initialization
            )
            feature_out_dim = embedding_dim
            
            # Store config for logging
            self.wsa_config = wsa_config
            self.log_attention = wsa_config.get('log_attention_weights', False) if wsa_config else False
            
        else:
            # Original single ResNet (for ablation comparison)
            from torchvision.models import resnet18, ResNet18_Weights
            
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.resnet_body = nn.Sequential(*list(resnet.children())[:-1])
            for param in self.resnet_body.parameters():
                param.requires_grad = False
            self.resnet_body.eval()
            self.resnet_body[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            feature_out_dim = 512
            
            self.log_attention = False
        
        self.embedding_dim = embedding_dim if use_wsa else feature_out_dim
        
        # Action heads
        self.action_mlp = nn.Sequential(
            nn.Linear(feature_out_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.discrete_head = nn.Linear(hidden_dim, num_discrete_actions)
        self.continuous_mu_head = nn.Linear(hidden_dim, continuous_dim)
        self.continuous_logstd_head = nn.Linear(hidden_dim, continuous_dim)
    
    def forward(self, node_tokens, graph_token, return_attention=False):
        """
        Args:
            node_tokens: [num_nodes, encoder_out_dim]
            graph_token: [encoder_out_dim]
            return_attention: Whether to return attention weights
        """
        num_nodes = node_tokens.shape[0]
        device = node_tokens.device
        
        graph_context = graph_token.unsqueeze(0).repeat(num_nodes, 1)
        combined_features = torch.cat([node_tokens, graph_context], dim=-1)
        
        # Project and reshape
        projected_features = self.feature_proj(combined_features)
        padded_features = F.pad(projected_features, (0, 576 - 512))
        image_like_features = padded_features.view(-1, 1, 24, 24)
        
        # Extract features
        if self.use_wsa:
            # Use graph token as graph features for context
            graph_features_batch = graph_token.unsqueeze(0).repeat(num_nodes, 1)
            
            if return_attention:
                shared_features, attention_weights = self.feature_extractor(
                    image_like_features,
                    graph_features_batch,
                    return_attention=True
                )
            else:
                shared_features = self.feature_extractor(
                    image_like_features,
                    graph_features_batch,
                    return_attention=False
                )
        else:
            # Original ResNet processing
            resnet_out = self.resnet_body(image_like_features)
            shared_features = resnet_out.view(num_nodes, -1)
        
        # Action heads
        shared_features = self.action_mlp(shared_features)
        discrete_logits = self.discrete_head(shared_features)
        continuous_mu = self.continuous_mu_head(shared_features)
        continuous_logstd = self.continuous_logstd_head(shared_features)
        
        if return_attention and self.use_wsa:
            return discrete_logits, continuous_mu, continuous_logstd, attention_weights
        return discrete_logits, continuous_mu, continuous_logstd


if __name__ == '__main__':
    print("🧪 Testing Multi-PTM Feature Extractor with WSA")
    print("=" * 60)
    
    # Test parameters
    batch_size = 8
    num_nodes = 5
    
    # Create test inputs
    image_features = torch.randn(num_nodes, 1, 24, 24)
    graph_features = torch.randn(num_nodes, 14)
    
    # Initialize WSA feature extractor
    print("\n✓ Initializing Multi-PTM Feature Extractor...")
    feature_extractor = MultiPTMFeatureExtractor(
        input_channels=1,
        input_height=24,
        input_width=24,
        graph_feature_dim=14,
        embedding_dim=256,
        context_dim=128
    )
    
    print(f"  - Number of PTMs: {feature_extractor.num_models}")
    print(f"  - Embedding dimension: {feature_extractor.embedding_dim}")
    print(f"  - Context dimension: {feature_extractor.context_dim}")
    
    # Forward pass
    print("\n✓ Running forward pass...")
    representation, attention_weights = feature_extractor(
        image_features,
        graph_features,
        return_attention=True
    )
    
    print(f"\n✅ Output representation shape: {representation.shape}")
    print(f"✅ Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights (per node):")
    for i in range(min(3, num_nodes)):
        weights = attention_weights[i].detach().numpy()
        print(f"  Node {i}: {weights} (sum={weights.sum():.4f})")
    
    # Test WSA-enhanced actor
    print("\n" + "=" * 60)
    print("✓ Testing WSA-Enhanced Actor...")
    
    actor = WSAEnhancedActor(
        encoder_out_dim=64,
        hidden_dim=128,
        num_discrete_actions=2,
        continuous_dim=4,
        dropout_rate=0.1,
        embedding_dim=256,
        use_wsa=True
    )
    
    node_tokens = torch.randn(num_nodes, 64)
    graph_token = torch.randn(64)
    
    discrete_logits, continuous_mu, continuous_logstd, attn = actor(
        node_tokens, graph_token, return_attention=True
    )
    
    print(f"\n✅ Discrete logits shape: {discrete_logits.shape}")
    print(f"✅ Continuous mu shape: {continuous_mu.shape}")
    print(f"✅ Attention weights shape: {attn.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! WSA architecture is ready.")
    print("=" * 60)
