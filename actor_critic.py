"""
Hybrid Actor-Critic Network for Durotaxis Environment

This module implements a hybrid actor-critic architecture that handles:
1. Discrete actions: spawn/delete decisions per node
2. Continuous actions: spawn parameters (gamma, alpha, noise, theta)
3. Multi-component value estimation for different reward components
4. Graph neural network integration via GraphInputEncoder

The architecture is designed to work with the durotaxis environment's
reward component dictionary structure for flexible learning updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
from encoder import GraphInputEncoder
from config_loader import ConfigLoader


class HybridActorCritic(nn.Module):
    """
    Hybrid Actor-Critic network for the durotaxis environment.
    
    Features:
    - Graph-based state representation using GraphInputEncoder
    - Discrete action head for spawn/delete decisions
    - Continuous action head for spawn parameters
    - Multi-head critic for different reward components
    - Node-level action prediction with graph context
    
    Parameters
    ----------
    encoder : GraphInputEncoder
        Pre-trained or fresh graph neural network encoder
    hidden_dim : int, default=128
        Hidden dimension for MLP layers
    num_discrete_actions : int, default=2
        Number of discrete actions (spawn=0, delete=1)
    continuous_dim : int, default=4
        Dimension of continuous action space (gamma, alpha, noise, theta)
    value_components : List[str], optional
        List of reward components to predict separately
        If None, predicts only total value
    dropout_rate : float, default=0.1
        Dropout rate for regularization
    """
    
    def __init__(self, 
                 encoder: GraphInputEncoder,
                 config_path: str = "config.yaml",
                 **overrides):
        """
        Initialize HybridActorCritic with configuration from YAML file
        
        Parameters
        ----------
        encoder : GraphInputEncoder
            Pre-configured graph input encoder
        config_path : str
            Path to configuration YAML file
        **overrides
            Parameter overrides for any configuration values
        """
        super().__init__()
        
        # Load configuration
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_actor_critic_config()
        
        # Apply overrides
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
        self.num_value_heads = len(value_components)
        
        # Shared feature extractor for node-level processing
        # Input: [node_features + graph_context] = [encoder.out_dim * 2]
        self.shared_node_mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
        )
        
        # === ACTOR HEADS ===
        
        # Discrete action head (spawn/delete per node)
        self.discrete_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.num_discrete_actions)
        )
        
        # Continuous action heads (spawn parameters)
        # Mean values for continuous actions
        self.continuous_mu_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.continuous_dim)
        )
        
        # Log standard deviation for continuous actions
        self.continuous_logstd_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, self.continuous_dim)
        )
        
        # === CRITIC HEADS ===
        
        # Graph-level value estimation using graph token
        self.graph_value_mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim // 2),
        )
        
        # Multi-component value heads
        self.value_heads = nn.ModuleDict()
        for component in self.value_components:
            self.value_heads[component] = nn.Linear(self.hidden_dim // 2, 1)
        
        # Initialize continuous action heads for better starting values
        self._init_continuous_heads()
        
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
    
    def _init_continuous_heads(self):
        """Initialize continuous action heads to produce reasonable starting values."""
        # Initialize mu head to output values that give mid-range results after sigmoid/tanh
        # For sigmoid: output = 0 gives sigmoid(0) = 0.5, which maps to mid-range
        # For tanh: output = 0 gives tanh(0) = 0, which maps to theta = 0
        
        # Initialize mu head final layer with positive bias to push away from boundary values
        nn.init.normal_(self.continuous_mu_head[-1].weight, mean=0.0, std=0.3)
        # Set bias to positive values to get mid-range after sigmoid
        # sigmoid(1.0) ≈ 0.73, which gives good mid-range values
        init_bias = torch.tensor([1.0, 1.0, 1.0, 0.0])  # gamma, alpha, noise, theta
        self.continuous_mu_head[-1].bias.data.copy_(init_bias)
        
        # Initialize logstd head to produce reasonable initial variance
        nn.init.normal_(self.continuous_logstd_head[-1].weight, mean=0.0, std=0.1)
        nn.init.constant_(self.continuous_logstd_head[-1].bias, -0.3)  # log(0.74) ≈ -0.3

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
        
        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            State dictionary from state_extractor.get_state_features()
            Must contain: node_features, graph_features, edge_attr, edge_index
        deterministic : bool, default=False
            Whether to use deterministic actions (for evaluation)
        action_mask : torch.Tensor, optional
            Boolean mask tensor of shape [num_nodes, num_discrete_actions]
            True for valid actions, False for invalid actions
            Invalid actions will have their logits set to -inf
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - discrete_logits: [num_nodes, num_discrete_actions]
            - discrete_probs: [num_nodes, num_discrete_actions] 
            - continuous_mu: [num_nodes, continuous_dim]
            - continuous_std: [num_nodes, continuous_dim]
            - value_predictions: Dict[str, torch.Tensor] for each component
            - actions (if sampled): discrete_actions, continuous_actions
            - log_probs (if sampled): discrete_log_probs, continuous_log_probs
        """
        # Extract components from state
        node_features = state_dict['node_features']
        graph_features = state_dict['graph_features']
        edge_features = state_dict['edge_attr']
        edge_index = state_dict['edge_index']
        
        # Convert edge_index from DGL tuple format to PyG tensor format if needed
        if isinstance(edge_index, tuple):
            src, dst = edge_index
            edge_index_tensor = torch.stack([src, dst], dim=0)  # [2, num_edges]
        else:
            edge_index_tensor = edge_index
        
        # Handle empty graphs
        num_nodes = node_features.shape[0]
        if num_nodes == 0:
            return self._empty_output()
        
        # Get embeddings from encoder
        encoder_out = self.encoder(
            graph_features=graph_features,
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index_tensor
        )  # [num_nodes+1, out_dim]
        
        # Split encoder output: first token is graph, rest are nodes
        graph_token = encoder_out[0]  # [out_dim]
        node_tokens = encoder_out[1:]   # [num_nodes, out_dim]
        
        # === ACTOR PROCESSING ===
        
        # Broadcast graph token to all nodes for context
        graph_context = graph_token.unsqueeze(0).repeat(num_nodes, 1)  # [num_nodes, out_dim]
        
        # Combine node tokens with graph context
        combined_features = torch.cat([node_tokens, graph_context], dim=-1)  # [num_nodes, out_dim*2]
        shared_features = self.shared_node_mlp(combined_features)  # [num_nodes, hidden_dim]
        
        # Discrete action logits and probabilities
        discrete_logits = self.discrete_head(shared_features)  # [num_nodes, num_discrete_actions]
        
        # Apply action masking to prevent invalid moves
        if action_mask is not None:
            # Ensure mask has correct shape
            if action_mask.shape != discrete_logits.shape:
                raise ValueError(f"Action mask shape {action_mask.shape} doesn't match logits shape {discrete_logits.shape}")
            
            # Set invalid action logits to -inf (will become 0 probability after softmax)
            discrete_logits = discrete_logits.masked_fill(~action_mask, -float('inf'))
        
        # Sanitize discrete_logits before softmax to prevent NaN
        # Clamp to prevent extreme values that cause softmax to produce NaN
        discrete_logits = torch.clamp(discrete_logits, min=-20.0, max=20.0)
        
        # Check for NaN/Inf in logits before softmax
        if torch.isnan(discrete_logits).any() or torch.isinf(discrete_logits).any():
            print(f"⚠️ WARNING: Invalid discrete_logits detected before softmax")
            discrete_logits = torch.where(torch.isnan(discrete_logits) | torch.isinf(discrete_logits), 
                                         torch.zeros_like(discrete_logits), 
                                         discrete_logits)
        
        discrete_probs = F.softmax(discrete_logits, dim=-1)
        
        # Additional sanitization: if softmax still produces NaN/Inf, use uniform distribution
        if torch.isnan(discrete_probs).any() or torch.isinf(discrete_probs).any():
            print(f"⚠️ WARNING: NaN/Inf in discrete_probs after softmax, using uniform distribution")
            num_actions = discrete_probs.shape[-1]
            discrete_probs = torch.ones_like(discrete_probs) / num_actions
        
        # Continuous action parameters
        continuous_mu = self.continuous_mu_head(shared_features)  # [num_nodes, continuous_dim]
        continuous_logstd = self.continuous_logstd_head(shared_features)  # [num_nodes, continuous_dim]
        
        # Sanitize continuous parameters to prevent NaN
        if torch.isnan(continuous_mu).any() or torch.isinf(continuous_mu).any():
            print(f"⚠️ WARNING: NaN/Inf detected in continuous_mu, replacing with zeros")
            continuous_mu = torch.where(torch.isnan(continuous_mu) | torch.isinf(continuous_mu),
                                       torch.zeros_like(continuous_mu),
                                       continuous_mu)
        
        if torch.isnan(continuous_logstd).any() or torch.isinf(continuous_logstd).any():
            print(f"⚠️ WARNING: NaN/Inf detected in continuous_logstd, replacing with zeros")
            continuous_logstd = torch.where(torch.isnan(continuous_logstd) | torch.isinf(continuous_logstd),
                                           torch.zeros_like(continuous_logstd),
                                           continuous_logstd)
        
        continuous_std = torch.exp(torch.clamp(continuous_logstd, -5, 2))  # Prevent extreme values
        
        # Apply parameter bounds to continuous actions
        continuous_mu_bounded = self._apply_bounds(continuous_mu)
        
        # === CRITIC PROCESSING ===
        
        # Handle batched vs single graph processing for value predictions
        if graph_features.dim() == 1:
            # Single graph case
            graph_value_features = self.graph_value_mlp(graph_token)  # [hidden_dim//2]
            
            # Multi-component value predictions
            value_predictions = {}
            for component in self.value_components:
                value_predictions[component] = self.value_heads[component](graph_value_features).squeeze(-1)
        else:
            # Batched graph case - we need to process each graph separately
            # since the current architecture expects single graph tokens
            batch_size = graph_features.shape[0]
            
            # For now, use the mean graph token for all graphs (simplified approach)
            # This works because graph_token is already the mean of all graph embeddings
            graph_value_features = self.graph_value_mlp(graph_token)  # [hidden_dim//2]
            
            # Multi-component value predictions - replicate for each graph in batch
            value_predictions = {}
            for component in self.value_components:
                # Get single value and replicate for each graph in batch
                single_value = self.value_heads[component](graph_value_features).squeeze(-1)  # scalar
                # Replicate to create one value per graph in batch
                values_batch = single_value.unsqueeze(0).repeat(batch_size)  # [batch_size]
                value_predictions[component] = values_batch
        
        # === OUTPUT PREPARATION ===
        
        output = {
            'encoder_out': encoder_out,
            'discrete_logits': discrete_logits,
            'discrete_probs': discrete_probs,
            'continuous_mu': continuous_mu_bounded,
            'continuous_std': continuous_std,
            'value_predictions': value_predictions,
            'graph_features': graph_value_features,
            'shared_node_features': shared_features
        }
        
        # Sample actions if not deterministic
        if not deterministic:
            # Sample discrete actions
            discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
            discrete_actions = discrete_dist.sample()  # [num_nodes]
            discrete_log_probs = discrete_dist.log_prob(discrete_actions)  # [num_nodes]
            
            # Sample continuous actions from unbounded distribution, then apply bounds once
            continuous_dist = torch.distributions.Normal(continuous_mu, continuous_std)
            continuous_actions_raw = continuous_dist.sample()  # [num_nodes, continuous_dim]
            continuous_log_probs = continuous_dist.log_prob(continuous_actions_raw).sum(dim=-1)  # [num_nodes]
            
            # Apply bounds to sampled continuous actions
            continuous_actions = self._apply_bounds(continuous_actions_raw)
            
            output.update({
                'discrete_actions': discrete_actions,
                'continuous_actions': continuous_actions,
                'discrete_log_probs': discrete_log_probs,
                'continuous_log_probs': continuous_log_probs,
                'total_log_probs': discrete_log_probs + continuous_log_probs
            })
        else:
            # Deterministic actions
            discrete_actions = torch.argmax(discrete_logits, dim=-1)
            continuous_actions = continuous_mu_bounded
            
            output.update({
                'discrete_actions': discrete_actions,
                'continuous_actions': continuous_actions
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
        value_predictions = {component: torch.tensor(0.0) for component in self.value_components}
        
        return {
            'encoder_out': torch.empty(1, self.encoder.out_dim),  # Just graph token
            'discrete_logits': torch.empty(0, self.num_discrete_actions),
            'discrete_probs': torch.empty(0, self.num_discrete_actions),
            'continuous_mu': torch.empty(0, self.continuous_dim),
            'continuous_std': torch.empty(0, self.continuous_dim),
            'value_predictions': value_predictions,
            'discrete_actions': torch.empty(0, dtype=torch.long),
            'continuous_actions': torch.empty(0, self.continuous_dim),
            'discrete_log_probs': torch.empty(0),
            'continuous_log_probs': torch.empty(0),
            'total_log_probs': torch.empty(0)
        }
    
    def evaluate_actions(self, state_dict: Dict[str, torch.Tensor], 
                        discrete_actions: torch.Tensor,
                        continuous_actions: torch.Tensor,
                        cached_output: Optional[Dict[str, torch.Tensor]] = None,
                        action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate given actions (used for policy updates).
        
        Parameters
        ----------
        state_dict : Dict[str, torch.Tensor]
            State dictionary
        discrete_actions : torch.Tensor
            Discrete actions to evaluate [num_nodes]
        continuous_actions : torch.Tensor
            Continuous actions to evaluate [num_nodes, continuous_dim]
        cached_output : Dict[str, torch.Tensor], optional
            Pre-computed forward pass output to avoid recomputation
            If None, will compute forward pass
        action_mask : torch.Tensor, optional
            Boolean mask tensor for valid actions [num_nodes, num_discrete_actions]
            Only used if cached_output is None
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing log probabilities and value predictions
        """
        # Use cached output if available, otherwise compute forward pass
        if cached_output is not None:
            output = cached_output
        else:
            output = self.forward(state_dict, deterministic=True, action_mask=action_mask)
        
        if output['discrete_logits'].shape[0] == 0:
            # Handle empty graph case
            return {
                'discrete_log_probs': torch.empty(0),
                'continuous_log_probs': torch.empty(0),
                'total_log_probs': torch.empty(0),
                'value_predictions': output['value_predictions'],
                'entropy': torch.tensor(0.0)
            }
        
        # Evaluate discrete actions
        # Final safety check before creating Categorical distribution
        discrete_probs = output['discrete_probs']
        if torch.isnan(discrete_probs).any() or torch.isinf(discrete_probs).any():
            print(f"⚠️ CRITICAL: NaN/Inf in discrete_probs at evaluate_actions, using uniform distribution")
            num_actions = discrete_probs.shape[-1]
            discrete_probs = torch.ones_like(discrete_probs) / num_actions
        
        # Ensure probs sum to 1 (numerical stability)
        discrete_probs = discrete_probs / (discrete_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        discrete_dist = torch.distributions.Categorical(probs=discrete_probs)
        discrete_log_probs = discrete_dist.log_prob(discrete_actions)
        discrete_entropy = discrete_dist.entropy()
        
        # Evaluate continuous actions
        continuous_dist = torch.distributions.Normal(output['continuous_mu'], output['continuous_std'])
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
        
        Parameters
        ----------
        output : Dict[str, torch.Tensor]
            Output from forward() call
            
        Returns
        -------
        Dict[int, str]
            Dictionary mapping node_id to action string ('spawn' or 'delete')
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
        
        Parameters
        ----------
        output : Dict[str, torch.Tensor]
            Output from forward() call
        node_id : int
            ID of the node to get parameters for
            
        Returns
        -------
        Tuple[float, float, float, float]
            (gamma, alpha, noise, theta) parameters for topology.spawn()
        """
        if 'continuous_actions' not in output or node_id >= output['continuous_actions'].shape[0]:
            # Return default parameters
            return (1.0, 1.0, 0.5, 0.0)
        
        params = output['continuous_actions'][node_id]
        return (params[0].item(), params[1].item(), params[2].item(), params[3].item())


class HybridPolicyAgent:
    """
    Agent wrapper for the HybridActorCritic network.
    
    Provides high-level interface for action selection and execution
    compatible with the existing durotaxis environment structure.
    """
    
    def __init__(self, topology, state_extractor, hybrid_network: HybridActorCritic):
        self.topology = topology
        self.state_extractor = state_extractor
        self.network = hybrid_network
        
    def get_actions_and_values(self, deterministic: bool = False,
                              action_mask: Optional[torch.Tensor] = None) -> Tuple[Dict[int, str], 
                                                                         Dict[int, Tuple[float, float, float, float]], 
                                                                         Dict[str, torch.Tensor]]:
        """
        Get actions, spawn parameters, and value predictions.
        
        Parameters
        ----------
        deterministic : bool, default=False
            Whether to use deterministic action selection
        action_mask : torch.Tensor, optional
            Boolean mask tensor for valid actions [num_nodes, num_discrete_actions]
            
        Returns
        -------
        Tuple containing:
            - actions: Dict[int, str] - Node actions ('spawn' or 'delete')
            - spawn_params: Dict[int, Tuple] - Spawn parameters per node
            - values: Dict[str, torch.Tensor] - Value predictions
        """
        # Get current state
        state = self.state_extractor.get_state_features(include_substrate=True)
        
        if state['num_nodes'] == 0:
            empty_values = {component: torch.tensor(0.0) for component in self.network.value_components}
            return {}, {}, empty_values
        
        # Forward pass through network
        output = self.network(state, deterministic=deterministic, action_mask=action_mask)
        
        # Convert to topology format
        actions = self.network.get_topology_actions(output)
        
        # Get spawn parameters for each node
        spawn_params = {}
        for node_id in range(state['num_nodes']):
            spawn_params[node_id] = self.network.get_spawn_parameters(output, node_id)
        
        return actions, spawn_params, output['value_predictions']
    
    def act_with_policy(self, deterministic: bool = False,
                       action_mask: Optional[torch.Tensor] = None) -> Dict[int, str]:
        """
        Execute actions using the hybrid network (compatible with existing interface).
        
        Parameters
        ----------
        deterministic : bool, default=False
            Whether to use deterministic action selection
        action_mask : torch.Tensor, optional
            Boolean mask tensor for valid actions [num_nodes, num_discrete_actions]
            
        Returns
        -------
        Dict[int, str]
            Executed actions
        """
        actions, spawn_params, _ = self.get_actions_and_values(deterministic, action_mask)
        
        # Execute actions (same logic as TopologyPolicyAgent)
        spawn_actions = {node_id: action for node_id, action in actions.items() if action == 'spawn'}
        delete_actions = {node_id: action for node_id, action in actions.items() if action == 'delete'}
        
        # Execute spawns first
        for node_id in spawn_actions:
            gamma, alpha, noise, theta = spawn_params[node_id]
            try:
                self.topology.spawn(node_id, gamma=gamma, alpha=alpha, noise=noise, theta=theta)
            except Exception as e:
                print(f"Failed to spawn from node {node_id}: {e}")
        
        # Execute deletions in reverse order
        delete_node_ids = sorted(delete_actions.keys(), reverse=True)
        for node_id in delete_node_ids:
            try:
                self.topology.delete(node_id)
            except Exception as e:
                print(f"Failed to delete node {node_id}: {e}")
        
        return actions


if __name__ == '__main__':
    # Test the HybridActorCritic network
    print("🧪 Testing HybridActorCritic Network")
    print("=" * 50)
    
    try:
        # Create test encoder
        encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=2)
        
        # Test with different value component configurations
        
        # 1. Single value head (traditional)
        print("\n1. Testing single value head...")
        hybrid_single = HybridActorCritic(
            encoder=encoder,
            hidden_dim=128,
            value_components=['total_value']
        )
        
        # 2. Multi-component value heads (for reward components)
        print("2. Testing multi-component value heads...")
        reward_components = ['total_value', 'graph_value', 'spawn_value', 'node_value', 'edge_value']
        hybrid_multi = HybridActorCritic(
            encoder=encoder,
            hidden_dim=128,
            value_components=reward_components
        )
        
        # Create dummy state data with correct dimensions
        num_nodes = 5
        num_edges = 4
        
        state_dict = {
            'node_features': torch.randn(num_nodes, 9),  # 9 node features expected
            'graph_features': torch.randn(14),  # 14 graph features expected
            'edge_attr': torch.randn(num_edges, 3),  # 3 edge features expected
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }
        
        print(f"\nTest state: {num_nodes} nodes, {num_edges} edges")
        
        # Test forward pass
        print("\n3. Testing forward pass...")
        
        for name, network in [("Single", hybrid_single), ("Multi", hybrid_multi)]:
            print(f"\n--- {name} Value Head Network ---")
            
            # Stochastic forward pass
            output_stochastic = network(state_dict, deterministic=False)
            print(f"Discrete actions shape: {output_stochastic['discrete_actions'].shape}")
            print(f"Continuous actions shape: {output_stochastic['continuous_actions'].shape}")
            print(f"Value predictions: {list(output_stochastic['value_predictions'].keys())}")
            
            # Deterministic forward pass
            output_deterministic = network(state_dict, deterministic=True)
            print(f"Deterministic mode works: ✅")
            
            # Test action evaluation
            eval_output = network.evaluate_actions(
                state_dict,
                output_stochastic['discrete_actions'],
                output_stochastic['continuous_actions']
            )
            print(f"Action evaluation works: ✅")
            print(f"Entropy: {eval_output['entropy']:.4f}")
            
            # Test topology action conversion
            topology_actions = network.get_topology_actions(output_stochastic)
            print(f"Topology actions: {topology_actions}")
            
            # Test spawn parameter extraction
            if len(topology_actions) > 0:
                node_id = list(topology_actions.keys())[0]
                spawn_params = network.get_spawn_parameters(output_stochastic, node_id)
                print(f"Spawn params for node {node_id}: gamma={spawn_params[0]:.3f}, alpha={spawn_params[1]:.3f}, noise={spawn_params[2]:.3f}, theta={spawn_params[3]:.3f}")
        
        # Test empty graph handling
        print("\n4. Testing empty graph handling...")
        empty_state = {
            'node_features': torch.empty(0, 9),  # 9 node features
            'graph_features': torch.randn(14),  # 14 graph features
            'edge_attr': torch.empty(0, 3),  # 3 edge features
            'edge_index': torch.empty(2, 0, dtype=torch.long),
            'num_nodes': 0,
            'num_edges': 0
        }
        
        empty_output = hybrid_multi(empty_state, deterministic=False)
        print(f"Empty graph handling: ✅")
        print(f"Empty actions shape: {empty_output['discrete_actions'].shape}")
        
        print(f"\n✅ All tests passed! HybridActorCritic is ready for use.")
        print(f"💡 Key features:")
        print(f"   - Discrete + continuous action spaces")
        print(f"   - Multi-component value estimation")
        print(f"   - Graph neural network integration")
        print(f"   - Compatible with existing durotaxis environment")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
