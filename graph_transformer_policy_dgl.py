import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATv2Conv


class GraphTransformerEncoder(nn.Module):
    """
    Graph encoder using stacked GATv2Conv layers.
    Returns (graph_embedding, node_embeddings).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_lin = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_feats=hidden_dim,
                out_feats=hidden_dim // num_heads,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                residual=True,
                activation=None,  # add ReLU after LN
                allow_zero_in_degree=True,  # Fix: Allow nodes with 0 in-degree
            )
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: DGLGraph
            node_feats: [num_nodes, in_dim]

        Returns:
            graph_emb: [batch_size, hidden_dim]
            node_emb: [num_nodes, hidden_dim]
        """
        # Handle empty graphs
        if node_feats.shape[0] == 0:
            return torch.zeros(1, self.input_lin.out_features), torch.empty(0, self.input_lin.out_features)
        
        # Add self-loops to handle isolated nodes (alternative fix)
        g = dgl.add_self_loop(g)
        
        h = self.input_lin(node_feats)
        for conv, ln in zip(self.layers, self.norms):
            h_new = conv(g, h).flatten(1)  # concat heads
            h = ln(h + self.dropout(h_new))
            h = F.relu(h)

        with g.local_scope():
            g.ndata["h"] = h
            graph_emb = dgl.mean_nodes(g, "h")  # [batch_size, hidden_dim]

        return graph_emb, h


class GraphPolicyNetwork(nn.Module):
    """
    Policy network on top of GraphTransformerEncoder (DGL).
    Per node, outputs:
      - spawn/delete logits (matching topology actions)
      - spawn params: gamma, alpha, noise, theta (matching topology.spawn parameters)
    """

    def __init__(self, encoder: GraphTransformerEncoder, hidden_dim: int, noise_scale: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.noise_scale = noise_scale

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action heads: spawn vs delete (matching topology.act() format)
        self.action_head = nn.Linear(hidden_dim, 2)  # [spawn_logit, delete_logit]
        
        # Spawn parameter heads (matching topology.spawn signature)
        self.gamma_head = nn.Linear(hidden_dim, 1)   # gamma parameter
        self.alpha_head = nn.Linear(hidden_dim, 1)   # alpha parameter  
        self.noise_head = nn.Linear(hidden_dim, 1)   # noise parameter
        self.theta_head = nn.Linear(hidden_dim, 1)   # theta (direction) parameter

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, deterministic: bool = False, max_gamma: float = 5.0, max_alpha: float = 2.0, max_noise: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning action probabilities and spawn parameters.
        
        Args:
            g: DGL graph from topology
            node_feats: Node features from embedding
            deterministic: Whether to sample or use deterministic actions
            max_gamma: Upper bound for gamma parameter
            max_alpha: Upper bound for alpha parameter
            max_noise: Upper bound for noise parameter
            
        Returns:
            Dictionary with action logits, probabilities, and spawn parameters
        """
        graph_emb, node_emb = self.encoder(g, node_feats)
        
        # Handle empty graphs
        if node_emb.shape[0] == 0:
            return {
                "graph_emb": graph_emb,
                "node_emb": node_emb,
                "action_logits": torch.empty(0, 2),
                "spawn_prob": torch.empty(0),
                "delete_prob": torch.empty(0),
                "gamma": torch.empty(0),
                "alpha": torch.empty(0), 
                "noise": torch.empty(0),
                "theta": torch.empty(0),
            }
        
        h = self.node_mlp(node_emb)

        # Action logits: [num_nodes, 2] where [:, 0] = spawn, [:, 1] = delete
        action_logits = self.action_head(h)
        action_probs = F.softmax(action_logits, dim=-1)
        spawn_prob = action_probs[:, 0]
        delete_prob = action_probs[:, 1]

        # Spawn parameters (matching topology.spawn defaults)
        gamma = max_gamma * torch.sigmoid(self.gamma_head(h).squeeze(-1))  # Scale to [0, max_gamma] to match default
        alpha = max_alpha * torch.sigmoid(self.alpha_head(h).squeeze(-1))  # Scale to [0, max_alpha] to match default  
        noise = max_noise * torch.sigmoid(self.noise_head(h).squeeze(-1))  # Scale to [0, max_noise] to match default
        theta = torch.tanh(self.theta_head(h).squeeze(-1)) * math.pi  # Scale to [-π, π]

        out = {
            "graph_emb": graph_emb,
            "node_emb": node_emb,
            "action_logits": action_logits,
            "spawn_prob": spawn_prob,
            "delete_prob": delete_prob,
            "gamma": gamma,
            "alpha": alpha,
            "noise": noise,
            "theta": theta,
        }

        if deterministic:
            # Deterministic actions
            action_choice = torch.argmax(action_logits, dim=-1)  # 0=spawn, 1=delete
            out["action_choice"] = action_choice
            out["spawn_params"] = torch.stack([gamma, alpha, noise, theta], dim=-1)
            return out

        # Stochastic sampling
        action_dist = torch.distributions.Categorical(probs=action_probs)
        action_sample = action_dist.sample()
        out["action_sample"] = action_sample  # 0=spawn, 1=delete
        
        # Add noise to continuous parameters for exploration
        spawn_params_mean = torch.stack([gamma, alpha, noise, theta], dim=-1)
        spawn_params_noise = torch.randn_like(spawn_params_mean) * self.noise_scale
        spawn_params_sample = spawn_params_mean + spawn_params_noise
        
        # Clamp to valid ranges
        spawn_params_sample[..., 0] = torch.clamp(spawn_params_sample[..., 0], 0.1, 10.0)  # gamma
        spawn_params_sample[..., 1] = torch.clamp(spawn_params_sample[..., 1], 0.1, 5.0)   # alpha
        spawn_params_sample[..., 2] = torch.clamp(spawn_params_sample[..., 2], 0.0, 2.0)   # noise
        spawn_params_sample[..., 3] = torch.clamp(spawn_params_sample[..., 3], -math.pi, math.pi)  # theta
        
        out["spawn_params_sample"] = spawn_params_sample

        return out

    def get_topology_actions(self, policy_output: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[int, str]:
        """
        Convert policy network output to topology-compatible actions.
        
        Args:
            policy_output: Output from forward()
            deterministic: Whether to use deterministic or sampled actions
            
        Returns:
            Dictionary mapping node_id to action string ('spawn' or 'delete')
        """
        if deterministic:
            actions = policy_output["action_choice"]
        else:
            actions = policy_output["action_sample"]
        
        # Convert to topology action format
        topology_actions = {}
        for node_id, action in enumerate(actions):
            if action.item() == 0:
                topology_actions[node_id] = 'spawn'
            else:
                topology_actions[node_id] = 'delete'
                
        return topology_actions

    def get_spawn_parameters(self, policy_output: Dict[str, torch.Tensor], node_id: int, deterministic: bool = False) -> Tuple[float, float, float, float]:
        """
        Get spawn parameters for a specific node.
        
        Args:
            policy_output: Output from forward()
            node_id: ID of the node to get parameters for
            deterministic: Whether to use deterministic or sampled parameters
            
        Returns:
            Tuple of (gamma, alpha, noise, theta) for topology.spawn()
        """
        if deterministic:
            params = policy_output["spawn_params"][node_id]
        else:
            params = policy_output["spawn_params_sample"][node_id]
            
        return params[0].item(), params[1].item(), params[2].item(), params[3].item()

    # TODO verify if to delete
    # @staticmethod
    # def compute_new_node_position(
    #     parent_pos: torch.Tensor,
    #     gamma: torch.Tensor,
    #     alpha: torch.Tensor,
    #     noise: torch.Tensor,
    #     theta: torch.Tensor,
    #     distance_scale: float = 1.0,
    # ) -> torch.Tensor:
    #     magnitude = distance_scale * (gamma * alpha + noise) # TODO use agent.hill_equation
    #     dx = magnitude * torch.cos(theta)
    #     dy = magnitude * torch.sin(theta)

    #     if parent_pos.shape[-1] == 2:
    #         offset = torch.stack([dx, dy], dim=-1)
    #         return parent_pos + offset
    #     elif parent_pos.shape[-1] == 3:
    #         dz = torch.zeros_like(dx)
    #         offset = torch.stack([dx, dy, dz], dim=-1)
    #         return parent_pos + offset
    #     else:
    #         raise ValueError("parent_pos must have last dim 2 or 3.")


class TopologyPolicyAgent:
    """
    Agent that combines our topology with the graph transformer policy.
    """
    
    def __init__(self, topology, embedding, policy_network):
        self.topology = topology
        self.embedding = embedding
        self.policy = policy_network
        
    def get_policy_actions(self, embedding_dim: int = 64, deterministic: bool = False) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float, float, float]]]:
        """
        Get actions and spawn parameters from the policy network.
        
        Returns:
            Tuple of (actions_dict, spawn_params_dict)
        """
        # Get current state embedding
        dgl_graph = self.embedding.to_dgl(embedding_dim=embedding_dim)
        
        if dgl_graph.num_nodes() == 0:
            return {}, {}
        
        # Forward pass through policy
        policy_output = self.policy(dgl_graph, dgl_graph.ndata['x'], deterministic=deterministic)
        
        # Convert to topology actions
        actions = self.policy.get_topology_actions(policy_output, deterministic)
        
        # Get spawn parameters for each node
        spawn_params = {}
        for node_id in range(dgl_graph.num_nodes()):
            spawn_params[node_id] = self.policy.get_spawn_parameters(policy_output, node_id, deterministic)
            
        return actions, spawn_params
    
    def act_with_policy(self, embedding_dim: int = 64, deterministic: bool = False):
        """
        Execute one step using the policy network.
        """
        actions, spawn_params = self.get_policy_actions(embedding_dim, deterministic)
        
        # Separate spawn and delete actions
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
    from agent import Topology
    from substrate import Substrate
    from embedding_dgl import GraphEmbedding
    
    print("Setting up topology and substrate...")
    
    # Create substrate and topology 
    substrate = Substrate((100, 50))
    substrate.create('linear', m=0.01, b=1.0)
    
    topology = Topology(substrate=substrate)
    topology.reset(init_num_nodes=5)
    
    # Create embedding 
    embedding = GraphEmbedding(topology)
    
    print(f"Initial topology: {topology.graph.num_nodes()} nodes, {topology.graph.num_edges()} edges")
    
    # Get the DGL graph with embeddings 
    dgl_graph = embedding.to_dgl(embedding_dim=64)
    node_features = dgl_graph.ndata['x']  
    
    print(f"DGL graph from embedding: {dgl_graph.num_nodes()} nodes, {dgl_graph.num_edges()} edges")
    print(f"Node features shape from embedding: {node_features.shape}")
    
    
    # Create policy network (using the correct input dimension from embedding)
    feature_dim = node_features.shape[1] if node_features.shape[0] > 0 else 64
    encoder = GraphTransformerEncoder(in_dim=feature_dim, hidden_dim=128, num_layers=2)
    policy = GraphPolicyNetwork(encoder, hidden_dim=128)
    
    # Test with actual graph and features
    if dgl_graph.num_nodes() > 0:
        print("Testing policy with the topology...")
        policy_output = policy(dgl_graph, node_features, deterministic=False)
        
        print(f"Policy output shapes:")
        print(f"  Action logits: {policy_output['action_logits'].shape}")
        print(f"  Spawn probabilities: {policy_output['spawn_prob'].shape}")
        print(f"  Delete probabilities: {policy_output['delete_prob'].shape}")
        print(f"  Spawn parameters sample: {policy_output['spawn_params_sample'].shape}")
        
        # Show actual values
        print(f"Spawn probabilities: {policy_output['spawn_prob']}")
        print(f"Delete probabilities: {policy_output['delete_prob']}")
        print(f"Gamma values: {policy_output['gamma']}")
        print(f"Alpha values: {policy_output['alpha']}")
        print(f"Graph embedding: {policy_output['graph_emb'].shape}")
    else:
        print("No nodes in topology - skipping policy test")
    
    # Create and test the agent
    print("\nCreating and testing policy agent...")
    agent = TopologyPolicyAgent(topology, embedding, policy)
    
    # Test policy-based actions
    actions, spawn_params = agent.get_policy_actions(embedding_dim=64, deterministic=True)
    print(f"Policy actions: {actions}")
    print(f"Spawn parameters for node 0: {spawn_params.get(0, 'N/A')}")
    
    # Execute actions
    print("\nExecuting policy actions...")
    executed_actions = agent.act_with_policy(embedding_dim=64, deterministic=False)
    print(f"Executed actions: {executed_actions}")
    print(f"Final topology: {topology.graph.num_nodes()} nodes, {topology.graph.num_edges()} edges")
    
    print("\nIntegration test completed successfully! 🚀")