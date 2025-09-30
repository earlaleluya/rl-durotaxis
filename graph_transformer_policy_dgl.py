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
        node_dim: int,
        graph_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Node feature processing
        self.node_input_lin = nn.Linear(node_dim, hidden_dim)
        
        # Graph feature processing
        self.graph_input_lin = nn.Linear(graph_dim, hidden_dim)
        
        # GAT layers
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
                activation=None,
                allow_zero_in_degree=True,
            )
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        
        # Graph-level fusion
        self.graph_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concat pooled nodes + graph features
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, graph_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: DGLGraph (can be None if using pre-computed embeddings)
            node_feats: [num_nodes, node_dim]
            graph_feats: [graph_dim]

        Returns:
            graph_emb: [hidden_dim] - Enhanced with both node pooling and graph features
            node_emb: [num_nodes, hidden_dim] - Context-aware node embeddings
        """
        # Handle empty graphs
        if node_feats.shape[0] == 0:
            empty_graph_emb = self.graph_input_lin(graph_feats).squeeze(0)
            return empty_graph_emb, torch.empty(0, self.node_input_lin.out_features)
        
        # Process node features
        h = self.node_input_lin(node_feats)
        
        # Process graph features
        graph_h = self.graph_input_lin(graph_feats.unsqueeze(0))  # [1, hidden_dim]
        
        # If we have a graph structure, apply GAT layers
        if g is not None and g.num_edges() > 0:
            # Add self-loops to handle isolated nodes
            g = dgl.add_self_loop(g)
            
            # Apply GAT layers
            for conv, ln in zip(self.layers, self.norms):
                h_new = conv(g, h).flatten(1)  # concat heads
                h = ln(h + self.dropout(h_new))
                h = F.relu(h)
        else:
            # No graph structure - just apply MLPs to node features
            for i, (conv, ln) in enumerate(zip(self.layers, self.norms)):
                # Skip GAT and just apply normalization and activation
                h_new = h  # No graph convolution
                h = ln(h + self.dropout(h_new))
                h = F.relu(h)

        # Create graph embedding by combining pooled nodes + graph features
        pooled_nodes = torch.mean(h, dim=0, keepdim=True)  # [1, hidden_dim]
        
        # Fuse pooled node features with graph-level features
        combined = torch.cat([pooled_nodes, graph_h], dim=-1)  # [1, hidden_dim * 2]
        graph_emb = self.graph_fusion(combined).squeeze(0)  # [hidden_dim]

        return graph_emb, h


class GraphPolicyNetwork(nn.Module):
    """
    Policy network on top of GraphTransformerEncoder (DGL) that uses full embedding state (graph-level and)
    Per node, outputs:
      - spawn/delete logits (matching topology actions)
      - spawn params: gamma, alpha, noise, theta (matching topology.spawn parameters)
    """

    def __init__(self, encoder: GraphTransformerEncoder, hidden_dim: int, noise_scale: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.noise_scale = noise_scale

        # Node-level MLP (enhanced with graph context)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Node features + graph context
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.action_head = nn.Linear(hidden_dim, 2)  # [spawn_logit, delete_logit]
        
        # Spawn parameter heads
        self.gamma_head = nn.Linear(hidden_dim, 1)
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.noise_head = nn.Linear(hidden_dim, 1)
        self.theta_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_dict: Dict[str, torch.Tensor], deterministic: bool = False, max_gamma: float = 5.0, max_alpha: float = 2.0, max_noise: float = 1.0) -> Dict[str, torch.Tensor]:
        """
        Forward pass using full embedding state.
        
        Args:
            state_dict: Full state from embedding.get_state_embedding()
            deterministic: Whether to sample or use deterministic actions
            max_gamma: Upper bound for gamma parameter
            max_alpha: Upper bound for alpha parameter
            max_noise: Upper bound for noise parameter
            
        Returns:
            Dictionary with action logits, probabilities, and spawn parameters
        """
        # Extract components from state
        node_embeddings = state_dict['node_embeddings']
        graph_embedding = state_dict['graph_embedding']
        edge_index = state_dict['edge_index']
        
        # Create DGL graph from edge index if edges exist
        g = None
        if isinstance(edge_index, torch.Tensor) and edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            g = dgl.graph((src, dst), num_nodes=node_embeddings.shape[0])
        
        # Get enhanced embeddings from encoder
        graph_emb, node_emb = self.encoder(
            g=g,
            node_feats=node_embeddings,
            graph_feats=graph_embedding
        )
        
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
        
        # Broadcast graph embedding to all nodes for context
        graph_context = graph_emb.unsqueeze(0).repeat(node_emb.shape[0], 1)
        
        # Combine node embeddings with graph context
        combined_features = torch.cat([node_emb, graph_context], dim=-1)
        h = self.node_mlp(combined_features)

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
        spawn_params_sample[..., 0] = torch.clamp(spawn_params_sample[..., 0], 0.1, max_gamma * 2)  # gamma
        spawn_params_sample[..., 1] = torch.clamp(spawn_params_sample[..., 1], 0.1, max_alpha * 2)   # alpha
        spawn_params_sample[..., 2] = torch.clamp(spawn_params_sample[..., 2], 0.0, max_noise * 2)   # noise
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



class TopologyPolicyAgent:
    """
    Agent that combines our topology with the graph transformer policy. Also, it uses full embedding state.
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
        # Get FULL state embedding (not just DGL graph)
        state = self.embedding.get_state_embedding(embedding_dim=embedding_dim)
        
        if state['num_nodes'] == 0:
            return {}, {}
        
        # Forward pass through policy using full state
        policy_output = self.policy(state, deterministic=deterministic)
        
        # Convert to topology actions
        actions = self.policy.get_topology_actions(policy_output, deterministic)
        
        # Get spawn parameters for each node
        spawn_params = {}
        for node_id in range(state['num_nodes']):
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
    
    # Get FULL embedding state (not just DGL graph)
    state = embedding.get_state_embedding(embedding_dim=64)
    
    print(f"Full embedding state:")
    print(f"  Node embeddings shape: {state['node_embeddings'].shape}")
    print(f"  Graph embedding shape: {state['graph_embedding'].shape}")
    print(f"  Node features shape: {state['node_features'].shape}")
    print(f"  Graph features shape: {state['graph_features'].shape}")
    
    # Safe edge handling
    edge_index = state['edge_index']
    if isinstance(edge_index, torch.Tensor):
        print(f"  Edge index shape: {edge_index.shape}")
    else:
        print(f"  Edge index type: {type(edge_index)} (likely empty graph)")
    
    edge_attr = state['edge_attr']
    if isinstance(edge_attr, torch.Tensor):
        print(f"  Edge attributes shape: {edge_attr.shape}")
    else:
        print(f"  Edge attributes type: {type(edge_attr)} (likely empty graph)")
    
    print(f"  Number of nodes: {state['num_nodes']}")
    print(f"  Number of edges: {state['num_edges']}")
    
    # Create enhanced policy network using both node and graph dimensions
    node_dim = state['node_embeddings'].shape[1] if state['num_nodes'] > 0 else 64
    graph_dim = state['graph_embedding'].shape[0]
    
    print(f"Creating encoder with node_dim={node_dim}, graph_dim={graph_dim}")
    
    encoder = GraphTransformerEncoder(
        node_dim=node_dim, 
        graph_dim=graph_dim, 
        hidden_dim=128, 
        num_layers=2
    )
    policy = GraphPolicyNetwork(encoder, hidden_dim=128)
    
    # Test with full state
    if state['num_nodes'] > 0:
        print("Testing policy with full embedding state...")
        policy_output = policy(state, deterministic=False)
        
        print(f"Policy output shapes:")
        print(f"  Action logits: {policy_output['action_logits'].shape}")
        print(f"  Spawn probabilities: {policy_output['spawn_prob'].shape}")
        print(f"  Enhanced graph embedding: {policy_output['graph_emb'].shape}")
        
        print(f"Values:")
        print(f"  Spawn probabilities: {policy_output['spawn_prob']}")
        print(f"  Delete probabilities: {policy_output['delete_prob']}")
        print(f"  Gamma values: {policy_output['gamma']}")
        print(f"  Alpha values: {policy_output['alpha']}")
        print(f"  Theta values: {policy_output['theta']}")
    else:
        print("No nodes in topology - skipping policy test")
    
    # Create and test enhanced agent
    print("\nCreating enhanced policy agent...")
    agent = TopologyPolicyAgent(topology, embedding, policy)
    
    # Test policy-based actions
    actions, spawn_params = agent.get_policy_actions(embedding_dim=64, deterministic=True)
    print(f"Policy actions: {actions}")
    if len(spawn_params) > 0:
        print(f"Sample spawn parameters for node 0: {spawn_params.get(0, 'N/A')}")
    
    # Execute actions
    print("\nExecuting policy actions...")
    executed_actions = agent.act_with_policy(embedding_dim=64, deterministic=False)
    print(f"Executed actions: {executed_actions}")
    print(f"Final topology: {topology.graph.num_nodes()} nodes, {topology.graph.num_edges()} edges")
    
    # Test a few action cycles to see graph evolution
    print("\nTesting multiple action cycles:")
    for i in range(3):
        print(f"\n--- Cycle {i+1} ---")
        before_nodes = topology.graph.num_nodes()
        before_edges = topology.graph.num_edges()
        
        executed_actions = agent.act_with_policy(embedding_dim=64, deterministic=False)
        
        after_nodes = topology.graph.num_nodes()
        after_edges = topology.graph.num_edges()
        
        print(f"Before: {before_nodes} nodes, {before_edges} edges")
        print(f"Actions: {executed_actions}")
        print(f"After: {after_nodes} nodes, {after_edges} edges")
        print(f"Change: {after_nodes - before_nodes:+d} nodes, {after_edges - before_edges:+d} edges")
    
    print("\nIntegration test completed successfully! 🚀")