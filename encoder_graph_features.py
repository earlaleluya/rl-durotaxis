import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple
from torch_geometric.nn import TransformerConv


class GraphInputEncoder(nn.Module):
    """
    Graph Input Encoder using Graph Transformer architecture for durotaxis simulation.
    
    This encoder processes graph-structured data representing cellular topology with:
    - Graph-level features (14 dimensions): Global properties [num_nodes, num_edges, density, centroid_x, centroid_y, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, bbow_width, bbox_height, bbox_area, hull_area, avg_degree]
    - Node-level features (8 dimensions): Per-cell properties [node_x, node_y, substrate_intensity, in_degree, out_degree, centrality, centroid_distance, is_boundary]  
    - Edge-level features (3 dimensions): Connection properties [distances, direction_norm_x, direction_norm_y]
    
    Architecture:
    1. Feature projection: Maps input features to common hidden dimension
    2. Virtual node: Graph features become a special "global" node token
    3. Graph Transformer: Multi-layer transformer with residual connections and multi-head attention
    4. Output: Per-node embeddings for downstream policy decisions
    
    The encoder uses PyTorch Geometric's TransformerConv for efficient graph attention
    with proper residual connections and edge-aware processing.
    
    Args:
        hidden_dim (int, optional): Hidden dimension for internal representations. Defaults to 128.
        out_dim (int, optional): Output embedding dimension per node. Defaults to 64.
        num_layers (int, optional): Number of transformer layers. Defaults to 4.
        
    Input Shapes:
        graph_features: [14] - Global graph properties
        node_features: [num_nodes, 8] - Per-node cell properties  
        edge_features: [num_edges, 3] - Per-edge connection properties
        edge_index: [2, num_edges] - Graph connectivity in COO format
        batch: [num_nodes] (optional) - Batch assignment for multiple graphs
        
    Output Shape:
        [num_nodes+1, out_dim] - Node embeddings with graph token as first element
        
    Example:
        >>> encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=3)
        >>> graph_feat = torch.randn(14)
        >>> node_feat = torch.randn(5, 8)  # 5 nodes
        >>> edge_feat = torch.randn(6, 3)  # 6 edges
        >>> edge_idx = torch.randint(0, 5, (2, 6))
        >>> out = encoder(graph_feat, node_feat, edge_feat, edge_idx)
        >>> print(out.shape)  # torch.Size([6, 64]) = [5 nodes + 1 graph token, 64]
    """
    
    def __init__(self, hidden_dim=128, out_dim=64, num_layers=4):
        super().__init__()
        # Graph-level MLP → virtual node
        self.graph_mlp = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Node projection
        self.node_proj = nn.Linear(8, hidden_dim)
        # Edge projection
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GraphTransformer backbone (using PyTorch Geometric's TransformerConv)
        self.gnn = MyGraphTransformer(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=num_layers,
            heads=4,
            dropout=0.1,
            edge_dim=hidden_dim   # must match edge_emb size
        )
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, graph_features, node_features, edge_features, edge_index, batch=None):
        """
        Forward pass through the Graph Input Encoder.
        
        Processes multi-modal graph data through feature projection, virtual node creation,
        and graph transformer layers to produce rich node embeddings for policy networks.
        
        Processing Pipeline:
        1. Project graph features to hidden dimension and create virtual "graph token"
        2. Project node features to hidden dimension  
        3. Concatenate graph token with node embeddings (graph token becomes first element)
        4. Project edge features to hidden dimension
        5. Apply multi-layer graph transformer with residual connections
        6. Return node embeddings with graph context
        
        Args:
            graph_features (torch.Tensor): Global graph properties [14]
                Features include: substrate stats, topology metrics, cell counts, etc.
            node_features (torch.Tensor): Per-node cell properties [num_nodes, 8]  
                Features include: position, velocity, substrate intensity, cell state, etc.
            edge_features (torch.Tensor): Per-edge connection properties [num_edges, 3]
                Features include: distance, angle, connection strength, etc.
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges]
                Each column [src, dst] represents an edge from node src to node dst
            batch (torch.Tensor, optional): Batch assignment [num_nodes]. Defaults to None.
                Used for processing multiple graphs simultaneously (single graph if None)
                
        Returns:
            torch.Tensor: Node embeddings with graph context [num_nodes+1, out_dim]
                - First element [0] is the graph token (global context)
                - Remaining elements [1:] are node embeddings
                - Each embedding captures local node properties + global graph context
                
        Note:
            The graph token (first output element) contains global graph information
            and can be used for graph-level predictions or as context for node decisions.
        """

        # Encode graph-level features as "virtual node"
        graph_emb = self.graph_mlp(graph_features).unsqueeze(0)   # [1, hidden_dim]

        # Encode node-level features
        node_emb = self.node_proj(node_features)  # [num_nodes, hidden_dim]

        # Concatenate graph token
        x = torch.cat([graph_emb, node_emb], dim=0)  # [num_nodes+1, hidden_dim]

        # Encode edge features
        edge_emb = self.edge_mlp(edge_features)  # [num_edges, hidden_dim]

        # If batch is None, assume single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Run GraphTransformer
        out = self.gnn(x, edge_index, edge_emb, batch)

        return out  # [num_nodes+1, out_dim]



class MyGraphTransformer(nn.Module):
    """
    Graph Transformer implementation using PyTorch Geometric's TransformerConv.
    This provides proper residual connections and multi-head attention.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, heads=4, dropout=0.1, edge_dim=None):
        super(MyGraphTransformer, self).__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer
        self.layers.append(
            TransformerConv(in_channels, hidden_channels // heads,
                            heads=heads, dropout=dropout, edge_dim=edge_dim)
        )
        self.norms.append(nn.LayerNorm(hidden_channels))
        self.dropouts.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                TransformerConv(hidden_channels, hidden_channels // heads,
                                heads=heads, dropout=dropout, edge_dim=edge_dim)
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))

        # Final layer
        self.layers.append(
            TransformerConv(hidden_channels, out_channels // heads,
                            heads=heads, dropout=dropout, edge_dim=edge_dim)
        )
        self.norms.append(nn.LayerNorm(out_channels))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Store dimensions for residual connections
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass with proper residual connections.
        
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim] (optional)
            batch: [num_nodes] (optional, for batch processing)
        """
        for i, (conv, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            residual = x
            
            # Apply transformer convolution
            x = conv(x, edge_index, edge_attr)
            
            # Residual connection (only if dimensions match)
            if i == 0:
                # First layer: input_dim -> hidden_dim
                if self.in_channels == self.hidden_channels:
                    x = x + residual
            elif i == len(self.layers) - 1:
                # Last layer: hidden_dim -> output_dim
                if self.hidden_channels == self.out_channels:
                    x = x + residual
            else:
                # Hidden layers: hidden_dim -> hidden_dim (always match)
                x = x + residual
            
            # Apply layer norm and activation
            x = norm(x)
            if i < len(self.layers) - 1:  # No activation on final layer
                x = F.relu(x)
            x = dropout(x)
            
        return x




class GraphPolicyNetwork(nn.Module):
    """
    Policy network on top of GraphInputEncoder that uses GraphTransformer.
    Per node, outputs:
      - spawn/delete logits (matching topology actions)
      - spawn params: gamma, alpha, noise, theta (matching topology.spawn parameters)
    """

    def __init__(self, encoder: GraphInputEncoder, hidden_dim: int, noise_scale: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.noise_scale = noise_scale
        self.hidden_dim = hidden_dim

        # Node-level MLP (enhanced with graph context)
        self.node_mlp = nn.Sequential(
            nn.Linear(encoder.out_dim * 2, hidden_dim),  # Node features + graph context
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
        Forward pass using embedding state.
        
        Args:
            state_dict: State from state_extractor.get_state_features()
            deterministic: Whether to sample or use deterministic actions
            max_gamma: Upper bound for gamma parameter
            max_alpha: Upper bound for alpha parameter
            max_noise: Upper bound for noise parameter
            
        Returns:
            Dictionary with action logits, probabilities, and spawn parameters
        """
        # Extract components from state
        node_features = state_dict['node_features']
        graph_features = state_dict['graph_features']
        edge_features = state_dict['edge_attr']
        edge_index = state_dict['edge_index']
        
        # Convert edge_index from DGL tuple format to PyG tensor format
        if isinstance(edge_index, tuple):
            src, dst = edge_index
            edge_index_tensor = torch.stack([src, dst], dim=0)  # [2, num_edges]
        else:
            edge_index_tensor = edge_index
        
        # Handle empty graphs
        if node_features.shape[0] == 0:
            return {
                "encoder_out": torch.empty(0, self.encoder.out_dim),
                "action_logits": torch.empty(0, 2),
                "spawn_prob": torch.empty(0),
                "delete_prob": torch.empty(0),
                "gamma": torch.empty(0),
                "alpha": torch.empty(0), 
                "noise": torch.empty(0),
                "theta": torch.empty(0),
            }
        
        # Get embeddings from encoder
        encoder_out = self.encoder(
            graph_features=graph_features,
            node_features=node_features,
            edge_features=edge_features,
            edge_index=edge_index_tensor
        )  # [num_nodes+1, out_dim]
        
        # Split encoder output: first token is graph, rest are nodes
        graph_token = encoder_out[0:1]  # [1, out_dim]
        node_tokens = encoder_out[1:]   # [num_nodes, out_dim]
        
        # Handle case where we only have graph token (no nodes)
        if node_tokens.shape[0] == 0:
            return {
                "encoder_out": encoder_out,
                "action_logits": torch.empty(0, 2),
                "spawn_prob": torch.empty(0),
                "delete_prob": torch.empty(0),
                "gamma": torch.empty(0),
                "alpha": torch.empty(0), 
                "noise": torch.empty(0),
                "theta": torch.empty(0),
            }
        
        # Broadcast graph token to all nodes for context
        graph_context = graph_token.repeat(node_tokens.shape[0], 1)
        
        # Combine node tokens with graph context
        combined_features = torch.cat([node_tokens, graph_context], dim=-1)
        h = self.node_mlp(combined_features)

        # Action logits: [num_nodes, 2] where [:, 0] = spawn, [:, 1] = delete
        action_logits = self.action_head(h)
        action_probs = F.softmax(action_logits, dim=-1)
        spawn_prob = action_probs[:, 0]
        delete_prob = action_probs[:, 1]

        # Spawn parameters (matching topology.spawn defaults)
        gamma = max_gamma * torch.sigmoid(self.gamma_head(h).squeeze(-1))  # Scale to [0, max_gamma]
        alpha = max_alpha * torch.sigmoid(self.alpha_head(h).squeeze(-1))  # Scale to [0, max_alpha]  
        noise = max_noise * torch.sigmoid(self.noise_head(h).squeeze(-1))  # Scale to [0, max_noise]
        theta = torch.tanh(self.theta_head(h).squeeze(-1)) * math.pi  # Scale to [-π, π]

        out = {
            "encoder_out": encoder_out,
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
    Agent that combines topology with the GraphInputEncoder policy.
    Uses GraphTransformer instead of GAT layers.
    """
    
    def __init__(self, topology, state_extractor, policy_network):
        self.topology = topology
        self.state_extractor = state_extractor
        self.policy = policy_network
        
    def get_policy_actions(self, embedding_dim: int = 64, deterministic: bool = False) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float, float, float]]]:
        """
        Get actions and spawn parameters from the policy network.
        
        Returns:
            Tuple of (actions_dict, spawn_params_dict)
        """
        # Get state features
        state = self.state_extractor.get_state_features(include_substrate=True)
        
        if state['num_nodes'] == 0:
            return {}, {}
        
        # Forward pass through policy using state
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
    # Usage example
    print("Testing GraphInputEncoder with GraphTransformer...")
    
    num_nodes = 5
    num_edges = 6

    graph_features = torch.randn(14)
    node_features = torch.randn(num_nodes, 8)
    edge_features = torch.randn(num_edges, 3)
    edge_index = torch.randint(0, num_nodes+1, (2, num_edges))  # random edges

    encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=3)

    out = encoder(graph_features, node_features, edge_features, edge_index)
    print(f"Output shape: {out.shape}")  # → [num_nodes+1, 64]
    
    print(f"Graph token (first row): {out[0].shape}")  # [64]
    print(f"Node tokens (remaining rows): {out[1:].shape}")  # [num_nodes, 64]
    
    # Test with real topology data
    try:
        from topology import Topology
        from substrate import Substrate
        from state import TopologyState
        
        print("\nTesting with real topology data...")
        
        # Create substrate and topology 
        substrate = Substrate((100, 50))
        substrate.create('linear', m=0.01, b=1.0)
        
        topology = Topology(substrate=substrate)
        topology.reset(init_num_nodes=5)
        
        # Create state extractor 
        state_extractor = TopologyState(topology)
        state = state_extractor.get_state_features(include_substrate=True)
        
        print(f"Real data shapes:")
        print(f"  Graph features: {state['graph_features'].shape}")
        print(f"  Node features: {state['node_features'].shape}")
        print(f"  Edge features: {state['edge_attr'].shape}")
        print(f"  Edge index: {state['edge_index'][0].shape if isinstance(state['edge_index'], tuple) else state['edge_index'].shape}")
        print(f"  Number of nodes: {state['num_nodes']}")
        print(f"  Number of edges: {state['num_edges']}")
        
        # Test encoder with real data
        encoder_real = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=3)
        
        if state['num_edges'] > 0:
            # Convert edge_index from tuple to tensor if needed
            if isinstance(state['edge_index'], tuple):
                src, dst = state['edge_index']
                edge_index_tensor = torch.stack([src, dst], dim=0)
            else:
                edge_index_tensor = state['edge_index']
            
            out_real = encoder_real(
                graph_features=state['graph_features'],
                node_features=state['node_features'],
                edge_features=state['edge_attr'],
                edge_index=edge_index_tensor
            )
            print(f"Real data output shape: {out_real.shape}")
            print(f"Expected: [{state['num_nodes']+1}, 64]")
        else:
            print("No edges in graph - skipping edge-dependent test")
        
        # Test GraphPolicyNetwork
        print("\nTesting GraphPolicyNetwork...")
        hidden_dim = 128
        encoder_policy = GraphInputEncoder(hidden_dim=hidden_dim, out_dim=64, num_layers=2)
        policy = GraphPolicyNetwork(encoder_policy, hidden_dim=hidden_dim)
        
        if state['num_nodes'] > 0:
            policy_output = policy(state, deterministic=False)
            
            print(f"Policy output shapes:")
            print(f"  Action logits: {policy_output['action_logits'].shape}")
            print(f"  Spawn probabilities: {policy_output['spawn_prob'].shape}")
            print(f"  Encoder output: {policy_output['encoder_out'].shape}")
            
            print(f"Values:")
            print(f"  Spawn probabilities: {policy_output['spawn_prob']}")
            print(f"  Delete probabilities: {policy_output['delete_prob']}")
            print(f"  Gamma values: {policy_output['gamma']}")
            print(f"  Alpha values: {policy_output['alpha']}")
            print(f"  Theta values: {policy_output['theta']}")
        else:
            print("No nodes in topology - skipping policy test")
        
        # Test TopologyPolicyAgent
        print("\nTesting TopologyPolicyAgent...")
        agent = TopologyPolicyAgent(topology, state_extractor, policy)
        
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
        
        # Test multiple action cycles
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
        
        print("\nGraphTransformer integration test completed successfully! 🚀")
        
    except ImportError as e:
        print(f"Could not import dependencies: {e}")
        print("Basic GraphTransformer test completed successfully!") 