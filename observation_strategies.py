"""
Alternative observation strategies for DurotaxisEnv.
Each strategy has different implications for learning and performance.
"""

import numpy as np
import torch
import dgl

def get_observation_strategy_1_graph_only(new_state):
    """
    Strategy 1: Graph embedding only 
    
    Pros:
    - Fixed size, works with all RL algorithms
    - Fast, low memory
    - Global perspective
    
    Cons:
    - Loses individual node information
    - Cannot learn node-specific behaviors
    - Limited spatial understanding
    
    Best for: Population-level policies, simple environments
    """
    return new_state['graph_embedding'].numpy().astype(np.float32)


def get_observation_strategy_2_padded_nodes(new_state, max_nodes=20, embedding_dim=64):
    """
    Strategy 2: Node embeddings with padding/truncation
    
    Pros:
    - Preserves individual node information
    - Fixed size for RL algorithms
    - Detailed spatial structure
    
    Cons:
    - Larger observation space
    - Padding can confuse learning
    - Arbitrary max_nodes limit
    
    Best for: Node-specific policies, detailed control
    """
    node_emb = new_state['node_embeddings'].numpy().astype(np.float32)
    
    if node_emb.shape[0] == 0:
        # No nodes - return zeros
        return np.zeros(max_nodes * embedding_dim, dtype=np.float32)
    
    if node_emb.shape[0] <= max_nodes:
        # Pad with zeros
        padding = np.zeros((max_nodes - node_emb.shape[0], node_emb.shape[1]))
        padded = np.concatenate([node_emb, padding], axis=0)
        return padded.flatten()
    else:
        # Truncate to max_nodes (could lose information!)
        return node_emb[:max_nodes].flatten()


def get_observation_strategy_3_combined(new_state, max_nodes=15):
    """
    Strategy 3: Graph embedding + padded node embeddings
    
    Pros:
    - Best of both worlds
    - Global + local information
    - Rich representation
    
    Cons:
    - Large observation space
    - More complex for RL algorithms
    - Higher computational cost
    
    Best for: Complex environments, high-performance requirements
    """
    graph_emb = new_state['graph_embedding'].numpy().astype(np.float32)
    node_emb = new_state['node_embeddings'].numpy().astype(np.float32)
    
    if node_emb.shape[0] == 0:
        # No nodes
        node_part = np.zeros(max_nodes * graph_emb.shape[0])
        return np.concatenate([graph_emb, node_part])
    
    if node_emb.shape[0] <= max_nodes:
        # Pad nodes
        padded_nodes = np.zeros((max_nodes, node_emb.shape[1]))
        padded_nodes[:node_emb.shape[0]] = node_emb
        return np.concatenate([graph_emb, padded_nodes.flatten()])
    else:
        # Truncate nodes
        return np.concatenate([graph_emb, node_emb[:max_nodes].flatten()])


def get_observation_strategy_4_statistical(new_state):
    """
    Strategy 4: Graph embedding + statistical node features
    
    Pros:
    - Fixed size regardless of node count
    - Preserves node distribution information
    - No arbitrary limits
    
    Cons:
    - Loses individual node identity
    - Statistical summaries may miss details
    - More complex to interpret
    
    Best for: Variable-size graphs, population dynamics
    """
    graph_emb = new_state['graph_embedding'].numpy().astype(np.float32)
    node_emb = new_state['node_embeddings'].numpy().astype(np.float32)
    
    if node_emb.shape[0] == 0:
        # No nodes - return only graph embedding
        return graph_emb
    
    # Compute statistical features of node embeddings
    node_stats = np.concatenate([
        np.mean(node_emb, axis=0),      # Average node features
        np.std(node_emb, axis=0),       # Variability in nodes
        np.min(node_emb, axis=0),       # Minimum values
        np.max(node_emb, axis=0),       # Maximum values
    ])
    
    return np.concatenate([graph_emb, node_stats])


def get_observation_strategy_5_attention_based(new_state, top_k=10):
    """
    Strategy 5: Graph embedding + top-k most important nodes
    
    Pros:
    - Adaptive selection of important nodes
    - Fixed size with relevant information
    - Can focus on boundary/central nodes
    
    Cons:
    - Requires defining "importance"
    - More computation needed
    - May miss relevant nodes
    
    Best for: Large graphs, attention-based policies
    """
    graph_emb = new_state['graph_embedding'].numpy().astype(np.float32)
    node_emb = new_state['node_embeddings'].numpy().astype(np.float32)
    
    if node_emb.shape[0] == 0:
        return graph_emb
    
    if node_emb.shape[0] <= top_k:
        # Fewer nodes than top_k, pad the rest
        padded = np.zeros((top_k, node_emb.shape[1]))
        padded[:node_emb.shape[0]] = node_emb
        return np.concatenate([graph_emb, padded.flatten()])
    
    # Simple importance: distance from centroid (boundary nodes are important)
    positions = new_state['node_features'][:, :2]  # x, y coordinates
    centroid = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    
    # Select top_k nodes with highest distance (boundary nodes)
    top_indices = np.argsort(distances)[-top_k:]
    important_nodes = node_emb[top_indices]
    
    return np.concatenate([graph_emb, important_nodes.flatten()])


def get_observation_strategy_6_attention_based_statistical(new_state, policy_network, device='cpu'):
    """
    Strategy 6: Enhanced graph embedding + statistical node features from GraphPolicyNetwork.
    
    Uses the GraphPolicyNetwork's forward pass to get:
    - Enhanced graph_emb (fusion of pooled nodes + graph features)  
    - Statistical summaries of processed node_emb (context-aware node features)
    
    Args:
        new_state: Full state from embedding.get_state_embedding()
        policy_network: Your GraphPolicyNetwork instance
        device: torch device
        
    Returns:
        np.ndarray: [graph_emb + node_stats] shape: [hidden_dim + hidden_dim*4]
    """
    
    # Convert state to tensors if needed
    if isinstance(new_state['node_embeddings'], np.ndarray):
        for key in ['node_embeddings', 'graph_embedding']:
            if key in new_state:
                new_state[key] = torch.from_numpy(new_state[key]).to(device)
    
    # Get enhanced embeddings from policy network
    with torch.no_grad():
        policy_output = policy_network(new_state, deterministic=True)
        
        # Extract enhanced features
        graph_emb = policy_output['graph_emb']  # [hidden_dim] - Enhanced graph representation
        node_emb = policy_output['node_emb']    # [num_nodes, hidden_dim] - Context-aware nodes
    
    # Convert graph embedding to numpy
    graph_features = graph_emb.cpu().numpy().astype(np.float32)
    
    # Handle empty graphs
    if node_emb.shape[0] == 0:
        # Return just graph embedding padded to expected size
        expected_node_stats_size = graph_emb.shape[0] * 4  # mean + std + min + max
        padding = np.zeros(expected_node_stats_size, dtype=np.float32)
        return np.concatenate([graph_features, padding])
    
    # Calculate statistical summaries of enhanced node embeddings
    node_emb_np = node_emb.cpu().numpy().astype(np.float32)
    
    node_stats = np.concatenate([
        np.mean(node_emb_np, axis=0),      # Average enhanced node properties
        np.std(node_emb_np, axis=0),       # Variability in enhanced features  
        np.min(node_emb_np, axis=0),       # Minimum enhanced values
        np.max(node_emb_np, axis=0),       # Maximum enhanced values
    ])
    
    # Combine enhanced graph embedding with node statistics
    observation = np.concatenate([graph_features, node_stats])
    
    return observation


def get_observation_strategy_6_lightweight(new_state, policy_network, device='cpu'):
    """
    Strategy 6 Lightweight: Only uses the encoder part (no policy heads).
    More efficient for observation generation.
    """
    
    # Convert state to tensors if needed  
    if isinstance(new_state['node_embeddings'], np.ndarray):
        for key in ['node_embeddings', 'graph_embedding']:
            if key in new_state:
                new_state[key] = torch.from_numpy(new_state[key]).to(device)
    
    # Extract components from state
    node_embeddings = new_state['node_embeddings']
    graph_embedding = new_state['graph_embedding']
    edge_index = new_state['edge_index']
    
    # Create DGL graph from edge index if edges exist
    g = None
    if isinstance(edge_index, torch.Tensor) and edge_index.shape[1] > 0:
        src, dst = edge_index[0], edge_index[1]
        g = dgl.graph((src, dst), num_nodes=node_embeddings.shape[0])
    
    # Get enhanced embeddings from encoder only (more efficient)
    with torch.no_grad():
        graph_emb, node_emb = policy_network.encoder(
            g=g,
            node_feats=node_embeddings,
            graph_feats=graph_embedding
        )
    
    # Convert to numpy
    graph_features = graph_emb.cpu().numpy().astype(np.float32)
    
    # Handle empty graphs
    if node_emb.shape[0] == 0:
        expected_node_stats_size = graph_emb.shape[0] * 4
        padding = np.zeros(expected_node_stats_size, dtype=np.float32)
        return np.concatenate([graph_features, padding])
    
    # Statistical summaries of enhanced node embeddings
    node_emb_np = node_emb.cpu().numpy().astype(np.float32)
    
    node_stats = np.concatenate([
        np.mean(node_emb_np, axis=0),
        np.std(node_emb_np, axis=0), 
        np.min(node_emb_np, axis=0),
        np.max(node_emb_np, axis=0),
    ])
    
    return np.concatenate([graph_features, node_stats])


# Observation space shapes for each strategy
def get_observation_space_shapes(embedding_dim=64, max_nodes=20, hidden_dim=128):
    """
    Returns observation space shapes for each strategy.
    """
    
    shapes = {
        "strategy_1_graph_only": (embedding_dim,),                    # 64
        "strategy_2_padded_nodes": (max_nodes * embedding_dim,),      # 1280
        "strategy_3_combined": (embedding_dim + max_nodes * embedding_dim,),  # 1344
        "strategy_4_statistical": (embedding_dim * 5,),              # 320 
        "strategy_5_attention_based": (embedding_dim + max_nodes * embedding_dim,),  # 1344
        "strategy_6_attention_based_statistical": (hidden_dim * 5,), # 640 (128 * 5)
        "strategy_6_lightweight": (hidden_dim * 5,),                 # 640 (128 * 5)
    }
    
    return shapes


if __name__ == "__main__":
    # Example usage and comparison
    print("Observation Strategy Comparison")
    print("=" * 50)
    
    # Mock state for demonstration
    embedding_dim = 64
    num_nodes = 8
    
    mock_state = {
        'graph_embedding': torch.randn(embedding_dim),
        'node_embeddings': torch.randn(num_nodes, embedding_dim),
        'node_features': torch.randn(num_nodes, 10),  # positions + other features
    }
    
    strategies = [
        ("Graph Only", get_observation_strategy_1_graph_only),
        ("Padded Nodes", lambda s: get_observation_strategy_2_padded_nodes(s, max_nodes=15)),
        ("Combined", lambda s: get_observation_strategy_3_combined(s, max_nodes=10)),
        ("Statistical", get_observation_strategy_4_statistical),
        ("Attention-based", lambda s: get_observation_strategy_5_attention_based(s, top_k=8)),
    ]
    
    for name, strategy_func in strategies:
        obs = strategy_func(mock_state)
        print(f"{name:15} | Shape: {obs.shape:20} | Size: {obs.size:6} elements")
    
    print("\nRecommendations:")
    print("- Start with Strategy 1 (current) for simplicity")
    print("- Use Strategy 4 for better node information without size issues")
    print("- Use Strategy 3 for maximum information (but larger obs space)")
    print("- Use Strategy 5 for large graphs with attention mechanisms")