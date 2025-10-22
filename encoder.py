import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple
from torch_geometric.nn import TransformerConv
from config_loader import ConfigLoader


class SimplicialEmbedding(nn.Module):
    """
    Simplicial Embedding (SEM) Layer.

    This layer constrains the latent features to lie on a product of geometric
    simplices. It partitions the input, applies a group-wise softmax, and
    introduces a geometric inductive bias that can stabilize training.

    Args:
        input_dim (int): The dimension of the input feature vector.
        num_groups (int): The number of groups (L) to partition the input into.
        temperature (float): The temperature parameter (τ) for the softmax.
    """
    def __init__(self, input_dim: int, num_groups: int, temperature: float = 1.0):
        super().__init__()
        if input_dim % num_groups != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by num_groups ({num_groups})")
        
        self.input_dim = input_dim
        self.num_groups = num_groups
        self.group_size = input_dim // num_groups
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the simplicial embedding transformation.

        Args:
            x (torch.Tensor): The input tensor of shape (..., input_dim).

        Returns:
            torch.Tensor: The transformed tensor with the same shape as the input.
        """
        # Reshape for group-wise softmax
        original_shape = x.shape
        x_reshaped = x.view(-1, self.num_groups, self.group_size)

        # Apply softmax with temperature to each group
        z = F.softmax(x_reshaped / self.temperature, dim=-1)

        # Reshape back to the original input shape (except for the last dim)
        return z.view(*original_shape)


class GraphInputEncoder(nn.Module):
    """
    Graph Input Encoder using Graph Transformer architecture for durotaxis simulation.
    
    This encoder processes graph-structured data representing cellular topology with:
    - Graph-level features (14 dimensions): Global properties [num_nodes, num_edges, density, centroid_x, centroid_y, bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y, bbow_width, bbox_height, bbox_area, hull_area, avg_degree]
    - Node-level features (9 dimensions): Per-cell properties [node_x, node_y, substrate_intensity, in_degree, out_degree, centrality, centroid_distance, is_boundary, new_node_flag]  
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
        node_features: [num_nodes, 9] - Per-node cell properties  
        edge_features: [num_edges, 3] - Per-edge connection properties
        edge_index: [2, num_edges] - Graph connectivity in COO format
        batch: [num_nodes] (optional) - Batch assignment for multiple graphs
        
    Output Shape:
        [num_nodes+1, out_dim] - Node embeddings with graph token as first element
        
    Example:
        >>> encoder = GraphInputEncoder(hidden_dim=128, out_dim=64, num_layers=3)
        >>> graph_feat = torch.randn(14)
        >>> node_feat = torch.randn(5, 9)  # 5 nodes, 9 features
        >>> edge_feat = torch.randn(6, 3)  # 6 edges
        >>> edge_idx = torch.randint(0, 5, (2, 6))
        >>> out = encoder(graph_feat, node_feat, edge_feat, edge_idx)
        >>> print(out.shape)  # torch.Size([6, 64]) = [5 nodes + 1 graph token, 64]
    """
    
    def __init__(self, config_path="config.yaml", **overrides):
        """
        Initialize GraphInputEncoder with configuration from YAML file
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        **overrides
            Parameter overrides for any configuration values
        """
        super().__init__()
        
        # Load configuration from YAML file
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_encoder_config()
        
        # Apply overrides
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
        
        # Set dimensions
        # Note: hidden_dim was moved to actor_critic section, use fallback default
        self.hidden_dim = config.get('hidden_dim', overrides.get('hidden_dim', 128))
        self.out_dim = config.get('out_dim', overrides.get('out_dim', 64))
        self.num_layers = config.get('num_layers', overrides.get('num_layers', 4))
            
        # Graph-level MLP → virtual node
        self.graph_mlp = nn.Sequential(
            nn.Linear(14, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        # Node projection
        self.node_proj = nn.Linear(9, self.hidden_dim)
        # Edge projection
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # GraphTransformer backbone (using PyTorch Geometric's TransformerConv)
        self.gnn = MyGraphTransformer(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            out_channels=self.out_dim,
            num_layers=self.num_layers,
            heads=4,
            dropout=0.1,
            edge_dim=self.hidden_dim  
        )

        # Simplicial Embedding configuration
        # Load from actor_critic section since that's where it's defined in config.yaml
        actor_critic_config = config_loader.get_actor_critic_config()
        self.sem_enabled = actor_critic_config.get('simplicial_embedding', {}).get('enabled', False)
        self.sem_layer = None
        if self.sem_enabled:
            sem_groups = actor_critic_config.get('simplicial_embedding', {}).get('num_groups', 16)
            sem_temperature = actor_critic_config.get('simplicial_embedding', {}).get('temperature', 1.0)
            
            # Ensure out_dim is divisible by num_groups
            if self.out_dim % sem_groups != 0:
                print(f"⚠️  WARNING: out_dim ({self.out_dim}) is not divisible by num_groups ({sem_groups})")
                print(f"    Adjusting num_groups to nearest divisor...")
                # Find nearest divisor
                for g in range(sem_groups, 0, -1):
                    if self.out_dim % g == 0:
                        sem_groups = g
                        print(f"    Using num_groups = {sem_groups}")
                        break
            
            self.sem_layer = SimplicialEmbedding(
                input_dim=self.out_dim,
                num_groups=sem_groups,
                temperature=sem_temperature
            )
            print(f"✅ Simplicial Embedding enabled in GraphInputEncoder:")
            print(f"   - Input/Output dim: {self.out_dim}")
            print(f"   - Num groups: {sem_groups}")
            print(f"   - Group size: {self.out_dim // sem_groups}")
            print(f"   - Temperature: {sem_temperature}")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights of linear layers with Xavier uniform initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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
            node_features (torch.Tensor): Per-node cell properties [num_nodes, 9]  
                Features include: position, velocity, substrate intensity, cell state, new_node flag, etc.
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
        
        # Safety check: replace NaN/Inf values with zeros to prevent propagation
        if torch.isnan(graph_features).any() or torch.isinf(graph_features).any():
            print(f"⚠️  WARNING: NaN/Inf detected in graph_features, replacing with zeros")
            graph_features = torch.nan_to_num(graph_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(node_features).any() or torch.isinf(node_features).any():
            print(f"⚠️  WARNING: NaN/Inf detected in node_features, replacing with zeros")
            node_features = torch.nan_to_num(node_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(edge_features).any() or torch.isinf(edge_features).any():
            print(f"⚠️  WARNING: NaN/Inf detected in edge_features, replacing with zeros")
            edge_features = torch.nan_to_num(edge_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Encode graph-level features as "virtual node"
        # Handle both single graph [feature_dim] and batched graphs [batch_size, feature_dim]
        if graph_features.dim() == 1:
            # Single graph case
            graph_emb = self.graph_mlp(graph_features).unsqueeze(0)   # [1, hidden_dim]
        else:
            # Batched graph case - process each graph separately and create graph tokens
            batch_size = graph_features.shape[0]
            if batch is None:
                # If no batch tensor provided, assume all nodes belong to first graph
                batch = torch.zeros(node_features.shape[0], dtype=torch.long, device=node_features.device)
            
            # Create one graph token per graph in the batch
            graph_embs = self.graph_mlp(graph_features)  # [batch_size, hidden_dim]
            
            # Create graph tokens by replicating for each graph
            graph_tokens = []
            for i in range(batch_size):
                graph_tokens.append(graph_embs[i].unsqueeze(0))  # [1, hidden_dim]
            
            # For now, use the mean of all graph embeddings as global context
            # This is a simplification - a more sophisticated approach would handle
            # multiple graphs properly in the transformer
            graph_emb = torch.mean(graph_embs, dim=0).unsqueeze(0)  # [1, hidden_dim]

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

        # Apply Simplicial Embedding if enabled
        if self.sem_layer is not None:
            out = self.sem_layer(out)

        return out  # [num_nodes+1, out_dim]



class MyGraphTransformer(nn.Module):
    """
    MyGraphTransformer implements a multi-layer Graph Transformer using PyTorch Geometric's TransformerConv.
    It supports multi-head attention, residual connections, layer normalization, and dropout.
        in_channels (int): Number of input node features.
        hidden_channels (int): Number of hidden node features for intermediate layers.
        out_channels (int): Number of output node features.
        num_layers (int, optional): Number of transformer layers. Default is 3.
        heads (int, optional): Number of attention heads in each TransformerConv. Default is 4.
        dropout (float, optional): Dropout probability applied after each layer. Default is 0.1.
        edge_dim (int, optional): Dimensionality of edge features (if any). Default is None.
    Forward Args:
        x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
        edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
        edge_attr (Tensor, optional): Edge feature matrix of shape [num_edges, edge_dim]. Default is None.
        batch (LongTensor, optional): Batch vector assigning each node to a specific graph. Default is None.
    Returns:
        Tensor: Output node features of shape [num_nodes, out_channels].
    Notes:
        - Residual connections are applied when input and output dimensions match.
        - Layer normalization and dropout are applied after each layer.
        - ReLU activation is used after all layers except the final layer.
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
                x = F.gelu(x)
            x = dropout(x)
            
        return x





if __name__ == '__main__':
    # Usage example
    print("Testing GraphInputEncoder with GraphTransformer...")
    
    num_nodes = 5
    num_edges = 6

    graph_features = torch.randn(14)
    node_features = torch.randn(num_nodes, 9)  # Updated to 9 dimensions
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
        
    except ImportError as e:
        print(f"Could not import dependencies: {e}")
        print("Basic GraphTransformer test completed successfully!")