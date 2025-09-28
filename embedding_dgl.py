import torch
import torch.nn.functional as F
import dgl # Import DGL
from scipy.spatial import ConvexHull
import numpy as np


class GraphEmbedding:
    """
    Graph embedding class that works with Topology instances to create
    state representations for RL algorithms.
    """
    
    def __init__(self, topology=None):
        """
        Initialize GraphEmbedding with a Topology instance.
        
        Parameters
        ----------
        topology : Topology
            The topology instance containing the graph and substrate
        """
        self.topology = topology
    
    def set_topology(self, topology):
        """Set or update the topology instance."""
        self.topology = topology
    
    @property
    def graph(self):
        """Access the graph from the topology."""
        if self.topology is None:
            raise ValueError("No topology set. Use set_topology() first.")
        # Assuming self.topology.graph is already a DGL graph object
        return self.topology.graph
    
    @property
    def substrate(self):
        """Access the substrate from the topology."""
        if self.topology is None:
            raise ValueError("No topology set. Use set_topology() first.")
        return self.topology.substrate

    def get_state_embedding(self, embedding_dim=64, include_substrate=True):
        """
        Get comprehensive state embedding combining graph-level and node-level features.
        
        Parameters
        ----------
        embedding_dim : int
            Dimension of the output embeddings
        include_substrate : bool
            Whether to include substrate intensity in node features
            
        Returns
        -------
        dict : State dictionary containing:
            - 'graph_embedding': Global graph representation
            - 'node_embeddings': Per-node feature matrix
            - 'edge_index': Edge connectivity for GNN (DGL style: (src, dst) tuples)
            - 'edge_attr': Edge attributes
            - 'graph_features': Raw graph-level features
            - 'node_features': Raw node-level features
        """
        
        # Get raw features
        node_features = self._get_node_features(include_substrate=include_substrate)
        edge_features = self._get_edge_features()
        graph_features = self._get_graph_features()
        
        # Create embeddings
        node_embeddings = self._create_node_embeddings(node_features, embedding_dim)
        graph_embedding = self._create_graph_embedding(node_embeddings, graph_features, embedding_dim)
        
        # Get edge information for GNN
        src, dst = self.graph.edges()
        
        state = {
            'graph_embedding': graph_embedding,          # Shape: [embedding_dim]
            'node_embeddings': node_embeddings,          # Shape: [num_nodes, embedding_dim]
            'edge_index': (src, dst),                    # DGL style edge representation
            'edge_attr': edge_features,                  # Shape: [num_edges, edge_feature_dim]
            'graph_features': graph_features,            # Shape: [graph_feature_dim]
            'node_features': node_features,              # Shape: [num_nodes, node_feature_dim]
            'num_nodes': self.graph.num_nodes(),
            'num_edges': self.graph.num_edges()
        }
        
        return state

    def _get_node_features(self, include_substrate=True):
        """
        Extract node-level features for each node in the graph.
        
        Returns
        -------
        torch.Tensor : Node feature matrix [num_nodes, feature_dim]
        """
        # Node features are directly stored in the DGL graph's ndata
        positions = self.graph.ndata['pos']
        num_nodes = positions.shape[0]
        
        features = []
        
        # Basic positional features
        features.append(positions) # [x, y] coordinates
        
        if include_substrate and self.substrate is not None:
            # Substrate intensity at each node position
            intensities = []
            for i in range(num_nodes):
                pos = positions[i].numpy()
                intensity = self.substrate.get_intensity(pos)
                intensities.append(intensity)
            substrate_features = torch.tensor(intensities, dtype=torch.float32).unsqueeze(1)
            features.append(substrate_features)
        
        # Topological features
        degrees = self._get_node_degrees()
        features.append(degrees)
        
        # Centrality measures
        centralities = self._get_node_centralities()
        features.append(centralities)
        
        # Distance from centroid
        centroid = torch.mean(positions, dim=0)
        distances_from_centroid = torch.norm(positions - centroid, dim=1, keepdim=True)
        features.append(distances_from_centroid)
        
        # Convex hull membership (is node on boundary?)
        boundary_flags = self._get_boundary_flags()
        features.append(boundary_flags)
        
        return torch.cat(features, dim=1)

    def _get_edge_features(self):
        """
        Extract edge-level features.
        
        Returns
        -------
        torch.Tensor : Edge feature matrix [num_edges, feature_dim]
        """
        if self.graph.num_edges() == 0:
            return torch.empty(0, 3, dtype=torch.float32)
        
        # Get edge endpoints
        src, dst = self.graph.edges()
        positions = self.graph.ndata['pos']
        
        # Calculate edge features
        src_pos = positions[src]
        dst_pos = positions[dst]
        
        # Euclidean distance
        distances = torch.norm(dst_pos - src_pos, dim=1, keepdim=True)
        
        # Direction vector (normalized)
        direction = dst_pos - src_pos
        direction_norm = F.normalize(direction, p=2, dim=1)
        
        edge_features = torch.cat([distances, direction_norm], dim=1)
        
        return edge_features

    def _get_graph_features(self):
        """
        Extract graph-level features.
        
        Returns
        -------
        torch.Tensor : Graph feature vector
        """
        positions = self.graph.ndata['pos']
        
        features = []
        
        # Basic graph statistics
        num_nodes = torch.tensor([self.graph.num_nodes()], dtype=torch.float32)
        num_edges = torch.tensor([self.graph.num_edges()], dtype=torch.float32)
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else torch.tensor([0.0])
        
        features.extend([num_nodes, num_edges, density])
        
        # Spatial statistics
        centroid = torch.mean(positions, dim=0)
        features.append(centroid)
        
        # Bounding box
        if num_nodes > 0:
            bbox_min = torch.min(positions, dim=0)[0]
            bbox_max = torch.max(positions, dim=0)[0]
            bbox_size = bbox_max - bbox_min
            bbox_area = torch.prod(bbox_size) if len(bbox_size) > 1 else bbox_size[0]
        else:
            bbox_min = torch.zeros(2)
            bbox_max = torch.zeros(2)
            bbox_size = torch.zeros(2)
            bbox_area = torch.tensor(0.0)
        
        features.extend([bbox_min, bbox_max, bbox_size, bbox_area.unsqueeze(0)])
        
        # Convex hull area (if possible)
        try:
            if num_nodes >= 3:
                hull = ConvexHull(positions.numpy())
                hull_area = torch.tensor([hull.volume], dtype=torch.float32)  # 'volume' is area in 2D
            else:
                hull_area = torch.tensor([0.0])
        except:
            hull_area = torch.tensor([0.0])
        
        features.append(hull_area)
        
        # Average node degree
        if num_nodes > 0:
            degrees = self._get_node_degrees()
            avg_degree = torch.mean(degrees)
        else:
            avg_degree = torch.tensor([0.0])
        
        features.append(avg_degree.unsqueeze(0))
        
        return torch.cat(features, dim=0)

    def _create_node_embeddings(self, node_features, embedding_dim):
        """
        Create node embeddings from raw features using a simple linear projection.
        """
        feature_dim = node_features.shape[1]
        
        # Simple linear projection (in practice, you'd use a learned network)
        if feature_dim < embedding_dim:
            # Pad with zeros if features are fewer than embedding dim
            padding = torch.zeros(node_features.shape[0], embedding_dim - feature_dim)
            embeddings = torch.cat([node_features, padding], dim=1)
        else:
            # Simple projection (you could use PCA or learned projection)
            weight = torch.randn(feature_dim, embedding_dim) * 0.1
            embeddings = torch.matmul(node_features, weight)
        
        return embeddings

    def _create_graph_embedding(self, node_embeddings, graph_features, embedding_dim):
        """
        Create graph-level embedding from node embeddings and graph features.
        """
        if node_embeddings.shape[0] == 0:
            return torch.zeros(embedding_dim)
        
        # Graph-level aggregation (mean pooling + graph features)
        node_pooled = torch.mean(node_embeddings, dim=0)
        
        # Combine with graph features
        if graph_features.shape[0] < embedding_dim:
            graph_padded = F.pad(graph_features, (0, embedding_dim - graph_features.shape[0]))
        else:
            graph_padded = graph_features[:embedding_dim]
        
        # Simple combination (you could use more sophisticated methods)
        graph_embedding = 0.7 * node_pooled + 0.3 * graph_padded
        
        return graph_embedding

    def _get_node_degrees(self):
        """Get node degree features."""
        if self.graph.num_nodes() == 0:
            return torch.empty(0, 2, dtype=torch.float32)
        
        in_degrees = self.graph.in_degrees().float().unsqueeze(1)
        out_degrees = self.graph.out_degrees().float().unsqueeze(1)
        return torch.cat([in_degrees, out_degrees], dim=1)

    def _get_node_centralities(self):
        """Get basic centrality measures."""
        num_nodes = self.graph.num_nodes()
        if num_nodes == 0:
            return torch.empty(0, 1, dtype=torch.float32)
        
        positions = self.graph.ndata['pos']
        
        # Simple centrality: inverse of average distance to all other nodes
        centralities = []
        for i in range(num_nodes):
            if num_nodes == 1:
                centralities.append(1.0)
            else:
                pos_i = positions[i]
                distances = torch.norm(positions - pos_i, dim=1)
                avg_distance = torch.mean(distances[distances > 0])  # exclude self
                centrality = 1.0 / (avg_distance + 1e-6)  # avoid division by zero
                centralities.append(centrality.item())
        
        return torch.tensor(centralities, dtype=torch.float32).unsqueeze(1)

    def _get_boundary_flags(self):
        """Get binary flags indicating if nodes are on the convex hull boundary."""
        num_nodes = self.graph.num_nodes()
        if num_nodes == 0:
            return torch.empty(0, 1, dtype=torch.float32)
        
        boundary_flags = torch.zeros(num_nodes, 1, dtype=torch.float32)
        
        try:
            outmost_indices = self.topology.get_outmost_nodes()
            boundary_flags[outmost_indices] = 1.0
        except:
            pass
        
        return boundary_flags

    def get_rl_ready_state(self, embedding_dim=64):
        """
        Get a state representation ready for RL algorithms.
        
        Returns a dictionary compatible with common GNN libraries like DGL.
        """
        state = self.get_state_embedding(embedding_dim=embedding_dim)
        
        # Add additional RL-specific information
        state['action_space_size'] = self.graph.num_nodes() * 2  # spawn or delete for each node
        state['observation_space_dim'] = embedding_dim
        
        return state

    def to_dgl(self, embedding_dim=64):
        """
        Convert the topology to a DGL Graph format.
        
        Returns
        -------
        dgl.DGLGraph : Graph data ready for GNN training
        """
        state = self.get_state_embedding(embedding_dim=embedding_dim)
        
        # Create a DGL graph from the edge list
        if state['num_edges'] > 0:
            src, dst = state['edge_index']
            dgl_graph = dgl.graph((src, dst), num_nodes=state['num_nodes'])
            
            # Set edge features if edges exist
            if state['edge_attr'].shape[0] > 0:
                dgl_graph.edata['edge_attr'] = state['edge_attr']
        else:
            # Create a graph with no edges
            dgl_graph = dgl.graph(([], []), num_nodes=state['num_nodes'])
    
        # Set node features
        dgl_graph.ndata['x'] = state['node_embeddings']
    
        # Add graph-level features as a single node feature
        # (This is one way to handle graph-level info in DGL)
        if state['num_nodes'] > 0:
            dgl_graph.ndata['graph_feat'] = state['graph_embedding'].unsqueeze(0).repeat(dgl_graph.num_nodes(), 1)
    
        return dgl_graph


if __name__ == '__main__':
    from agent import Topology
    from substrate import Substrate
    
    # Create substrate and topology
    substrate = Substrate((100, 50))
    substrate.create('linear', m=0.01, b=1.0)
    
    # Use the existing Topology class - it already creates DGL graphs
    topology = Topology(substrate=substrate)
    topology.reset(init_num_nodes=10)
    
    # Verify that we have a DGL graph
    print(f"Graph type: {type(topology.graph)}")
    print(f"Number of nodes: {topology.graph.num_nodes()}")
    print(f"Number of edges: {topology.graph.num_edges()}")
    print(f"Node positions shape: {topology.graph.ndata['pos'].shape}")
    
    # Create embedding instance
    embedding = GraphEmbedding(topology)
    
    # Get state for GNN-based RL
    state = embedding.get_rl_ready_state(embedding_dim=128)
    print(f"Graph embedding shape: {state['graph_embedding'].shape}")
    print(f"Node embeddings shape: {state['node_embeddings'].shape}")
    
    # Check edge index
    if state['num_edges'] > 0:
        src, dst = state['edge_index']
        print(f"Edge index - src shape: {src.shape}, dst shape: {dst.shape}")
    else:
        print("No edges in the graph")
    
    # Convert to DGL format
    dgl_graph = embedding.to_dgl(embedding_dim=64)
    print(f"DGL graph: {dgl_graph}")
    print(f"Node features 'x' shape: {dgl_graph.ndata['x'].shape}")
    if 'edge_attr' in dgl_graph.edata and dgl_graph.num_edges() > 0:
        print(f"Edge features 'edge_attr' shape: {dgl_graph.edata['edge_attr'].shape}")
    else:
        print("No edge features (no edges in graph)")
    
    # Get embeddings for transformer input
    state = embedding.get_state_embedding(embedding_dim=256)
    node_sequence = state['node_embeddings']  # [num_nodes, embedding_dim]
    graph_context = state['graph_embedding']  # [embedding_dim]
    
    print(f"Node sequence shape: {node_sequence.shape}")
    print(f"Graph context shape: {graph_context.shape}")
    
    # Test with some actions to create edges
    print("\n--- Testing with graph actions ---")
    actions = topology.act()
    print(f"Actions taken: {actions}")
    print(f"After actions - Number of nodes: {topology.graph.num_nodes()}")
    print(f"After actions - Number of edges: {topology.graph.num_edges()}")
    
    # Get updated embeddings
    updated_state = embedding.get_rl_ready_state(embedding_dim=128)
    print(f"Updated graph embedding shape: {updated_state['graph_embedding'].shape}")
    print(f"Updated node embeddings shape: {updated_state['node_embeddings'].shape}")