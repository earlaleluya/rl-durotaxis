import torch
import torch.nn.functional as F
import dgl 
from scipy.spatial import ConvexHull


class TopologyState:
    """
    Topology State class that extracts graph features, node features, and edge features
    from Topology instances for RL algorithms and graph neural networks.
    """
    
    def __init__(self, topology=None):
        """
        Initialize TopologyState with a Topology instance.
        
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

    def get_state_features(self, include_substrate=True, node_age=None, node_stagnation=None):
        """
        Get topology state features: graph features, node features, and edge features.
        
        Parameters
        ----------
        include_substrate : bool
            Whether to include substrate intensity in node features
        node_age : dict, optional
            Dictionary mapping persistent_id to node age (number of steps existed)
        node_stagnation : dict, optional
            Dictionary mapping persistent_id to stagnation info {'pos': array, 'count': int}
            
        Returns
        -------
        dict : State dictionary containing:
            - 'graph_features': Graph-level feature vector
            - 'node_features': Node feature matrix [num_nodes, node_feature_dim]
            - 'edge_attr': Edge feature matrix [num_edges, edge_feature_dim]
            - 'edge_index': Edge connectivity (src, dst) tuples for GNN
            - 'num_nodes': Number of nodes
            - 'num_edges': Number of edges
            - 'persistent_id': Node persistent IDs for tracking
        """
        
        # Extract the three main feature types
        node_features = self._get_node_features(
            include_substrate=include_substrate,
            node_age=node_age,
            node_stagnation=node_stagnation
        )
        edge_features = self._get_edge_features()
        graph_features = self._get_graph_features()
        
        # Get edge connectivity for GNN
        src, dst = self.graph.edges()
        
        # Get persistent IDs for node tracking
        persistent_ids = self.graph.ndata.get('persistent_id', None)
        
        state = {
            'topology': self.topology,                   
            'graph_features': graph_features,            # Shape: [graph_feature_dim]
            'node_features': node_features,              # Shape: [num_nodes, node_feature_dim]
            'edge_attr': edge_features,                  # Shape: [num_edges, edge_feature_dim]
            'edge_index': (src, dst),                    # Edge connectivity
            'persistent_id': persistent_ids,             # Node persistent IDs for tracking
            'num_nodes': self.graph.num_nodes(),
            'num_edges': self.graph.num_edges()
        }
        
        return state

    def _get_node_features(self, include_substrate=True, node_age=None, node_stagnation=None):
        """
        Extract node-level features for each node in the graph.
        
        Parameters
        ----------
        include_substrate : bool
            Whether to include substrate intensity in node features
        node_age : dict, optional
            Dictionary mapping persistent_id to node age
        node_stagnation : dict, optional
            Dictionary mapping persistent_id to stagnation counter
        
        Returns
        -------
        torch.Tensor : Node feature matrix [num_nodes, feature_dim]
            Features include:
            - Position coordinates [x, y]
            - Substrate intensity (optional)
            - Node degree [in_degree, out_degree]
            - Centrality measure
            - Distance from graph centroid
            - Boundary flag (convex hull membership)
            - New node flag (1.0 if newly spawned, 0.0 if existing)
            - Age (normalized, 0.0 if not available)
            - Stagnation count (normalized, 0.0 if not available)
        """
        positions = self.graph.ndata['pos']
        num_nodes = positions.shape[0]
        
        features = []
        
        # 1. Positional features
        features.append(positions)  # [x, y] coordinates
        
        # 2. Substrate features (optional)
        if include_substrate and self.substrate is not None:
            substrate_features = self.topology.get_substrate_intensities()
            features.append(substrate_features)
        
        # 3. Topological features
        degrees = self._get_node_degrees()
        features.append(degrees)
        
        # 4. Centrality measures
        centralities = self._get_node_centralities()
        features.append(centralities)
        
        # 5. Distance from centroid
        centroid = torch.mean(positions, dim=0)
        distances_from_centroid = torch.norm(positions - centroid, dim=1, keepdim=True)
        features.append(distances_from_centroid)
        
        # 6. Boundary membership
        boundary_flags = self._get_boundary_flags()
        features.append(boundary_flags)
        
        # 7. New node flag
        if 'new_node' in self.graph.ndata:
            new_node_flags = self.graph.ndata['new_node'].unsqueeze(1)
        else:
            # Default to all zeros if new_node flag not initialized
            # Infer device from positions
            device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata and self.graph.ndata['pos'].numel() > 0 else torch.device('cpu')
            new_node_flags = torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)
        features.append(new_node_flags)
        
        # 8. Age feature (normalized for numerical stability)
        age_features = self._get_age_features(num_nodes, node_age)
        features.append(age_features)
        
        # 9. Stagnation feature (normalized for numerical stability)
        stagnation_features = self._get_stagnation_features(num_nodes, node_stagnation)
        features.append(stagnation_features)
        
        return torch.cat(features, dim=1)

    def _get_edge_features(self):
        """
        Extract edge-level features.
        
        Returns
        -------
        torch.Tensor : Edge feature matrix [num_edges, feature_dim]
            Features include:
            - Euclidean distance between nodes
            - Normalized direction vector [dx, dy]
        """
        if self.graph.num_edges() == 0:
            # Infer device from positions if available
            device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata and self.graph.ndata['pos'].numel() > 0 else torch.device('cpu')
            return torch.empty(0, 3, dtype=torch.float32, device=device)
        
        # Get edge endpoints
        src, dst = self.graph.edges()
        positions = self.graph.ndata['pos']
        
        # Calculate edge features
        src_pos = positions[src]
        dst_pos = positions[dst]
        
        # 1. Euclidean distance
        distances = torch.norm(dst_pos - src_pos, dim=1, keepdim=True)
        
        # 2. Normalized direction vector
        direction = dst_pos - src_pos
        # Add small epsilon to avoid division by zero when src and dst are at same position
        direction_norm = F.normalize(direction + 1e-8, p=2, dim=1)
        
        edge_features = torch.cat([distances, direction_norm], dim=1)
        
        return edge_features

    def _get_graph_features(self):
        """
        Extract graph-level features.
        
        Returns
        -------
        torch.Tensor : Graph feature vector [feature_dim]
            Features include:
            - Basic statistics: num_nodes, num_edges, density
            - Spatial statistics: centroid, bounding box, convex hull area
            - Topological statistics: average degree
        """
        positions = self.graph.ndata['pos']
        
        features = []
        
        # Infer device from existing positions tensor for device consistency
        device = positions.device if positions.numel() > 0 else torch.device('cpu')
        
        # 1. Basic graph statistics
        num_nodes = torch.tensor([self.graph.num_nodes()], dtype=torch.float32, device=device)
        num_edges = torch.tensor([self.graph.num_edges()], dtype=torch.float32, device=device)
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else torch.tensor([0.0], device=device)
        
        features.extend([num_nodes, num_edges, density])
        
        # 2. Spatial statistics
        if num_nodes > 0:
            centroid = torch.mean(positions, dim=0)  # [2] for 2D positions
            
            # Bounding box features
            bbox_min = torch.min(positions, dim=0)[0]  # [2]
            bbox_max = torch.max(positions, dim=0)[0]  # [2]
            bbox_size = bbox_max - bbox_min  # [2]
            bbox_area = torch.prod(bbox_size) if len(bbox_size) > 1 else bbox_size[0]  # scalar
            bbox_area = bbox_area.unsqueeze(0) if bbox_area.dim() == 0 else bbox_area  # [1]
        else:
            centroid = torch.zeros(2, device=device)
            bbox_min = torch.zeros(2, device=device)
            bbox_max = torch.zeros(2, device=device)
            bbox_size = torch.zeros(2, device=device)
            bbox_area = torch.zeros(1, device=device)
        
        # Add spatial features
        features.append(centroid)     # [2]
        features.append(bbox_min)     # [2]
        features.append(bbox_max)     # [2]
        features.append(bbox_size)    # [2]
        features.append(bbox_area)    # [1]
        
        # 3. Convex hull area
        try:
            if num_nodes >= 3:
                hull = ConvexHull(positions.cpu().numpy())  # Convert to CPU for scipy
                hull_area = torch.tensor([hull.volume], dtype=torch.float32, device=device)  # [1]
            else:
                hull_area = torch.tensor([0.0], device=device)  # [1]
        except:
            hull_area = torch.tensor([0.0], device=device)  # [1]
        
        features.append(hull_area)
        
        # 4. Average node degree
        if num_nodes > 0:
            degrees = self._get_node_degrees()
            avg_degree = torch.mean(degrees).unsqueeze(0)  # [1]
        else:
            avg_degree = torch.tensor([0.0], device=device)  # [1]
        
        features.append(avg_degree)
        
        return torch.cat(features, dim=0)

    def _get_node_degrees(self):
        """Get node degree features."""
        if self.graph.num_nodes() == 0:
            device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata and self.graph.ndata['pos'].numel() > 0 else torch.device('cpu')
            return torch.empty(0, 2, dtype=torch.float32, device=device)
        
        in_degrees = self.graph.in_degrees().float().unsqueeze(1)
        out_degrees = self.graph.out_degrees().float().unsqueeze(1)
        return torch.cat([in_degrees, out_degrees], dim=1)

    def _get_node_centralities(self):
        """Get basic centrality measures."""
        num_nodes = self.graph.num_nodes()
        if num_nodes == 0:
            device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata and self.graph.ndata['pos'].numel() > 0 else torch.device('cpu')
            return torch.empty(0, 1, dtype=torch.float32, device=device)
        
        positions = self.graph.ndata['pos']
        device = positions.device  # Get device from positions tensor
        
        # Simple centrality: inverse of average distance to all other nodes
        centralities = []
        for i in range(num_nodes):
            if num_nodes == 1:
                centralities.append(1.0)
            else:
                pos_i = positions[i]
                distances = torch.norm(positions - pos_i, dim=1)
                non_zero_distances = distances[distances > 1e-6]  # exclude self and nearly-identical positions
                
                # Handle case where all nodes are at same position
                if len(non_zero_distances) == 0:
                    # All nodes at same position -> use default centrality
                    centrality = 1.0
                else:
                    avg_distance = torch.mean(non_zero_distances)
                    centrality = 1.0 / (avg_distance + 1e-6)  # avoid division by zero
                
                centralities.append(centrality.item() if isinstance(centrality, torch.Tensor) else centrality)
        
        return torch.tensor(centralities, dtype=torch.float32, device=device).unsqueeze(1)

    def _get_boundary_flags(self):
        """Get binary flags indicating if nodes are on the convex hull boundary."""
        num_nodes = self.graph.num_nodes()
        if num_nodes == 0:
            device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata and self.graph.ndata['pos'].numel() > 0 else torch.device('cpu')
            return torch.empty(0, 1, dtype=torch.float32, device=device)
        
        positions = self.graph.ndata['pos']
        device = positions.device
        boundary_flags = torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)
        
        try:
            outmost_indices = self.topology.get_outmost_nodes()
            boundary_flags[outmost_indices] = 1.0
        except Exception as e:
            # Silently handle convex hull errors (collinear points, etc.)
            # All nodes marked as non-boundary (zeros) in this case
            pass
        
        return boundary_flags

    def _get_age_features(self, num_nodes, node_age):
        """
        Extract and normalize age features for each node.
        
        Age represents how many steps a node has existed. Normalized by dividing by 100
        to keep values in a reasonable range for neural network training.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
            node_age (dict or None): Dictionary mapping persistent_id -> age (int).
                                     If None, all nodes get age 0.0.
        
        Returns:
            torch.Tensor: Age features [num_nodes, 1], normalized to ~[0, 1] range.
        """
        # Infer device from existing graph data
        device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata else torch.device('cpu')
        
        if num_nodes == 0:
            return torch.empty(0, 1, dtype=torch.float32, device=device)
        
        # Initialize with zeros (default age)
        age_features = torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)
        
        # If no age data provided, return zeros
        if node_age is None or len(node_age) == 0:
            return age_features
        
        # Get persistent IDs for nodes
        if 'persistent_id' not in self.graph.ndata:
            return age_features
        
        persistent_ids = self.graph.ndata['persistent_id'].cpu().numpy()
        
        # Fill in age values from the dictionary
        for i, pid in enumerate(persistent_ids):
            if pid in node_age:
                # Normalize age by dividing by 100 (typical episode length)
                # This keeps values roughly in [0, 1] range for most episodes
                age_features[i, 0] = node_age[pid] / 100.0
        
        return age_features

    def _get_stagnation_features(self, num_nodes, node_stagnation):
        """
        Extract and normalize stagnation features for each node.
        
        Stagnation represents how many consecutive steps a node has not moved.
        Normalized by dividing by 50 to keep values in a reasonable range.
        
        Args:
            num_nodes (int): Number of nodes in the graph.
            node_stagnation (dict or None): Dictionary mapping persistent_id -> {'pos': tuple, 'count': int}.
                                            If None, all nodes get stagnation 0.0.
        
        Returns:
            torch.Tensor: Stagnation features [num_nodes, 1], normalized to ~[0, 1] range.
        """
        # Infer device from existing graph data
        device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata else torch.device('cpu')
        
        if num_nodes == 0:
            return torch.empty(0, 1, dtype=torch.float32, device=device)
        
        # Initialize with zeros (default stagnation)
        stagnation_features = torch.zeros(num_nodes, 1, dtype=torch.float32, device=device)
        
        # If no stagnation data provided, return zeros
        if node_stagnation is None or len(node_stagnation) == 0:
            return stagnation_features
        
        # Get persistent IDs for nodes
        if 'persistent_id' not in self.graph.ndata:
            return stagnation_features
        
        persistent_ids = self.graph.ndata['persistent_id'].cpu().numpy()
        
        # Fill in stagnation values from the dictionary
        for i, pid in enumerate(persistent_ids):
            if pid in node_stagnation:
                # Extract count from stagnation dict (structure: {'pos': pos, 'count': count})
                stagnation_info = node_stagnation[pid]
                if isinstance(stagnation_info, dict) and 'count' in stagnation_info:
                    count = stagnation_info['count']
                else:
                    # Fallback: if it's just a number, use it directly
                    count = stagnation_info
                
                # Normalize stagnation by dividing by 50 (reasonable threshold)
                # Values >1.0 indicate highly stagnant nodes
                stagnation_features[i, 0] = count / 50.0
        
        return stagnation_features



if __name__ == '__main__':
    from topology import Topology
    from substrate import Substrate
    
    # Create substrate and topology
    substrate = Substrate((100, 50))
    substrate.create('linear', m=0.01, b=1.0)
    
    # Create topology with DGL graph
    topology = Topology(substrate=substrate)
    topology.reset(init_num_nodes=10)
    
    # Verify that we have a DGL graph
    print(f"Graph type: {type(topology.graph)}")
    print(f"Number of nodes: {topology.graph.num_nodes()}")
    print(f"Number of edges: {topology.graph.num_edges()}")
    print(f"Node positions shape: {topology.graph.ndata['pos'].shape}")
    
    # Create topology state extractor
    state_extractor = TopologyState(topology)
    
    # Extract state features
    state = state_extractor.get_state_features()
    
    print(f"\n--- Extracted State Features ---")
    print(f"Graph features shape: {state['graph_features'].shape}")
    print(f"Node features shape: {state['node_features'].shape}")
    print(f"Edge features shape: {state['edge_attr'].shape}")
    print(f"Number of nodes: {state['num_nodes']}")
    print(f"Number of edges: {state['num_edges']}")
    
    # Check edge index
    if state['num_edges'] > 0:
        src, dst = state['edge_index']
        print(f"Edge index - src shape: {src.shape}, dst shape: {dst.shape}")
    else:
        print("No edges in the graph")
    
    # Test with some actions to create edges
    print("\n--- Testing with graph actions ---")
    actions = topology.act()
    print(f"Actions taken: {actions}")
    print(f"After actions - Number of nodes: {topology.graph.num_nodes()}")
    print(f"After actions - Number of edges: {topology.graph.num_edges()}")
    
    # Extract features again after topology changes
    state_after = state_extractor.get_state_features()
    print(f"\n--- State Features After Actions ---")
    print(f"Graph features shape: {state_after['graph_features'].shape}")
    print(f"Node features shape: {state_after['node_features'].shape}")
    print(f"Edge features shape: {state_after['edge_attr'].shape}")
    print(f"Number of nodes: {state_after['num_nodes']}")
    print(f"Number of edges: {state_after['num_edges']}")
    