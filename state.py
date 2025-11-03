import torch
import torch.nn.functional as F
import dgl 
from scipy.spatial import ConvexHull
from device import cpu_numpy


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
        
        # Track previous state for delta features
        self._prev_centroid_x = None
        self._prev_num_nodes = None
        self._prev_avg_intensity = None
    
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
        Get IMMUTABLE snapshot of topology state features.
        All tensors are cloned/detached and moved to consistent device to prevent aliasing.
        
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
        dict : Immutable state dictionary containing:
            - 'graph_features': Graph-level feature vector [G] (cloned)
            - 'node_features': Node feature matrix [N, F] (cloned)
            - 'edge_attr': Edge feature matrix [E, F] (cloned)
            - 'edge_index': Edge connectivity (src, dst) tuples (cloned)
            - 'persistent_id': Node persistent IDs [N] (cloned)
            - 'to_delete': Node deletion flags [N] (cloned)
            - 'num_nodes': Number of nodes (int)
            - 'num_edges': Number of edges (int)
            - 'centroid_x': Graph centroid x-coordinate (float)
            - 'goal_x': Goal x-position (float)
            
        Note: 'topology' reference is NOT included to prevent aliasing.
        """
        
        # Determine device from topology (consistent device for all tensors)
        device = self.topology.device if self.topology is not None else torch.device('cpu')
        
        # Extract features (these return tensors on the graph's device)
        node_features = self._get_node_features(
            include_substrate=include_substrate,
            node_age=node_age,
            node_stagnation=node_stagnation
        )
        edge_features = self._get_edge_features()
        graph_features = self._get_graph_features()
        
        # Get edge connectivity for GNN
        src, dst = self.graph.edges()
        
        # Get persistent IDs - CLONE to create immutable snapshot
        persistent_ids = self.graph.ndata.get('persistent_id', None)
        if persistent_ids is not None:
            persistent_ids = persistent_ids.clone().detach().to(device)
        else:
            persistent_ids = torch.empty(0, dtype=torch.long, device=device)
        
        # Get to_delete flags - CLONE to create immutable snapshot
        to_delete_flags = self.graph.ndata.get('to_delete', None)
        if to_delete_flags is not None:
            to_delete_flags = to_delete_flags.clone().detach().to(device)
        else:
            to_delete_flags = torch.zeros(self.graph.num_nodes(), dtype=torch.float32, device=device)
        
        # Calculate centroid x-coordinate for distance-based rewards
        centroid_x = 0.0
        avg_intensity = 0.0
        if self.graph.num_nodes() > 0:
            positions = self.graph.ndata['pos']
            centroid_x = float(torch.mean(positions[:, 0]).item())
            
            # Calculate average substrate intensity if available
            if 'substrate_intensity' in self.graph.ndata:
                avg_intensity = float(torch.mean(self.graph.ndata['substrate_intensity']).item())
        
        # Calculate delta features (change from previous state)
        delta_centroid_x = 0.0
        delta_num_nodes = 0
        delta_avg_intensity = 0.0
        
        if self._prev_centroid_x is not None:
            delta_centroid_x = centroid_x - self._prev_centroid_x
        if self._prev_num_nodes is not None:
            delta_num_nodes = self.graph.num_nodes() - self._prev_num_nodes
        if self._prev_avg_intensity is not None:
            delta_avg_intensity = avg_intensity - self._prev_avg_intensity
        
        # Update previous values for next iteration
        self._prev_centroid_x = centroid_x
        self._prev_num_nodes = self.graph.num_nodes()
        self._prev_avg_intensity = avg_intensity
        
        # Calculate goal x-position (rightmost substrate boundary)
        goal_x = float(self.substrate.width - 1) if self.substrate is not None else 1.0
        
        # Build immutable state dict with ALL tensors cloned/detached/device-consistent
        # DO NOT include 'topology' to prevent aliasing
        state = {
            'graph_features': graph_features.clone().detach().to(device),  # [G]
            'node_features': node_features.clone().detach().to(device),    # [N, F]
            'edge_attr': edge_features.clone().detach().to(device),        # [E, F]
            'edge_index': (src.clone().detach().to(device), dst.clone().detach().to(device)),  # Cloned tuple
            'persistent_id': persistent_ids,                                # [N] (already cloned above)
            'to_delete': to_delete_flags,                                   # [N] (already cloned above)
            'num_nodes': int(self.graph.num_nodes()),
            'num_edges': int(self.graph.num_edges()),
            'centroid_x': centroid_x,
            'goal_x': goal_x,
            # Delta features for temporal context (reduces partial observability)
            'delta_centroid_x': delta_centroid_x,
            'delta_num_nodes': delta_num_nodes,
            'delta_avg_intensity': delta_avg_intensity
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
            # Infer device from topology
            device = self.topology.device if self.topology is not None else torch.device('cpu')
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
            # Infer device from topology
            device = self.topology.device if self.topology is not None else torch.device('cpu')
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
        Extract graph-level features with RICHER pooling for better representation.
        
        Returns
        -------
        torch.Tensor : Graph feature vector [feature_dim]
            Features include:
            - Basic statistics: num_nodes, num_edges, density
            - Spatial statistics: centroid, bounding box, convex hull area
            - Topological statistics: average degree
            - ENHANCED: mean/max/sum pooling of node features for richer context
        """
        positions = self.graph.ndata['pos']
        
        features = []
        
        # Infer device from topology for device consistency
        device = self.topology.device if self.topology is not None else (positions.device if positions.numel() > 0 else torch.device('cpu'))
        
        # 1. Basic graph statistics
        num_nodes = torch.tensor([self.graph.num_nodes()], dtype=torch.float32, device=device)
        num_edges = torch.tensor([self.graph.num_edges()], dtype=torch.float32, device=device)
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else torch.tensor([0.0], device=device)
        
        features.extend([num_nodes, num_edges, density])
        
        # 2. Spatial statistics
        if num_nodes > 0:
            centroid = torch.mean(positions, dim=0).to(device)  # [2] for 2D positions
            
            # Bounding box features
            bbox_min = torch.min(positions, dim=0)[0].to(device)  # [2]
            bbox_max = torch.max(positions, dim=0)[0].to(device)  # [2]
            bbox_size = bbox_max - bbox_min  # [2]
            bbox_area = torch.prod(bbox_size) if len(bbox_size) > 1 else bbox_size[0]  # scalar
            bbox_area = bbox_area.unsqueeze(0) if bbox_area.dim() == 0 else bbox_area  # [1]
            bbox_area = bbox_area.to(device)  # Ensure on correct device
            
            # ENHANCED: Add max pooling of positions for better shape representation
            pos_max = torch.max(positions, dim=0)[0].to(device)  # [2] - rightmost and highest positions
            pos_min = torch.min(positions, dim=0)[0].to(device)  # [2] - leftmost and lowest positions
        else:
            centroid = torch.zeros(2, device=device)
            bbox_min = torch.zeros(2, device=device)
            bbox_max = torch.zeros(2, device=device)
            bbox_size = torch.zeros(2, device=device)
            bbox_area = torch.zeros(1, device=device)
            pos_max = torch.zeros(2, device=device)
            pos_min = torch.zeros(2, device=device)
        
        # Add spatial features
        features.append(centroid)     # [2]
        features.append(bbox_min)     # [2]
        features.append(bbox_max)     # [2]
        features.append(bbox_size)    # [2]
        features.append(bbox_area)    # [1]
        
        # 3. Convex hull area
        try:
            if num_nodes >= 3:
                hull = ConvexHull(cpu_numpy(positions))  # Convert to CPU for scipy safely
                hull_area = torch.tensor([hull.volume], dtype=torch.float32, device=device)  # [1]
            else:
                hull_area = torch.tensor([0.0], device=device)  # [1]
        except Exception:
            hull_area = torch.tensor([0.0], device=device)  # [1]
        
        features.append(hull_area)
        
        # 4. Average node degree
        if num_nodes > 0:
            degrees = self._get_node_degrees()
            avg_degree = torch.mean(degrees).unsqueeze(0).to(device)  # [1]
            max_degree = torch.max(degrees).unsqueeze(0).to(device)  # [1] - ENHANCED
            sum_degree = torch.sum(degrees).unsqueeze(0).to(device)  # [1] - ENHANCED
        else:
            avg_degree = torch.tensor([0.0], device=device)  # [1]
            max_degree = torch.tensor([0.0], device=device)  # [1]
            sum_degree = torch.tensor([0.0], device=device)  # [1]
        
        features.append(avg_degree)
        features.append(max_degree)  # NEW
        features.append(sum_degree)  # NEW
        
        # 5. ENHANCED: Substrate intensity statistics (mean/max/sum pooling)
        if num_nodes > 0 and 'substrate_intensity' in self.graph.ndata:
            intensities = self.graph.ndata['substrate_intensity'].unsqueeze(1) if self.graph.ndata['substrate_intensity'].dim() == 1 else self.graph.ndata['substrate_intensity']
            mean_intensity = torch.mean(intensities).unsqueeze(0).to(device)  # [1]
            max_intensity = torch.max(intensities).unsqueeze(0).to(device)   # [1]
            sum_intensity = torch.sum(intensities).unsqueeze(0).to(device)   # [1]
        else:
            mean_intensity = torch.tensor([0.0], device=device)
            max_intensity = torch.tensor([0.0], device=device)
            sum_intensity = torch.tensor([0.0], device=device)
        
        features.append(mean_intensity)  # NEW
        features.append(max_intensity)   # NEW
        features.append(sum_intensity)   # NEW
        
        return torch.cat(features, dim=0)

    def _get_node_degrees(self):
        """Get node degree features."""
        device = self.topology.device if self.topology is not None else torch.device('cpu')
        if self.graph.num_nodes() == 0:
            return torch.empty(0, 2, dtype=torch.float32, device=device)
        
        in_degrees = self.graph.in_degrees().float().unsqueeze(1).to(device)
        out_degrees = self.graph.out_degrees().float().unsqueeze(1).to(device)
        return torch.cat([in_degrees, out_degrees], dim=1)

    def _get_node_centralities(self):
        """Get basic centrality measures."""
        num_nodes = self.graph.num_nodes()
        if num_nodes == 0:
            device = self.topology.device if self.topology is not None else torch.device('cpu')
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
            device = self.topology.device if self.topology is not None else torch.device('cpu')
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
        # Infer device from topology
        device = self.topology.device if self.topology is not None else torch.device('cpu')
        
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
        # Infer device from topology
        device = self.topology.device if self.topology is not None else torch.device('cpu')
        
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
    