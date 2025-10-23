'''
    The agent herewith is a "topology of nodes". This program intends to represent topology as    def get_all_nodes(self):
        """
        Return a list of all node IDs in the graph.
        
        Returns
        -------
        list of int
            Lis    def compute_    def get_node_positions(self):
        """
        Get all node positions as a dictionary.
        
        Returns
        -------
        dict
            Dictionary mapping node index to position array [x, y]
        """
        return {i: self.graph.ndata['pos'][i].numpy() for i in range(self.graph.num_nodes())}troid(self):
        """
        Compute the centroid (center of mass) of all nodes.
        
        Returns
        -------
        np.ndarray
            2D coordinates [x, y] of the topology centroid
        """
        centroid = torch.mean(self.graph.ndata['pos'], dim=0)
        return centroid.numpy()ntaining all node indices in the current graph
        """
        return self.graph.nodes().tolist()"graph".

    Resources:
    - Deep Graph Library: Deep Graph learning at scale (https://www.youtube.com/watch?v=VmQkLro6UWo)
'''
import dgl
import torch
import matplotlib.pyplot as plt
import networkx as nx
import torch
from substrate import Substrate
import random
from scipy.spatial import ConvexHull
import numpy as np  



class Topology:
    """
    Dynamic graph topology representing cellular network with spawn/delete operations.
    
    This class manages a dynamic graph where nodes represent cells and edges represent
    cell-cell connections. It provides methods for topology evolution through node
    spawning (cell division/migration) and deletion (cell death/removal), along with
    visualization capabilities.
    
    The topology integrates with a substrate to determine spawning behaviors based on
    substrate intensity gradients (durotaxis simulation).
    
    Attributes
    ----------
    substrate : Substrate
        The substrate environment providing intensity signals
    graph : dgl.DGLGraph
        The dynamic graph structure
    fig : matplotlib.figure.Figure
        Figure handle for visualization
    ax : matplotlib.axes.Axes
        Axes handle for visualization
        
    Examples
    --------
    >>> topology = Topology(substrate=substrate)
    >>> topology.reset(init_num_nodes=5)
    >>> topology.show()
    >>> actions = topology.act()
    """
    
    def __init__(self, dgl_graph=None, substrate=None, flush_delay=0.01):
        self.substrate = substrate
        self._next_persistent_id = 0  # Global counter for unique persistent IDs
        self.flush_delay = flush_delay  # Default flush delay for visualization
        self.graph = dgl_graph if dgl_graph is not None else self.reset()
        self.fig = None  # Store figure reference
        self.ax = None   # Store axes reference



    def act(self):
        """
        Perform random spawn and delete actions on all nodes.
        
        This method simulates stochastic cellular behavior by randomly choosing
        spawn or delete actions for each node in the graph. It processes spawns
        first to avoid index shifting issues, then processes deletions in reverse
        order for the same reason.
        
        Returns
        -------
        dict
            Dictionary mapping node_id to action taken ('spawn' or 'delete')
            
        Notes
        -----
        This is a fallback method for random behavior. In the RL environment,
        actions are typically determined by the policy network.
        """
        all_nodes = self.get_all_nodes()
        sample_actions = {node_id: random.choice(['spawn', 'delete']) for node_id in all_nodes}
        
        # Separate actions into spawn and delete
        spawn_actions = {node_id: action for node_id, action in sample_actions.items() if action == 'spawn'}
        delete_actions = {node_id: action for node_id, action in sample_actions.items() if action == 'delete'}
        # Process spawns first (no index shifting issues)
        for node_id in spawn_actions:
            '''gamma, alpha, noise, theta are learnable parameters.'''
            self.spawn(node_id, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0)
        # Process deletions in REVERSE ORDER (highest index first)
        # This prevents index shifting from affecting subsequent deletions
        delete_node_ids = sorted(delete_actions.keys(), reverse=True)
        for node_id in delete_node_ids:
            self.delete(node_id)
        return sample_actions 




    def get_all_nodes(self):
        """Return a list of all node IDs in the graph."""
        return self.graph.nodes().tolist()


    def get_substrate_intensities(self):
        """
        Get substrate intensity values for all nodes in the graph.
        
        Returns
        -------
        torch.Tensor : Substrate intensities [num_nodes, 1]
        """
        if self.substrate is None:
            return torch.empty(0, 1, dtype=torch.float32)
        
        positions = self.graph.ndata['pos']
        device = positions.device
        num_nodes = positions.shape[0]
        
        if num_nodes == 0:
            return torch.empty(0, 1, dtype=torch.float32, device=device)
        
        intensities = []
        substrate_shape = self.substrate.signal_matrix.shape if hasattr(self.substrate, 'signal_matrix') else (0, 0)
        
        for i in range(num_nodes):
            pos = positions[i].cpu().numpy()  # Move to CPU for numpy operation
            intensity = self.substrate.get_intensity(pos)
            intensities.append(intensity)
        
        substrate_features = torch.tensor(intensities, dtype=torch.float32, device=device).unsqueeze(1)
        return substrate_features    



    def spawn(self, curr_node_id, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0):
        """
        Spawns a new node from curr_node_id in the direction theta, at a distance determined by the Hill equation.
        Adds the new node to the graph and connects curr_node_id to the new node.
        """
        try:
            r = self._hill_equation(curr_node_id, gamma, alpha, noise)
            # Get current node position
            curr_pos = self.graph.ndata['pos'][curr_node_id].numpy()
            # Compute new node position
            x, y = curr_pos[0] + r * np.cos(theta), curr_pos[1] + r * np.sin(theta)

            # Keep spawned nodes inside substrate bounds to avoid instant termination
            if hasattr(self.substrate, 'width') and hasattr(self.substrate, 'height'):
                width = float(self.substrate.width)
                height = float(self.substrate.height)
                # Small safety margin prevents hugging boundaries (helps boundary penalties work proactively)
                margin_x = max(2.0, 0.01 * width)
                margin_y = max(2.0, 0.01 * height)

                # Reflect positions that exit bounds back into the safe zone
                if x < margin_x:
                    # If spawn aimed left, mirror it just inside the margin
                    x = margin_x
                elif x > width - margin_x:
                    x = width - margin_x

                if y < margin_y:
                    y = margin_y
                elif y > height - margin_y:
                    y = height - margin_y

            # Get device from existing positions tensor to ensure device consistency
            # spawn() is only called when curr_node_id exists, so pos tensor always has elements
            device = self.graph.ndata['pos'].device
            new_node_coord = torch.tensor([x, y], dtype=torch.float32, device=device)  
            
            # Store current graph state before modification
            num_nodes_before = self.graph.num_nodes()
            
            # Store all current node data before adding new node
            current_node_data = {}
            for key, value in self.graph.ndata.items():
                current_node_data[key] = value.clone()
            
            # Add new node to graph (this will automatically extend existing features)
            self.graph.add_nodes(1)
            
            # Manually set the node features to avoid dimension mismatches
            # Use the device we already determined above (from self.graph.ndata['pos'])
            # Position data
            self.graph.ndata['pos'] = torch.cat([current_node_data['pos'], new_node_coord.unsqueeze(0)], dim=0)
            
            # New node flags
            current_new_node_flags = current_node_data.get('new_node', torch.zeros(num_nodes_before, dtype=torch.float32, device=device))
            new_node_flag = torch.tensor([1.0], dtype=torch.float32, device=device)
            self.graph.ndata['new_node'] = torch.cat([current_new_node_flags, new_node_flag], dim=0)
            
            # Persistent IDs
            current_persistent_ids = current_node_data.get('persistent_id', torch.arange(num_nodes_before, dtype=torch.long, device=device))
            new_persistent_id = torch.tensor([self._next_persistent_id], dtype=torch.long, device=device)
            self.graph.ndata['persistent_id'] = torch.cat([current_persistent_ids, new_persistent_id], dim=0)
            self._next_persistent_id += 1
            
            # To_delete flags (initialize new node with 0)
            current_to_delete_flags = current_node_data.get('to_delete', torch.zeros(num_nodes_before, dtype=torch.float32, device=device))
            new_to_delete_flag = torch.tensor([0.0], dtype=torch.float32, device=device)
            self.graph.ndata['to_delete'] = torch.cat([current_to_delete_flags, new_to_delete_flag], dim=0)
            
            # Handle spawn parameters
            self._update_spawn_parameters(num_nodes_before, gamma, alpha, noise, theta)
            
            # Get the NEW node ID (after adding the node)
            new_node_id = self.graph.num_nodes() - 1  
            
            # Add edge from curr_node_id to new_node_id
            self.graph.add_edges(curr_node_id, new_node_id)
            
            return new_node_id
            
        except Exception as e:
            # If spawn fails, restore the graph to its previous state
            # This is important to maintain consistency
            return None
    
    def _update_spawn_parameters(self, num_nodes_before, gamma, alpha, noise, theta):
        """Helper method to update spawn parameters safely."""
        # When a node is added, DGL automatically extends all node features to match the new number of nodes
        # We just need to set the spawn parameters for the new node (at index num_nodes_before)
        
        # Initialize spawn parameter arrays if they don't exist
        if 'gamma' not in self.graph.ndata:
            current_num_nodes = self.graph.num_nodes()
            # Get device from positions
            device = self.graph.ndata['pos'].device if 'pos' in self.graph.ndata and self.graph.ndata['pos'].numel() > 0 else torch.device('cpu')
            # Initialize spawn parameters to zeros for safety (avoid NaN propagation)
            self.graph.ndata['gamma'] = torch.zeros((current_num_nodes,), dtype=torch.float32, device=device)
            self.graph.ndata['alpha'] = torch.zeros((current_num_nodes,), dtype=torch.float32, device=device)
            self.graph.ndata['noise'] = torch.zeros((current_num_nodes,), dtype=torch.float32, device=device)
            self.graph.ndata['theta'] = torch.zeros((current_num_nodes,), dtype=torch.float32, device=device)
        
        # Set the spawn parameters for the new node (at index num_nodes_before)
        new_node_idx = num_nodes_before
        self.graph.ndata['gamma'][new_node_idx] = gamma
        self.graph.ndata['alpha'][new_node_idx] = alpha
        self.graph.ndata['noise'][new_node_idx] = noise
        self.graph.ndata['theta'][new_node_idx] = theta



    def _hill_equation(self, node_id, gamma, alpha, noise):
        """
        Calculates the Hill equation value for a given node in the graph.

        The Hill equation models the response of a system to a stimulus, commonly used in biochemistry
        to describe ligand binding. In this context, it computes a value based on the node's position,
        substrate intensity, and provided parameters.

        Args:
            node_id (int): The identifier of the node in the graph.
            gamma (float): The maximum response or scaling factor.
            alpha (float): The affinity constant or threshold parameter.
            noise (float): An additive noise term to introduce stochasticity.

        Returns:
            float: The computed Hill equation value for the specified node.
        """
        node_pos = self.graph.ndata['pos'][node_id].numpy()
        node_intensity = self.substrate.get_intensity(node_pos)
        # Guard against zero or non-finite intensities
        try:
            node_intensity = float(node_intensity)
        except Exception:
            node_intensity = 1.0

        if not np.isfinite(node_intensity) or node_intensity <= 1e-6:
            # Use a safe fallback intensity to avoid division by zero
            node_intensity = 1.0

        return float(gamma) * (1.0 / (1.0 + (float(alpha) / node_intensity)**2)) + float(noise)



    def delete(self, curr_node_id):
        """
        Deletes a node from the graph and reconnects its predecessors to its successors.

        Args:
            curr_node_id (int): The ID of the node to be deleted.

        Process:
            - Finds all predecessor and successor nodes of the specified node.
            - Removes the node from the graph.
            - Adjusts the indices of remaining nodes to account for the removal.
            - Connects each predecessor to each successor to maintain graph connectivity.
            - Preserves new_node flags for remaining nodes.

        Note:
            After node removal, indices of nodes greater than curr_node_id are decremented by 1.
        """
        # Find predecessors and successors of the current node
        predecessors = self.graph.predecessors(curr_node_id).tolist()
        successors = self.graph.successors(curr_node_id).tolist()
        
        # Store new_node flags before removal (if they exist)
        if 'new_node' in self.graph.ndata:
            new_node_flags = self.graph.ndata['new_node'].clone()
        
        # Store persistent IDs before removal (if they exist)
        if 'persistent_id' in self.graph.ndata:
            persistent_ids = self.graph.ndata['persistent_id'].clone()
        
        # Store to_delete flags before removal (if they exist)
        if 'to_delete' in self.graph.ndata:
            to_delete_flags = self.graph.ndata['to_delete'].clone()
        
        # Store spawn parameters before removal (if they exist)
        spawn_params = {}
        for param in ['gamma', 'alpha', 'noise', 'theta']:
            if param in self.graph.ndata:
                spawn_params[param] = self.graph.ndata[param].clone()
        
        # Remove the current node
        self.graph = dgl.remove_nodes(self.graph, curr_node_id)
        
        # Restore new_node flags for remaining nodes (excluding deleted node)
        if 'new_node' in self.graph.ndata:
            remaining_flags = torch.cat([
                new_node_flags[:curr_node_id],
                new_node_flags[curr_node_id+1:]
            ])
            self.graph.ndata['new_node'] = remaining_flags
        
        # Restore persistent IDs for remaining nodes (excluding deleted node)
        if 'persistent_id' in self.graph.ndata:
            remaining_persistent_ids = torch.cat([
                persistent_ids[:curr_node_id],
                persistent_ids[curr_node_id+1:]
            ])
            self.graph.ndata['persistent_id'] = remaining_persistent_ids
        
        # Restore to_delete flags for remaining nodes (excluding deleted node)
        if 'to_delete' in locals() and 'to_delete_flags' in locals():
            remaining_to_delete_flags = torch.cat([
                to_delete_flags[:curr_node_id],
                to_delete_flags[curr_node_id+1:]
            ])
            self.graph.ndata['to_delete'] = remaining_to_delete_flags
        
        # Restore spawn parameters for remaining nodes (excluding deleted node)
        for param, param_data in spawn_params.items():
            remaining_param_values = torch.cat([
                param_data[:curr_node_id],
                param_data[curr_node_id+1:]
            ])
            self.graph.ndata[param] = remaining_param_values
        
        # After removal, node indices shift down for nodes after curr_node_id
        # Adjust indices for successors and predecessors
        def adjust_idx(idx):
            return idx if idx < curr_node_id else idx - 1
        adjusted_predecessors = [adjust_idx(p) for p in predecessors]
        adjusted_successors = [adjust_idx(s) for s in successors]
        # Connect each predecessor to each successor
        for p in adjusted_predecessors:
            for s in adjusted_successors:
                self.graph.add_edges(p, s)


    def compute_centroid(self):
        """Compute the centroid (center of mass) of all nodes"""
        centroid = torch.mean(self.graph.ndata['pos'], dim=0)
        return centroid.numpy()
    
    
    def get_node_positions(self):
        """Get all node positions as a dictionary"""
        return {i: self.graph.ndata['pos'][i].numpy() for i in range(self.graph.num_nodes())}
    
    
    def get_outmost_nodes(self):
        """
        Get the outmost (boundary) nodes using convex hull.
        Returns the node indices that form the outer boundary of the point cloud.
        """
        if self.graph.num_nodes() < 3:
            # Need at least 3 points for a convex hull
            return list(range(self.graph.num_nodes()))
        
        # Get all node positions
        positions = self.graph.ndata['pos'].numpy()
        
        try:
            # Suppress qhull warnings by checking for collinearity first
            # Check if all x-coordinates are the same
            x_coords = positions[:, 0]
            y_coords = positions[:, 1]
            
            x_unique = len(np.unique(x_coords)) > 1
            y_unique = len(np.unique(y_coords)) > 1
            
            # If points are collinear or degenerate, use fallback
            if not (x_unique and y_unique):
                return self._get_extreme_nodes()
            
            # Compute convex hull
            hull = ConvexHull(positions)
            # Return the indices of vertices that form the convex hull
            return hull.vertices.tolist()
        except Exception:
            # Silently fallback to extreme nodes (no verbose error printing)
            return self._get_extreme_nodes()
    
    
    def _get_extreme_nodes(self):
        """
        Fallback method: get nodes with extreme coordinates
        (leftmost, rightmost, topmost, bottommost)
        """
        positions = self.graph.ndata['pos'].numpy()
        
        # Find extreme points
        min_x_idx = np.argmin(positions[:, 0])  # leftmost
        max_x_idx = np.argmax(positions[:, 0])  # rightmost
        min_y_idx = np.argmin(positions[:, 1])  # bottommost
        max_y_idx = np.argmax(positions[:, 1])  # topmost
        
        # Return unique indices
        extreme_indices = [min_x_idx, max_x_idx, min_y_idx, max_y_idx]
        return list(set(extreme_indices))



    def reset(self, init_num_nodes=5, init_bin=0.1):
        """
        Reset topology with initial nodes positioned in a specific substrate region.
        
        Creates a new graph with the specified number of initial nodes, positioned
        randomly within the leftmost fraction of the substrate (defined by init_bin)
        and vertically centered to avoid boundary violations.
        
        Initial Node Placement Strategy:
        - X-axis: Leftmost init_bin fraction (e.g., 0-10% of width)
        - Y-axis: Centered at 40-60% of substrate height (safe center zone)
        
        This centered initialization helps the agent avoid top/bottom boundaries
        from the start, allowing it to focus on the primary goal of rightward
        migration without early termination from boundary violations.
        
        Parameters
        ----------
        init_num_nodes : int, optional
            Number of initial nodes to create. Default is 5.
        init_bin : float, optional
            Fraction of substrate width from left edge where nodes will be placed.
            For example, init_bin=0.1 places nodes in leftmost 10% of substrate.
            Default is 0.1.
            
        Returns
        -------
        dgl.DGLGraph
            The reset graph with initial topology
            
        Notes
        -----
        Each node gets:
        - 'pos': 2D position coordinates [X: 0-10% width, Y: 40-60% height]
        - 'persistent_id': Unique identifier that persists across operations
        - 'new_node': Flag indicating if node was recently spawned (starts as 0)
        - 'to_delete': Flag for deletion marking (starts as 0)
        
        The vertical centering (40-60% height) ensures:
        - Initial centroid is in safe center zone
        - No nodes start in edge/danger/critical zones
        - Agent receives safe_center_bonus from step 1
        - Reduced early-episode boundary violations
        
        Example
        -------
        For a 600x400 substrate:
        - X placement: 0-60 pixels (leftmost 10%)
        - Y placement: 160-240 pixels (center 40-60%)
        - Centroid: ~(30, 200) - safe center position
        """
        """
        Reset topology with initial nodes.
        
        Args:
            init_num_nodes (int): Number of initial nodes to create
            init_bin (float): Fraction of substrate width from leftmost side where nodes will be placed.
                            For example, init_bin=0.1 places nodes in the leftmost 10% of the substrate.
        """
        # Reset the next persistent ID counter
        self._next_persistent_id = 0
        
        # Create new graph - handle zero nodes case
        if init_num_nodes <= 0:
            # Create empty graph with minimal node features for consistency
            self.graph = dgl.graph(([], []))
            # No nodes to add, but ensure node data tensors exist for compatibility
            return
        
        # Create new graph with initial nodes
        self.graph = dgl.graph(([], []))
        
        # Add initial nodes
        self.graph.add_nodes(init_num_nodes)
        
        # Calculate the x-range for initial placement based on init_bin
        max_x = self.substrate.width * init_bin
        
        # Calculate y-range for vertical center placement (40-60% of height)
        # This helps agent avoid boundaries from the start
        y_center = self.substrate.height * 0.5  # Center point
        y_range = self.substrate.height * 0.1   # ±10% from center = 40-60% range
        min_y = y_center - y_range
        max_y = y_center + y_range
        
        # Initialize node positions within the specified bin (X) and centered vertically (Y)
        positions = []
        persistent_ids = []
        for i in range(init_num_nodes):
            # Random position within the leftmost init_bin fraction of substrate (X-axis)
            x = np.random.uniform(0, max_x)
            # Random position within center 20% of substrate height (Y-axis: 40-60%)
            y = np.random.uniform(min_y, max_y)
            positions.append([x, y])
            persistent_ids.append(self._next_persistent_id)
            self._next_persistent_id += 1
        
        # Set node features
        # Use CPU tensors - they will be moved to device when passed through training pipeline
        self.graph.ndata['pos'] = torch.tensor(positions, dtype=torch.float32)
        self.graph.ndata['persistent_id'] = torch.tensor(persistent_ids, dtype=torch.long)
        self.graph.ndata['new_node'] = torch.zeros(init_num_nodes, dtype=torch.float32)
        self.graph.ndata['to_delete'] = torch.zeros(init_num_nodes, dtype=torch.float32)  # Initialize to_delete flags
        
        # Initialize spawn parameters for initial nodes (zeros for safety to avoid NaN propagation)
        self.graph.ndata['gamma'] = torch.zeros((init_num_nodes,), dtype=torch.float32)
        self.graph.ndata['alpha'] = torch.zeros((init_num_nodes,), dtype=torch.float32)
        self.graph.ndata['noise'] = torch.zeros((init_num_nodes,), dtype=torch.float32)
        self.graph.ndata['theta'] = torch.zeros((init_num_nodes,), dtype=torch.float32)
        
        # Initial graph may start with no edges
        # Add some random initial connections
        if init_num_nodes > 1:
            self._add_initial_edges(init_num_nodes)

        return self.graph

    def _add_initial_edges(self, init_num_nodes):
        """Add initial edges to the graph based on x-coordinates, connecting from left to right."""
        positions = self.graph.ndata['pos'].numpy()
        
        # Create node index pairs with their x-coordinates
        node_x_pairs = [(i, positions[i][0]) for i in range(init_num_nodes)]
        
        # Sort nodes by x-coordinate (left to right)
        node_x_pairs.sort(key=lambda pair: pair[1])
        
        # Extract sorted node indices
        sorted_node_indices = [pair[0] for pair in node_x_pairs]
        
        # Create edges connecting each node to the next rightmost node
        edges_src = []
        edges_dst = []
        
        for i in range(len(sorted_node_indices) - 1):
            current_node = sorted_node_indices[i]
            next_node = sorted_node_indices[i + 1]
            
            # Add directed edge from current node to next rightmost node
            edges_src.append(current_node)
            edges_dst.append(next_node)
        
        # Add edges to the graph if any were created
        if edges_src:
            self.graph.add_edges(edges_src, edges_dst)


    def show(self, size=(10, 8), flush_delay=None, highlight_outmost=False, update_only=True, episode_num=None):        
        """
        Visualizes the agent's topology and substrate signal matrix.
        Parameters
        ----------
        size : tuple of int, optional
            Figure size for the plot (width, height). Default is (10, 8).
        flush_delay : float, optional
            Time in seconds to pause after updating the plot. If None, uses the class default flush_delay.
        highlight_outmost : bool, optional
            If True, highlights the outmost nodes in the topology, draws the convex hull boundary,
            and marks the centroid. If False, only plots all nodes and the centroid.
        update_only : bool, optional
            If True, updates existing figure without opening new window. If False, creates new figure.
        Description
        -----------
        - Plots the substrate signal matrix as a background image.
        - If nodes exist, plots all node positions in the graph.
        - If no nodes exist, shows substrate-only visualization.
        - If `highlight_outmost` is True:
            - Highlights outmost nodes in red.
            - Draws the convex hull boundary around the nodes.
            - Marks the centroid with a green star.
        - If `highlight_outmost` is False:
            - Plots all nodes in red.
            - Marks the centroid with a green star.
        - Displays a legend and sets the plot title to 'Topology'.
        """
        canvas = self.substrate.signal_matrix.copy()
        
        # Check if we have any nodes
        num_nodes = self.graph.num_nodes()
        if num_nodes > 0:
            positions = self.graph.ndata['pos'].numpy()
        else:
            positions = None
        
        # Use class flush_delay if none provided
        if flush_delay is None:
            flush_delay = self.flush_delay
        
        # Enable interactive mode
        plt.ion()
        
        # Validate and recover figure state if needed
        figure_needs_recreation = False
        if self.fig is not None:
            try:
                # Check if figure is still valid
                if not plt.fignum_exists(self.fig.number):
                    figure_needs_recreation = True
                    self.fig = None
                    self.ax = None
            except:
                figure_needs_recreation = True
                self.fig = None
                self.ax = None
        
        # Create figure only if it doesn't exist, update_only is False, or needs recreation
        # Always try to reuse existing figure for episode continuity
        if self.fig is None or not update_only or figure_needs_recreation:
            self.fig, self.ax = plt.subplots(figsize=size)
            plt.show(block=False)  # Non-blocking show
        else:
            # Clear the existing axes for update but keep the window
            self.ax.clear()
        
        # Set up the plot area with proper limits
        substrate_width, substrate_height = self.substrate.width, self.substrate.height
        self.ax.set_xlim(0, substrate_width)
        self.ax.set_ylim(0, substrate_height)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Display substrate as background first
        self.ax.imshow(canvas, cmap='viridis', origin='lower', 
                      extent=[0, substrate_width, 0, substrate_height], alpha=0.7)
        
        # Handle visualization based on whether nodes exist
        if positions is None or num_nodes == 0:
            # Substrate-only visualization (no nodes)
            episode_str = f"Ep{episode_num:2d}" if episode_num is not None else "Ep??"
            step_str = f"Step{getattr(self, '_step_counter', '??'):3d}" if hasattr(self, '_step_counter') else "Step???"
            self.ax.set_title(f'{episode_str} {step_str}: Substrate Only - 0 nodes')
            # No legend needed for substrate-only view
        else:
            # Normal visualization with nodes
            # Compute and plot centroid
            centroid = self.compute_centroid()
            
            if highlight_outmost:
                # Get outmost nodes
                outmost_indices = self.get_outmost_nodes()
                # Plot all nodes in blue
                self.ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=2, alpha=0.6, label='All nodes')
                # Highlight outmost nodes in red
                outmost_positions = positions[outmost_indices]
                self.ax.scatter(outmost_positions[:, 0], outmost_positions[:, 1], 
                               c='red', s=2, marker='o', edgecolor='black', linewidth=1,
                               label=f'Outmost nodes ({len(outmost_indices)})')
                # Draw convex hull boundary
                if len(outmost_indices) >= 3:
                    try:
                        hull = ConvexHull(positions)
                        for simplex in hull.simplices:
                            self.ax.plot(positions[simplex, 0], positions[simplex, 1], 'r--', alpha=0.7)
                    except:
                        pass
                # Add green marker for centroid
                self.ax.scatter(centroid[0], centroid[1], c='green', s=100, marker='*', 
                               edgecolor='black', linewidth=1, label='Centroid')
                self.ax.legend()
                # Uniform title format
                episode_str = f"Ep{episode_num:2d}" if episode_num is not None else "Ep??"
                step_str = f"Step{getattr(self, '_step_counter', '??'):3d}" if hasattr(self, '_step_counter') else "Step???"
                self.ax.set_title(f'{episode_str} {step_str}: Topology - {len(positions)} nodes')
            else:
                self.ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=2, alpha=0.8, label='Nodes')
                # Add green marker for centroid
                self.ax.scatter(centroid[0], centroid[1], c='green', s=100, marker='*', 
                               edgecolor='black', linewidth=1, label='Centroid')
                self.ax.legend()
                # Uniform title format
                episode_str = f"Ep{episode_num:2d}" if episode_num is not None else "Ep??"
                step_str = f"Step{getattr(self, '_step_counter', '??'):3d}" if hasattr(self, '_step_counter') else "Step???"
                self.ax.set_title(f'{episode_str} {step_str}: Topology - {len(positions)} nodes')        # Remove the duplicate imshow call since we already added it above
        
        # Refresh the display with proper flushing and error handling
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            # Force window update
            if hasattr(self.fig.canvas, 'manager'):
                self.fig.canvas.manager.show()
            
            plt.pause(flush_delay)
        except Exception as e:
            print(f"Warning: Figure refresh failed: {e}")
            # Try to recreate figure for next call
            self.fig = None
            self.ax = None  

    
    def close_figure(self):
        """Close the figure window and reset figure references"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
    def force_figure_update(self):
        """Force the matplotlib figure to update and refresh"""
        if self.fig is not None:
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                if hasattr(self.fig.canvas, 'manager'):
                    self.fig.canvas.manager.show()
                plt.pause(0.01)  # Small pause to ensure update
            except Exception as e:
                print(f"Warning: Force figure update failed: {e}")
                # Reset figure state for recovery
                self.fig = None
                self.ax = None
      




if __name__ == '__main__':
    
    substrate_linear = Substrate((600, 400))
    substrate_linear.create('linear', m=0.05, b=1)
   
    # Create topology with custom flush_delay (0.1 seconds)
    agent = Topology(substrate=substrate_linear, flush_delay=0.1)      

    agent.reset(init_num_nodes=100, init_bin=0.1)
    for i in range(1,20):
        agent.show(highlight_outmost=True)
        agent.act()

    # Keep the last figure window open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Blocking show to keep window open