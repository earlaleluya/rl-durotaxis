'''
    The agent herewith is a "topology of nodes". This program intends to represent topology as a "graph".

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
    
    def __init__(self, graph=None, substrate=None):
        self.substrate = substrate
        self.graph = graph if graph is not None else self.reset()
        self.fig = None  # Store figure reference
        self.ax = None   # Store axes reference



    def act(self):
        """This method still simulates the sample actions to be taken, which based on random choice of either spawn or delete."""
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
        num_nodes = positions.shape[0]
        
        if num_nodes == 0:
            return torch.empty(0, 1, dtype=torch.float32)
        
        intensities = []
        for i in range(num_nodes):
            pos = positions[i].numpy()
            intensity = self.substrate.get_intensity(pos)
            intensities.append(intensity)
        
        substrate_features = torch.tensor(intensities, dtype=torch.float32).unsqueeze(1)
        return substrate_features    



    def spawn(self, curr_node_id, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0):
        """
        Spawns a new node from curr_node_id in the direction theta, at a distance determined by the Hill equation.
        Adds the new node to the graph and connects curr_node_id to the new node.
        """
        r = self._hill_equation(curr_node_id, gamma, alpha, noise)
        # Get current node position
        curr_pos = self.graph.ndata['pos'][curr_node_id].numpy()
        # Compute new node position
        x, y = curr_pos[0] + r * np.cos(theta), curr_pos[1] + r * np.sin(theta)
        new_node_coord = torch.tensor([x, y], dtype=torch.float32)  
        # Store current positions before adding node
        current_positions = self.graph.ndata['pos']
        # Add new node to graph
        self.graph.add_nodes(1)
        # Update position data by concatenating the new coordinate
        updated_positions = torch.cat([current_positions, new_node_coord.unsqueeze(0)], dim=0)
        self.graph.ndata['pos'] = updated_positions
        # Get the NEW node ID (after adding the node)
        new_node_id = self.graph.num_nodes() - 1  
        # Add edge from curr_node_id to new_node_id
        self.graph.add_edges(curr_node_id, new_node_id)
        return new_node_id



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
        return gamma * (1 / (1 + (alpha / node_intensity)**2)) + noise



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

        Note:
            After node removal, indices of nodes greater than curr_node_id are decremented by 1.
        """
        # Find predecessors and successors of the current node
        predecessors = self.graph.predecessors(curr_node_id).tolist()
        successors = self.graph.successors(curr_node_id).tolist()
        # Remove the current node
        self.graph = dgl.remove_nodes(self.graph, curr_node_id)
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




    def try_show(self, g):
        G = g.to_networkx()
        nx.draw(G, pos=g.ndata['pos'], with_labels=True, node_color='lightgreen', node_size=500, font_size=16)
        plt.show()



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
            # Compute convex hull
            hull = ConvexHull(positions)
            # Return the indices of vertices that form the convex hull
            return hull.vertices.tolist()
        except Exception as e:
            print(f"Error computing convex hull: {e}")
            # Fallback: return nodes with extreme coordinates
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



    def reset(self, init_num_nodes=1, init_bin=0.05):
        """
        Reset the topology by generating random node coordinates.
        The x-coordinate is in [0, init_bin * substrate.width].
        The y-coordinate is in [0, substrate.height].
        """
        x_max = init_bin * self.substrate.width
        y_max = self.substrate.height
        # Randomize coordinates
        coordinates = torch.stack([
            torch.rand(init_num_nodes, dtype=torch.float32) * x_max,  # x-coordinates
            torch.rand(init_num_nodes, dtype=torch.float32) * y_max   # y-coordinates
        ], dim=1)
        # Sort nodes by x-coordinate
        x_coords = coordinates[:, 0]
        sorted_indices = torch.argsort(x_coords)
        # Create edges: node i -> node i+1 in sorted order
        src = sorted_indices[:-1]
        dst = sorted_indices[1:]
        # Create a directed graph with these edges
        g = dgl.graph((src, dst), num_nodes=init_num_nodes)
        g.ndata['pos'] = coordinates
        self.graph = g
        return g



    def show(self, size=(10, 8), highlight_outmost=False, update_only=True):        
        """
        Visualizes the agent's topology and substrate signal matrix.
        Parameters
        ----------
        size : tuple of int, optional
            Figure size for the plot (width, height). Default is (10, 8).
        highlight_outmost : bool, optional
            If True, highlights the outmost nodes in the topology, draws the convex hull boundary,
            and marks the centroid. If False, only plots all nodes and the centroid.
        update_only : bool, optional
            If True, updates existing figure without opening new window. If False, creates new figure.
        Description
        -----------
        - Plots the substrate signal matrix as a background image.
        - Plots all node positions in the graph.
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
        positions = self.graph.ndata['pos'].numpy()
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure only if it doesn't exist or update_only is False
        if self.fig is None or not update_only:
            self.fig, self.ax = plt.subplots(figsize=size)
            plt.show(block=False)  # Non-blocking show
        else:
            # Clear the existing axes for update
            self.ax.clear()
        
        # Compute and plot centroid
        centroid = self.compute_centroid()
        
        if highlight_outmost:
            # Get outmost nodes
            outmost_indices = self.get_outmost_nodes()
            # Plot all nodes in blue
            self.ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=10, alpha=0.6, label='All nodes')
            # Highlight outmost nodes in red
            outmost_positions = positions[outmost_indices]
            self.ax.scatter(outmost_positions[:, 0], outmost_positions[:, 1], 
                           c='red', s=50, marker='o', edgecolor='black', linewidth=1,
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
            self.ax.set_title('Topology')
        else:
            self.ax.scatter(positions[:, 0], positions[:, 1], c='red', s=10)
            # Add green marker for centroid
            self.ax.scatter(centroid[0], centroid[1], c='green', s=100, marker='*', 
                           edgecolor='black', linewidth=1, label='Centroid')
            self.ax.legend()
            self.ax.set_title('Topology')        
        
        self.ax.imshow(canvas, cmap='viridis', origin='lower')
        
        # Refresh the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to ensure update
    
    
    def close_figure(self):
        """Close the figure window and reset figure references"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
      




if __name__ == '__main__':
    
    substrate_linear = Substrate((600, 400))
    substrate_linear.create('linear', m=0.05, b=1)
   
    agent = Topology(substrate=substrate_linear)      



    agent.reset(init_num_nodes=100, init_bin=0.1)
    for i in range(1,20):
        agent.show(highlight_outmost=True)
        agent.act()


    # Keep the last figure window open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Blocking show to keep window open