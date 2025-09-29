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
    



class Topology:
    
    def __init__(self, graph=None):
        self.graph = graph
    
    def compute_centroid(self):
        """Compute the centroid (center of mass) of all nodes"""
        if self.graph is None or 'pos' not in self.graph.ndata:
            raise ValueError("Graph or node positions not available")
        centroid = torch.mean(self.graph.ndata['pos'], dim=0)
        return centroid.numpy()
    
    def get_node_positions(self):
        """Get all node positions as a dictionary"""
        if self.graph is None or 'pos' not in self.graph.ndata:
            return {}
        return {i: self.graph.ndata['pos'][i].numpy() for i in range(self.graph.num_nodes())}




# TODO topology can compute centroid
# TODO topology can perform graph embeddings



if __name__ == '__main__':
    
   

    
    # Define explicit coordinates for nodes 0, 1, 2, 3
    coordinates = torch.tensor([[0.0, 0.0],    # node 0 at (0,0)
                               [2.12, 0.0],    # node 1 at (2,0)
                               [2.40, 2.0],    # node 2 at (2,2)
                               [0.67, 2.0]])   # node 3 at (0,2)

    # Create a DGL graph
    sources = [0, 0, 1, 1, 2, 3, 0]
    destinations = [1, 2, 2, 3, 3, 0, 0]
    g = dgl.graph((sources, destinations))


    g.ndata['pos'] = coordinates  # Store coordinates in node data

    # Compute centroid of the graph
    centroid = torch.mean(g.ndata['pos'], dim=0)
    print(f"Graph centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")

    print("Is g on CUDA?", g.device.type == 'cuda')
    g = g.to('cuda')
    print("Is g on CUDA?", g.device.type == 'cuda')
    g = g.cpu()


    # Convert to NetworkX for visualization
    G = g.to_networkx()

    # Draw with explicit positions
    nx.draw(G, pos=g.ndata['pos'], with_labels=True, node_color='lightgreen', node_size=500, font_size=16)
    plt.show()


    # Create Topology instance and compute centroid
    topology = Topology(g)
    centroid_coords = topology.compute_centroid()
    
    # Extract positions for NetworkX plotting
    pos_dict = topology.get_node_positions()
    
    # Draw with explicit positions
    nx.draw(G, pos=pos_dict, with_labels=True, node_color='lightgreen', 
            node_size=500, font_size=16)
    
    # Add centroid point
    plt.scatter(centroid_coords[0], centroid_coords[1], color='red', s=100, 
                marker='x', linewidths=3, label='Centroid')
    
    plt.title(f"Graph with centroid at ({centroid_coords[0]:.2f}, {centroid_coords[1]:.2f})")
    plt.legend()
    plt.axis('equal')  # Equal scaling for x and y axes
    plt.show()
