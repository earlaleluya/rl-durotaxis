'''
    This file intends to provide the various substrate for the durotaxis of physarum. The following are the available substrate:
    - linear 
    - exponential
'''
import numpy as np
import matplotlib.pyplot as plt

class Substrate:
    """
    This class generates linear and exponential substrates for simulating durotaxis.
    Durotaxis is the directed movement of cells in response to a gradient of
    substrate stiffness.
    """

    def __init__(self, size):
        width, height = size
        self.height = height
        self.width = width
        self.signal_matrix = None


    def create(self, kind, m=0.05, b=1.0):
        """
        Creates a new signal matrix based on the specified kind (linear or exponential).
        """
        x_coords = np.arange(self.width)
        if kind == 'linear':
            self.signal_matrix = self._create_linear(x_coords, m, b)
        elif kind == 'exponential':
            self.signal_matrix = self._create_exponential(x_coords, m, b)
        else:
            raise ValueError("Invalid substrate kind. Choose 'linear' or 'exponential'.")


    def _create_linear(self, x_coords, m, b):
        """Vectorized creation of a linear gradient."""
        gradient = m * x_coords + b
        return np.tile(gradient, (self.height, 1))


    def _create_exponential(self, x_coords, m, b):
        """Vectorized creation of an exponential gradient."""
        gradient = b * np.exp(m * x_coords)
        return np.tile(gradient, (self.height, 1))

    def show(self):
        """Display the signal matrix as a heatmap."""
        if self.signal_matrix is None:
            print("No substrate has been created. Call the 'create' method first.")
            return
        plt.figure(figsize=(10, 6))
        plt.imshow(self.signal_matrix, cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(label='Signal Intensity')
        plt.title('Substrate Signal Matrix')
        plt.xlabel('X-coordinate (width)')
        plt.ylabel('Y-coordinate (height)')
        plt.show()



        

if __name__ == '__main__':
    # Create and display a linear substrate
    substrate_linear = Substrate((200, 200))
    substrate_linear.create('linear', m=0.05, b=1)
    substrate_linear.show()

    # Create and display an exponential substrate
    substrate_exponential = Substrate((200, 100))
    substrate_exponential.create('exponential', m=0.05, b=1)
    substrate_exponential.show()