import numpy as np
import gymnasium as gym  
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Define the Environment ---
class DurotacticEnvironment(gym.Env):
    def __init__(self, size=200, m=0.005, b=1.0, epsilon=0.001):
        super(DurotacticEnvironment, self).__init__()
        self.size = size
        self.critical_nodes = 20
        self.epsilon = epsilon
        self.action_space = spaces.Discrete(9)
        # what are the 9 actions?

        self.observation_space = spaces.Box(low=0, high=size, shape=(2,), dtype=np.float32)
        # (x,y) for x and y in [0,200] range in a 200x200 grid 
        
        self.signal_field = self._create_signal_field(m, b)
        self.active_nodes = []

    def _create_signal_field(self, m, b):
        signal_matrix = np.zeros((self.size, self.size))
        for x in range(self.size):
            signal_matrix[:, x] = m * x + b
        return signal_matrix

    def step(self, action):
        current_x, current_y = self.state
        reward = 0

        # Add (0, 0) for the 9th action (no movement)
        #      [(-1,-1)] [(0,-1)] [(1,-1)]
        #      [(-1,0)]  [(0,0)]  [(1,0)]
        #      [(-1,1)]  [(0,1)]  [(1,1)]
        # ['no movement', 'bottom', 'right', 'bottom-right', 'top-right', 'left', 'bottom-left', 'top-left', 'no movement']
        # TODO: fix bug by adding top (0,-1)
        dx = [0, 0, 1, 1,  1, -1, -1, -1, 0]
        dy = [0, 1, 0, 1, -1,  0,  1, -1, 0]
        

        new_x = np.clip(current_x + dx[action], 0, self.size - 1)
        new_y = np.clip(current_y + dy[action], 0, self.size - 1)

        new_state = (new_x, new_y)

        current_signal = self.signal_field[current_x, current_y]
        new_signal = self.signal_field[new_x, new_y]
        delta = 0.5

        if new_signal - current_signal >= delta:
            reward += 10

        terminated = False # Initialize terminated flag
        truncated = False  # Initialize truncated flag

        if len(self.active_nodes) > self.critical_nodes:
            reward -= 50
            terminated = True

        if new_signal > self.epsilon:
            reward += 1
        else:
            reward -= 1

        self.state = new_state
        self.active_nodes.append(self.state)

        return np.array(self.state), reward, terminated, truncated, {} # Updated return values

    def reset(self, seed=None, options=None): # New parameters for gymnasium
        super().reset(seed=seed)
        self.state = (self.np_random.integers(0, self.size), self.np_random.integers(0, self.size))
        self.active_nodes = []
        self.active_nodes.append(self.state)
        return np.array(self.state), {} # Updated return values

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.signal_field.T, cmap='viridis', origin='lower')
        ax.set_title("Durotactic Signal Gradient")
        if self.active_nodes:
            x_coords, y_coords = zip(*self.active_nodes)
            ax.plot(x_coords, y_coords, 'ro', markersize=4)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        plt.show()

    def render_animation(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.signal_field.T, cmap='viridis', origin='lower')
        ax.set_title("Node Movement in Durotactic Gradient")
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        node_plot, = ax.plot([], [], 'ro', markersize=4)

        def update_plot(frame):
            if self.active_nodes[:frame]: # Check if the list is not empty
                x_coords, y_coords = zip(*self.active_nodes[:frame])
                node_plot.set_data(x_coords, y_coords)
            return node_plot,

        ani = animation.FuncAnimation(fig, update_plot, frames=len(self.active_nodes), blit=True)
        return ani


# --- 2. Create the Agent (Q-learning) ---
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = np.zeros((env.size, env.size, env.action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            x, y = state
            return np.argmax(self.q_table[int(x), int(y)])

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        best_next_action_value = np.max(self.q_table[int(next_x), int(next_y)])
        update_value = reward + self.gamma * best_next_action_value - self.q_table[int(x), int(y), action]
        self.q_table[int(x), int(y), action] += self.lr * update_value


if __name__ == "__main__":
    env = DurotacticEnvironment()
    agent = QLearningAgent(env)

    num_episodes = 100
    all_paths = []

    # Define a filename for the trajectories
    output_filename = "node_trajectories.txt"

    # Open the file in write mode to clear it before a new run
    with open(output_filename, "w") as f:
        f.write("Episode,X,Y\n") # Write a header

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False

        # ... (rest of the training loop is the same) ...
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.learn(state, action, reward, next_state)
            state = next_state

        # Store the path for visualization and save to file
        current_path = list(env.active_nodes)
        all_paths.append(current_path)

        # --- New code to save the trajectory to a text file ---
        with open(output_filename, "a") as f:
            for x, y in current_path:
                f.write(f"{episode},{x},{y}\n")

        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if episode % 20 == 0:
            print(f"Episode {episode}: Path length: {len(env.active_nodes)}")

    print("Training complete. Trajectories saved to", output_filename)

    # ... (rest of the visualization code is the same) ...
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(env.signal_field.T, cmap='viridis', origin='lower')
    ax.set_title("Durotactic Gradient with Node Paths")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    for path in all_paths:
        if path:
            x_coords, y_coords = zip(*path)
            ax.plot(x_coords, y_coords, color='red', alpha=0.5)
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=6)
    plt.show()

    print("Generating animation...")
    ani = env.render_animation()
    plt.show()