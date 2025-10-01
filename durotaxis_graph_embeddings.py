import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO

from topology import Topology
from substrate import Substrate
from embedding_dgl import GraphEmbedding
from encoder_graph_embeddings import GraphTransformerEncoder, GraphPolicyNetwork, TopologyPolicyAgent
from observation_strategies import get_observation_strategy_6_lightweight


class DurotaxisEnv(gym.Env):
    """
    A durotaxis environment that uses graph transformer policy for topology evolution.
    The environment simulates cellular durotaxis using a dynamic graph topology.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, 
                 substrate_size=(600, 400),
                 substrate_type='linear',
                 substrate_params={'m': 0.01, 'b': 1.0},
                 init_num_nodes=1,
                 max_nodes=50,
                 max_steps=1000,
                 embedding_dim=64,
                 hidden_dim=128,  
                 render_mode=None,
                 policy_agent=None):
        super().__init__()
        
        # Environment parameters
        self.substrate_size = substrate_size
        self.substrate_type = substrate_type
        self.substrate_params = substrate_params
        self.init_num_nodes = init_num_nodes
        self.max_nodes = max_nodes
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim 
        self.current_step = 0
        
        # 1. Action Space
        # Since the agent uses graph_transformer_policy_dgl.act_with_policy(),
        # we define a dummy action space for compatibility with RL frameworks.
        # The actual actions are determined by the policy network based on graph embeddings.
        self.action_space = spaces.Discrete(1)  # Dummy action - actual actions come from policy network

        # 2. Observation Space
        # The observation space accommodates graph embeddings from the GraphEmbedding class.
        # Since graph size is dynamic, we use a fixed-size embedding representation.
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.embedding_dim,), 
            dtype=np.float32
        )
        
        # 3. Initialize environment components
        self._setup_environment()
        
        # 4. Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Create policy network for enhanced observations
        if policy_agent is not None:
            self.observation_policy = policy_agent.policy
        else:
            # Create a separate encoder just for observations
            node_dim = embedding_dim  # From your embedding
            graph_dim = embedding_dim  # From your embedding
            encoder = GraphTransformerEncoder(
                node_dim=node_dim,
                graph_dim=graph_dim, 
                hidden_dim=self.hidden_dim,  # Use instance variable
                num_layers=2
            )
            self.observation_policy = GraphPolicyNetwork(encoder, hidden_dim=self.hidden_dim)  # Use instance variable

        # Update observation space for Strategy 6
        obs_dim = self.hidden_dim * 5  # hidden_dim * 5 (graph + mean + std + min + max)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _setup_environment(self):
        """Initialize substrate, topology, embedding, and policy components."""
        # Create substrate
        self.substrate = Substrate(self.substrate_size)
        self.substrate.create(self.substrate_type, **self.substrate_params)
        
        # Create topology
        self.topology = Topology(substrate=self.substrate)
        
        # Create embedding system
        self.embedding = GraphEmbedding(self.topology)
        
        # Initialize policy network (will be set up after first reset)
        self.policy_agent = None
        self._policy_initialized = False

    def _initialize_policy(self):
        """Initialize the policy network based on the current graph state."""
        if self._policy_initialized:
            return
            
        # Get initial state to determine dimensions
        state = self.embedding.get_state_embedding(embedding_dim=self.embedding_dim)
        
        node_dim = state['node_embeddings'].shape[1] if state['num_nodes'] > 0 else self.embedding_dim
        graph_dim = state['graph_embedding'].shape[0]
        
        # Create policy network
        encoder = GraphTransformerEncoder(
            node_dim=node_dim,
            graph_dim=graph_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        policy = GraphPolicyNetwork(encoder, hidden_dim=self.hidden_dim, noise_scale=0.1)
        
        # Create policy agent
        self.policy_agent = TopologyPolicyAgent(self.topology, self.embedding, policy)
        self._policy_initialized = True

    def step(self, action):
        """
        Execute one time step using the graph transformer policy.
        The 'action' parameter is ignored as we use the policy network.
        """
        self.current_step += 1
        
        # Store previous state for reward calculation
        prev_state = self.embedding.get_state_embedding(embedding_dim=self.embedding_dim)
        prev_num_nodes = prev_state['num_nodes']
        
        # Get previous DGL graph 
        prev_dgl = self.embedding.to_dgl(embedding_dim=self.embedding_dim)
        
        # Execute actions using the policy network
        if self.policy_agent is not None and prev_num_nodes > 0:
            try:
                executed_actions = self.policy_agent.act_with_policy(
                    embedding_dim=self.embedding_dim, 
                    deterministic=False
                )
            except Exception as e:
                print(f"Policy execution failed: {e}")
                executed_actions = {}
        else:
            # Fallback to random actions if policy fails
            executed_actions = self.topology.act()
        
        # Get new state
        new_state = self.embedding.get_state_embedding(embedding_dim=self.embedding_dim)
        
        # Get next DGL graph
        next_dgl = self.embedding.to_dgl(embedding_dim=self.embedding_dim)
        
        # Strategy 6: Enhanced observation using policy network
        observation = get_observation_strategy_6_lightweight(
            new_state, 
            self.observation_policy, 
            device='cpu'
        )
        
        # Calculate reward with DGL graphs
        reward = self._calculate_reward(prev_state, new_state, executed_actions, prev_dgl, next_dgl)
        
        # Check termination conditions
        terminated = self._check_terminated(new_state)
        truncated = self.current_step >= self.max_steps
        
        # Info dictionary
        info = {
            'num_nodes': new_state['num_nodes'],
            'num_edges': new_state['num_edges'],
            'actions_taken': len(executed_actions),
            'step': self.current_step,
            'policy_initialized': self._policy_initialized
        }
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_state, new_state, actions, prev_dgl=None, next_dgl=None):
        """
        Calculate reward based on topology evolution and substrate exploration.
        Now includes DGL graph analysis for more sophisticated reward calculation.
        """
        reward = 0.0
        
        # Basic survival reward
        reward += 0.1
        
        # DGL-based reward calculations
        if prev_dgl is not None and next_dgl is not None:
            # Example: Calculate graph-based metrics using DGL
            import dgl
            
            # Connectivity analysis
            if next_dgl.num_nodes() > 1:
                # Graph density
                density = next_dgl.num_edges() / (next_dgl.num_nodes() * (next_dgl.num_nodes() - 1))
                reward += density * 2.0
                
                # Clustering coefficient (if available)
                try:
                    # You can add more sophisticated graph metrics here
                    # Example: average clustering coefficient, betweenness centrality, etc.
                    pass
                except:
                    pass
            
            # Structural changes reward
            if prev_dgl.num_nodes() > 0:
                # Reward for growth
                node_growth = next_dgl.num_nodes() - prev_dgl.num_nodes()
                edge_growth = next_dgl.num_edges() - prev_dgl.num_edges()
                
                # Moderate growth is good
                if 0 <= node_growth <= 2:
                    reward += node_growth * 0.5
                elif node_growth > 2:
                    reward -= (node_growth - 2) * 0.3  # Penalty for too rapid growth
                
                if edge_growth > 0:
                    reward += edge_growth * 0.2
        
        # Fallback to original reward if DGL graphs not available
        else:
            # Reward for maintaining connectivity
            if new_state['num_nodes'] > 1:
                density = new_state['num_edges'] / (new_state['num_nodes'] * (new_state['num_nodes'] - 1))
                reward += density * 2.0
        
        # Reward for substrate exploration (using graph features)
        graph_features = new_state['graph_features']
        if len(graph_features) > 10:  # Ensure we have enough features
            # Reward based on spatial coverage (bbox area)
            bbox_area = graph_features[10].item() if len(graph_features) > 10 else 0.0
            reward += bbox_area * 0.01
            
            # Reward based on convex hull area
            hull_area = graph_features[-2].item() if len(graph_features) > 12 else 0.0
            reward += hull_area * 0.005
        
        # Penalty for too many or too few nodes
        num_nodes = new_state['num_nodes']
        if num_nodes < 2:
            reward -= 5.0  # Strong penalty for losing all connectivity
        elif num_nodes > self.max_nodes:
            reward -= 2.0  # Penalty for excessive growth
        
        # Small reward for taking actions (encourages exploration)
        reward += len(actions) * 0.05
        
        return reward

    def _check_terminated(self, state):
        """Check if the episode should terminate."""
        # Terminate if no nodes left
        if state['num_nodes'] == 0:
            return True
        
        # Terminate if too many nodes
        if state['num_nodes'] > self.max_nodes:
            return True
            
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset topology
        self.topology.reset(init_num_nodes=self.init_num_nodes)
        
        # Initialize policy if not done yet
        self._initialize_policy()
        
        # Get initial observation
        state = self.embedding.get_state_embedding(embedding_dim=self.embedding_dim)
        observation = state['graph_embedding'].numpy().astype(np.float32)
        
        info = {
            'num_nodes': state['num_nodes'],
            'num_edges': state['num_edges'],
            'step': self.current_step,
            'policy_initialized': self._policy_initialized
        }
        
        return observation, info

    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            state = self.embedding.get_state_embedding(embedding_dim=self.embedding_dim)
            print(f"Step {self.current_step}: {state['num_nodes']} nodes, {state['num_edges']} edges")
            
            # Optional: visualize the topology
            if hasattr(self.topology, 'plot') and state['num_nodes'] > 0:
                try:
                    self.topology.plot()
                except Exception as e:
                    print(f"Plotting failed: {e}")

    def close(self):
        """Clean up resources."""
        if hasattr(self.topology, 'close'):
            self.topology.close()


class PolicyWrapper:
    """
    A wrapper that makes the graph transformer policy compatible with stable-baselines3.
    This allows training the policy network directly using RL algorithms.
    """
    
    def __init__(self, env):
        self.env = env
        
    def predict(self, obs, deterministic=True):
        """
        Make a prediction using the policy network.
        Returns dummy action since actual actions come from policy network.
        """
        return np.array([0]), None  # Dummy action


if __name__ == '__main__':
    print("Setting up DurotaxisEnv with Graph Transformer Policy...")
    
    # Create the durotaxis environment
    env = DurotaxisEnv(
        substrate_size=(100, 50),
        substrate_type='linear',
        substrate_params={'m': 0.01, 'b': 1.0},
        init_num_nodes=5,
        max_nodes=30,
        max_steps=100,
        embedding_dim=64,
        hidden_dim=128,
        render_mode="human"
    )
    
    print("Environment created successfully!")
    
    # Test the environment
    print("\n--- Testing Environment ---")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps to test the integration
    total_reward = 0
    for step in range(10):
        action = 0  # Dummy action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Reward={reward:.3f}, Nodes={info['num_nodes']}, Edges={info['num_edges']}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    
    # Example of training with a simple RL algorithm
    print("\n--- Testing with PPO (Meta-Learning Approach) ---")
    
    # Note: This creates a meta-learning setup where PPO learns to optimize
    # the high-level environment dynamics while the graph transformer policy
    # handles the low-level graph actions
    
    try:
        # Create a simple policy that works with the environment
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        
        print("Training for a short period...")
        model.learn(total_timesteps=1000)
        
        print("Testing trained model...")
        obs, info = env.reset()
        for _ in range(20):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
                
    except Exception as e:
        print(f"PPO training failed: {e}")
        print("This is expected - the environment is primarily designed for the graph transformer policy")
    
    # Clean up
    env.close()
    
    print("\n--- Direct Policy Testing ---")
    
    # Test the graph transformer policy directly
    env2 = DurotaxisEnv(
        init_num_nodes=8,
        max_steps=50,
        embedding_dim=128,
        hidden_dim=128,
        render_mode="human"
    )
    
    obs, info = env2.reset()
    print(f"Direct policy test - Initial: {info}")
    
    for step in range(15):
        obs, reward, terminated, truncated, info = env2.step(0)
        print(f"Step {step + 1}: Reward={reward:.3f}, Nodes={info['num_nodes']}, Edges={info['num_edges']}")
        
        if terminated or truncated:
            print("Episode ended")
            obs, info = env2.reset()
    
    env2.close()
    print("\nDurotaxis environment integration test completed! 🚀")