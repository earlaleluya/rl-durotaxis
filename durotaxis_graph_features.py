import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO

from topology import Topology
from substrate import Substrate
from embedding_dgl import GraphEmbedding
from encoder_graph_features import GraphInputEncoder, GraphPolicyNetwork, TopologyPolicyAgent



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
        # The observation space uses encoder_out from GraphPolicyNetwork (GraphInputEncoder output).
        # Shape: [num_nodes+1, out_dim] where first element is graph token, rest are node embeddings
        max_nodes_plus_graph = self.max_nodes + 1  # +1 for graph token
        encoder_out_dim = 64  # Default out_dim from GraphInputEncoder
        obs_dim = max_nodes_plus_graph * encoder_out_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 3. Initialize environment components
        self._setup_environment()
        
        # 4. Rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # Create policy network for encoder-based observations
        if policy_agent is not None:
            self.observation_policy = policy_agent.policy
        else:
            # Create a separate encoder for observations
            encoder = GraphInputEncoder(
                hidden_dim=self.hidden_dim,
                out_dim=64,  # This matches encoder_out_dim above
                num_layers=2
            )
            self.observation_policy = GraphPolicyNetwork(encoder, hidden_dim=self.hidden_dim)

        # Store encoder output dimension for observation processing
        self.encoder_out_dim = 64

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
            
        # Create policy network using GraphInputEncoder
        encoder = GraphInputEncoder(
            hidden_dim=self.hidden_dim,
            out_dim=64,
            num_layers=3
        )
        
        policy = GraphPolicyNetwork(encoder, hidden_dim=self.hidden_dim, noise_scale=0.1)
        
        # Create policy agent
        self.policy_agent = TopologyPolicyAgent(self.topology, self.embedding, policy)
        self._policy_initialized = True

    def _get_encoder_observation(self, state):
        """
        Get observation from GraphPolicyNetwork encoder_out with semantic pooling.
        
        Args:
            state: State dictionary from embedding.get_state_embedding()
            
        Returns:
            np.ndarray: Fixed-size observation vector from encoder_out
            
        Note:
            Uses semantic pooling when num_nodes > max_nodes to preserve representative
            nodes based on their features rather than arbitrary truncation.
        """
        try:
            # Get policy output which includes encoder_out
            policy_output = self.observation_policy(state, deterministic=True)
            encoder_out = policy_output['encoder_out']  # Shape: [num_nodes+1, out_dim]
            
            # Check for potential information reduction
            actual_nodes = encoder_out.shape[0] - 1  # -1 for graph token
            max_size = (self.max_nodes + 1) * self.encoder_out_dim
            
            if actual_nodes <= self.max_nodes:
                # Normal case: no pooling needed
                print(f"✅ No pooling needed: {actual_nodes} nodes ≤ {self.max_nodes} max_nodes")
                encoder_flat = encoder_out.flatten().detach().cpu().numpy()
                
                # Pad with zeros if smaller than max size
                padded = np.zeros(max_size, dtype=np.float32)
                padded[:len(encoder_flat)] = encoder_flat
                return padded
            
            else:
                # 🧠 SEMANTIC POOLING: Intelligently select representative nodes
                print(f"🧠 Applying semantic pooling: {actual_nodes} nodes > {self.max_nodes} max_nodes")
                
                graph_token = encoder_out[0:1]  # [1, out_dim] - Always preserve graph token
                node_embeddings = encoder_out[1:]  # [actual_nodes, out_dim]
                
                # Extract semantic features for clustering
                selected_indices = self._semantic_node_selection(
                    node_embeddings, 
                    state, 
                    target_count=self.max_nodes
                )
                
                # Select representative nodes
                selected_nodes = node_embeddings[selected_indices]
                
                # Combine graph token with selected nodes
                pooled_embeddings = torch.cat([graph_token, selected_nodes], dim=0)
                
                print(f"  📊 Pooling results:")
                print(f"    - Original nodes: {actual_nodes}")
                print(f"    - Selected nodes: {len(selected_indices)}")
                print(f"    - Selected indices: {sorted(selected_indices.tolist())}")
                print(f"    - Information preserved: {len(selected_indices)/actual_nodes*100:.1f}%")
                
                # Flatten and pad to fixed size
                pooled_flat = pooled_embeddings.flatten().detach().cpu().numpy()
                padded = np.zeros(max_size, dtype=np.float32)
                padded[:len(pooled_flat)] = pooled_flat
                
                return padded
                
        except Exception as e:
            print(f"Error getting encoder observation: {e}")
            # Fallback to zero observation
            return np.zeros((self.max_nodes + 1) * self.encoder_out_dim, dtype=np.float32)

    def _semantic_node_selection(self, node_embeddings, state, target_count):
        """
        Intelligently select representative nodes using semantic features.
        
        Args:
            node_embeddings: [num_nodes, out_dim] tensor of node embeddings
            state: State dictionary with node features
            target_count: Number of nodes to select
            
        Returns:
            torch.Tensor: Indices of selected nodes
        """
        import torch
        import numpy as np
        from sklearn.cluster import KMeans
        
        num_nodes = node_embeddings.shape[0]
        
        try:
            # Strategy 1: Use raw node features for semantic clustering
            node_features = state['node_features']  # [num_nodes, 8] - rich feature set
            
            if node_features.shape[0] != num_nodes:
                # Fallback to uniform sampling if feature mismatch
                print(f"    ⚠️ Feature mismatch, using uniform sampling")
                indices = torch.linspace(0, num_nodes-1, target_count, dtype=torch.long)
                return indices
            
            # Convert to numpy for clustering
            features_np = node_features.detach().cpu().numpy()
            
            # Strategy 2: Multi-criteria semantic selection
            selected_indices = []
            
            # A. Spatial diversity (based on position features - first 2 dims)
            if features_np.shape[1] >= 2:
                positions = features_np[:, :2]  # x, y coordinates
                spatial_clusters = min(target_count // 3, 8)  # 1/3 for spatial diversity
                
                if spatial_clusters > 1:
                    try:
                        kmeans_spatial = KMeans(n_clusters=spatial_clusters, random_state=42, n_init=10)
                        spatial_labels = kmeans_spatial.fit_predict(positions)
                        
                        # Select one representative from each spatial cluster
                        for cluster_id in range(spatial_clusters):
                            cluster_nodes = np.where(spatial_labels == cluster_id)[0]
                            if len(cluster_nodes) > 0:
                                # Choose node closest to cluster centroid
                                centroid = kmeans_spatial.cluster_centers_[cluster_id]
                                distances = np.linalg.norm(positions[cluster_nodes] - centroid, axis=1)
                                best_idx = cluster_nodes[np.argmin(distances)]
                                selected_indices.append(best_idx)
                        
                        print(f"    🗺️ Spatial clustering: {len(selected_indices)} nodes from {spatial_clusters} regions")
                    except:
                        pass
            
            # B. Feature diversity (based on all features)
            remaining_slots = target_count - len(selected_indices)
            if remaining_slots > 0:
                # Select nodes not already chosen
                available_indices = list(set(range(num_nodes)) - set(selected_indices))
                
                if len(available_indices) > remaining_slots:
                    try:
                        # Cluster remaining nodes by feature similarity
                        available_features = features_np[available_indices]
                        feature_clusters = min(remaining_slots, len(available_indices))
                        
                        if feature_clusters > 1:
                            kmeans_features = KMeans(n_clusters=feature_clusters, random_state=42, n_init=10)
                            feature_labels = kmeans_features.fit_predict(available_features)
                            
                            # Select representative from each feature cluster
                            for cluster_id in range(feature_clusters):
                                cluster_mask = (feature_labels == cluster_id)
                                if np.any(cluster_mask):
                                    cluster_indices = np.array(available_indices)[cluster_mask]
                                    # Choose node closest to cluster centroid
                                    centroid = kmeans_features.cluster_centers_[cluster_id]
                                    distances = np.linalg.norm(available_features[cluster_mask] - centroid, axis=1)
                                    best_local_idx = np.argmin(distances)
                                    best_global_idx = cluster_indices[best_local_idx]
                                    selected_indices.append(best_global_idx)
                            
                            print(f"    🎯 Feature clustering: +{len(selected_indices) - len(selected_indices[:spatial_clusters if 'spatial_clusters' in locals() else 0])} diverse nodes")
                        else:
                            # Just take the available nodes
                            selected_indices.extend(available_indices[:remaining_slots])
                    except Exception as e:
                        print(f"    ⚠️ Feature clustering failed: {e}")
                        # Fallback: uniform sampling from remaining
                        selected_indices.extend(available_indices[:remaining_slots])
                else:
                    # Take all remaining nodes
                    selected_indices.extend(available_indices)
            
            # C. Fill remaining slots with uniform sampling if needed
            if len(selected_indices) < target_count:
                available_indices = list(set(range(num_nodes)) - set(selected_indices))
                remaining_needed = target_count - len(selected_indices)
                
                if available_indices:
                    # Uniform sampling from remaining
                    step = max(1, len(available_indices) // remaining_needed)
                    additional = available_indices[::step][:remaining_needed]
                    selected_indices.extend(additional)
                    print(f"    📐 Uniform fill: +{len(additional)} nodes to reach target")
            
            # Ensure we don't exceed target count
            selected_indices = selected_indices[:target_count]
            
            # Sort indices for consistent ordering
            selected_indices = sorted(selected_indices)
            
            return torch.tensor(selected_indices, dtype=torch.long)
            
        except Exception as e:
            print(f"    ❌ Semantic selection failed: {e}")
            # Fallback to uniform sampling
            indices = torch.linspace(0, num_nodes-1, target_count, dtype=torch.long)
            print(f"    🔄 Fallback to uniform sampling: {target_count} nodes")
            return indices

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
        
        # Get observation from GraphInputEncoder output
        observation = self._get_encoder_observation(new_state)
        
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
        
        # Get initial observation using encoder_out
        state = self.embedding.get_state_embedding(embedding_dim=self.embedding_dim)
        
        # Get observation from GraphInputEncoder output
        observation = self._get_encoder_observation(state)
        
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