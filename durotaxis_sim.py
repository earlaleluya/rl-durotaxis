import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO
import os
from datetime import datetime
import json

from topology import Topology
from substrate import Substrate
from state import TopologyState
from encoder import GraphInputEncoder
from policy import GraphPolicyNetwork, TopologyPolicyAgent



class DurotaxisEnv(gym.Env):
    """
    A durotaxis environment that uses graph transformer policy for topology evolution.
    
    This environment simulates cellular durotaxis (movement in response to substrate stiffness)
    using a dynamic graph topology where nodes represent cells and edges represent connections.
    The environment uses reinforcement learning with graph neural networks to learn optimal
    cell migration and proliferation strategies.
    
    The environment provides:
    - Dynamic graph topology with node spawn/delete operations
    - Substrate with gradient-based intensity signals
    - Comprehensive reward system for different behaviors
    - Termination conditions for success/failure scenarios
    - Real-time visualization capabilities
    
    Attributes
    ----------
    substrate : Substrate
        The substrate environment with intensity gradients
    topology : Topology
        The dynamic graph representing cell topology
    action_space : gym.Space
        The action space (discrete actions)
    observation_space : gym.Space
        The observation space (graph embeddings)
    
    Examples
    --------
    >>> env = DurotaxisEnv(substrate_size=(600, 400), init_num_nodes=5)
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, 
                 substrate_size=(600, 400),
                 substrate_type='linear',
                 substrate_params={'m': 0.01, 'b': 1.0},
                 init_num_nodes=1,
                 max_nodes=50,
                 num_critical_nodes=200,
                 max_steps=1000,
                 embedding_dim=64,
                 hidden_dim=128,  
                 delta_time=3,
                 delta_intensity=2.50,
                 # Grouped reward parameters
                 graph_rewards={
                     'connectivity_penalty': 10.0,  # Penalty for losing connectivity (N < 2)
                     'growth_penalty': 10.0,  # Penalty for excessive growth (N > max_nodes)
                     'survival_reward': 0.01,  # Basic survival reward for valid topology
                     'action_reward': 0.005,  # Reward multiplier per action taken
                 },
                 node_rewards={
                     'movement_reward': 0.01,  # Reward multiplier for rightward movement
                     'intensity_penalty': 5.0,  # Penalty for below-average substrate intensity
                     'intensity_bonus': 0.01,  # Bonus for above-average substrate intensity
                     'substrate_reward': 0.05,  # Reward multiplier for substrate intensity
                 },
                 edge_reward={
                    'rightward_bonus': 0.1, 
                    'leftward_penalty': 0.1},  # Edge direction rewards
                 spawn_rewards={
                     'spawn_success_reward': 1.0,  # Reward for successful durotaxis spawning
                     'spawn_failure_penalty': 1.0,  # Penalty for failed durotaxis spawning
                 },
                 delete_reward={
                    'proper_deletion': 2.0, 
                    'persistence_penalty': 2.0},  # Deletion compliance rewards
                 position_rewards={
                     'boundary_bonus': 0.1,  # Bonus for boundary/frontier nodes
                     'left_edge_penalty': 0.2,  # Penalty for being near left edge
                     'edge_position_penalty': 0.1,  # Penalty for being near top/bottom edges
                 },
                 termination_rewards={
                     'success_reward': 100.0,  # Large reward for reaching rightmost location
                     'out_of_bounds_penalty': -30.0,  # Large penalty for nodes going out of bounds
                     'no_nodes_penalty': -30.0,  # Penalty for losing all nodes
                     'leftward_drift_penalty': -30.0,  # Penalty for consistent leftward movement
                     'timeout_penalty': -10.0,  # Small penalty for reaching max time steps
                     'critical_nodes_penalty': -25.0,  # Penalty for exceeding critical node threshold
                 },
                 render_mode=None,
                 policy_agent=None,
                 flush_delay=0.01,  # Visualization flush delay
                 enable_visualization=True,  # Enable/disable topology.show() visualization
                 model_path="./saved_models"):  # Path to save models
        super().__init__()
        
        # Environment parameters
        self.substrate_size = substrate_size
        self.substrate_type = substrate_type
        self.substrate_params = substrate_params
        self.init_num_nodes = init_num_nodes
        self.max_nodes = max_nodes
        self.num_critical_nodes = num_critical_nodes
        self.max_steps = max_steps
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim 
        self.delta_time = delta_time
        self.delta_intensity = delta_intensity  
        self.flush_delay = flush_delay  # Store flush delay for visualization
        self.enable_visualization = enable_visualization  # Control topology.show() activation  
        
        # Model saving configuration
        self.model_path = model_path
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.algorithm_name = "GraphTransformer"  # Default algorithm name
        self.external_model = None  # Store external RL model (PPO, DQN, etc.)
        
        # Determine run number and create run-specific directory
        self.run_number = self._get_next_run_number()
        self.run_directory = os.path.join(self.model_path, f"run{self.run_number:04d}")
        self._ensure_model_directory()
        
        # Unpack directional reward dictionaries
        self.delete_reward = delete_reward
        self.delete_proper_reward = delete_reward['proper_deletion']
        self.delete_persistence_penalty = delete_reward['persistence_penalty']
        
        self.edge_reward = edge_reward
        self.edge_rightward_bonus = edge_reward['rightward_bonus']
        self.edge_leftward_penalty = edge_reward['leftward_penalty']
        
        # Unpack grouped reward parameters
        self.graph_rewards = graph_rewards
        self.connectivity_penalty = graph_rewards['connectivity_penalty']
        self.growth_penalty = graph_rewards['growth_penalty']
        self.survival_reward = graph_rewards['survival_reward']
        self.action_reward = graph_rewards['action_reward']
        
        self.node_rewards = node_rewards
        self.movement_reward = node_rewards['movement_reward']
        self.intensity_penalty = node_rewards['intensity_penalty']
        self.intensity_bonus = node_rewards['intensity_bonus']
        self.substrate_reward = node_rewards['substrate_reward']
        
        self.position_rewards = position_rewards
        self.boundary_bonus = position_rewards['boundary_bonus']
        self.left_edge_penalty = position_rewards['left_edge_penalty']
        self.edge_position_penalty = position_rewards['edge_position_penalty']
        
        self.spawn_rewards = spawn_rewards
        self.spawn_success_reward = spawn_rewards['spawn_success_reward']
        self.spawn_failure_penalty = spawn_rewards['spawn_failure_penalty']
        
        self.termination_rewards = termination_rewards
        self.success_reward = termination_rewards['success_reward']
        self.out_of_bounds_penalty = termination_rewards['out_of_bounds_penalty']
        self.no_nodes_penalty = termination_rewards['no_nodes_penalty']
        self.leftward_drift_penalty = termination_rewards['leftward_drift_penalty']
        self.timeout_penalty = termination_rewards['timeout_penalty']
        self.critical_nodes_penalty = termination_rewards['critical_nodes_penalty']
        self.critical_nodes_penalty = termination_rewards['critical_nodes_penalty']
        
        self.current_step = 0
        self.current_episode = 0
        
        # Centroid tracking for fail termination
        self.centroid_history = []  # Store centroid x-coordinates
        self.consecutive_left_moves = 0  # Count consecutive leftward moves
        self.fail_threshold = 2 * self.delta_time  # Threshold for fail termination

        # Topology tracking for substrate intensity comparison
        self.topology_history = []
        self.dequeued_topology = None  # Store the most recently dequeued topology
        
        # Node-level reward tracking
        self.prev_node_positions = []  # Store previous node positions for movement rewards
        self.last_reward_breakdown = None  # Store detailed reward information
        
        # 1. Action Space
        # Since the agent uses graph_transformer_policy_dgl.act_with_policy(),
        # we define a dummy action space for compatibility with RL frameworks.
        # The actual actions are determined by the policy network based on graph embeddings.
        self.action_space = spaces.Discrete(1)  # Dummy action - actual actions come from policy network

        # 2. Observation Space
        # The observation space uses output from GraphInputEncoder directly.
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
        
        # Create GraphInputEncoder for observations
        self.observation_encoder = GraphInputEncoder(
            hidden_dim=self.hidden_dim,
            out_dim=64,  # This matches encoder_out_dim above
            num_layers=4
        )

        # Store encoder output dimension for observation processing
        self.encoder_out_dim = 64

    def _get_next_run_number(self):
        """
        Get the next run number by finding existing run directories.
        
        Returns
        -------
        int
            Next available run number (1, 2, 3, ...)
        """
        if not os.path.exists(self.model_path):
            return 1
        
        existing_runs = []
        for item in os.listdir(self.model_path):
            if os.path.isdir(os.path.join(self.model_path, item)) and item.startswith('run'):
                try:
                    run_num_str = item[3:]  # Remove "run" prefix
                    run_num = int(run_num_str)
                    existing_runs.append(run_num)
                except ValueError:
                    continue
        
        return max(existing_runs) + 1 if existing_runs else 1

    def _ensure_model_directory(self):
        """
        Ensure the model directory and run-specific directory exist.
        """
        # Create base model directory
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            print(f"📁 Created base model directory: {self.model_path}")
        
        # Create run-specific directory
        if not os.path.exists(self.run_directory):
            os.makedirs(self.run_directory)
            print(f"📁 Created run directory: {self.run_directory} (Run #{self.run_number})")
    
    def set_algorithm_name(self, algorithm_name):
        """
        Set the algorithm name for model saving.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the algorithm (e.g., 'PPO', 'DQN', 'GraphTransformer')
        """
        self.algorithm_name = algorithm_name
        print(f"🔧 Algorithm name set to: {self.algorithm_name}")
    
    def set_external_model(self, model):
        """
        Set an external RL model for saving.
        
        Parameters
        ----------
        model : stable_baselines3 model
            The RL model to save (PPO, DQN, etc.)
        """
        self.external_model = model
        if hasattr(model, '__class__'):
            # Automatically set algorithm name from model class
            self.algorithm_name = model.__class__.__name__
        print(f"🎯 External model set: {self.algorithm_name}")
    
    def save_model(self, episode_num=None):
        """
        Save the current model(s) with episode-specific naming in run directory.
        
        Parameters
        ----------
        episode_num : int, optional
            Episode number for naming. If None, uses current episode.
        """
        if episode_num is None:
            episode_num = self.current_episode
        
        # Create filename with algorithm and episode (timestamp now in directory structure)
        base_filename = f"{self.algorithm_name}_ep{episode_num:05d}"
        
        saved_files = []
        
        # Save external RL model if available (PPO, DQN, etc.)
        if self.external_model is not None:
            model_file = os.path.join(self.run_directory, f"{base_filename}")
            self.external_model.save(model_file)
            saved_files.append(model_file + ".zip")
        
        # Save internal graph policy network if available
        if (hasattr(self, 'policy_agent') and 
            self.policy_agent is not None and 
            hasattr(self.policy_agent, 'policy_network')):
            
            policy_file = os.path.join(self.run_directory, f"{base_filename}_policy.pth")
            torch.save({
                'policy_state_dict': self.policy_agent.policy_network.state_dict(),
                'encoder_state_dict': self.policy_agent.policy_network.encoder.state_dict(),
                'episode': episode_num,
                'hidden_dim': self.hidden_dim,
                'embedding_dim': self.embedding_dim,
                'run_number': self.run_number,
                'run_timestamp': self.run_timestamp
            }, policy_file)
            saved_files.append(policy_file)
        
        # Save environment metadata
        metadata_file = os.path.join(self.run_directory, f"{base_filename}_metadata.json")
        metadata = {
            'algorithm': self.algorithm_name,
            'episode': episode_num,
            'run_number': self.run_number,
            'run_timestamp': self.run_timestamp,
            'substrate_size': self.substrate_size,
            'max_nodes': self.max_nodes,
            'max_steps': self.max_steps,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'total_reward': getattr(self, 'episode_total_reward', 0.0),
            'save_timestamp': datetime.now().isoformat()
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files.append(metadata_file)
        
        print(f"💾 Saved model files for episode {episode_num} (Run #{self.run_number:04d}):")
        for file in saved_files:
            print(f"   📄 {os.path.basename(file)}")
        
        return saved_files
    
    def load_model(self, model_file_path):
        """
        Load a previously saved model.
        
        Parameters
        ----------
        model_file_path : str
            Path to the saved model file
        """
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        if model_file_path.endswith('.zip'):
            # Load stable-baselines3 model
            if self.algorithm_name == 'PPO':
                self.external_model = PPO.load(model_file_path, env=self)
            elif self.algorithm_name == 'DQN':
                self.external_model = DQN.load(model_file_path, env=self)
            else:
                raise ValueError(f"Unsupported algorithm for loading: {self.algorithm_name}")
        
        elif model_file_path.endswith('.pth'):
            # Load PyTorch model (graph policy network)
            checkpoint = torch.load(model_file_path)
            if (hasattr(self, 'policy_agent') and 
                self.policy_agent is not None and 
                hasattr(self.policy_agent, 'policy_network')):
                
                self.policy_agent.policy_network.load_state_dict(checkpoint['policy_state_dict'])
                self.policy_agent.policy_network.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                print(f"✅ Loaded graph policy network from episode {checkpoint.get('episode', 'unknown')}")
        
        print(f"📥 Model loaded from: {os.path.basename(model_file_path)}")
    
    def list_saved_models(self):
        """
        List all saved models organized by run directories.
        
        Returns
        -------
        dict
            Dictionary with run directories and their model files
        """
        if not os.path.exists(self.model_path):
            print(f"📁 Model directory does not exist: {self.model_path}")
            return {}
        
        runs_info = {}
        
        # Look for run directories
        for item in os.listdir(self.model_path):
            item_path = os.path.join(self.model_path, item)
            if os.path.isdir(item_path) and item.startswith('run'):
                try:
                    run_num_str = item[3:]  # Remove "run" prefix
                    run_num = int(run_num_str)
                    
                    # Get files in this run directory
                    run_files = os.listdir(item_path)
                    models = {
                        'rl_models': [f for f in run_files if f.endswith('.zip')],
                        'policy_models': [f for f in run_files if f.endswith('.pth')],
                        'metadata': [f for f in run_files if f.endswith('.json')]
                    }
                    
                    runs_info[f"run{run_num:04d}"] = models
                    
                except ValueError:
                    continue
        
        # Also check for legacy files in base directory (backward compatibility)
        base_files = [f for f in os.listdir(self.model_path) 
                      if os.path.isfile(os.path.join(self.model_path, f))]
        if base_files:
            legacy_models = {
                'rl_models': [f for f in base_files if f.endswith('.zip')],
                'policy_models': [f for f in base_files if f.endswith('.pth')],
                'metadata': [f for f in base_files if f.endswith('.json')]
            }
            if any(legacy_models.values()):
                runs_info['legacy'] = legacy_models
        
        print(f"📚 Models in {self.model_path}:")
        for run_name, models in sorted(runs_info.items()):
            print(f"  📂 {run_name}:")
            for model_type, file_list in models.items():
                if file_list:
                    print(f"    {model_type}: {len(file_list)} files")
                    for file in sorted(file_list)[:5]:  # Show first 5 files
                        print(f"      📄 {file}")
                    if len(file_list) > 5:
                        print(f"      ... and {len(file_list) - 5} more")
                else:
                    print(f"    {model_type}: No files")
        
        return runs_info

    def _setup_environment(self):
        """Initialize substrate, topology, state extractor, and policy components."""
        # Create substrate
        self.substrate = Substrate(self.substrate_size)
        self.substrate.create(self.substrate_type, **self.substrate_params)
        
        # Create topology
        self.topology = Topology(substrate=self.substrate, flush_delay=self.flush_delay)
        
        # Create state extractor
        self.state_extractor = TopologyState(self.topology)
        
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
            num_layers=4
        )
        
        policy = GraphPolicyNetwork(encoder, hidden_dim=self.hidden_dim, noise_scale=0.1)
        
        # Create policy agent
        self.policy_agent = TopologyPolicyAgent(self.topology, self.state_extractor, policy)
        self._policy_initialized = True

    def _get_encoder_observation(self, state):
        """
        Get observation from GraphInputEncoder output with semantic pooling.
        
        Args:
            state: State dictionary from state_extractor.get_state_features()
            
        Returns:
            np.ndarray: Fixed-size observation vector from encoder output
            
        Note:
            Uses semantic pooling when num_nodes > max_nodes to preserve representative
            nodes based on their features rather than arbitrary truncation.
        """
        try:
            # Extract components from state
            node_features = state['node_features']
            graph_features = state['graph_features']
            edge_features = state['edge_attr']
            edge_index = state['edge_index']
            
            # Convert edge_index from DGL tuple format to PyG tensor format if needed
            if isinstance(edge_index, tuple):
                src, dst = edge_index
                edge_index_tensor = torch.stack([src, dst], dim=0)  # [2, num_edges]
            else:
                edge_index_tensor = edge_index
            
            # Handle empty graphs
            if node_features.shape[0] == 0:
                return np.zeros((self.max_nodes + 1) * self.encoder_out_dim, dtype=np.float32)
            
            # Get encoder output directly
            encoder_out = self.observation_encoder(
                graph_features=graph_features,
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index_tensor
            )  # Shape: [num_nodes+1, out_dim]
            
            # Check for potential information reduction
            actual_nodes = encoder_out.shape[0] - 1  # -1 for graph token
            max_size = (self.max_nodes + 1) * self.encoder_out_dim
            
            if actual_nodes <= self.max_nodes:
                # Normal case: no pooling needed
                # print(f"✅ No pooling needed: {actual_nodes} nodes ≤ {self.max_nodes} max_nodes")
                encoder_flat = encoder_out.flatten().detach().cpu().numpy()
                
                # Pad with zeros if smaller than max size
                padded = np.zeros(max_size, dtype=np.float32)
                padded[:len(encoder_flat)] = encoder_flat
                return padded
            
            else:
                # 🧠 SEMANTIC POOLING: Intelligently select representative nodes
                # print(f"🧠 Applying semantic pooling: {actual_nodes} nodes > {self.max_nodes} max_nodes")
                
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
                
                # print(f"  📊 Pooling results:")
                # print(f"    - Original nodes: {actual_nodes}")
                # print(f"    - Selected nodes: {len(selected_indices)}")
                # print(f"    - Selected indices: {sorted(selected_indices.tolist())}")
                # print(f"    - Information preserved: {len(selected_indices)/actual_nodes*100:.1f}%")
                
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
        Perform semantic node selection using attention-based clustering.
        
        This method uses graph neural network embeddings to identify semantically
        similar nodes and select a diverse subset based on attention weights.
        The selection aims to choose nodes that are representative of different
        regions or characteristics in the topology.
        
        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings from the graph neural network, shape (num_nodes, embedding_dim)
        state : dict
            Current state dictionary containing graph information
        target_count : int
            Desired number of nodes to select
            
        Returns
        -------
        torch.Tensor
            Indices of selected nodes, shape (target_count,)
            
        Notes
        -----
        The algorithm uses attention mechanisms to compute node importance scores
        and applies diversity constraints to avoid selecting clustered nodes.
        """
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
        
        The action parameter is ignored as this environment uses an internal
        graph transformer policy to determine node actions (spawn/delete operations).
        The policy analyzes the current graph state and substrate conditions to
        make decisions about topology evolution.
        
        Parameters
        ----------
        action : Any
            Ignored parameter (maintained for gym.Env compatibility)
            
        Returns
        -------
        observation : np.ndarray
            Graph embedding observation from the encoder
        reward : float
            Scalar reward combining graph, node, and termination rewards
        terminated : bool
            True if episode terminated (success/failure conditions met)
        truncated : bool
            True if episode truncated (max steps reached)
        info : dict
            Additional information including reward breakdown and statistics
            
        Notes
        -----
        The reward system includes multiple components:
        - Graph-level rewards (connectivity, growth penalties)
        - Node-level rewards (movement, substrate interaction)
        - Termination rewards (success/failure bonuses/penalties)
        """
        self.current_step += 1
        
        # Store previous state for reward calculation
        prev_state = self.state_extractor.get_state_features(include_substrate=True)
        prev_num_nodes = prev_state['num_nodes']
        
        # Enqueue previous topology to history (maintain max capacity of delta_time)
        self.topology_history.append(prev_state['topology'])
        if len(self.topology_history) > self.delta_time:
            # Dequeue the oldest topology to maintain capacity
            self.dequeued_topology = self.topology_history.pop(0)  # Remove from front (FIFO)
        
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
        new_state = self.state_extractor.get_state_features(include_substrate=True)
        
        # Get observation from GraphInputEncoder output
        observation = self._get_encoder_observation(new_state)
        
        # Calculate reward 
        reward = self._calculate_reward(prev_state, new_state, executed_actions)
        
        # Reset new_node flags after reward calculation (they've served their purpose)
        self._reset_new_node_flags()
        
        # Check termination conditions
        terminated, termination_reward = self._check_terminated(new_state)
        
        # Add termination reward to the total reward
        if terminated:
            reward += termination_reward
            # Update reward breakdown to include termination reward
            if self.last_reward_breakdown:
                self.last_reward_breakdown['termination_reward'] = termination_reward
                self.last_reward_breakdown['total_reward'] = reward
        
        # Accumulate episode total reward
        self.episode_total_reward += reward
        
        truncated = self.current_step >= self.max_steps
        
        # Info dictionary
        info = {
            'num_nodes': new_state['num_nodes'],
            'num_edges': new_state['num_edges'],
            'actions_taken': len(executed_actions),
            'step': self.current_step,
            'policy_initialized': self._policy_initialized,
            'reward_breakdown': self.last_reward_breakdown  # Detailed reward information
        }
        
        # 📊 One-line performance summary
        centroid_x = new_state.get('graph_features', [0, 0, 0, 0])[3] if new_state['num_nodes'] > 0 else 0
        centroid_direction = "→" if len(self.centroid_history) >= 2 and centroid_x > self.centroid_history[-2] else "←" if len(self.centroid_history) >= 2 and centroid_x < self.centroid_history[-2] else "="
        spawn_r = self.last_reward_breakdown.get('spawn_reward', 0) if self.last_reward_breakdown else 0
        node_r = self.last_reward_breakdown.get('total_node_reward', 0) if self.last_reward_breakdown else 0
        edge_r = self.last_reward_breakdown.get('edge_reward', 0) if self.last_reward_breakdown else 0
        print(f"📊 Ep{self.current_episode:2d} Step{self.current_step:3d}: N={new_state['num_nodes']:2d} E={new_state['num_edges']:2d} | R={reward:+6.3f} (S:{spawn_r:+4.1f} N:{node_r:+4.1f} E:{edge_r:+4.1f}) | C={centroid_x:5.1f}{centroid_direction} | A={len(executed_actions):2d} | T={terminated} {truncated}")
        
        # Auto-render after each step to ensure visualization is always updated
        # This ensures visualization works even when PPO doesn't call render()
        if self.enable_visualization:
            self.render()
        
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, prev_state, new_state, actions):
        """
        Calculate reward based on topology evolution and substrate exploration.
        Returns scalar reward (graph-level + aggregated node-level).
        """
        # Initialize reward components
        graph_reward = 0.0
        node_rewards = []
        
        # === GRAPH-LEVEL REWARDS ===
        
        # Penalty for too many (N > Nc) or too few nodes (N < 2)
        num_nodes = new_state['num_nodes']
        if num_nodes < 2:
            graph_reward -= self.connectivity_penalty  # Strong penalty for losing all connectivity
        elif num_nodes > self.max_nodes:
            graph_reward -= self.growth_penalty  # Penalty for excessive growth, N > Nc
        else:
            graph_reward += self.survival_reward # Basic survival reward

        # Small reward for taking actions (encourages exploration)
        graph_reward += len(actions) * self.action_reward
        
        # === SPAWN REWARD: Durotaxis-based spawning ===
        spawn_reward = self._calculate_spawn_reward(prev_state, new_state, actions)
        graph_reward += spawn_reward
        
        # === DELETE REWARD: Proper deletion compliance ===
        delete_reward = self._calculate_delete_reward(prev_state, new_state, actions)
        graph_reward += delete_reward
        
        # === EDGE REWARD: Directional bias toward rightward movement ===
        edge_reward = self._calculate_edge_reward(prev_state, new_state, actions)
        graph_reward += edge_reward
        
        
        # === NODE-LEVEL REWARDS ===
        
        if num_nodes > 0:
            node_features = new_state['node_features']
            
            for i in range(num_nodes):
                node_reward = 0.0
                
                # 1. Position-based rewards (durotaxis progression)
                node_x = node_features[i][0].item()  # x-coordinate
                node_y = node_features[i][1].item()  # y-coordinate
                
                # Reward for moving rightward (positive durotaxis)
                if hasattr(self, 'prev_node_positions') and i < len(self.prev_node_positions):
                    prev_x = self.prev_node_positions[i][0]
                    x_movement = node_x - prev_x
                    node_reward += x_movement * self.movement_reward  # Reward rightward movement
                
                # 2. Dequeued topology comparison reward
                if self.dequeued_topology is not None:
                    # Get node's persistent ID for reliable tracking
                    node_persistent_id = self._get_node_persistent_id(i)
                    
                    # Check if this node was present in the dequeued topology using persistent ID
                    if node_persistent_id is not None:
                        node_was_in_dequeued = self._check_persistent_id_in_topology(node_persistent_id, self.dequeued_topology)
                    else:
                        # Fallback to spatial matching if persistent IDs not available
                        node_was_in_dequeued = self._node_exists_in_topology(node_x, node_y, self.dequeued_topology)
                    
                    if node_was_in_dequeued:
                        # Compute average substrate intensity of all nodes in current topology
                        current_intensities = []
                        for j in range(num_nodes):
                            if len(node_features[j]) > 2:
                                current_intensities.append(node_features[j][2].item())
                        
                        if current_intensities:
                            avg_intensity = sum(current_intensities) / len(current_intensities)
                            current_node_intensity = node_features[i][2].item() if len(node_features[i]) > 2 else 0.0
                            
                            # Set to_delete flag based on intensity comparison
                            if current_node_intensity < avg_intensity:
                                node_reward -= self.intensity_penalty  # Penalty for being below average
                                # Note: Automatic deletion based on intensity is disabled to prevent topology collapse
                                # print(f"📉 Node {i} (PID: {node_persistent_id}) below average intensity: {current_node_intensity:.3f} < {avg_intensity:.3f} (penalty: -{self.intensity_penalty})")
                            else:
                                node_reward += self.intensity_bonus  # Basic survival reward
                                # Note: Automatic deletion based on intensity is disabled to prevent topology collapse
                                # print(f"📈 Node {i} (PID: {node_persistent_id}) above/at average intensity: {current_node_intensity:.3f} >= {avg_intensity:.3f} (bonus: +{self.intensity_bonus})")

                # 3. Substrate intensity rewards
                if len(node_features[i]) > 2: 
                    substrate_intensity = node_features[i][2].item()
                    node_reward += substrate_intensity * self.substrate_reward  # Reward higher stiffness areas                
                
                # 4. Boundary position rewards
                if len(node_features[i]) > 7:  # Assuming boundary flag is at index 7
                    is_boundary = node_features[i][7].item()
                    if is_boundary > 0.5:  # Node is on boundary
                        node_reward += self.boundary_bonus  # Small bonus for frontier nodes
                
                # 5. Positional penalties (avoid substrate edges)
                substrate_width = self.substrate.width
                substrate_height = self.substrate.height
                
                # Penalty for being too close to left edge (opposite of durotaxis)
                if node_x < substrate_width * 0.1:
                    node_reward -= self.left_edge_penalty
                
                # Penalty for being too close to top/bottom edges
                if node_y < substrate_height * 0.1 or node_y > substrate_height * 0.9:
                    node_reward -= self.edge_position_penalty
                
                node_rewards.append(node_reward)
            
            # Store current positions for next step
            self.prev_node_positions = [(node_features[i][0].item(), node_features[i][1].item()) 
                                       for i in range(num_nodes)]
        
        # === COMBINE REWARDS ===
        
        # Aggregate node rewards (you can use different strategies)
        if node_rewards:
            # Strategy 1: Simple sum
            total_node_reward = sum(node_rewards)
            
            # Strategy 2: Average (uncomment to use)
            # total_node_reward = sum(node_rewards) / len(node_rewards)
            
            # Strategy 3: Weighted combination (uncomment to use)
            # total_node_reward = sum(node_rewards) * (num_nodes / self.max_nodes)
        else:
            total_node_reward = 0.0
        
        # Final combined reward
        total_reward = graph_reward + total_node_reward
        
        # Store detailed reward information for analysis
        self.last_reward_breakdown = {
            'total_reward': total_reward,
            'graph_reward': graph_reward,
            'spawn_reward': spawn_reward,
            'delete_reward': delete_reward,
            'edge_reward': edge_reward,
            'node_rewards': node_rewards,
            'total_node_reward': total_node_reward,
            'num_nodes': num_nodes
        }
        
        return total_reward

    def _calculate_spawn_reward(self, prev_state, new_state, actions):
        """
        Calculate reward for durotaxis-based spawning.
        
        Reward rule: If a node spawns a new node and the substrate intensity 
        of the new node is >= delta_intensity higher than the spawning node,
        then reward += spawn_success_reward, otherwise penalty -= spawn_failure_penalty
        
        Args:
            prev_state: Previous topology state
            new_state: Current topology state  
            actions: Actions taken (should include spawn actions)
            
        Returns:
            float: Spawn reward (spawn_success_reward per qualifying spawn, 0.0 otherwise)
        """
        spawn_reward = 0.0
        
        # Get node features from both states
        new_node_features = new_state['node_features']
        new_num_nodes = new_state['num_nodes']
        
        if new_num_nodes == 0:
            return spawn_reward
        
        # Use new_node flag to identify newly spawned nodes
        # The new_node flag is the last feature in the node feature vector
        for node_idx in range(new_num_nodes):
            if node_idx < len(new_node_features):
                node_feature_vector = new_node_features[node_idx]
                
                # Check if this is a newly spawned node (new_node flag = 1.0)
                if len(node_feature_vector) > 0:
                    new_node_flag = node_feature_vector[-1].item()  # Last feature is new_node flag
                    
                    if new_node_flag == 1.0:  # This is a newly spawned node
                        # Get substrate intensity (3rd feature, index 2)
                        if len(node_feature_vector) > 2:
                            new_node_intensity = node_feature_vector[2].item()
                            
                            # Find parent node from previous state by checking actions
                            # For now, we'll use spatial proximity as backup
                            best_parent_intensity = None
                            min_distance = float('inf')
                            
                            new_node_pos = node_feature_vector[:2]  # x, y coordinates
                            
                            # Check against previous state nodes
                            if 'node_features' in prev_state:
                                prev_node_features = prev_state['node_features']
                                for prev_idx, prev_node in enumerate(prev_node_features):
                                    if len(prev_node) > 2:
                                        prev_node_pos = prev_node[:2]
                                        distance = ((new_node_pos[0] - prev_node_pos[0])**2 + 
                                                   (new_node_pos[1] - prev_node_pos[1])**2)**0.5
                                        
                                        if distance < min_distance:
                                            min_distance = distance
                                            best_parent_intensity = prev_node[2].item()
                            
                            # Check spawn reward condition
                            if best_parent_intensity is not None:
                                intensity_difference = new_node_intensity - best_parent_intensity
                                
                                if intensity_difference >= self.delta_intensity:
                                    spawn_reward += self.spawn_success_reward
                                    # print(f"🎯 Spawn reward! New node intensity: {new_node_intensity:.3f}, "
                                    #       f"Parent intensity: {best_parent_intensity:.3f}, "
                                    #       f"Difference: {intensity_difference:.3f} >= {self.delta_intensity}")
                                else:
                                    spawn_reward -= self.spawn_failure_penalty
                                    # print(f"❌ Spawn penalty! New node intensity: {new_node_intensity:.3f}, "
                                    #       f"Parent intensity: {best_parent_intensity:.3f}, "
                                    #       f"Difference: {intensity_difference:.3f} < {self.delta_intensity}")
        
        return spawn_reward

    def _calculate_delete_reward(self, prev_state, new_state, actions):
        """
        Calculate reward/penalty based on deletion compliance with to_delete flags.
        
        Logic:
        - If a node from previous topology was marked to_delete=1 AND no longer exists: +delete_reward
        - If a node from previous topology was marked to_delete=1 BUT still exists: -delete_reward
        
        Args:
            prev_state: Previous state dict containing topology
            new_state: Current state dict containing topology  
            actions: Actions taken this step
            
        Returns:
            float: Delete reward (positive for proper deletions, negative for persistence)
        """
        delete_reward = 0.0
        
        # Need previous topology to check to_delete flags
        if 'topology' not in prev_state or prev_state['topology'] is None:
            return 0.0
            
        prev_topology = prev_state['topology']
        
        # Check if previous topology had any nodes
        if prev_topology.graph.num_nodes() == 0:
            return 0.0
            
        # Check if previous topology had to_delete flags
        if 'to_delete' not in prev_topology.graph.ndata:
            return 0.0
            
        # Get previous topology data
        prev_to_delete_flags = prev_topology.graph.ndata['to_delete']
        prev_persistent_ids = prev_topology.graph.ndata['persistent_id']
        
        # Current topology data
        current_topology = new_state['topology']
        if current_topology.graph.num_nodes() > 0:
            current_persistent_ids = current_topology.graph.ndata['persistent_id'].tolist()
        else:
            current_persistent_ids = []
        
        # Check each node from previous topology
        for i, to_delete_flag in enumerate(prev_to_delete_flags):
            if to_delete_flag.item() > 0.5:  # Node was marked for deletion
                prev_persistent_id = prev_persistent_ids[i].item()
                
                # Check if this persistent ID still exists in current topology
                if prev_persistent_id in current_persistent_ids:
                    # Node was marked for deletion but still exists - penalty
                    delete_reward -= self.delete_persistence_penalty
                    # print(f"🔴 Delete penalty! Node PID:{prev_persistent_id} was marked but still exists (-{self.delete_persistence_penalty})")
                else:
                    # Node was marked for deletion and was actually deleted - reward
                    delete_reward += self.delete_proper_reward
                    # print(f"🟢 Delete reward! Node PID:{prev_persistent_id} was properly deleted (+{self.delete_proper_reward})")
        
        return delete_reward

    def _calculate_edge_reward(self, prev_state, new_state, actions):
        """
        Calculate reward/penalty based on edge directions to encourage rightward movement.
        
        Logic:
        - For each edge, calculate direction vector from source to destination node
        - If edge points rightward (positive x-direction): +edge_reward
        - If edge points leftward (negative x-direction): -edge_reward
        - Vertical edges (same x-coordinate) get no reward/penalty
        
        Args:
            prev_state: Previous state dict (not used but kept for consistency)
            new_state: Current state dict containing topology
            actions: Actions taken this step (not used but kept for consistency)
            
        Returns:
            float: Edge reward (positive for rightward bias, negative for leftward bias)
        """
        edge_reward = 0.0
        
        # Need current topology to analyze edges
        if 'topology' not in new_state or new_state['topology'] is None:
            return 0.0
            
        current_topology = new_state['topology']
        
        # Check if topology has any nodes and edges
        if current_topology.graph.num_nodes() == 0 or current_topology.graph.num_edges() == 0:
            return 0.0
        
        # Get node positions and edge information
        node_positions = current_topology.graph.ndata['pos']  # Shape: [num_nodes, 2]
        edges = current_topology.graph.edges()  # Returns (src_nodes, dst_nodes)
        src_nodes, dst_nodes = edges
        
        rightward_edges = 0
        leftward_edges = 0
        
        # Analyze each edge direction
        for i in range(len(src_nodes)):
            src_idx = src_nodes[i].item()
            dst_idx = dst_nodes[i].item()
            
            # Get positions of source and destination nodes
            src_pos = node_positions[src_idx]  # [x, y]
            dst_pos = node_positions[dst_idx]  # [x, y]
            
            # Calculate direction vector
            dx = dst_pos[0].item() - src_pos[0].item()  # x-direction component
            
            # Categorize edge direction
            if dx > 0.01:  # Rightward (with small threshold to avoid numerical issues)
                rightward_edges += 1
                edge_reward += self.edge_rightward_bonus
            elif dx < -0.01:  # Leftward
                leftward_edges += 1
                edge_reward -= self.edge_leftward_penalty
            # If |dx| <= 0.01, consider it vertical/neutral (no reward/penalty)
        
        # Log edge direction analysis for debugging
        if rightward_edges > 0 or leftward_edges > 0:
            total_edges = current_topology.graph.num_edges()
            # print(f"🔀 Edge analysis: {rightward_edges} rightward (+), {leftward_edges} leftward (-), "
            #       f"{total_edges - rightward_edges - leftward_edges} vertical/neutral (0) "
            #       f"| Reward: {edge_reward:.3f}")
        
        return edge_reward

    def _node_exists_in_topology(self, node_x, node_y, topology, tolerance=None):
        """
        Check if a node with similar position exists in the given topology.
        Now uses persistent IDs for reliable tracking instead of spatial proximity.
        
        Args:
            node_x, node_y: Current node position (still used for fallback)
            topology: Topology object to search in
            tolerance: Maximum distance to consider nodes as "same" (fallback only)
            
        Returns:
            bool: True if a similar node exists in the topology
        """
        if topology is None:
            return False
            
        try:
            # Method 1: Use persistent IDs if available (preferred method)
            current_persistent_ids = self.topology.graph.ndata.get('persistent_id', None)
            dequeued_persistent_ids = topology.graph.ndata.get('persistent_id', None)
            
            if current_persistent_ids is not None and dequeued_persistent_ids is not None:
                # Find current node's persistent ID by position matching
                current_positions = self.topology.graph.ndata['pos']
                current_node_persistent_id = None
                
                for i, pos in enumerate(current_positions):
                    curr_x, curr_y = pos[0].item(), pos[1].item()
                    if abs(curr_x - node_x) < 0.01 and abs(curr_y - node_y) < 0.01:  # Very tight match for same node
                        current_node_persistent_id = current_persistent_ids[i].item()
                        break
                
                if current_node_persistent_id is not None:
                    # Check if this persistent ID exists in dequeued topology
                    for pid in dequeued_persistent_ids:
                        if pid.item() == current_node_persistent_id:
                            return True
                    return False
            
            # Method 2: Fallback to spatial proximity (original method)
            # print("⚠️ Falling back to spatial matching (persistent IDs not available)")
            
            # Calculate adaptive tolerance based on substrate and environment characteristics
            if tolerance is None:
                tolerance = self._calculate_adaptive_tolerance()
            
            # Get node positions from the topology
            if hasattr(topology, 'graph') and topology.graph.num_nodes() > 0:
                positions = topology.graph.ndata.get('pos', None)
                if positions is not None:
                    for pos in positions:
                        dequeued_x, dequeued_y = pos[0].item(), pos[1].item()
                        distance = ((node_x - dequeued_x)**2 + (node_y - dequeued_y)**2)**0.5
                        if distance <= tolerance:
                            return True
            return False
            
        except Exception as e:
            print(f"Error checking node existence in topology: {e}")
            return False

    def _calculate_adaptive_tolerance(self):
        """
        Calculate adaptive tolerance for node matching based on substrate characteristics.
        
        Returns:
            float: Adaptive tolerance value
        """
        # Base tolerance factors
        base_tolerance = 2.0
        
        # Factor 1: Substrate gradient steepness
        # For linear gradients: y = mx + b, steeper slope (larger |m|) = more movement
        if hasattr(self.substrate_params, 'get'):
            slope = abs(self.substrate_params.get('m', 0.01))
        elif isinstance(self.substrate_params, dict):
            slope = abs(self.substrate_params.get('m', 0.01))
        else:
            slope = 0.01  # Default slope
        
        # Adjust tolerance based on slope: steeper slope = larger tolerance
        slope_factor = max(1.0, slope * 100)  # Scale slope to reasonable range
        
        # Factor 2: Substrate size - larger substrates may need larger tolerances
        substrate_diagonal = (self.substrate_size[0]**2 + self.substrate_size[1]**2)**0.5
        size_factor = max(1.0, substrate_diagonal / 500)  # Normalize by reference size
        
        # Factor 3: Delta time - more steps between comparisons = more potential movement
        time_factor = max(1.0, self.delta_time / 3.0)  # Normalize by default delta_time
        
        # Factor 4: Current step - nodes might move more early in episode
        if self.current_step > 0:
            step_factor = max(0.5, 1.0 - (self.current_step / self.max_steps) * 0.3)
        else:
            step_factor = 1.0
        
        # Combine factors
        adaptive_tolerance = base_tolerance * slope_factor * size_factor * time_factor * step_factor
        
        # Clamp to reasonable bounds
        adaptive_tolerance = max(0.5, min(adaptive_tolerance, 10.0))
        
        # Debug output (can be removed in production)
        if hasattr(self, '_tolerance_debug_counter'):
            self._tolerance_debug_counter += 1
        else:
            self._tolerance_debug_counter = 1
        
        # if self._tolerance_debug_counter <= 3:  # Only print first few times
            # print(f"🎯 Adaptive tolerance: {adaptive_tolerance:.2f} "
            #       f"(slope_factor: {slope_factor:.2f}, size_factor: {size_factor:.2f}, "
            #       f"time_factor: {time_factor:.2f}, step_factor: {step_factor:.2f})")
        
        return adaptive_tolerance

    def _get_node_persistent_id(self, node_idx):
        """
        Get the persistent ID for a node at the given index.
        
        Args:
            node_idx: Current index of the node in the graph
            
        Returns:
            int or None: Persistent ID of the node, or None if not available
        """
        try:
            if hasattr(self.topology, 'graph') and 'persistent_id' in self.topology.graph.ndata:
                persistent_ids = self.topology.graph.ndata['persistent_id']
                if node_idx < len(persistent_ids):
                    return persistent_ids[node_idx].item()
            return None
        except Exception as e:
            print(f"Error getting persistent ID for node {node_idx}: {e}")
            return None

    def _check_persistent_id_in_topology(self, persistent_id, topology):
        """
        Check if a persistent ID exists in the given topology.
        
        Args:
            persistent_id: The persistent ID to search for
            topology: Topology object to search in
            
        Returns:
            bool: True if the persistent ID exists in the topology
        """
        try:
            if topology is None or persistent_id is None:
                return False
                
            if hasattr(topology, 'graph') and 'persistent_id' in topology.graph.ndata:
                dequeued_persistent_ids = topology.graph.ndata['persistent_id']
                for pid in dequeued_persistent_ids:
                    if pid.item() == persistent_id:
                        return True
            return False
        except Exception as e:
            print(f"Error checking persistent ID {persistent_id} in topology: {e}")
            return False

    def _reset_new_node_flags(self):
        """
        Reset all new_node flags to 0.0 after reward calculation.
        This ensures that nodes are only considered "new" for one step.
        """
        if hasattr(self.topology, 'graph') and 'new_node' in self.topology.graph.ndata:
            num_nodes = self.topology.graph.num_nodes()
            self.topology.graph.ndata['new_node'] = torch.zeros(num_nodes, dtype=torch.float32)

    def _check_terminated(self, state):
        """
        Check if the episode should terminate.
        
        Returns:
            tuple: (terminated: bool, termination_reward: float)
        """
        # 1. Terminate if number of nodes exceeds critical threshold ('fail termination')
        if state['num_nodes'] > self.num_critical_nodes:
            print(f"🚨 Episode terminated: Too many nodes ({state['num_nodes']} > {self.num_critical_nodes} critical threshold)")
            return True, self.critical_nodes_penalty
        
        # 2. Terminate if number of nodes becomes 0 ('fail termination')
        if state['num_nodes'] == 0:
            print(f"⚪ Episode terminated: No nodes remaining")
            return True, self.no_nodes_penalty

        # 3. Terminate if any node is out of bounds from the substrate ('fail termination')
        if state['num_nodes'] > 0:
            substrate_width = self.substrate.width
            substrate_height = self.substrate.height
            node_features = state['node_features']
            
            for i in range(state['num_nodes']):
                node_x = node_features[i][0].item()  # x-coordinate
                node_y = node_features[i][1].item()  # y-coordinate
                
                # Check if node is out of bounds
                if (node_x < 0 or node_y < 0 or 
                    node_x >= substrate_width or node_y >= substrate_height):
                    print(f"❌ Episode terminated: Node {i} out of bounds at ({node_x:.2f}, {node_y:.2f})")
                    print(f"   Substrate bounds: (0, 0) to ({substrate_width}, {substrate_height})")
                    return True, self.out_of_bounds_penalty

        # 4. Terminate if graph's centroid keeps going left consequently for 2*self.delta_time times ('fail termination')
        if state['num_nodes'] > 0:
            # Get current centroid x-coordinate directly from graph_features
            centroid_x = state['graph_features'][3].item()
            
            # Update centroid history
            self.centroid_history.append(centroid_x)
            
            # Check for consecutive leftward movement
            if len(self.centroid_history) >= 2:
                current_centroid = self.centroid_history[-1]
                previous_centroid = self.centroid_history[-2]
                
                if current_centroid < previous_centroid:  # Moving left
                    self.consecutive_left_moves += 1
                else:  # Not moving left (right, same, or first step)
                    self.consecutive_left_moves = 0
                
                # Check fail termination condition
                if self.consecutive_left_moves >= self.fail_threshold:
                    print(f"❌ Episode terminated: Centroid moved left {self.consecutive_left_moves} consecutive times (threshold: {self.fail_threshold})")
                    print(f"   Current centroid: {current_centroid:.2f}, Previous: {previous_centroid:.2f}")
                    return True, self.leftward_drift_penalty

        # 5. Terminate if one node from the graph reaches the rightmost location ('success termination')
        if state['num_nodes'] > 0:
            # Get substrate width to determine rightmost position
            substrate_width = self.substrate.width
            rightmost_x = substrate_width - 1  # Rightmost valid x-coordinate
            
            # Check each node's x-position (first element of node_features)
            node_features = state['node_features']
            for i in range(state['num_nodes']):
                node_x = node_features[i][0].item()  # x-coordinate
                if node_x >= rightmost_x:
                    print(f"🎯 Episode terminated: Node {i} reached rightmost location (x={node_x:.2f} >= {rightmost_x}) - SUCCESS!")
                    return True, self.success_reward

        # 6. Terminate if max_time_steps is reached
        if self.current_step >= self.max_steps:
            print(f"⏰ Episode terminated: Max time steps reached ({self.current_step}/{self.max_steps})")
            return True, self.timeout_penalty
        
        return False, 0.0  # No termination, no termination reward

    def get_topology_history(self):
        """
        Get the current topology history queue.
        
        Returns:
            list: List of previous topologies (max length = delta_time)
                 Index 0 = oldest, Index -1 = most recent
        """
        return self.topology_history.copy()  # Return a copy to prevent external modification

    def get_topology_history_length(self):
        """Get the current length of topology history queue."""
        return len(self.topology_history)

    def get_dequeued_topology(self):
        """
        Get the most recently dequeued topology.
        
        Returns:
            Topology or None: The topology that was most recently removed from 
                            the queue, or None if no topology has been dequeued yet
        """
        return self.dequeued_topology

    def get_node_rewards(self):
        """
        Get the individual node rewards from the last step.
        
        Returns:
            list: List of individual node rewards, empty if no rewards calculated yet
        """
        if self.last_reward_breakdown:
            return self.last_reward_breakdown['node_rewards']
        return []

    def get_reward_breakdown(self):
        """
        Get the complete reward breakdown from the last step.
        
        Returns:
            dict: Dictionary containing detailed reward information
        """
        return self.last_reward_breakdown if self.last_reward_breakdown else {}

    def get_average_node_reward(self):
        """
        Get the average node reward from the last step.
        
        Returns:
            float: Average node reward, 0.0 if no nodes
        """
        node_rewards = self.get_node_rewards()
        if node_rewards:
            return sum(node_rewards) / len(node_rewards)
        return 0.0

    def get_spawn_reward(self):
        """
        Get the spawn reward from the last step.
        
        Returns:
            float: Spawn reward, 0.0 if no qualifying spawns
        """
        if self.last_reward_breakdown:
            return self.last_reward_breakdown.get('spawn_reward', 0.0)
        return 0.0

    def get_delete_reward(self):
        """
        Get the delete reward from the last step.
        
        Returns:
            float: Delete reward (positive for proper deletions, negative for persistence), 0.0 if none
        """
        if self.last_reward_breakdown:
            return self.last_reward_breakdown.get('delete_reward', 0.0)
        return 0.0

    def get_edge_reward(self):
        """
        Get the edge reward from the last step.
        
        Returns:
            float: Edge reward (positive for rightward bias, negative for leftward bias), 0.0 if none
        """
        if self.last_reward_breakdown:
            return self.last_reward_breakdown.get('edge_reward', 0.0)
        return 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Save model from previous episode if it completed
        if (self.current_episode > 0 and 
            hasattr(self, 'episode_total_reward') and
            (self.external_model is not None or self._policy_initialized)):
            
            try:
                self.save_model(episode_num=self.current_episode)
            except Exception as e:
                print(f"⚠️  Model saving failed for episode {self.current_episode}: {e}")
        
        # Reset step counter and increment episode counter
        self.current_step = 0
        self.current_episode += 1
        
        # Initialize episode reward tracking
        self.episode_total_reward = 0.0
        
        # Reset centroid tracking for fail termination
        self.centroid_history = []
        self.consecutive_left_moves = 0
        
        # Reset topology history
        self.topology_history = []
        self.dequeued_topology = None
        
        # Reset node-level reward tracking
        self.prev_node_positions = []
        self.last_reward_breakdown = None
        
        # Reset topology
        self.topology.reset(init_num_nodes=self.init_num_nodes)
        
        # Initialize to_delete flags for all nodes to 0 (not marked for deletion)
        self._clear_all_to_delete_flags()
        
        # Initialize policy if not done yet
        self._initialize_policy()
        
        # Get initial observation using state features
        state = self.state_extractor.get_state_features(include_substrate=True)
        
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
        # Always show visualization if enabled, regardless of render_mode
        # This ensures visualization works even when wrapped by PPO/Monitor
        state = self.state_extractor.get_state_features(include_substrate=True)
        
        # Debug: Track render calls
        # print(f"DEBUG: render() called - Episode {self.current_episode}, Step {self.current_step}")
        
        # Visualize the topology using the show method (only if enabled)
        # Check actual topology node count, not processed state node count
        actual_num_nodes = self.topology.graph.num_nodes()
        
        if self.enable_visualization and hasattr(self.topology, 'show'):
            try:
                # Set step counter for visualization
                self.topology._step_counter = self.current_step
                
                # Show additional info if nodes exceed max_nodes
                if actual_num_nodes > self.max_nodes:
                    print(f"  🔍 Visualizing full topology: {actual_num_nodes} nodes (exceeds max_nodes={self.max_nodes})")
                elif actual_num_nodes == 0:
                    print(f"  🔍 Showing substrate-only: 0 nodes")
                
                # Always call topology.show() - it will handle zero nodes gracefully
                self.topology.show(highlight_outmost=True, update_only=True, episode_num=self.current_episode)
                
                # Force figure update to ensure visualization continues across episodes
                if hasattr(self.topology, 'force_figure_update'):
                    self.topology.force_figure_update()
                    
            except Exception as e:
                print(f"Visualization failed: {e}")
                # Try to recover visualization for next call
                if hasattr(self.topology, 'fig'):
                    self.topology.fig = None
                    self.topology.ax = None
        elif not self.enable_visualization:
            print("  📊 Visualization disabled (terminal output only)")
        # If visualization is enabled but topology.show doesn't exist, silently continue    # ============ to_delete Flag Management Methods ============
    
    def _set_node_to_delete_flag(self, node_idx, flag_value):
        """Set the to_delete flag for a specific node."""
        if self.topology.graph.num_nodes() == 0:
            return
        
        if 0 <= node_idx < self.topology.graph.num_nodes():
            self.topology.graph.ndata['to_delete'][node_idx] = float(flag_value)
    
    def _get_node_to_delete_flag(self, node_idx):
        """Get the to_delete flag for a specific node."""
        if self.topology.graph.num_nodes() == 0:
            return False
        
        if 0 <= node_idx < self.topology.graph.num_nodes():
            return bool(self.topology.graph.ndata['to_delete'][node_idx].item())
        return False
    
    def _get_nodes_marked_for_deletion(self):
        """Get indices of all nodes marked for deletion."""
        if self.topology.graph.num_nodes() == 0:
            return []
        
        to_delete_flags = self.topology.graph.ndata['to_delete']
        marked_indices = []
        for i, flag in enumerate(to_delete_flags):
            if flag.item() > 0.5:  # Consider > 0.5 as marked
                marked_indices.append(i)
        return marked_indices
    
    def _count_nodes_marked_for_deletion(self):
        """Count how many nodes are marked for deletion."""
        return len(self._get_nodes_marked_for_deletion())
    
    def _clear_all_to_delete_flags(self):
        """Clear all to_delete flags (set to 0.0)."""
        if self.topology.graph.num_nodes() > 0:
            self.topology.graph.ndata['to_delete'] = torch.zeros(
                self.topology.graph.num_nodes(), dtype=torch.float32
            )
    
    def get_deletion_analysis(self):
        """
        Get comprehensive analysis of nodes marked for deletion.
        
        Returns:
            dict: Analysis containing counts, percentages, and persistent IDs
        """
        if self.topology.graph.num_nodes() == 0:
            return {
                'total_nodes': 0,
                'nodes_marked_for_deletion': 0,
                'deletion_percentage': 0.0,
                'marked_node_indices': [],
                'persistent_ids_marked': [],
                'persistent_ids_safe': []
            }
        
        total_nodes = self.topology.graph.num_nodes()
        marked_indices = self._get_nodes_marked_for_deletion()
        marked_count = len(marked_indices)
        
        # Get persistent IDs
        persistent_ids = self.topology.graph.ndata['persistent_id'].tolist()
        
        marked_pids = [persistent_ids[i] for i in marked_indices]
        safe_indices = [i for i in range(total_nodes) if i not in marked_indices]
        safe_pids = [persistent_ids[i] for i in safe_indices]
        
        deletion_percentage = (marked_count / total_nodes * 100) if total_nodes > 0 else 0.0
        
        return {
            'total_nodes': total_nodes,
            'nodes_marked_for_deletion': marked_count,
            'deletion_percentage': deletion_percentage,
            'marked_node_indices': marked_indices,
            'persistent_ids_marked': marked_pids,
            'persistent_ids_safe': safe_pids
        }

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
    print("Setting up DurotaxisEnv with Graph Transformer Policy and Model Saving...")
    
    # Create the durotaxis environment with model saving
    env = DurotaxisEnv(
        substrate_size=(200, 200),
        substrate_type='linear',
        substrate_params={'m': 0.01, 'b': 1.0},
        init_num_nodes=5,
        max_nodes=30,
        max_steps=100,
        embedding_dim=64,
        hidden_dim=128,
        render_mode="human",  # Enable rendering
        enable_visualization=True,  # Enable topology.show()
        flush_delay=0.0001,
        model_path="./saved_models"  # Model saving directory
    )
    
    print("Environment created successfully!")
    print(f"🏠 Model saving directory: {env.model_path}")
    print(f"⏰ Run timestamp: {env.run_timestamp}")
    
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
        
        # Render the environment to show visualization
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    
    # Example of training with PPO and model saving
    print("\n--- Testing with PPO (Meta-Learning Approach) + Model Saving ---")
    
    try: 
        # Create a PPO model and register it with the environment
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        env.set_external_model(model)  # Register the model for saving
        
        print("Training for a short period with model saving...")
        
        # Train for multiple episodes to demonstrate saving
        episodes_to_train = 3
        steps_per_episode = 300
        
        for episode in range(episodes_to_train):
            print(f"\n🎯 Training Episode {episode + 1}/{episodes_to_train}")
            
            # Train for one episode worth of steps
            model.learn(total_timesteps=steps_per_episode)
            
            # Test the trained model for this episode
            print(f"🧪 Testing episode {episode + 1}...")
            obs, info = env.reset()  # This will save the model from previous episode
            
            episode_reward = 0
            for step in range(20):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                env.render()
                
                if terminated or truncated:
                    print(f"✅ Episode {episode + 1} completed with reward: {episode_reward:.3f}")
                    break
        
        # Save final model
        print(f"\n💾 Saving final model...")
        env.save_model()
        
    except Exception as e:
        print(f"❌ PPO training failed: {e}")
        print("This is expected - the environment is primarily designed for the graph transformer policy")
    
    # Clean up
    env.close()
    
    print("\n--- Direct Policy Testing with Model Saving ---")
    
    # Test the graph transformer policy directly with model saving
    env2 = DurotaxisEnv(
        init_num_nodes=1,
        max_steps=50,
        num_critical_nodes=200,
        embedding_dim=128,
        hidden_dim=128,
        render_mode="human",
        enable_visualization=True,
        model_path="./saved_models"
    )
    
    env2.set_algorithm_name("GraphTransformerDirect")  # Set custom algorithm name
    
    print(f"🔬 Direct policy test with saving to: {env2.model_path}")
    
    # Run multiple episodes to test saving
    for episode in range(3):
        print(f"\n🎮 Episode {episode + 1}/3")
        obs, info = env2.reset()  # This will save previous episode's model
        print(f"Initial: {info}")
        
        episode_reward = 0
        for step in range(15):
            obs, reward, terminated, truncated, info = env2.step(0)
            episode_reward += reward
            
            # Render the environment to show visualization
            env2.render()
            
            if terminated or truncated:
                print(f"✅ Episode {episode + 1} ended with reward: {episode_reward:.3f}")
                break
    
    # Save final model
    env2.save_model()
    
    env2.close()
    print("\n🚀 Durotaxis environment with model saving test completed!")
    
    # Show saved models
    import os
    if os.path.exists("./saved_models"):
        print(f"\n📁 Saved models in ./saved_models:")
        for file in sorted(os.listdir("./saved_models")):
            print(f"   � {file}")
    else:
        print("\n⚠️  No saved models directory found")