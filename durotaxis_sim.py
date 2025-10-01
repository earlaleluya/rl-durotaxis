import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO

from topology import Topology
from substrate import Substrate
from state import TopologyState
from encoder import GraphInputEncoder
from policy import GraphPolicyNetwork, TopologyPolicyAgent



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
                 delta_time=3,
                 delta_intensity=2.50,  
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
        self.delta_time = delta_time
        self.delta_intensity = delta_intensity  
        self.current_step = 0
        
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

    def _setup_environment(self):
        """Initialize substrate, topology, state extractor, and policy components."""
        # Create substrate
        self.substrate = Substrate(self.substrate_size)
        self.substrate.create(self.substrate_type, **self.substrate_params)
        
        # Create topology
        self.topology = Topology(substrate=self.substrate)
        
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
        terminated = self._check_terminated(new_state)
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
            graph_reward -= 5.0  # Strong penalty for losing all connectivity
        elif num_nodes > self.max_nodes:
            graph_reward -= 10.0  # Penalty for excessive growth, N > Nc
        else:
            graph_reward += 0.01 # Basic survival reward

        # Small reward for taking actions (encourages exploration)
        graph_reward += len(actions) * 0.005
        
        # === SPAWN REWARD: Durotaxis-based spawning ===
        spawn_reward = self._calculate_spawn_reward(prev_state, new_state, actions)
        graph_reward += spawn_reward
        
        
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
                    node_reward += x_movement * 0.01  # Reward rightward movement
                
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
                            
                            if current_node_intensity < avg_intensity:
                                node_reward -= 5.0  # Penalty for being below average
                                print(f"📉 Node {i} (PID: {node_persistent_id}) below average intensity: {current_node_intensity:.3f} < {avg_intensity:.3f} (penalty: -5.0)")
                            else:
                                node_reward += 0.01  # Basic survival reward
                                print(f"📈 Node {i} (PID: {node_persistent_id}) above/at average intensity: {current_node_intensity:.3f} >= {avg_intensity:.3f} (bonus: +0.01)")

                # 3. Substrate intensity rewards
                if len(node_features[i]) > 2: 
                    substrate_intensity = node_features[i][2].item()
                    node_reward += substrate_intensity * 0.05  # Reward higher stiffness areas                
                
                # 4. Boundary position rewards
                if len(node_features[i]) > 7:  # Assuming boundary flag is at index 7
                    is_boundary = node_features[i][7].item()
                    if is_boundary > 0.5:  # Node is on boundary
                        node_reward += 0.1  # Small bonus for frontier nodes
                
                # 5. Positional penalties (avoid substrate edges)
                substrate_width = self.substrate.width
                substrate_height = self.substrate.height
                
                # Penalty for being too close to left edge (opposite of durotaxis)
                if node_x < substrate_width * 0.1:
                    node_reward -= 0.2
                
                # Penalty for being too close to top/bottom edges
                if node_y < substrate_height * 0.1 or node_y > substrate_height * 0.9:
                    node_reward -= 0.1
                
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
        then reward += 1.0
        
        Args:
            prev_state: Previous topology state
            new_state: Current topology state  
            actions: Actions taken (should include spawn actions)
            
        Returns:
            float: Spawn reward (1.0 per qualifying spawn, 0.0 otherwise)
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
                                    spawn_reward += 1.0
                                    print(f"🎯 Spawn reward! New node intensity: {new_node_intensity:.3f}, "
                                          f"Parent intensity: {best_parent_intensity:.3f}, "
                                          f"Difference: {intensity_difference:.3f} >= {self.delta_intensity}")
                                else:
                                    spawn_reward -= 1.0
                                    print(f"❌ Spawn penalty! New node intensity: {new_node_intensity:.3f}, "
                                          f"Parent intensity: {best_parent_intensity:.3f}, "
                                          f"Difference: {intensity_difference:.3f} < {self.delta_intensity}")
        
        return spawn_reward

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
            print("⚠️ Falling back to spatial matching (persistent IDs not available)")
            
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
        
        if self._tolerance_debug_counter <= 3:  # Only print first few times
            print(f"🎯 Adaptive tolerance: {adaptive_tolerance:.2f} "
                  f"(slope_factor: {slope_factor:.2f}, size_factor: {size_factor:.2f}, "
                  f"time_factor: {time_factor:.2f}, step_factor: {step_factor:.2f})")
        
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
        """Check if the episode should terminate."""
        # Terminate if no nodes left ('fail termination')
        if state['num_nodes'] == 0:
            return True
        
        # Terminate if graph's centroid keeps going left consequently for 2*self.delta_time times ('fail termination')
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
                    return True

        # Terminate if one node from the graph reaches the rightmost location ('success termination')
        if state['num_nodes'] > 0:
            # Get substrate width to determine rightmost position
            substrate_width = self.substrate.width
            rightmost_x = substrate_width - 1  # Rightmost valid x-coordinate
            
            # Check each node's x-position (first element of node_features)
            node_features = state['node_features']
            for i in range(state['num_nodes']):
                node_x = node_features[i][0].item()  # x-coordinate
                if node_x >= rightmost_x:
                    print(f"🎯 Episode terminated: Node {i} reached rightmost location (x={node_x:.2f} >= {rightmost_x})")
                    return True
        
        return False

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

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
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
        if self.render_mode == "human":
            state = self.state_extractor.get_state_features(include_substrate=True)
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