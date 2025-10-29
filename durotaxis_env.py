import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import os

from topology import Topology
from substrate import Substrate
from state import TopologyState
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic, HybridPolicyAgent
from config_loader import ConfigLoader



class DurotaxisEnv(gym.Env):
    """
    A sophisticated durotaxis environment using graph transformer policies for cellular topology evolution.
    
    This environment simulates cellular durotaxis (directed migration in response to substrate stiffness gradients)
    using dynamic graph topologies where nodes represent individual cells and edges represent cellular connections.
    The environment employs reinforcement learning with graph neural networks to learn optimal cell migration,
    proliferation, and deletion strategies that maximize rightward movement along substrate gradients.
    
    The environment features a multi-component reward system that balances graph-level constraints (connectivity,
    growth control), node-level behaviors (movement, substrate interaction), edge directionality preferences,
    and sophisticated termination conditions. It includes semantic pooling for handling variable graph sizes
    and run-based model organization for systematic experiment management.
    
    Key Features
    ------------
    - **Dynamic Graph Topology**: Real-time node spawn/delete operations with persistent node tracking
    - **Substrate Gradients**: Configurable intensity fields (linear, exponential, custom) for durotaxis simulation
    - **Multi-Component Rewards**: Graph, node, edge, spawn, deletion, and termination reward components
    - **Semantic Pooling**: Intelligent node selection for large graphs using spatial and feature clustering
    - **Termination Conditions**: Success (rightmost boundary), failure (node limits, drift), and timeout scenarios
    - **Run Organization**: Automatic model saving with run directories (run0001, run0002, etc.)
    - **Real-time Visualization**: Optional topology rendering with configurable update rates
    
    Parameters
    ----------
    substrate_size : tuple of int, default=(600, 400)
        Dimensions of the substrate environment (width, height) in pixels.
    substrate_type : str, default='linear'
        Type of substrate gradient ('linear', 'exponential', 'gaussian', 'custom').
    substrate_params : dict, default={'m': 0.01, 'b': 1.0}
        Parameters for substrate generation. For linear: {'m': slope, 'b': intercept}.
    init_num_nodes : int, default=1
        Initial number of nodes (cells) when environment resets.
    max_critical_nodes : int, default=50
        Maximum allowed nodes before applying growth penalties.
    threshold_critical_nodes : int, default=200
        Critical threshold - episode terminates if exceeded (fail condition).
    max_steps : int, default=1000
        Maximum steps per episode before timeout termination.
    encoder_hidden_dim : int, default=128
        Hidden layer dimension for the GraphInputEncoder network.
    encoder_output_dim : int, default=64
        Output dimension for the GraphInputEncoder network embeddings.
    encoder_num_layers : int, default=4
        Number of layers in the GraphInputEncoder network.
    delta_time : int, default=3
        Time window for topology history comparison (affects reward calculations).
    delta_intensity : float, default=2.50
        Minimum intensity difference required for successful durotaxis spawning.
    
    graph_rewards : dict, default={'connectivity_penalty': 10.0, 'growth_penalty': 10.0, 'survival_reward': 0.01, 'action_reward': 0.005}
        Graph-level reward components:
        
        - 'connectivity_penalty': Penalty when nodes < 2 (loss of connectivity)
        - 'growth_penalty': Penalty when nodes > max_critical_nodes (excessive growth)
        - 'survival_reward': Base reward for maintaining valid topology
        - 'action_reward': Reward multiplier per action taken (encourages exploration)
    
    node_rewards : dict, default={'movement_reward': 0.01, 'intensity_penalty': 5.0, 'intensity_bonus': 0.01, 'substrate_reward': 0.05}
        Node-level reward components:
        
        - 'movement_reward': Reward multiplier for rightward movement (durotaxis)
        - 'intensity_penalty': Penalty for nodes below average substrate intensity
        - 'intensity_bonus': Bonus for nodes at/above average substrate intensity
        - 'substrate_reward': Reward multiplier for substrate intensity values
    
    edge_reward : dict, default={'rightward_bonus': 0.1, 'leftward_penalty': 0.1}
        Edge direction rewards:
        
        - 'rightward_bonus': Reward for edges pointing rightward (positive x-direction)
        - 'leftward_penalty': Penalty for edges pointing leftward (negative x-direction)
    
    spawn_rewards : dict, default={'spawn_success_reward': 1.0, 'spawn_failure_penalty': 1.0}
        Spawning behavior rewards:
        
        - 'spawn_success_reward': Reward for successful durotaxis-based spawning
        - 'spawn_failure_penalty': Penalty for spawning without sufficient intensity gradient
    
    delete_reward : dict, default={'proper_deletion': 2.0, 'persistence_penalty': 2.0}
        Deletion compliance rewards:
        
        - 'proper_deletion': Reward for deleting nodes marked with to_delete flag
        - 'persistence_penalty': Penalty for keeping nodes marked for deletion
    
    position_rewards : dict, default={'boundary_bonus': 0.1, 'left_edge_penalty': 0.2, 'edge_position_penalty': 0.1}
        Positional behavior rewards:
        
        - 'boundary_bonus': Bonus for nodes on topology boundary (frontier exploration)
        - 'left_edge_penalty': Penalty for nodes near left substrate edge
        - 'edge_position_penalty': Penalty for nodes near top/bottom substrate edges
    
    termination_rewards : dict, default={'success_reward': 100.0, 'out_of_bounds_penalty': -30.0, 'no_nodes_penalty': -30.0, 'leftward_drift_penalty': -30.0, 'timeout_penalty': -10.0, 'critical_nodes_penalty': -25.0}
        Episode termination rewards:
        
        - 'success_reward': Large reward for reaching rightmost substrate boundary
        - 'out_of_bounds_penalty': Penalty for nodes moving outside substrate bounds
        - 'no_nodes_penalty': Penalty for losing all nodes (topology collapse)
        - 'leftward_drift_penalty': Penalty for consistent leftward centroid movement
        - 'timeout_penalty': Small penalty for reaching maximum time steps
        - 'critical_nodes_penalty': Penalty for exceeding critical node threshold
    
    flush_delay : float, default=0.0001
        Delay between visualization updates (seconds) for rendering control.
    enable_visualization : bool, default=True
        Enable/disable automatic topology visualization during episodes.
    
    Attributes
    ----------
    substrate : Substrate
        The substrate environment containing intensity gradients and spatial information.
    topology : Topology
        Dynamic graph structure representing cellular topology with DGL backend.
    state_extractor : TopologyState
        Component for extracting graph features and node attributes from topology.
    observation_encoder : GraphInputEncoder
        Graph neural network encoder for converting topology to fixed-size observations.
    policy_agent : HybridPolicyAgent
        Graph transformer policy for intelligent action selection.
    action_space : gym.Space
        Discrete action space (dummy - actual actions determined by policy network).
    observation_space : gym.Space
        Box space for flattened graph embeddings with semantic pooling support.
    
    Methods
    -------
    reset(seed=None, options=None)
        Reset environment to initial state with optional seeding.
    step(action)
        Execute one timestep using graph transformer policy (action ignored).
    render(mode='human')
        Visualize current topology state with optional mode specification.
    
    Notes
    -----
    **Observation Space**: The environment uses semantic pooling when the number of nodes exceeds max_critical_nodes.
    This intelligently selects representative nodes using spatial and feature clustering rather than arbitrary
    truncation, preserving graph structure information for the policy network.
    
    **Reward System**: The multi-component reward system balances competing objectives:
    
    - Graph connectivity vs. growth control
    - Individual cell movement vs. collective behavior  
    - Exploration vs. exploitation of substrate gradients
    - Short-term actions vs. long-term durotaxis success
    
    **Termination Logic**: Episodes can terminate due to:
    
    - Success: Any node reaches rightmost substrate boundary
    - Failure: Node count exceeds critical threshold, all nodes lost, or persistent leftward drift
    - Timeout: Maximum steps reached without success/failure
    
    **Model Organization**: Each environment instance automatically creates run directories (run0001, run0002, etc.)
    for systematic experiment tracking. Models, metadata, and episode information are saved separately per run.
    
    Examples
    --------
    Basic environment setup:
    
    >>> env = Durotaxis(substrate_size=(800, 600), init_num_nodes=3, max_critical_nodes=30)
    >>> obs, info = env.reset()
    >>> obs, reward, terminated, truncated, info = env.step(0)  # Action ignored
    
    Custom reward configuration:
    
    >>> custom_rewards = {
    ...     'graph_rewards': {'connectivity_penalty': 15.0, 'survival_reward': 0.02},
    ...     'node_rewards': {'movement_reward': 0.02, 'substrate_reward': 0.1},
    ...     'termination_rewards': {'success_reward': 200.0}
    ... }
    >>> env = Durotaxis(**custom_rewards)
    
    Model saving and loading:
    
    Advanced substrate configuration:
    
    >>> env = Durotaxis(
    ...     substrate_type='exponential',
    ...     substrate_params={'base': 1.0, 'rate': 0.02, 'direction': 'x'},
    ...     delta_intensity=3.0,  # Higher threshold for durotaxis
    ...     threshold_critical_nodes=150  # Lower critical limit
    ... )
    """
    metadata = {"render_fps": 30}

    def __init__(self, 
                 config_path: str | dict | None = None,
                 **kwargs):
        """
        Initialize DurotaxisEnv with configuration from YAML file
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        **overrides
            Parameter overrides for any configuration values
        """
        super().__init__()
        
        # Load configuration
        config_loader = ConfigLoader(config_path)
        config = config_loader.get_environment_config()
        
        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:
                config[key] = value
        
        # Environment parameters
        self.substrate_size = tuple(config.get('substrate_size', [200, 200]))
        self.substrate_type = config.get('substrate_type', 'linear')
        self.substrate_params = config.get('substrate_params', {'m': 0.01, 'b': 1.0})
        self.init_num_nodes = config.get('init_num_nodes', 1)
        self.max_critical_nodes = config.get('max_critical_nodes', 50)
        self.threshold_critical_nodes = config.get('threshold_critical_nodes', 200)
        self.max_steps = config.get('max_steps', 200)
        
        # Encoder configuration from trainer overrides or config
        encoder_config = config_loader.get_encoder_config()
        # Note: hidden_dim was moved to actor_critic section, so we use trainer overrides or fallback
        self.encoder_hidden_dim = kwargs.get('encoder_hidden_dim', 128)  # Default fallback since hidden_dim was removed from encoder config
        self.encoder_out_dim = kwargs.get('encoder_output_dim', encoder_config.get('out_dim', 64))
        self.encoder_num_layers = kwargs.get('encoder_num_layers', encoder_config.get('num_layers', 4))
        
        # Simulation parameters
        self.delta_time = config.get('delta_time', 3)
        self.delta_intensity = config.get('delta_intensity', 2.50)
        self.flush_delay = config.get('flush_delay', 0.0001)
        self.enable_visualization = config.get('enable_visualization', True)
        
        # Simple delete-only mode flag
        self.simple_delete_only_mode = config.get('simple_delete_only_mode', False)
        
        # Centroid-to-goal distance-only mode flag
        self.centroid_distance_only_mode = config.get('centroid_distance_only_mode', False)
        
        # Include termination rewards flag (for special modes)
        # Default behavior:
        #   - If both modes are False: Always use termination rewards (default True, ignored)
        #   - If either mode is True: User must explicitly set this to True to include termination rewards
        self.include_termination_rewards = config.get('include_termination_rewards', False)

        # Distance-mode parameters
        dm = config.get('distance_mode', {})
        self.dm_use_delta = bool(dm.get('use_delta_distance', True))
        self.dm_dist_scale = float(dm.get('distance_reward_scale', 5.0))
        self.dm_term_scale = float(dm.get('terminal_reward_scale', 0.02))
        self.dm_term_clip = bool(dm.get('clip_terminal_rewards', True))
        self.dm_term_clip_val = float(dm.get('terminal_reward_clip_value', 10.0))
        self.dm_delete_scale = float(dm.get('delete_penalty_scale', 1.0))
        # Scheduler knobs already wired in trainer (optional)
        self._prev_centroid_x = None

        # Unpack reward dictionaries from config
        self.graph_rewards = config.get('graph_rewards', {
            'connectivity_penalty': 10.0,
            'growth_penalty': 10.0,
            'survival_reward': 0.01,
            'action_reward': 0.005
        })
        
        self.node_rewards = config.get('node_rewards', {
            'movement_reward': 0.01,
            'intensity_penalty': 5.0,
            'intensity_bonus': 0.01,
            'substrate_reward': 0.05
        })
        
        self.edge_reward = config.get('edge_reward', {
            'rightward_bonus': 0.1,
            'leftward_penalty': 0.1
        })
        
        self.spawn_rewards = config.get('spawn_rewards', {
            'spawn_success_reward': 1.0,
            'spawn_failure_penalty': 1.0
        })
        
        self.delete_reward = config.get('delete_reward', {
            'proper_deletion': 2.0,
            'persistence_penalty': 2.0,
            'improper_deletion_penalty': 2.0
        })
        
        self.position_rewards = config.get('position_rewards', {
            'boundary_bonus': 0.1,
            'left_edge_penalty': 0.2,
            'edge_position_penalty': 0.1
        })
        
        self.termination_rewards = config.get('termination_rewards', {
            'success_reward': 100.0,
            'out_of_bounds_penalty': -30.0,
            'no_nodes_penalty': -30.0,
            'leftward_drift_penalty': -30.0,
            'timeout_penalty': -10.0,
            'critical_nodes_penalty': -25.0
        })
        
        # Unpack reward component values for direct access
        self.delete_proper_reward = self.delete_reward['proper_deletion']
        self.delete_persistence_penalty = self.delete_reward['persistence_penalty']
        self.delete_improper_penalty = self.delete_reward.get('improper_deletion_penalty', 2.0)
        
        self.edge_rightward_bonus = self.edge_reward['rightward_bonus']
        self.edge_leftward_penalty = self.edge_reward['leftward_penalty']
        
        # Unpack grouped reward parameters
        self.connectivity_penalty = self.graph_rewards['connectivity_penalty']
        self.growth_penalty = self.graph_rewards['growth_penalty']
        self.survival_reward = self.graph_rewards['survival_reward']
        self.action_reward = self.graph_rewards['action_reward']
        
        self.centroid_movement_reward = self.graph_rewards.get('centroid_movement_reward', 0.0)
        
        self.movement_reward = self.node_rewards['movement_reward']
        self.leftward_penalty = self.node_rewards.get('leftward_penalty', 0.0)
        self.intensity_penalty = self.node_rewards['intensity_penalty']
        self.intensity_bonus = self.node_rewards['intensity_bonus']
        self.substrate_reward = self.node_rewards['substrate_reward']
        
        self.boundary_bonus = self.position_rewards['boundary_bonus']
        self.left_edge_penalty = self.position_rewards['left_edge_penalty']
        self.edge_position_penalty = self.position_rewards['edge_position_penalty']
        
        # Enhanced boundary avoidance parameters
        self.danger_zone_penalty = self.position_rewards.get('danger_zone_penalty', 2.0)
        self.critical_zone_penalty = self.position_rewards.get('critical_zone_penalty', 5.0)
        self.edge_zone_threshold = self.position_rewards.get('edge_zone_threshold', 0.15)
        self.danger_zone_threshold = self.position_rewards.get('danger_zone_threshold', 0.08)
        self.critical_zone_threshold = self.position_rewards.get('critical_zone_threshold', 0.03)
        self.safe_center_bonus = self.position_rewards.get('safe_center_bonus', 0.05)
        self.safe_center_range = self.position_rewards.get('safe_center_range', 0.30)
        
        self.spawn_success_reward = self.spawn_rewards['spawn_success_reward']
        self.spawn_failure_penalty = self.spawn_rewards['spawn_failure_penalty']
        
        # Boundary-aware spawn penalties
        self.spawn_near_boundary_penalty = self.spawn_rewards.get('spawn_near_boundary_penalty', 3.0)
        self.spawn_in_danger_zone_penalty = self.spawn_rewards.get('spawn_in_danger_zone_penalty', 8.0)
        self.spawn_boundary_check = self.spawn_rewards.get('spawn_boundary_check', True)
        
        self.success_reward = self.termination_rewards['success_reward']
        self.out_of_bounds_penalty = self.termination_rewards['out_of_bounds_penalty']
        self.no_nodes_penalty = self.termination_rewards['no_nodes_penalty']
        self.leftward_drift_penalty = self.termination_rewards['leftward_drift_penalty']
        self.timeout_penalty = self.termination_rewards['timeout_penalty']
        self.critical_nodes_penalty = self.termination_rewards['critical_nodes_penalty']

        # Empty-graph recovery configuration (prevent instant termination when possible)
        default_recovery_penalty = self.no_nodes_penalty * 0.5 if self.no_nodes_penalty < 0 else -abs(self.no_nodes_penalty)
        recovery_config = config.get('empty_graph_recovery', {})
        self.enable_empty_graph_recovery = config.get(
            'empty_graph_recovery_enabled',
            recovery_config.get('enabled', True)
        )
        recovery_nodes = config.get(
            'empty_graph_recovery_nodes',
            recovery_config.get('spawn_nodes', max(1, self.init_num_nodes))
        )
        self.empty_graph_recovery_nodes = max(1, int(recovery_nodes))
        recovery_penalty_cfg = config.get(
            'empty_graph_recovery_penalty',
            recovery_config.get('penalty', default_recovery_penalty)
        )
        self.empty_graph_recovery_penalty = recovery_penalty_cfg if recovery_penalty_cfg <= 0 else -abs(recovery_penalty_cfg)
        self.empty_graph_recovery_max_attempts = config.get(
            'empty_graph_recovery_max_attempts',
            recovery_config.get('max_attempts', 2)
        )
        self.empty_graph_recovery_noise = recovery_config.get('position_noise', 5.0)
        
        self.survival_reward_config = config.get('survival_reward_config', {'enabled': False})
        self.milestone_rewards = config.get('milestone_rewards', {'enabled': False})

        self.current_step = 0
        self.current_episode = 0
        
        # Milestone tracking (reset each episode)
        self._milestones_reached = set()
        
        # Centroid tracking for fail termination
        self.centroid_history = []  # Store centroid x-coordinates
        self.consecutive_left_moves = 0  # Count consecutive leftward moves
        
        # KMeans caching for performance optimization
        self._kmeans_cache = {}  # Cache for fitted KMeans models
        self._last_features_hash = None  # Hash of last features for cache invalidation
        self._cache_hit_count = 0  # Statistics for cache performance
        self._cache_miss_count = 0  # Statistics for cache performance
        
        # Encoder observation optimization
        self._max_obs_size = (self.max_critical_nodes + 1) * self.encoder_out_dim
        self._pre_allocated_obs = np.zeros(self._max_obs_size, dtype=np.float32)  # Reusable zero array
        self._edge_index_cache = None  # Cache for edge_index tensor conversion
        self._last_edge_structure_hash = None  # Hash of edge structure for cache invalidation
        self._edge_cache_hit_count = 0  # Statistics for edge cache performance
        self._edge_cache_miss_count = 0  # Statistics for edge cache performance
        self.fail_threshold = 2 * self.delta_time  # Threshold for fail termination

        # Topology tracking for substrate intensity comparison
        self.topology_history = []
        self.dequeued_topology = None  # Store the most recently dequeued topology
        
        # NEW: Advanced node feature tracking for intelligent deletion
        self._node_age = {}  # Tracks age of each node by persistent_id
        self._node_stagnation = {}  # Tracks stagnation counter for each node
        self._stagnation_threshold = 5.0  # Distance threshold to be considered "stagnant"
        
        # Node-level reward tracking
        self.prev_node_positions = []  # Store previous node positions for movement rewards
        self.last_reward_breakdown = None  # Store detailed reward information

        # Empty graph recovery bookkeeping
        self.empty_graph_recovery_attempts = 0
        self.empty_graph_recoveries_this_episode = 0
        self.total_empty_graph_recoveries = 0
        self.empty_graph_recovery_last_step = None
        
        # Curriculum learning parameters (initialized to defaults)
        self._curriculum_penalty_multiplier = 1.0
        self._survival_config = {}
        self.min_nodes_for_spawn = config.get('min_nodes_for_spawn', 2)
        self.max_nodes_for_spawn = config.get('max_nodes_for_spawn', self.max_critical_nodes)
        self.consecutive_left_moves_limit = config.get('consecutive_left_moves_limit', self.fail_threshold)
        
        # 1. Action Space
        # Since the agent uses graph_transformer_policy_dgl.act_with_policy(),
        # we define a dummy action space for compatibility with RL frameworks.
        # The actual actions are determined by the policy network based on graph embeddings.
        self.action_space = spaces.Discrete(1)  # Dummy action - actual actions come from policy network

        # 2. Observation Space
        # The observation space uses output from GraphInputEncoder directly.
        # Shape: [num_nodes+1, out_dim] where first element is graph token, rest are node embeddings
        max_critical_nodes_plus_graph = self.max_critical_nodes + 1  # +1 for graph token
        obs_dim = max_critical_nodes_plus_graph * self.encoder_out_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 3. Initialize environment components
        self._setup_environment()
        
        # Goal X position (rightmost substrate boundary) - set after substrate is created
        self.goal_x = self.substrate.width - 1
        
        # 4. Rendering setup complete
        
        # Create GraphInputEncoder for observations
        self.observation_encoder = GraphInputEncoder(
            hidden_dim=self.encoder_hidden_dim,
            out_dim=self.encoder_out_dim,
            num_layers=self.encoder_num_layers
        )

        # Store encoder output dimension for observation processing

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
            hidden_dim=self.encoder_hidden_dim,
            out_dim=self.encoder_out_dim,
            num_layers=self.encoder_num_layers
        )
        policy = HybridActorCritic(encoder, hidden_dim=self.encoder_hidden_dim)
        # Create policy agent
        self.policy_agent = HybridPolicyAgent(self.topology, self.state_extractor, policy)
        self._policy_initialized = True

    def _get_edge_structure_hash(self, edge_index):
        """
        Generate a hash for edge structure to detect changes.
        Uses edge count and structural characteristics for efficient comparison.
        """
        import hashlib
        
        if isinstance(edge_index, tuple):
            src, dst = edge_index
            # Convert to consistent format for hashing
            edge_data = torch.stack([src, dst], dim=0)
        else:
            edge_data = edge_index
            
        # Create hash from edge structure characteristics
        if edge_data.numel() == 0:
            return "empty_graph"
        
        # Use shape and a sample of edge connections for hash
        shape_str = str(edge_data.shape)
        
        # Sample edges for hash (avoid hashing entire large edge set)
        if edge_data.shape[1] > 100:
            # For large graphs, sample key edges
            indices = torch.linspace(0, edge_data.shape[1]-1, 20, dtype=torch.long, device=edge_data.device)
            sample = edge_data[:, indices].flatten()
        else:
            # For small graphs, use all edges
            sample = edge_data.flatten()
        
        hash_data = f"{shape_str}_{sample.cpu().numpy().tobytes().hex()}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    def _get_cached_edge_index(self, edge_index):
        """
        Get cached edge_index tensor or create new one.
        Returns (edge_index_tensor, is_cache_hit)
        """
        structure_hash = self._get_edge_structure_hash(edge_index)
        
        if (self._last_edge_structure_hash == structure_hash and 
            self._edge_index_cache is not None):
            self._edge_cache_hit_count += 1
            return self._edge_index_cache, True
        else:
            # Convert edge_index from DGL tuple format to PyG tensor format
            if isinstance(edge_index, tuple):
                src, dst = edge_index
                edge_index_tensor = torch.stack([src, dst], dim=0)  # [2, num_edges]
            else:
                edge_index_tensor = edge_index
            
            # Cache the converted tensor
            self._edge_index_cache = edge_index_tensor
            self._last_edge_structure_hash = structure_hash
            self._edge_cache_miss_count += 1
            
            return edge_index_tensor, False

    def _get_encoder_observation(self, state):
        """
        Get observation from GraphInputEncoder output with semantic pooling.
        
        Args:
            state: State dictionary from state_extractor.get_state_features()
            
        Returns:
            np.ndarray: Fixed-size observation vector from encoder output
            
        Note:
            Uses semantic pooling when num_nodes > max_critical_nodes to preserve representative
            nodes based on their features rather than arbitrary truncation.
        """
        try:
            # Extract components from state
            node_features = state['node_features']
            graph_features = state['graph_features']
            edge_features = state['edge_attr']
            edge_index = state['edge_index']
            
            # Get cached edge_index tensor (avoids repeated conversions)
            edge_index_tensor, edge_cache_hit = self._get_cached_edge_index(edge_index)
            
            # Handle empty graphs
            if node_features.shape[0] == 0:
                # Reuse pre-allocated array (avoid new allocation)
                self._pre_allocated_obs.fill(0.0)  # Reset to zeros
                return self._pre_allocated_obs.copy()  # Return copy to avoid mutation
            
            # Get encoder output directly
            encoder_out = self.observation_encoder(
                graph_features=graph_features,
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index_tensor
            )  # Shape: [num_nodes+1, out_dim]
            
            # ✅ SIMPLICIAL EMBEDDING HANDLES VARIABLE NODE COUNTS
            # No pooling needed - Simplicial Embedding in encoder provides geometric structure
            # that naturally handles variable graph sizes without information loss
            actual_nodes = encoder_out.shape[0] - 1  # -1 for graph token
            
            # Flatten encoder output directly (no pooling)
            encoder_flat = encoder_out.flatten().detach().cpu().numpy()
            
            # Dynamically handle variable-sized observations
            if len(encoder_flat) > len(self._pre_allocated_obs):
                # Resize pre-allocated array if needed
                self._pre_allocated_obs = np.zeros(len(encoder_flat), dtype=np.float32)
                self._max_obs_size = len(encoder_flat)
            
            # Use pre-allocated array (avoid new allocation for same-size or smaller obs)
            self._pre_allocated_obs.fill(0.0)  # Reset to zeros
            self._pre_allocated_obs[:len(encoder_flat)] = encoder_flat
            return self._pre_allocated_obs[:len(encoder_flat)].copy()  # Return only the used portion
                
        except Exception as e:
            print(f"Error getting encoder observation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to zero observation using pre-allocated array
            self._pre_allocated_obs.fill(0.0)
            return self._pre_allocated_obs.copy()

    def _get_features_cache_key(self, features_np, n_clusters):
        """
        Generate a cache key for KMeans based on feature characteristics.
        Uses data shape, sample of values, and clustering parameters.
        """
        import hashlib
        import numpy as np
        
        # Create a deterministic hash from feature characteristics
        shape_str = str(features_np.shape)
        
        # Sample key points for hash (to avoid hashing entire array)
        if features_np.size > 1000:
            # For large arrays, sample strategically
            indices = np.linspace(0, features_np.shape[0]-1, 20, dtype=int)
            sample = features_np.flat[indices]
        else:
            # For small arrays, use statistical summary
            sample = np.array([
                features_np.mean(), features_np.std(), 
                features_np.min(), features_np.max(),
                np.median(features_np.flat)
            ])
        
        # Combine shape, sample, and clustering params
        key_data = f"{shape_str}_{n_clusters}_{sample.tobytes().hex()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_kmeans(self, features_np, n_clusters, cache_prefix):
        """
        Get cached KMeans model or create new one.
        Returns (kmeans_model, is_cache_hit)
        """
        from sklearn.cluster import MiniBatchKMeans
        
        cache_key = f"{cache_prefix}_{self._get_features_cache_key(features_np, n_clusters)}"
        
        if cache_key in self._kmeans_cache:
            self._cache_hit_count += 1
            return self._kmeans_cache[cache_key], True
        else:
            # Create new KMeans model
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=256, n_init='auto', random_state=42)
            kmeans.fit(features_np)
            
            # Cache the fitted model
            self._kmeans_cache[cache_key] = kmeans
            self._cache_miss_count += 1
            
            # Limit cache size to prevent memory issues
            if len(self._kmeans_cache) > 50:  # Keep only 50 most recent models
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._kmeans_cache))
                del self._kmeans_cache[oldest_key]
            
            return kmeans, False

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
            Node embeddings from the graph neural network, shape (num_nodes, out_dim)
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
        from sklearn.cluster import MiniBatchKMeans
        
        num_nodes = node_embeddings.shape[0]
        
        try:
            # Strategy 1: Use raw node features for semantic clustering
            node_features = state['node_features']  # [num_nodes, 8] - rich feature set
            
            if node_features.shape[0] != num_nodes:
                # Fallback to uniform sampling if feature mismatch
                print(f"    ⚠️ Feature mismatch, using uniform sampling")
                device = node_features.device
                indices = torch.linspace(0, num_nodes-1, target_count, dtype=torch.long, device=device)
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
                        kmeans_spatial, is_cache_hit = self._get_cached_kmeans(positions, spatial_clusters, "spatial")
                        spatial_labels = kmeans_spatial.predict(positions)
                        
                        if is_cache_hit:
                            print(f"    🗺️ Spatial clustering: Using cached model for {spatial_clusters} clusters")
                        
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
                            kmeans_features, is_cache_hit = self._get_cached_kmeans(available_features, feature_clusters, "features")
                            feature_labels = kmeans_features.predict(available_features)
                            
                            if is_cache_hit:
                                print(f"    🎯 Feature clustering: Using cached model for {feature_clusters} clusters")
                            
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
            
            # Infer device from node_features
            device = node_features.device if hasattr(node_features, 'device') else torch.device('cpu')
            return torch.tensor(selected_indices, dtype=torch.long, device=device)
            
        except Exception as e:
            print(f"    ❌ Semantic selection failed: {e}")
            # Fallback to uniform sampling
            device = node_features.device if hasattr(node_features, 'device') else torch.device('cpu')
            indices = torch.linspace(0, num_nodes-1, target_count, dtype=torch.long, device=device)
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
        reward : dict
            Dictionary containing reward components with keys:
            - 'total_reward': float - Total scalar reward (sum of all components)
            - 'graph_reward': float - Graph-level rewards (connectivity, growth, actions)
            - 'spawn_reward': float - Durotaxis-based spawning rewards
            - 'delete_reward': float - Deletion compliance rewards
            - 'edge_reward': float - Edge direction rewards
            - 'total_node_reward': float - Aggregated node-level rewards
            - 'node_rewards': list - Individual node reward values
            - 'num_nodes': int - Number of nodes in current state
            - 'termination_reward': float - Termination reward (only if episode terminated)
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
        prev_state = self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )
        prev_num_nodes = prev_state['num_nodes']
        
        # Store a snapshot of topology for history tracking (not stored in state to avoid aliasing)
        # We store just the persistent_ids and positions for history purposes
        topology_snapshot = {
            'persistent_ids': prev_state['persistent_id'].clone() if 'persistent_id' in prev_state else torch.empty(0),
            'num_nodes': prev_state['num_nodes'],
            'num_edges': prev_state['num_edges'],
            'centroid_x': prev_state['centroid_x']
        }
        
        # Enqueue topology snapshot to history (maintain max capacity of delta_time)
        self.topology_history.append(topology_snapshot)
        if len(self.topology_history) > self.delta_time:
            # Dequeue the oldest snapshot to maintain capacity
            self.dequeued_topology = self.topology_history.pop(0)  # Remove from front (FIFO)
        
        # Check for empty graph BEFORE executing actions to prevent policy from seeing invalid state
        empty_graph_recovered_pre = False
        if self.enable_empty_graph_recovery and prev_num_nodes == 0:
            print(f"⚠️  Pre-action empty graph detected at step {self.current_step}, recovering...")
            recovered_state = self._recover_empty_graph(prev_state)
            if recovered_state is not None and recovered_state['num_nodes'] > 0:
                prev_state = recovered_state
                prev_num_nodes = recovered_state['num_nodes']
                empty_graph_recovered_pre = True
                # Update state extractor to point to recovered topology
                self.state_extractor.set_topology(self.topology)

        # Execute actions using the policy network (now guaranteed to have nodes)
        if self.policy_agent is not None and prev_num_nodes > 0:
            try:
                executed_actions = self.policy_agent.act_with_policy(
                    deterministic=False
                )
            except Exception as e:
                print(f"⚠️  Policy execution failed: {e}")
                executed_actions = {}
        else:
            # Fallback to random actions if policy fails or no nodes
            if prev_num_nodes > 0:
                executed_actions = self.topology.act()
            else:
                executed_actions = {}
        
        # Get new state after actions
        new_state = self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )

        # NEW: Update and inject advanced node features (age, stagnation)
        self._update_and_inject_node_features(new_state)

        # NEW: Heuristically mark nodes for deletion based on lab rules
        # This provides intelligent guidance to the agent about what to prune
        self._heuristically_mark_nodes_for_deletion()

        # Attempt graceful recovery if topology collapsed to zero nodes AFTER actions
        empty_graph_recovered_post = False
        if self.enable_empty_graph_recovery and new_state['num_nodes'] == 0:
            recovered_state = self._recover_empty_graph(prev_state)
            if recovered_state is not None and recovered_state['num_nodes'] > 0:
                new_state = recovered_state
                empty_graph_recovered_post = True

        # Track if ANY recovery happened this step
        empty_graph_recovered = empty_graph_recovered_pre or empty_graph_recovered_post

        # Get observation from GraphInputEncoder output (after any recovery adjustments)
        observation = self._get_encoder_observation(new_state)
        
        # Calculate reward components (returns detailed breakdown)
        reward_components = self._calculate_reward(prev_state, new_state, executed_actions)

        # Apply penalty when empty-graph recovery was needed (discourage aggressive deletions)
        # Only apply if enable_empty_graph_recovery is True (controlled by config flag)
        if empty_graph_recovered and self.enable_empty_graph_recovery:
            recovery_penalty = self.empty_graph_recovery_penalty
            reward_components['empty_graph_recovery_penalty'] = recovery_penalty
            # Apply recovery penalty to all modes when the flag is enabled
            reward_components['total_reward'] += recovery_penalty
        
        # Reset new_node flags after reward calculation (they've served their purpose)
        self._reset_new_node_flags()
        
        # Check termination conditions
        terminated, termination_reward = self._check_terminated(new_state)
        
        # Add termination reward to the reward components
        if terminated:
            reward_components['termination_reward'] = termination_reward
            
            # Determine if termination rewards should be included
            is_normal_mode = not self.simple_delete_only_mode and not self.centroid_distance_only_mode
            is_combined_mode = self.simple_delete_only_mode and self.centroid_distance_only_mode
            is_special_mode_with_termination = (self.simple_delete_only_mode or self.centroid_distance_only_mode) and self.include_termination_rewards
            
            # Include termination rewards if:
            # 1. Normal mode (both special modes disabled), OR
            # 2. Special mode with include_termination_rewards=True
            if is_normal_mode or is_special_mode_with_termination:
                if is_combined_mode:
                    # Combined mode: Distance + Delete + Scaled Termination
                    # Apply scaled and clipped termination (prioritize distance signal)
                    scaled_termination = termination_reward * self.dm_term_scale
                    if self.dm_term_clip:
                        scaled_termination = max(-self.dm_term_clip_val, 
                                                 min(self.dm_term_clip_val, scaled_termination))
                    reward_components['total_reward'] += scaled_termination
                    reward_components['termination_reward_scaled'] = scaled_termination
                elif self.simple_delete_only_mode:
                    # Simple delete mode only: Replace total with graph_reward + termination
                    reward_components['total_reward'] = (
                        reward_components.get('graph_reward', 0.0) + termination_reward
                    )
                elif self.centroid_distance_only_mode:
                    # Centroid distance mode only: Apply scaled and clipped termination
                    scaled_termination = termination_reward * self.dm_term_scale
                    if self.dm_term_clip:
                        scaled_termination = max(-self.dm_term_clip_val, 
                                                 min(self.dm_term_clip_val, scaled_termination))
                    reward_components['total_reward'] += scaled_termination
                    # Store the scaled version for logging
                    reward_components['termination_reward_scaled'] = scaled_termination
                else:
                    # Normal mode: Add termination reward to existing total
                    reward_components['total_reward'] += termination_reward
            # else: Special mode without include_termination_rewards flag - ignore termination rewards
        
        # Accumulate episode total reward (using scalar total for tracking)
        scalar_reward = reward_components['total_reward']
        self.episode_total_reward += scalar_reward
        
        truncated = self.current_step >= self.max_steps
        
        # Info dictionary
        info = {
            'num_nodes': new_state['num_nodes'],
            'num_edges': new_state['num_edges'],
            'actions_taken': len(executed_actions),
            'step': self.current_step,
            'policy_initialized': self._policy_initialized,
            'reward_breakdown': reward_components,  # Detailed reward information as dictionary
            'empty_graph_recovered': empty_graph_recovered,
            'empty_graph_recovery_attempts': self.empty_graph_recovery_attempts,
            'empty_graph_recoveries_this_episode': self.empty_graph_recovery_attempts
        }
        
        # 📊 One-line performance summary with boundary warnings
        centroid_x = new_state.get('graph_features', [0, 0, 0, 0])[3] if new_state['num_nodes'] > 0 else 0
        centroid_direction = "→" if len(self.centroid_history) >= 2 and centroid_x > self.centroid_history[-2] else "←" if len(self.centroid_history) >= 2 and centroid_x < self.centroid_history[-2] else "="
        spawn_r = reward_components.get('spawn_reward', 0)
        node_r = reward_components.get('total_node_reward', 0)
        edge_r = reward_components.get('edge_reward', 0)
        
        # Check if any nodes are in boundary danger zones
        boundary_warning = ""
        if new_state['num_nodes'] > 0:
            node_features = new_state['node_features']
            substrate_height = self.substrate.height
            nodes_in_danger = 0
            nodes_in_critical = 0
            
            for i in range(new_state['num_nodes']):
                y_pos = node_features[i][1].item()
                dist_from_top = y_pos / substrate_height
                dist_from_bottom = (substrate_height - y_pos) / substrate_height
                min_dist = min(dist_from_top, dist_from_bottom)
                
                if min_dist < self.critical_zone_threshold:
                    nodes_in_critical += 1
                elif min_dist < self.danger_zone_threshold:
                    nodes_in_danger += 1
            
            if nodes_in_critical > 0:
                boundary_warning = f" 🚨CRITICAL:{nodes_in_critical}"
            elif nodes_in_danger > 0:
                boundary_warning = f" ⚠️DANGER:{nodes_in_danger}"
        
        recovery_flag = " ♻️" if empty_graph_recovered else ""
        print(
            f"📊 Ep{self.current_episode:2d} Step{self.current_step:3d}: "
            f"N={new_state['num_nodes']:2d} E={new_state['num_edges']:2d} | "
            f"R={scalar_reward:+6.3f} (S:{spawn_r:+4.1f} N:{node_r:+4.1f} E:{edge_r:+4.1f}) | "
            f"C={centroid_x:5.1f}{centroid_direction}{boundary_warning}{recovery_flag} | "
            f"A={len(executed_actions):2d} | T={terminated} {truncated}"
        )
        
        # Auto-render after each step to ensure visualization is always updated
        # This ensures visualization works consistently
        if self.enable_visualization:
            self.render()
        
        return observation, reward_components, terminated, truncated, info

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
        elif num_nodes > self.max_critical_nodes:
            # Scaled penalty: grows quadratically with excess nodes to strongly discourage runaway growth
            excess_nodes = num_nodes - self.max_critical_nodes
            scaled_penalty = self.growth_penalty * (1 + excess_nodes / self.max_critical_nodes)
            graph_reward -= scaled_penalty
        else:
            graph_reward += self.survival_reward # Basic survival reward

        # Small reward for taking actions (encourages exploration)
        graph_reward += len(actions) * self.action_reward
        
        # === CENTROID MOVEMENT REWARD: Collective rightward progress ===
        centroid_reward = self._calculate_centroid_movement_reward(prev_state, new_state)
        graph_reward += centroid_reward
        
        # === SPAWN REWARD: Durotaxis-based spawning ===
        spawn_reward = self._calculate_spawn_reward(prev_state, new_state, actions)
        graph_reward += spawn_reward
        
        # === DELETE REWARD: Proper deletion compliance ===
        delete_reward = self._calculate_delete_reward(prev_state, new_state, actions)
        graph_reward += delete_reward
        
        # === DELETION EFFICIENCY REWARD: Tidiness bonus for smart pruning ===
        efficiency_reward = self._calculate_deletion_efficiency_reward(prev_state, new_state)
        graph_reward += efficiency_reward
        
        # === EDGE REWARD: Directional bias toward rightward movement ===
        edge_reward = self._calculate_edge_reward(prev_state, new_state, actions)
        graph_reward += edge_reward
        
        
        # === NODE-LEVEL REWARDS (VECTORIZED) ===
        
        if num_nodes > 0:
            import numpy as np
            node_features = new_state['node_features']
            
            # Convert to numpy for efficient vectorized operations
            node_features_np = node_features.detach().cpu().numpy()  # [num_nodes, feature_dim]
            
            # Initialize vectorized reward array
            node_rewards_vec = np.zeros(num_nodes, dtype=np.float32)
            
            # 1. VECTORIZED Position-based rewards (durotaxis progression)
            node_positions = node_features_np[:, :2]  # [num_nodes, 2] - x, y coordinates
            
            # Reward/penalize based on movement direction - VECTORIZED
            if hasattr(self, 'prev_node_positions') and len(self.prev_node_positions) > 0:
                # Convert previous positions to numpy array
                prev_positions = np.array(self.prev_node_positions[:num_nodes])  # Handle size mismatches
                if prev_positions.shape[0] == num_nodes:
                    # Vectorized movement calculation
                    x_movement = node_positions[:, 0] - prev_positions[:, 0]  # [num_nodes]
                    
                    # Reward rightward movement, penalize leftward movement
                    movement_rewards = np.where(
                        x_movement > 0,
                        x_movement * self.movement_reward,  # Rightward: strong reward
                        x_movement * self.leftward_penalty  # Leftward: penalty (x_movement is negative, so this is negative)
                    )
                    node_rewards_vec += movement_rewards
            
            # 2. VECTORIZED Substrate intensity rewards
            if node_features_np.shape[1] > 2:
                substrate_intensities = node_features_np[:, 2]  # [num_nodes] - intensity values
                intensity_rewards = substrate_intensities * self.substrate_reward  # [num_nodes]
                node_rewards_vec += intensity_rewards
            
            # 3. VECTORIZED Boundary position rewards  
            if node_features_np.shape[1] > 7:
                boundary_flags = node_features_np[:, 7]  # [num_nodes] - boundary indicators
                boundary_rewards = np.where(boundary_flags > 0.5, self.boundary_bonus, 0.0)  # [num_nodes]
                node_rewards_vec += boundary_rewards
            
            # 4. VECTORIZED Positional penalties (avoid substrate edges) - ENHANCED BOUNDARY AVOIDANCE
            substrate_width = self.substrate.width
            substrate_height = self.substrate.height
            
            # Left edge penalty - VECTORIZED
            left_edge_mask = node_positions[:, 0] < substrate_width * 0.1  # [num_nodes]
            left_edge_penalties = np.where(left_edge_mask, -self.left_edge_penalty, 0.0)  # [num_nodes]
            node_rewards_vec += left_edge_penalties
            
            # ENHANCED: Progressive top/bottom edge penalties - VECTORIZED
            # Calculate distance from top and bottom boundaries
            y_positions = node_positions[:, 1]  # [num_nodes]
            dist_from_top = y_positions / substrate_height  # 0.0 = top, 1.0 = bottom
            dist_from_bottom = (substrate_height - y_positions) / substrate_height  # 0.0 = bottom, 1.0 = top
            
            # Get minimum distance to either boundary (0.0 = at boundary, 0.5 = center)
            min_dist_to_boundary = np.minimum(dist_from_top, dist_from_bottom)  # [num_nodes]
            
            # Define zone thresholds
            edge_threshold = self.edge_zone_threshold      # 15% - edge zone
            danger_threshold = self.danger_zone_threshold  # 8% - danger zone
            critical_threshold = self.critical_zone_threshold  # 3% - critical zone
            safe_center_range = self.safe_center_range     # 30% - safe center
            
            # Progressive penalties based on proximity to boundary
            boundary_penalties = np.zeros(num_nodes, dtype=np.float32)
            
            # Critical zone: SEVERE penalty (within 3% of boundary)
            critical_mask = min_dist_to_boundary < critical_threshold
            boundary_penalties = np.where(critical_mask, -self.critical_zone_penalty, boundary_penalties)
            
            # Danger zone: Strong penalty (within 8% of boundary)
            danger_mask = (min_dist_to_boundary >= critical_threshold) & (min_dist_to_boundary < danger_threshold)
            boundary_penalties = np.where(danger_mask, -self.danger_zone_penalty, boundary_penalties)
            
            # Edge zone: Moderate penalty (within 15% of boundary)
            edge_mask = (min_dist_to_boundary >= danger_threshold) & (min_dist_to_boundary < edge_threshold)
            boundary_penalties = np.where(edge_mask, -self.edge_position_penalty, boundary_penalties)
            
            # Safe center zone: Small bonus (center 30% of height)
            center_dist = np.abs(y_positions - substrate_height/2) / (substrate_height/2)  # 0.0 = center, 1.0 = edge
            safe_center_mask = center_dist < safe_center_range
            safe_center_rewards = np.where(safe_center_mask, self.safe_center_bonus, 0.0)
            
            node_rewards_vec += boundary_penalties
            node_rewards_vec += safe_center_rewards
            
            # 5. Per-node intensity comparison (requires loop for complexity)
            if self.dequeued_topology is not None and node_features_np.shape[1] > 2:
                # Compute average intensity once for all nodes - VECTORIZED
                current_intensities = node_features_np[:, 2]  # [num_nodes]
                avg_intensity = np.mean(current_intensities)
                
                # Process each node for intensity comparison (some complexity requires per-node handling)
                for i in range(num_nodes):
                    node_x = node_positions[i, 0]
                    node_y = node_positions[i, 1]
                    
                    # Get node's persistent ID for reliable tracking
                    node_persistent_id = self._get_node_persistent_id(i)
                    
                    # Check if this node was present in the dequeued topology
                    if node_persistent_id is not None:
                        node_was_in_dequeued = self._check_persistent_id_in_topology(node_persistent_id, self.dequeued_topology)
                    else:
                        # Fallback to spatial matching if persistent IDs not available
                        node_was_in_dequeued = self._node_exists_in_topology(node_x, node_y, self.dequeued_topology)
                    
                    if node_was_in_dequeued:
                        current_node_intensity = current_intensities[i]
                        
                        # Set penalties/bonuses based on intensity comparison
                        if current_node_intensity < avg_intensity:
                            node_rewards_vec[i] -= self.intensity_penalty  # Penalty for being below average
                        else:
                            node_rewards_vec[i] += self.intensity_bonus  # Basic survival reward
            
            # Convert back to list for compatibility with existing code
            node_rewards = node_rewards_vec.tolist()
            
            # Store current positions for next step - VECTORIZED
            self.prev_node_positions = node_positions.tolist()  # Convert back to list format
          
        # === COMBINE REWARDS ===
        
        # Aggregate node rewards (you can use different strategies)
        if node_rewards:
            # Strategy 1: Simple sum
            total_node_reward = sum(node_rewards)
            
            # Strategy 2: Average (uncomment to use)
            # total_node_reward = sum(node_rewards) / len(node_rewards)
            
            # Strategy 3: Weighted combination (uncomment to use)
            # total_node_reward = sum(node_rewards) * (num_nodes / self.max_critical_nodes)
        else:
            total_node_reward = 0.0
        
        # Apply curriculum penalty multiplier to negative rewards
        penalty_multiplier = getattr(self, '_curriculum_penalty_multiplier', 1.0)
        if penalty_multiplier != 1.0:
            # Apply penalty multiplier to negative components
            if graph_reward < 0:
                graph_reward *= penalty_multiplier
            if total_node_reward < 0:
                total_node_reward *= penalty_multiplier
        
        # Add survival reward if configured
        survival_reward = self.get_survival_reward(self.current_step)
        
        # Add milestone rewards for reaching distance thresholds
        milestone_reward = self._calculate_milestone_reward(new_state)
        
        # Final combined reward
        total_reward = graph_reward + total_node_reward + survival_reward + milestone_reward
        
        # === COMBINED MODE: Distance Shaping + Delete Penalties ===
        # When BOTH modes are enabled, combine distance shaping with delete penalties
        if self.simple_delete_only_mode and self.centroid_distance_only_mode:
            # === 1. Calculate Distance Signal (Dense, Directional) ===
            centroid_x = 0.0
            if num_nodes > 0:
                try:
                    graph_features = new_state.get('graph_features')
                    if graph_features is not None:
                        if isinstance(graph_features, torch.Tensor):
                            centroid_x = graph_features[3].item()  # Index 3 is centroid_x
                        else:
                            centroid_x = graph_features[3]
                except (IndexError, TypeError):
                    # Fallback: calculate centroid from node positions
                    node_features = new_state.get('node_features', [])
                    if len(node_features) > 0:
                        x_positions = [node[0].item() if isinstance(node[0], torch.Tensor) else node[0] 
                                     for node in node_features]
                        centroid_x = sum(x_positions) / len(x_positions)
            
            # Use delta distance shaping (potential-based) for rightward migration
            if self.dm_use_delta and self._prev_centroid_x is not None and self.goal_x > 0:
                # Delta distance: reward ∝ (cx_t - cx_{t-1}) / goal_x
                # Positive when moving right, negative when moving left
                delta_x = centroid_x - self._prev_centroid_x
                distance_signal = self.dm_dist_scale * (delta_x / self.goal_x)
            else:
                # Fallback: static distance penalty
                if self.goal_x > 0:
                    distance_signal = -(self.goal_x - centroid_x) / self.goal_x
                else:
                    distance_signal = 0.0
            
            # Update previous centroid for next step
            self._prev_centroid_x = centroid_x
            
            # === 2. Calculate Delete Penalties (Efficient Node Management) ===
            # Extract only the delete penalties from delete_reward
            delete_penalty_only = delete_reward if delete_reward < 0 else 0.0
            
            # Rule 0: Growth penalty (when num_nodes > max_critical_nodes)
            growth_penalty_only = 0.0
            if num_nodes > self.max_critical_nodes:
                excess_nodes = num_nodes - self.max_critical_nodes
                growth_penalty_only = -self.growth_penalty * (1 + excess_nodes / self.max_critical_nodes)
            
            # Combine delete penalties
            delete_penalties_total = growth_penalty_only + delete_penalty_only
            
            # === 3. Combine Distance + Delete ===
            total_reward = distance_signal + delete_penalties_total
            graph_reward = distance_signal + delete_penalties_total
            
            # Zero out all other components (keep reward focused)
            spawn_reward = 0.0
            delete_reward = delete_penalties_total  # Keep for logging
            efficiency_reward = 0.0
            edge_reward = 0.0
            centroid_reward = 0.0
            milestone_reward = 0.0
            total_node_reward = 0.0
            survival_reward = 0.0
        
        # === SIMPLE DELETE-ONLY MODE ===
        # When enabled, zero out all rewards except delete penalties (Rule 1 & 2) and growth penalty (Rule 0)
        elif self.simple_delete_only_mode:
            # Extract only the delete penalties from delete_reward
            # In simple mode, we want ONLY penalties, no positive rewards
            delete_penalty_only = delete_reward if delete_reward < 0 else 0.0
            
            # Rule 0: Growth penalty (when num_nodes > max_critical_nodes)
            growth_penalty_only = 0.0
            if num_nodes > self.max_critical_nodes:
                excess_nodes = num_nodes - self.max_critical_nodes
                growth_penalty_only = -self.growth_penalty * (1 + excess_nodes / self.max_critical_nodes)
            
            # Combine penalties: Rule 0 (growth) + Rule 1 (persistence) + Rule 2 (improper deletion)
            total_reward = growth_penalty_only + delete_penalty_only
            graph_reward = growth_penalty_only + delete_penalty_only
            
            # Zero out all other components
            spawn_reward = 0.0
            efficiency_reward = 0.0
            edge_reward = 0.0
            centroid_reward = 0.0
            milestone_reward = 0.0
            total_node_reward = 0.0
            survival_reward = 0.0
        
        
        # === CENTROID-TO-GOAL DISTANCE-ONLY MODE ===
        # When enabled, provide ONLY distance-based penalty from centroid to goal
        elif self.centroid_distance_only_mode:
            # Calculate centroid x position
            centroid_x = 0.0
            if num_nodes > 0:
                try:
                    graph_features = new_state.get('graph_features')
                    if graph_features is not None:
                        if isinstance(graph_features, torch.Tensor):
                            centroid_x = graph_features[3].item()  # Index 3 is centroid_x
                        else:
                            centroid_x = graph_features[3]
                except (IndexError, TypeError) as e:
                    # Fallback: calculate centroid from node positions
                    node_features = new_state.get('node_features', [])
                    if len(node_features) > 0:
                        x_positions = [node[0].item() if isinstance(node[0], torch.Tensor) else node[0] 
                                     for node in node_features]
                        centroid_x = sum(x_positions) / len(x_positions)
            
            # === DISTANCE MODE OPTIMIZATION ===
            # Use delta distance shaping (potential-based) for faster learning
            if self.dm_use_delta and self._prev_centroid_x is not None and self.goal_x > 0:
                # Delta distance shaping: reward = scale × (cx_t - cx_{t-1}) / goal_x
                # Potential-based: Φ(s) = cx / goal_x, preserves optimal policy
                # Positive when moving right, negative when moving left
                delta_x = centroid_x - self._prev_centroid_x
                distance_signal = self.dm_dist_scale * (delta_x / self.goal_x)
            else:
                # Fallback: Static distance penalty (original behavior)
                # Calculate distance penalty: -(goal_x - centroid_x) / goal_x
                # As centroid approaches goal, penalty approaches 0
                # If centroid is at or past goal, penalty is 0 or positive (reward)
                if self.goal_x > 0:
                    distance_signal = -(self.goal_x - centroid_x) / self.goal_x
                else:
                    distance_signal = 0.0
            
            # Update previous centroid for next step
            self._prev_centroid_x = centroid_x
            
            # Set total reward to distance signal only
            total_reward = distance_signal
            graph_reward = distance_signal
            
            # Zero out all other components
            spawn_reward = 0.0
            delete_reward = 0.0
            efficiency_reward = 0.0
            edge_reward = 0.0
            centroid_reward = 0.0
            milestone_reward = 0.0
            total_node_reward = 0.0
            survival_reward = 0.0
        
        # OPTIMIZATION 2: Use preallocated template for faster dict creation
        reward_breakdown = dict(self._reward_components_template)
        reward_breakdown['total_reward'] = total_reward
        reward_breakdown['graph_reward'] = graph_reward
        reward_breakdown['spawn_reward'] = spawn_reward
        reward_breakdown['delete_reward'] = delete_reward
        reward_breakdown['deletion_efficiency_reward'] = efficiency_reward if 'efficiency_reward' in locals() else 0.0
        reward_breakdown['edge_reward'] = edge_reward
        reward_breakdown['centroid_reward'] = centroid_reward if 'centroid_reward' in locals() else 0.0
        reward_breakdown['milestone_reward'] = milestone_reward
        reward_breakdown['node_rewards'] = node_rewards
        reward_breakdown['total_node_reward'] = total_node_reward
        reward_breakdown['survival_reward'] = survival_reward
        reward_breakdown['num_nodes'] = num_nodes
        
        # Store for backward compatibility (some methods might still use this)
        self.last_reward_breakdown = reward_breakdown
        
        return reward_breakdown

    def _recover_empty_graph(self, prev_state):
        """Attempt to recover from an empty topology without terminating the episode."""
        if self.empty_graph_recovery_max_attempts <= 0:
            return None
        if self.empty_graph_recovery_attempts >= self.empty_graph_recovery_max_attempts:
            print(f"⚠️  Empty graph recovery skipped: max attempts reached ({self.empty_graph_recovery_max_attempts})")
            return None

        self.empty_graph_recovery_attempts += 1
        self.empty_graph_recoveries_this_episode += 1
        self.total_empty_graph_recoveries += 1
        self.empty_graph_recovery_last_step = self.current_step

        spawn_nodes = max(1, int(self.empty_graph_recovery_nodes))
        print(
            f"♻️  Empty graph recovery triggered at step {self.current_step} "
            f"(attempt {self.empty_graph_recovery_attempts}/{self.empty_graph_recovery_max_attempts}) — respawning {spawn_nodes} node(s)"
        )

        # Reset topology with minimal nodes then reposition near previous centroid if possible
        self.topology.reset(init_num_nodes=spawn_nodes)

        if prev_state is not None and prev_state.get('num_nodes', 0) > 0:
            self._reposition_recovered_nodes(prev_state)

        # Clear deletion flags and reset trackers to align with new graph
        self._clear_all_to_delete_flags()
        self.prev_node_positions = []
        self.topology_history = []
        self.dequeued_topology = None

        # Ensure state extractor references the updated topology
        if hasattr(self, 'state_extractor'):
            self.state_extractor.set_topology(self.topology)

        return self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )

    def _reposition_recovered_nodes(self, prev_state):
        """Shift recovered nodes near the previous centroid to preserve spatial continuity."""
        if self.topology.graph.num_nodes() == 0:
            return

        try:
            positions = self.topology.graph.ndata['pos']
            device = positions.device
            
            prev_graph_features = prev_state.get('graph_features')
            if isinstance(prev_graph_features, torch.Tensor):
                target_centroid = prev_graph_features[3:5].detach().cpu()
            else:
                target_centroid = torch.tensor(prev_graph_features[3:5], dtype=torch.float32, device=device)
        except Exception as exc:
            print(f"⚠️  Empty graph recovery repositioning failed: {exc}")
            return

        positions = self.topology.graph.ndata['pos']
        num_nodes = positions.shape[0]
        
        # IMPORTANT: Add diversity to prevent collinear nodes
        # Generate random offsets for each node to ensure they're not all at same position
        base_noise = max(5.0, self.empty_graph_recovery_noise if self.empty_graph_recovery_noise else 5.0)
        
        # Create diverse positions by radiating from target centroid
        if num_nodes == 1:
            # Single node: place at target with small noise
            adjusted_positions = target_centroid.to(positions.device).unsqueeze(0)
            noise = torch.randn_like(adjusted_positions) * (base_noise * 0.5)
            adjusted_positions = adjusted_positions + noise
        else:
            # Multiple nodes: arrange in a circle around target centroid to ensure diversity
            device = positions.device
            angles = torch.linspace(0, 2 * np.pi, num_nodes + 1, device=device)[:-1]  # Evenly spaced angles
            radius = base_noise * 2.0  # Radius of circle
            
            adjusted_positions = torch.zeros_like(positions)
            target_x, target_y = target_centroid[0].item(), target_centroid[1].item()
            
            for i in range(num_nodes):
                angle = angles[i].item()
                # Place on circle
                x = target_x + radius * np.cos(angle)
                y = target_y + radius * np.sin(angle)
                # Add small random jitter to break perfect symmetry
                x += np.random.uniform(-base_noise * 0.3, base_noise * 0.3)
                y += np.random.uniform(-base_noise * 0.3, base_noise * 0.3)
                adjusted_positions[i] = torch.tensor([x, y], dtype=positions.dtype, device=positions.device)

        # Clamp to substrate bounds
        width = float(self.substrate.width)
        height = float(self.substrate.height)
        adjusted_positions[:, 0] = torch.clamp(adjusted_positions[:, 0], 0.0, max(width - 1e-3, 0.0))
        adjusted_positions[:, 1] = torch.clamp(adjusted_positions[:, 1], 0.0, max(height - 1e-3, 0.0))

        self.topology.graph.ndata['pos'] = adjusted_positions

    def _calculate_centroid_movement_reward(self, prev_state, new_state):
        """
        Calculate reward based on collective centroid movement to the right.
        
        This reward encourages the entire cell colony to migrate rightward as a group,
        providing a strong signal for the primary goal of durotaxis.
        
        Args:
            prev_state: Previous state dict containing graph_features with centroid
            new_state: Current state dict containing graph_features with centroid
            
        Returns:
            float: Positive reward for rightward movement, negative for leftward
        """
        if new_state['num_nodes'] == 0 or prev_state['num_nodes'] == 0:
            return 0.0
        
        # Get centroid x-coordinates from graph features
        prev_centroid_x = prev_state['graph_features'][3].item()
        curr_centroid_x = new_state['graph_features'][3].item()
        
        # Calculate movement
        centroid_movement = curr_centroid_x - prev_centroid_x
        
        # Apply reward multiplier
        reward = centroid_movement * self.centroid_movement_reward
        
        return reward
    
    def _calculate_milestone_reward(self, new_state):
        """
        Calculate reward for reaching distance milestones.
        
        Provides progressive rewards as the agent reaches certain percentages
        of the substrate width, creating intermediate goals that guide learning.
        
        Args:
            new_state: Current state dict
            
        Returns:
            float: Milestone reward if a new milestone is reached, 0.0 otherwise
        """
        if not hasattr(self, 'milestone_rewards') or not self.milestone_rewards.get('enabled', False):
            return 0.0
        
        # Skip milestone rewards and printing when in centroid distance only mode
        if self.centroid_distance_only_mode:
            return 0.0
        
        if new_state['num_nodes'] == 0:
            return 0.0
        
        # Get the rightmost node position
        node_features = new_state['node_features']
        max_x = max(node_features[:, 0]).item()
        
        substrate_width = self.substrate.width
        progress_percent = (max_x / substrate_width) * 100
        
        # Check if we've reached a new milestone (track in episode)
        if not hasattr(self, '_milestones_reached'):
            self._milestones_reached = set()
        
        reward = 0.0
        
        # Check each milestone threshold
        milestones = [
            (25, 'distance_25_percent'),
            (50, 'distance_50_percent'),
            (75, 'distance_75_percent'),
            (90, 'distance_90_percent')
        ]
        
        for threshold, key in milestones:
            if progress_percent >= threshold and threshold not in self._milestones_reached:
                self._milestones_reached.add(threshold)
                milestone_reward = self.milestone_rewards.get(key, 0.0)
                reward += milestone_reward
                print(f"🎯 MILESTONE REACHED! {threshold}% of substrate width! Reward: +{milestone_reward}")
        
        return reward
    
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
                            new_node_pos = node_feature_vector[:2]  # x, y coordinates
                            new_node_y = new_node_pos[1].item()
                            
                            # Check if spawned near boundary (boundary-aware spawning)
                            if self.spawn_boundary_check:
                                substrate_height = self.substrate.height
                                
                                # Calculate distance from top/bottom boundaries
                                dist_from_top = new_node_y / substrate_height
                                dist_from_bottom = (substrate_height - new_node_y) / substrate_height
                                min_dist_to_boundary = min(dist_from_top, dist_from_bottom)
                                
                                # Apply penalties for spawning near boundaries
                                if min_dist_to_boundary < self.danger_zone_threshold:  # Within 8% - danger zone
                                    spawn_reward -= self.spawn_in_danger_zone_penalty
                                    # print(f"⚠️ Spawn in DANGER ZONE! Y={new_node_y:.1f}, "
                                    #       f"dist={min_dist_to_boundary:.2%} < {self.danger_zone_threshold:.2%}, "
                                    #       f"penalty: -{self.spawn_in_danger_zone_penalty}")
                                elif min_dist_to_boundary < self.edge_zone_threshold:  # Within 15% - edge zone
                                    spawn_reward -= self.spawn_near_boundary_penalty
                                    # print(f"⚠️ Spawn near boundary! Y={new_node_y:.1f}, "
                                    #       f"dist={min_dist_to_boundary:.2%} < {self.edge_zone_threshold:.2%}, "
                                    #       f"penalty: -{self.spawn_near_boundary_penalty}")
                            
                            # Find parent node from previous state by checking actions
                            # For now, we'll use spatial proximity as backup
                            best_parent_intensity = None
                            min_distance = float('inf')
                            
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
        - If a node from previous topology was marked to_delete=1 BUT still exists: -delete_reward (persistence)
        - If a node from previous topology was NOT marked (to_delete=0) BUT was deleted: -delete_reward (improper)
        
        Args:
            prev_state: Previous state dict containing topology
            new_state: Current state dict containing topology  
            actions: Actions taken this step
            
        Returns:
            float: Delete reward (positive for proper deletions, negative for persistence/improper deletions)
        """
        delete_reward = 0.0
        
        # Need previous state with persistent_id and to_delete data  
        if 'persistent_id' not in prev_state or prev_state['persistent_id'] is None:
            return 0.0
        
        if 'to_delete' not in prev_state or prev_state['to_delete'] is None:
            return 0.0
            
        if prev_state['num_nodes'] == 0:
            return 0.0
        
        # Use the cloned data from prev_state (captured at the time of state extraction)
        prev_to_delete_flags = prev_state['to_delete']
        prev_persistent_ids = prev_state['persistent_id']
        
        # Get current persistent IDs
        if new_state['num_nodes'] > 0 and 'persistent_id' in new_state:
            current_persistent_ids = set(new_state['persistent_id'].cpu().tolist())
        else:
            current_persistent_ids = set()
        
        # Check each node from previous state
        for i, to_delete_flag in enumerate(prev_to_delete_flags):
            prev_persistent_id = prev_persistent_ids[i].item()
            node_was_deleted = prev_persistent_id not in current_persistent_ids
            
            if to_delete_flag.item() > 0.5:  # Node was marked for deletion
                if node_was_deleted:
                    # Node was marked for deletion and was actually deleted - reward
                    # In simple_delete_only_mode, we give 0 instead of positive reward
                    if not self.simple_delete_only_mode:
                        delete_reward += self.delete_proper_reward
                    # print(f"🟢 Delete reward! Node PID:{prev_persistent_id} was properly deleted (+{self.delete_proper_reward})")
                else:
                    # Node was marked for deletion but still exists - penalty (RULE 1)
                    delete_reward -= self.delete_persistence_penalty
                    # print(f"🔴 Delete penalty! Node PID:{prev_persistent_id} was marked but still exists (-{self.delete_persistence_penalty})")
            else:  # Node was NOT marked for deletion (to_delete=0)
                if node_was_deleted:
                    # Node was NOT marked but was deleted anyway - penalty (RULE 2)
                    delete_reward -= self.delete_improper_penalty
                    # print(f"🔴 Improper delete penalty! Node PID:{prev_persistent_id} was deleted without marking (-{self.delete_improper_penalty})")
                # else: node not marked and still exists - neutral (expected behavior)
        
        return delete_reward
    
    def _calculate_deletion_efficiency_reward(self, prev_state, new_state):
        """
        Reward the agent for deleting old, stagnant, or strategically unimportant nodes.
        
        This "tidiness" reward encourages the agent to prune its network efficiently,
        particularly targeting nodes that are:
        - Old (existed for many steps without contributing)
        - Stagnant (not moving/exploring)
        - On low-quality substrate (marked by the heuristic system)
        
        Args:
            prev_state: Previous state dict (using snapshots)
            new_state: Current state dict (using snapshots)
            
        Returns:
            float: Efficiency reward for smart deletions
        """
        # Use snapshot data instead of topology references
        if prev_state['num_nodes'] == 0 or 'persistent_id' not in prev_state:
            return 0.0

        efficiency_reward = 0.0
        
        # Get persistent IDs from snapshots
        prev_pids = set(prev_state['persistent_id'].cpu().tolist())
        current_pids = set(new_state['persistent_id'].cpu().tolist()) if new_state['num_nodes'] > 0 else set()
        
        deleted_pids = prev_pids - current_pids

        if not deleted_pids:
            return 0.0

        # Define reward multipliers (tuned for balanced learning)
        AGE_REWARD_MULTIPLIER = 0.05      # Reward for deleting old nodes
        STAGNATION_REWARD_MULTIPLIER = 0.1  # Reward for deleting stagnant nodes
        
        for pid in deleted_pids:
            reward = 0.0
            
            # Reward for deleting old nodes (>50 steps old)
            if pid in self._node_age and self._node_age[pid] > 50:
                age_bonus = (self._node_age[pid] - 50) * AGE_REWARD_MULTIPLIER
                reward += age_bonus

            # Reward for deleting stagnant nodes (>20 steps without movement)
            if pid in self._node_stagnation and self._node_stagnation[pid]['count'] > 20:
                stagnation_bonus = (self._node_stagnation[pid]['count'] - 20) * STAGNATION_REWARD_MULTIPLIER
                reward += stagnation_bonus
            
            if reward > 0:
                # Optional: Debug logging (uncomment for debugging)
                # print(f"🧹 Tidiness reward! Deleted node {pid} "
                #       f"(Age: {self._node_age.get(pid, 0)}, "
                #       f"Stagnant: {self._node_stagnation.get(pid, {}).get('count', 0)}). "
                #       f"Reward: +{reward:.2f}")
                efficiency_reward += reward
                
        return efficiency_reward

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
            # Infer device from existing graph data
            device = self.topology.graph.ndata['pos'].device if 'pos' in self.topology.graph.ndata else torch.device('cpu')
            self.topology.graph.ndata['new_node'] = torch.zeros(num_nodes, dtype=torch.float32, device=device)

    def _check_terminated(self, state):
        """
        Check if the episode should terminate.
        
        Returns:
            tuple: (terminated: bool, termination_reward: float)
        """
        # 1. Terminate if number of nodes exceeds critical threshold ('fail termination')
        if state['num_nodes'] > self.threshold_critical_nodes:
            print(f"🚨 Episode terminated: Too many nodes ({state['num_nodes']} > {self.threshold_critical_nodes} critical threshold)")
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
                if self.consecutive_left_moves >= self.consecutive_left_moves_limit:
                    print(f"❌ Episode terminated: Centroid moved left {self.consecutive_left_moves} consecutive times (threshold: {self.consecutive_left_moves_limit})")
                    print(f"   Current centroid: {current_centroid:.2f}, Previous: {previous_centroid:.2f}")
                    return True, self.leftward_drift_penalty

        # 5. Terminate if one node from the graph reaches the rightmost area ('success termination')
        if state['num_nodes'] > 0:
            # Get substrate width to determine success threshold (last 1% of width)
            substrate_width = self.substrate.width
            success_threshold = substrate_width * 0.99  # Success when reaching 99% of width (last 1% area)
            
            # Check each node's x-position (first element of node_features)
            node_features = state['node_features']
            for i in range(state['num_nodes']):
                node_x = node_features[i][0].item()  # x-coordinate
                if node_x >= success_threshold:
                    print(f"🎯 Episode terminated: Node {i} reached rightmost area (x={node_x:.2f} >= {success_threshold:.2f}, {(node_x/substrate_width)*100:.1f}% of width) - SUCCESS!")
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
    
    def _update_and_inject_node_features(self, state):
        """
        Update age and stagnation for each node.
        
        This tracking provides the environment with information needed to make
        intelligent deletion decisions based on node lifetime and activity.
        """
        if state['num_nodes'] == 0:
            self._node_age.clear()
            self._node_stagnation.clear()
            return

        current_pids = set()
        if 'persistent_id' in self.topology.graph.ndata:
            pids = self.topology.graph.ndata['persistent_id'].cpu().numpy()
            positions = self.topology.graph.ndata['pos'].cpu().numpy()

            for i, pid in enumerate(pids):
                current_pids.add(pid)
                pos = positions[i]

                # Update age
                self._node_age[pid] = self._node_age.get(pid, 0) + 1

                # Update stagnation
                if pid in self._node_stagnation:
                    prev_pos = self._node_stagnation[pid]['pos']
                    distance = np.linalg.norm(pos - prev_pos)
                    if distance < self._stagnation_threshold:
                        self._node_stagnation[pid]['count'] += 1
                    else:
                        self._node_stagnation[pid] = {'pos': pos, 'count': 0}
                else:
                    self._node_stagnation[pid] = {'pos': pos, 'count': 0}
        
        # Clean up trackers for deleted nodes
        deleted_pids = set(self._node_age.keys()) - current_pids
        for pid in deleted_pids:
            if pid in self._node_age:
                del self._node_age[pid]
            if pid in self._node_stagnation:
                del self._node_stagnation[pid]
    
    def _heuristically_mark_nodes_for_deletion(self):
        """
        Mark nodes for deletion based on the two rules from physical experiments.
        
        Rule 1: Only apply deletion logic if num_nodes > max_critical_nodes
        Rule 2: Mark the rightmost nodes whose intensity at t-delta_time was
                below the current average intensity
        
        This implements the domain-specific deletion strategy observed in lab
        experiments, where the organism prunes inefficient parts of its network.
        """
        # Rule 1: Only apply deletion logic if the number of nodes exceeds the critical threshold
        if self.topology.graph.num_nodes() <= self.max_critical_nodes:
            return

        # Ensure we have a past topology to compare against
        if self.dequeued_topology is None:
            return
        
        # Check if dequeued topology has the required data
        if not isinstance(self.dequeued_topology, dict):
            return
        
        if 'persistent_id' not in self.dequeued_topology or 'node_features' not in self.dequeued_topology:
            return

        # Rule 2: Identify nodes whose intensity at t-delta_time was less than current average
        
        # Get current graph info
        current_state = self.state_extractor.get_state_features(
            include_substrate=False,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )
        if current_state['num_nodes'] == 0:
            return
            
        current_intensities = current_state['node_features'][:, 2].cpu().numpy()
        current_avg_intensity = np.mean(current_intensities)
        
        # Get previous graph info from the dequeued topology
        prev_pids = self.dequeued_topology['persistent_id'].cpu().numpy()
        prev_intensities = self.dequeued_topology['node_features'][:, 2].cpu().numpy()

        # Find candidate nodes: those with previous intensity below current average
        candidate_pids = []
        for i, pid in enumerate(prev_pids):
            if prev_intensities[i] < current_avg_intensity:
                candidate_pids.append(pid)

        if not candidate_pids:
            return

        # Find these candidates in the current graph and mark the rightmost ones
        current_pids_list = current_state['persistent_id'].cpu().numpy().tolist()
        nodes_to_mark = []
        
        for pid in candidate_pids:
            try:
                # Find the index of the candidate node in the current graph
                current_idx = current_pids_list.index(pid)
                pos_x = current_state['node_features'][current_idx, 0].item()
                nodes_to_mark.append((current_idx, pos_x, pid))
            except ValueError:
                # Node no longer exists, skip
                continue
        
        if not nodes_to_mark:
            return

        # Sort candidates by their x-position (rightmost first)
        nodes_to_mark.sort(key=lambda x: x[1], reverse=True)
        
        # Mark the top N rightmost candidates for deletion
        # Strategy: Mark 10-20% of excess nodes to encourage gradual, controlled pruning
        excess_nodes = self.topology.graph.num_nodes() - self.max_critical_nodes
        num_to_mark = max(1, min(int(excess_nodes * 0.15), len(nodes_to_mark)))
        
        nodes_marked_count = 0
        for i in range(num_to_mark):
            node_idx_to_mark, pos_x, pid = nodes_to_mark[i]
            self._set_node_to_delete_flag(node_idx_to_mark, 1.0)
            nodes_marked_count += 1
        
        # Optional: Debug logging (uncomment for debugging)
        # if nodes_marked_count > 0:
        #     print(f"🚩 Marked {nodes_marked_count} rightmost low-intensity nodes for deletion "
        #           f"(excess: {excess_nodes}, total: {self.topology.graph.num_nodes()})")

    def apply_curriculum_config(self, curriculum_config: dict):
        """Apply curriculum learning configuration to the environment."""
        if not curriculum_config:
            return
            
        # Apply penalty multiplier if specified
        if 'penalty_multiplier' in curriculum_config:
            self._curriculum_penalty_multiplier = curriculum_config['penalty_multiplier']
        else:
            self._curriculum_penalty_multiplier = 1.0
            
        # Apply reduced spawn requirements if specified
        spawn_config = curriculum_config.get('spawn_requirements', {})
        if 'min_nodes' in spawn_config:
            self.min_nodes_for_spawn = spawn_config['min_nodes']
        if 'max_nodes' in spawn_config:
            self.max_nodes_for_spawn = spawn_config['max_nodes']
            
        # Apply relaxed termination if specified
        termination_config = curriculum_config.get('termination', {})
        if 'consecutive_left_moves' in termination_config:
            self.consecutive_left_moves_limit = termination_config['consecutive_left_moves']
            
    def get_survival_reward(self, step_count: int) -> float:
        """
        Calculate time-based survival reward that increases with episode length.
        
        Encourages the agent to survive longer by providing:
        1. Base reward for each step
        2. Bonus reward after reaching a threshold
        3. Progressive scaling based on max_steps
        
        Args:
            step_count: Current step number in episode
            
        Returns:
            float: Survival reward for current step
        """
        # Check if survival rewards are configured and enabled
        if not hasattr(self, 'survival_reward_config'):
            return 0.0
        
        survival_config = self.survival_reward_config
        if not survival_config.get('enabled', False):
            return 0.0
        
        base_reward = survival_config.get('base_reward', 0.02)
        bonus_threshold = survival_config.get('bonus_threshold', 100)
        bonus_reward = survival_config.get('bonus_reward', 0.05)
        max_step_factor = survival_config.get('max_step_factor', 0.8)
        
        # Base survival reward for each step
        reward = base_reward
        
        # Bonus for reaching threshold
        if step_count >= bonus_threshold:
            reward += bonus_reward
        
        # Progressive scaling: reward increases as episode progresses
        # This creates a strong incentive to reach longer episodes
        if step_count > 0 and self.max_steps > 0:
            progress_factor = min(step_count / (self.max_steps * max_step_factor), 1.0)
            reward *= (1.0 + progress_factor)  # Up to 2x multiplier
        
        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Save model from previous episode if it completed
        # Reset step counter and increment episode counter
        self.current_step = 0
        self.current_episode += 1
        
        # Initialize episode reward tracking
        self.episode_total_reward = 0.0
        
        # Reset centroid tracking for fail termination
        self.centroid_history = []
        self.consecutive_left_moves = 0
        
        # Reset previous centroid for delta distance computation (distance mode optimization)
        self._prev_centroid_x = None
        
        # Reset topology history
        self.topology_history = []
        self.dequeued_topology = None
        
        # Reset node-level reward tracking
        self.prev_node_positions = []
        self.last_reward_breakdown = None
        
        # NEW: Reset advanced feature trackers
        self._node_age.clear()
        self._node_stagnation.clear()
        
        # Reset milestone tracking for new episode
        self._milestones_reached = set()

        # Reset empty-graph recovery tracking for new episode
        self.empty_graph_recovery_attempts = 0
        self.empty_graph_recoveries_this_episode = 0
        self.empty_graph_recovery_last_step = None
        
        # OPTIMIZATION 2: Preallocate reward dict template to reduce per-step allocations
        self._reward_components_template = {
            'total_reward': 0.0,
            'graph_reward': 0.0,
            'spawn_reward': 0.0,
            'delete_reward': 0.0,
            'deletion_efficiency_reward': 0.0,
            'edge_reward': 0.0,
            'centroid_reward': 0.0,
            'milestone_reward': 0.0,
            'node_rewards': [],
            'total_node_reward': 0.0,
            'survival_reward': 0.0,
            'empty_graph_recovery_penalty': 0.0,
            'termination_reward': 0.0,
            'num_nodes': 0
        }
        
        # Reset topology
        self.topology.reset(init_num_nodes=self.init_num_nodes)
        
        # Verify initial centroid is in safe center zone
        if self.init_num_nodes > 0:
            initial_state = self.state_extractor.get_state_features(
                include_substrate=True,
                node_age=self._node_age,
                node_stagnation=self._node_stagnation
            )
            if initial_state['num_nodes'] > 0:
                initial_centroid_y = initial_state['graph_features'][4].item()  # Centroid Y
                substrate_height = self.substrate.height
                y_percentage = (initial_centroid_y / substrate_height) * 100
                
                # Check if in safe zone (40-60%)
                in_safe_zone = 40 <= y_percentage <= 60
                zone_indicator = "✅" if in_safe_zone else "⚠️"
                
                print(f"   {zone_indicator} Initial centroid Y: {initial_centroid_y:.1f} ({y_percentage:.1f}% of height) - {'SAFE CENTER' if in_safe_zone else 'NOT CENTERED'}")
        
        # Initialize to_delete flags for all nodes to 0 (not marked for deletion)
        self._clear_all_to_delete_flags()
        
        # Initialize policy if not done yet
        self._initialize_policy()
        
        # Get initial observation using state features
        state = self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )
        
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
        """Render the current state of the environment using enable_visualization setting."""
        # Always show visualization if enabled, based on enable_visualization setting
        # This ensures visualization works consistently across different usage patterns
        state = self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )
        
        # Debug: Track render calls
        # print(f"DEBUG: render() called - Episode {self.current_episode}, Step {self.current_step}")
        
        # Visualize the topology using the show method (only if enabled)
        # Check actual topology node count, not processed state node count
        actual_num_nodes = self.topology.graph.num_nodes()
        
        if self.enable_visualization and hasattr(self.topology, 'show'):
            try:
                # Set step counter for visualization
                self.topology._step_counter = self.current_step
                
                # Show additional info if nodes exceed max_critical_nodes
                if actual_num_nodes > self.max_critical_nodes:
                    print(f"  🔍 Visualizing full topology: {actual_num_nodes} nodes (exceeds max_critical_nodes={self.max_critical_nodes})")
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
    
    def get_kmeans_cache_stats(self):
        """Get KMeans cache performance statistics."""
        total_requests = self._cache_hit_count + self._cache_miss_count
        hit_rate = (self._cache_hit_count / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self._cache_hit_count,
            'cache_misses': self._cache_miss_count,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cached_models_count': len(self._kmeans_cache)
        }
    
    def clear_kmeans_cache(self):
        """Clear the KMeans cache and reset statistics."""
        self._kmeans_cache.clear()
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def get_encoder_cache_stats(self):
        """Get encoder observation cache performance statistics."""
        total_edge_requests = self._edge_cache_hit_count + self._edge_cache_miss_count
        edge_hit_rate = (self._edge_cache_hit_count / total_edge_requests * 100) if total_edge_requests > 0 else 0.0
        
        return {
            'edge_cache_hits': self._edge_cache_hit_count,
            'edge_cache_misses': self._edge_cache_miss_count,
            'total_edge_requests': total_edge_requests,
            'edge_hit_rate_percent': edge_hit_rate,
            'pre_allocated_array_size': self._max_obs_size,
            'memory_saved_per_call': 'Reuses single array instead of allocating new'
        }
    
    def clear_encoder_cache(self):
        """Clear the encoder observation cache and reset statistics."""
        self._edge_index_cache = None
        self._last_edge_structure_hash = None
        self._edge_cache_hit_count = 0
        self._edge_cache_miss_count = 0
        # Note: Pre-allocated array is kept for continued reuse

    def close(self):
        """Clean up resources."""
        # Clear all caches
        self._kmeans_cache.clear()
        self._edge_index_cache = None
        
        if hasattr(self.topology, 'close'):
            self.topology.close()



