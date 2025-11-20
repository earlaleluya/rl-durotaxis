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
    
    The environment features a streamlined 3-component reward system with Potential-Based Reward Shaping (PBRS):
    Delete Reward (Priority 1), Distance Reward (Priority 2), and Termination Reward (Priority 3). Each component
    can optionally include PBRS terms for smoother learning signals. The reward system is designed to balance
    proper node deletion, centroid progression toward the goal, and sparse terminal success/failure signals.
    
    Key Features
    ------------
    - **Dynamic Graph Topology**: Real-time node spawn/delete operations with persistent node tracking
    - **Substrate Gradients**: Configurable intensity fields (linear, exponential, custom) for durotaxis simulation
    - **3-Component Reward System**: Delete > Distance > Termination with optional PBRS for each component
    - **Potential-Based Reward Shaping (PBRS)**: Optional per-component shaping terms for smoother learning
    - **Fast TopK Selection**: O(N log K) node selection for fixed-size observations when graphs exceed max_critical_nodes
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
    
    reward_weights : dict, default={'delete_weight': 1.0, 'distance_weight': 1.0, 'termination_weight': 1.0}
        Environment-level weights for reward component composition:
        
        - 'delete_weight': Weight for delete reward component (Priority 1)
        - 'distance_weight': Weight for distance reward component (Priority 2)
        - 'termination_weight': Weight for termination reward component (Priority 3)
        
        Total reward = delete_weight * delete + distance_weight * distance + termination_weight * termination
    
    delete_reward : dict, default={'proper_deletion': 1.0, 'persistence_penalty': 1.0, 'improper_deletion_penalty': 1.0}
        Delete reward configuration (Priority 1: Proper deletion compliance):
        
        - 'proper_deletion': Reward for proper node deletion (scaled by 1/num_nodes for [-1, 1] range)
        - 'persistence_penalty': Penalty for keeping nodes that should be deleted
        - 'improper_deletion_penalty': Penalty for deleting nodes that should be kept
        
        Optional PBRS shaping terms:
        - 'pbrs.enabled': Enable potential-based reward shaping
        - 'pbrs.shaping_coeff': Scaling factor for shaping term (default: 0.1)
        - 'pbrs.phi_weight_pending_marked': Weight for pending/marked nodes in potential
        - 'pbrs.phi_weight_safe_unmarked': Weight for safe/unmarked nodes in potential
    
    spawn_rewards : dict, default={'spawn_success_reward': 1.0, 'spawn_failure_penalty': 1.0}
        Spawn reward configuration (for spawn action implementation):
        
        - 'spawn_success_reward': Reward for successful durotaxis-based spawning (ΔI ≥ delta_intensity)
        - 'spawn_failure_penalty': Penalty for spawning without sufficient intensity gradient
        
        Optional PBRS shaping terms:
        - 'pbrs.enabled': Enable potential-based reward shaping
        - 'pbrs.shaping_coeff': Scaling factor for shaping term (default: 0.05)
        - 'pbrs.phi_weight_spawnable': Weight for spawnable nodes in potential
    
    distance_mode : dict, default={'use_delta_distance': True, 'distance_reward_scale': 5.0}
        Distance reward configuration (Priority 2: Centroid movement toward goal):
        
        - 'use_delta_distance': Use delta-distance (change in centroid position)
        - 'distance_reward_scale': Scaling factor for distance reward
        - 'substrate_aware_scaling': Use substrate gradient for normalization
        - 'use_tanh_normalization': Use tanh (True) or softsign (False) for bounding
        - 'target_delta_x': Expected good step size for tuning (default: 0.05)
        - 'tanh_scale': atanh(0.9) ≈ 1.4722 for sensitivity tuning
        - 'gradient_cap': Optional cap for exponential substrate gradients
        
        Optional PBRS shaping terms:
        - 'pbrs.enabled': Enable potential-based reward shaping
        - 'pbrs.shaping_coeff': Scaling factor for shaping term (default: 0.05)
        - 'pbrs.phi_distance_scale': Scale for distance-to-goal potential
    
    termination_rewards : dict, default={'success_reward': 1.0, 'critical_nodes_penalty': -1.0, 'no_nodes_penalty': -1.0, 'out_of_bounds_penalty': -1.0, 'leftward_drift_penalty': -1.0, 'timeout_penalty': 0.0}
        Termination reward configuration (Priority 3: Sparse terminal signals):
        
        - 'success_reward': Binary +1 reward for reaching rightmost substrate boundary
        - 'critical_nodes_penalty': Binary -1 penalty for exceeding critical node threshold
        - 'no_nodes_penalty': Binary -1 penalty for losing all nodes (graph collapse)
        - 'out_of_bounds_penalty': Binary -1 penalty for nodes moving outside substrate bounds
        - 'leftward_drift_penalty': Binary -1 penalty for consistent leftward centroid movement
        - 'timeout_penalty': Neutral 0 for reaching maximum time steps (no penalty)
    
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
        Box space for flattened graph embeddings with fast topk selection.
        Shape: [(max_critical_nodes + 1) * encoder_out_dim] - fixed size.
    
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
    **Observation Space**: The environment uses fast topk selection when the number of nodes exceeds max_critical_nodes.
    This efficiently selects representative nodes using O(N log K) torch.topk operation based on saliency scores,
    ensuring fixed-size observations for neural network compatibility.
    
    **Reward System**: The 3-component reward system with optional PBRS shaping:
    
    1. **Delete Reward (Priority 1)**: Proper deletion compliance
       - Scaled to [-1, 1] range by dividing by num_nodes
       - Optional PBRS: Potential based on pending/marked vs. safe/unmarked nodes
       - Encourages deleting marked nodes, keeping unmarked nodes
    
    2. **Distance Reward (Priority 2)**: Centroid movement toward goal
       - Delta-distance with substrate-aware normalization
       - Bounded to [-1, 1] range using tanh or softsign
       - Optional PBRS: Potential based on distance-to-goal
       - Encourages rightward centroid movement
    
    3. **Termination Reward (Priority 3)**: Sparse terminal signals
       - Binary rewards: +1 for success, -1 for failures, 0 for timeout
       - Applied only at episode termination
       - Provides clear success/failure signals without shaping during episode
    
    **PBRS (Potential-Based Reward Shaping)**: Each component can optionally use PBRS for smoother learning:
    - Maintains policy invariance (doesn't change optimal policy)
    - Provides denser learning signals while preserving optimality
    - Configurable shaping coefficients per component (delete: 0.1, spawn: 0.05, distance: 0.05)
    
    **Termination Logic**: Episodes can terminate due to:
    
    - Success: Any node reaches rightmost substrate boundary
    - Failure: Node count exceeds critical threshold, all nodes lost, or persistent leftward drift
    - Timeout: Maximum steps reached without success/failure
    
    **Model Organization**: Each environment instance automatically creates run directories (run0001, run0002, etc.)
    for systematic experiment tracking. Models, metadata, and episode information are saved separately per run.
    
    Examples
    --------
    Basic environment setup via config.yaml:
    
    >>> from config_loader import ConfigLoader
    >>> config_loader = ConfigLoader('config.yaml')
    >>> env = DurotaxisEnv(config_loader.config.get('environment', {}))
    >>> obs, info = env.reset()
    >>> obs, reward_dict, terminated, truncated, info = env.step(actions)
    
    Custom reward weights (environment-level composition):
    
    >>> env_config = {
    ...     'reward_weights': {
    ...         'delete_weight': 1.0,      # Priority 1: Deletion compliance
    ...         'distance_weight': 1.0,    # Priority 2: Centroid movement  
    ...         'termination_weight': 1.0  # Priority 3: Terminal signals
    ...     }
    ... }
    >>> env = DurotaxisEnv(env_config)
    
    Enable PBRS for smoother learning:
    
    >>> env_config = {
    ...     'delete_reward': {
    ...         'pbrs': {
    ...             'enabled': True,
    ...             'shaping_coeff': 0.1,
    ...             'phi_weight_pending_marked': 1.0,
    ...             'phi_weight_safe_unmarked': 0.25
    ...         }
    ...     },
    ...     'distance_mode': {
    ...         'pbrs': {
    ...             'enabled': True,
    ...             'shaping_coeff': 0.05
    ...         }
    ...     }
    ... }
    >>> env = DurotaxisEnv(env_config)
    
    Special ablation modes (isolate specific components):
    
    >>> # Delete-only mode
    >>> env_config = {'simple_delete_only_mode': True}
    >>> env = DurotaxisEnv(env_config)
    >>> 
    >>> # Centroid distance-only mode
    >>> env_config = {'centroid_distance_only_mode': True}
    >>> env = DurotaxisEnv(env_config)
    
    Advanced substrate configuration:
    
    >>> env_config = {
    ...     'substrate_size': [800, 600],
    ...     'substrate_params': {'m': 0.01, 'b': 1.0},
    ...     'delta_intensity': 3.0,
    ...     'threshold_critical_nodes': 150
    ... }
    >>> env = DurotaxisEnv(env_config)
    """
    metadata = {"render_fps": 30}

    def __init__(self, 
                 config_path: str | dict | None = None,
                 device=None,
                 **kwargs):
        """
        Initialize DurotaxisEnv with configuration from YAML file
        
        Parameters
        ----------
        config_path : str
            Path to configuration YAML file
        device : torch.device or str, optional
            Device to use for tensors ('cpu', 'cuda', or None for auto-detect)
        **overrides
            Parameter overrides for any configuration values
        """
        super().__init__()
        
        # Set device first
        from device import get_device
        self.device = device if device is not None else get_device()
        
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
        
        # Simple spawn-only mode flag
        self.simple_spawn_only_mode = config.get('simple_spawn_only_mode', False)
        # Legacy: spawn_reward scalar kept for backward compatibility (no longer used)
        self.spawn_reward = float(config.get('spawn_reward', 2.0))
        
        # Include termination rewards flag (for special modes)
        # Default behavior:
        #   - If both modes are False: Always use termination rewards (default True, ignored)
        #   - If either mode is True: User must explicitly set this to True to include termination rewards
        self.include_termination_rewards = config.get('include_termination_rewards', False)

        # Reward composition weights (Priority: Delete > Spawn > Distance)
        rw = config.get('reward_weights', {})
        self._w_delete = float(rw.get('delete_weight', 1.0))
        self._w_distance = float(rw.get('distance_weight', 0.5))
        self._w_termination = float(rw.get('termination_weight', 0.25))

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
        
        # Observation selection parameters (fast topk approach)
        sel_cfg = config.get('observation_selection', {})
        self.obs_sel_method = sel_cfg.get('method', 'topk_x')  # topk_x (default), topk_mixed
        self.obs_sel_w_x = float(sel_cfg.get('w_x', 1.0))  # weight for x-coordinate (rightward bias)
        self.obs_sel_w_intensity = float(sel_cfg.get('w_intensity', 0.0))  # weight for intensity
        self.obs_sel_w_norm = float(sel_cfg.get('w_norm', 0.0))  # weight for embedding norm

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
            'success_reward': 1.0,
            'critical_nodes_penalty': -1.0,
            'no_nodes_penalty': -1.0,
            'out_of_bounds_penalty': -1.0,
            'leftward_drift_penalty': -1.0,
            'timeout_penalty': 0.0
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
        
        # ============================================================================
        # POTENTIAL-BASED REWARD SHAPING (PBRS) PARAMETERS
        # ============================================================================
        # Get gamma from algorithm config for PBRS (preserves optimal policy)
        algo_config = config_loader.config.get('algorithm', {})
        self._pbrs_gamma = float(algo_config.get('gamma', 0.99))
        
        # Delete reward PBRS parameters
        pbrs_delete = self.delete_reward.get('pbrs', {})
        self._pbrs_delete_enabled = pbrs_delete.get('enabled', False)
        self._pbrs_delete_coeff = float(pbrs_delete.get('shaping_coeff', 0.0))
        self._pbrs_delete_w_pending = float(pbrs_delete.get('phi_weight_pending_marked', 1.0))
        self._pbrs_delete_w_safe = float(pbrs_delete.get('phi_weight_safe_unmarked', 0.25))
        
        # Centroid distance PBRS parameters
        pbrs_centroid = self.graph_rewards.get('pbrs_centroid', {})
        self._pbrs_centroid_enabled = pbrs_centroid.get('enabled', False)
        self._pbrs_centroid_coeff = float(pbrs_centroid.get('shaping_coeff', 0.0))
        self._pbrs_centroid_scale = float(pbrs_centroid.get('phi_distance_scale', 1.0))
        
        # Substrate-aware distance reward normalization parameters
        distance_mode_cfg = config.get('distance_mode', {})
        self._dist_substrate_aware = distance_mode_cfg.get('substrate_aware_scaling', True)
        self._dist_use_tanh = distance_mode_cfg.get('use_tanh_normalization', True)
        self._dist_target_dx = float(distance_mode_cfg.get('target_delta_x', 0.05))
        self._dist_tanh_scale = float(distance_mode_cfg.get('tanh_scale', 1.4722))  # atanh(0.9) ≈ 1.4722
        self._dist_gradient_cap = distance_mode_cfg.get('gradient_cap', None)  # Optional cap for exponential gradients
        
        # Spawn reward PBRS parameters
        pbrs_spawn = self.spawn_rewards.get('pbrs', {})
        self._pbrs_spawn_enabled = pbrs_spawn.get('enabled', False)
        self._pbrs_spawn_coeff = float(pbrs_spawn.get('shaping_coeff', 0.0))
        self._pbrs_spawn_w_spawnable = float(pbrs_spawn.get('phi_weight_spawnable', 1.0))

        self.current_step = 0
        self.current_episode = 0
        
        # Milestone tracking (reset each episode)
        self._milestones_reached = set()
        
        # Centroid tracking for fail termination
        self.centroid_history = []  # Store centroid x-coordinates
        self.consecutive_left_moves = 0  # Count consecutive leftward moves
        
        # Encoder observation optimization
        self._max_obs_size = (self.max_critical_nodes + 1) * self.encoder_out_dim
        self._pre_allocated_obs = np.zeros(self._max_obs_size, dtype=np.float32)  # Reusable zero array for fixed-size observations
        self._edge_index_cache = None  # Cache for edge_index tensor conversion
        self._last_edge_structure_hash = None  # Hash of edge structure for cache invalidation
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
        
        # Create GraphInputEncoder for observations and move to device
        self.observation_encoder = GraphInputEncoder(
            hidden_dim=self.encoder_hidden_dim,
            out_dim=self.encoder_out_dim,
            num_layers=self.encoder_num_layers
        ).to(self.device)

        # Store encoder output dimension for observation processing

    def _setup_environment(self):
        """Initialize substrate, topology, state extractor, and policy components."""
        # Create substrate
        self.substrate = Substrate(self.substrate_size)
        self.substrate.create(self.substrate_type, **self.substrate_params)
        
        # Create topology with device
        self.topology = Topology(substrate=self.substrate, flush_delay=self.flush_delay, device=self.device)
        
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
        Returns edge_index_tensor.
        """
        structure_hash = self._get_edge_structure_hash(edge_index)
        
        if (self._last_edge_structure_hash == structure_hash and 
            self._edge_index_cache is not None):
            return self._edge_index_cache
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
            
            return edge_index_tensor

    def _get_encoder_observation(self, state):
        """
        Get observation from GraphInputEncoder output with fast fixed-size selection.
        Returns a fixed (max_critical_nodes+1) * encoder_out_dim flattened vector.
        
        Args:
            state: State dictionary from state_extractor.get_state_features()
            
        Returns:
            np.ndarray: Fixed-size observation vector from encoder output
            
        Note:
            - Uses fast topk selection (O(N log K)) instead of slow semantic pooling
            - When num_nodes > max_critical_nodes: selects top-K nodes by saliency score
            - When num_nodes <= max_critical_nodes: pads with zeros to fixed size
            - Works with both SEM enabled and disabled
            - Device-agnostic (works on CPU/GPU)
        """
        try:
            # Extract components from state
            node_features = state['node_features']  # [N, F] torch tensor
            graph_features = state['graph_features']  # [G] torch tensor
            edge_features = state['edge_attr']
            edge_index = state['edge_index']
            
            # Get cached edge_index tensor (avoids repeated conversions)
            edge_index_tensor = self._get_cached_edge_index(edge_index)
            
            # Handle empty graphs - return fixed-size zero observation
            if node_features.shape[0] == 0:
                self._pre_allocated_obs.fill(0.0)
                return self._pre_allocated_obs.copy()
            
            # Get encoder output (SEM is applied internally if enabled)
            encoder_out = self.observation_encoder(
                graph_features=graph_features,
                node_features=node_features,
                edge_features=edge_features,
                edge_index=edge_index_tensor
            )  # Shape: [N+1, D] where row 0 is graph token, rows 1:N+1 are node embeddings
            
            # Separate graph token from node embeddings
            graph_token = encoder_out[0:1]  # [1, D]
            node_embeddings = encoder_out[1:]  # [N, D]
            N, D = node_embeddings.shape
            K = int(self.max_critical_nodes)
            
            # Fast path: N <= K nodes → pad if needed and return fixed size
            if N <= K:
                if N < K:
                    # Pad with zeros to reach K nodes
                    pad_rows = K - N
                    pad = node_embeddings.new_zeros((pad_rows, D))
                    node_block = torch.cat([node_embeddings, pad], dim=0)  # [K, D]
                else:
                    node_block = node_embeddings  # [K, D]
                
                # Combine graph token with node block
                fixed = torch.cat([graph_token, node_block], dim=0)  # [K+1, D]
                flat = fixed.reshape(-1).detach().cpu().numpy()
                
                # Use pre-allocated buffer
                expected_size = (K + 1) * D
                if self._pre_allocated_obs.shape[0] != expected_size:
                    self._pre_allocated_obs = np.zeros(expected_size, dtype=np.float32)
                    self._max_obs_size = expected_size
                
                self._pre_allocated_obs[:len(flat)] = flat
                return self._pre_allocated_obs.copy()
            
            # N > K: select top-K representative nodes using fast topk
            # Build saliency score: s = w_x * x + w_intensity * I + w_norm * ||h||
            with torch.no_grad():
                device = node_embeddings.device
                scores = torch.zeros((N,), dtype=node_embeddings.dtype, device=device)
                
                # Component 1: x-coordinate (rightward bias) - feature 0
                if node_features.shape[1] >= 1 and self.obs_sel_w_x != 0.0:
                    x = node_features[:, 0]
                    # Normalize to [0, 1] for numerical stability
                    x_min, x_max = x.min(), x.max()
                    if x_max > x_min:
                        x_norm = (x - x_min) / (x_max - x_min + 1e-6)
                        scores.add_(self.obs_sel_w_x * x_norm)
                
                # Component 2: intensity - feature 2
                if node_features.shape[1] >= 3 and self.obs_sel_w_intensity != 0.0:
                    inten = node_features[:, 2]
                    # Z-score normalization
                    inten_mean, inten_std = inten.mean(), inten.std(unbiased=False)
                    if inten_std > 1e-6:
                        inten_norm = (inten - inten_mean) / (inten_std + 1e-6)
                        scores.add_(self.obs_sel_w_intensity * inten_norm)
                
                # Component 3: embedding norm (representational importance)
                if self.obs_sel_w_norm != 0.0:
                    h_norm = node_embeddings.norm(p=2, dim=1)  # [N]
                    # Z-score normalization
                    h_mean, h_std = h_norm.mean(), h_norm.std(unbiased=False)
                    if h_std > 1e-6:
                        h_norm_z = (h_norm - h_mean) / (h_std + 1e-6)
                        scores.add_(self.obs_sel_w_norm * h_norm_z)
                
                # Fallback: if all weights are zero, use rightmost x-coordinate
                if torch.all(scores == 0):
                    if node_features.shape[1] >= 1:
                        scores = node_features[:, 0]  # x-coordinate
                    else:
                        scores = node_embeddings.norm(p=2, dim=1)  # embedding norm
                
                # Select top-K nodes by score (O(N log K)) - device-agnostic
                topk_result = torch.topk(scores, k=K, largest=True, sorted=False)
                sel_idx = topk_result.indices  # [K]
                
                # Optional: sort selected indices by x-coordinate for positional consistency
                if node_features.shape[1] >= 1:
                    x_sel = node_features[sel_idx, 0]
                    order = torch.argsort(x_sel)  # left → right ordering
                    sel_idx = sel_idx[order]
                
                # Extract selected node embeddings
                selected = node_embeddings[sel_idx]  # [K, D]
            
            # Combine graph token with selected nodes
            fixed = torch.cat([graph_token, selected], dim=0)  # [K+1, D]
            flat = fixed.reshape(-1).detach().cpu().numpy()
            
            # Use pre-allocated buffer (constant size)
            expected_size = (K + 1) * D
            if self._pre_allocated_obs.shape[0] != expected_size:
                self._pre_allocated_obs = np.zeros(expected_size, dtype=np.float32)
                self._max_obs_size = expected_size
            
            self._pre_allocated_obs[:expected_size] = flat[:expected_size]
            return self._pre_allocated_obs[:expected_size].copy()
            
        except Exception as e:
            print(f"Error getting encoder observation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to zero observation with fixed size
            expected_size = (self.max_critical_nodes + 1) * self.encoder_out_dim
            if self._pre_allocated_obs.shape[0] != expected_size:
                self._pre_allocated_obs = np.zeros(expected_size, dtype=np.float32)
            self._pre_allocated_obs.fill(0.0)
            return self._pre_allocated_obs.copy()

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
            - 'total_reward': float - Total scalar reward (weighted sum of components)
            - 'delete_reward': float - Deletion compliance rewards (scaled to [-1, 1])
            - 'distance_reward': float - Centroid movement toward goal
            - 'num_nodes': int - Number of nodes in current state
            - 'empty_graph_recovery_penalty': float - Penalty if empty graph recovery occurred
            - 'termination_reward': float - Termination reward (only if episode terminated)
        terminated : bool
            True if episode terminated (success/failure conditions met)
        truncated : bool
            True if episode truncated (max steps reached)
        info : dict
            Additional information including reward breakdown and statistics
            
        Notes
        -----
        The reward system includes 3 components:
        - Delete reward (deletion compliance, scaled to [-1, 1] range)
        - Distance reward (centroid movement toward goal)
        - Total reward (weighted combination of delete + distance)
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
        # We store persistent_ids, node_features (for intensity), and basic stats
        topology_snapshot = {
            'persistent_id': prev_state['persistent_id'].clone() if 'persistent_id' in prev_state else torch.empty(0, device=self.device),
            'node_features': prev_state['node_features'].clone() if 'node_features' in prev_state else torch.empty(0, 3, device=self.device),
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
        # Store action parameters for logging
        action_params = {'delete_ratio': 0.0, 'gamma': 0.0, 'alpha': 0.0, 'noise': 0.0, 'theta': 0.0}
        
        if self.policy_agent is not None and prev_num_nodes > 0:
            try:
                executed_actions = self.policy_agent.act_with_policy(
                    deterministic=False
                )
                # Try to extract action parameters if available
                if hasattr(self.policy_agent, 'last_continuous_actions') and self.policy_agent.last_continuous_actions is not None:
                    last_actions = self.policy_agent.last_continuous_actions
                    if len(last_actions) >= 5:
                        # Extract single global action parameters (shape: [5])
                        action_params['delete_ratio'] = float(last_actions[0].item())
                        action_params['gamma'] = float(last_actions[1].item())
                        action_params['alpha'] = float(last_actions[2].item())
                        action_params['noise'] = float(last_actions[3].item())
                        action_params['theta'] = float(last_actions[4].item())
                    else:
                        print(f"⚠️  DEBUG: last_continuous_actions has wrong shape: {last_actions.shape if hasattr(last_actions, 'shape') else len(last_actions)}")
                else:
                    if not hasattr(self.policy_agent, 'last_continuous_actions'):
                        print(f"⚠️  DEBUG: policy_agent has no attribute 'last_continuous_actions'")
                    elif self.policy_agent.last_continuous_actions is None:
                        print(f"⚠️  DEBUG: last_continuous_actions is None (prev_num_nodes={prev_num_nodes})")
            except Exception as e:
                print(f"⚠️  Policy execution failed: {e}")
                import traceback
                traceback.print_exc()
                executed_actions = {}
        else:
            # Fallback to random actions if policy fails or no nodes
            if prev_num_nodes > 0:
                executed_actions = self.topology.act()
            else:
                executed_actions = {}
        
        # Get initial new state after actions (without updated marking)
        new_state = self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )

        # Update and inject advanced node features (age, stagnation)
        self._update_and_inject_node_features(new_state)

        # Heuristically mark nodes for deletion based on lab rules
        # This modifies topology's to_delete flags based on dequeued topology comparison
        self._heuristically_mark_nodes_for_deletion()
        
        # Re-capture new_state to include updated to_delete flags from marking
        # This ensures reward calculation sees the marks that were just applied
        new_state = self.state_extractor.get_state_features(
            include_substrate=True,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )

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
        
        # Check termination conditions and update termination reward
        terminated, termination_reward_value = self._check_terminated(new_state)
        
        # Update termination reward in reward_components (always included in composition)
        if terminated:
            reward_components['termination_reward'] = termination_reward_value
            # Add weighted termination reward to total_reward
            reward_components['total_reward'] += self._w_termination * termination_reward_value
        
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
        
        # Extract reward components
        delete_r = reward_components.get('delete_reward', 0.0)
        distance_r = reward_components.get('distance_reward', 0.0)
        termination_r = reward_components.get('termination_reward', 0.0)
        
        print(
            f"📊 Ep{self.current_episode:2d} Step{self.current_step:3d}: "
            f"N={new_state['num_nodes']:2d} E={new_state['num_edges']:2d} | "
            f"R={scalar_reward:+6.3f} (D:{delete_r:+4.3f} Dist:{distance_r:+4.3f} Term:{termination_r:+4.3f}) | "
            f"C={centroid_x:5.1f}{centroid_direction}{boundary_warning}{recovery_flag} | "
            f"Actions: dr={action_params['delete_ratio']:.3f} γ={action_params['gamma']:.2f} "
            f"α={action_params['alpha']:.2f} n={action_params['noise']:.3f} θ={action_params['theta']:.3f} | "
            f"Termi={terminated} Trunc={truncated}"
        )
        
        # Auto-render after each step to ensure visualization is always updated
        # This ensures visualization works consistently
        if self.enable_visualization:
            self.render()
        
        return observation, reward_components, terminated, truncated, info

    def _compute_substrate_gradient(self, x_position):
        """
        Compute the local gradient magnitude |∂(substrate)/∂x| at position x.
        
        For linear substrate f(x) = m*x + b:
            g(x) = |m| (constant)
        
        For exponential substrate f(x) = b * exp(m*x):
            g(x) = |b * m * exp(m*x)|
        
        Args:
            x_position: float, x-coordinate where gradient is computed
            
        Returns:
            float: gradient magnitude at x_position
        """
        try:
            # Get substrate parameters (prefer substrate's stored params for random mode)
            if hasattr(self.substrate, 'current_m') and self.substrate.current_m is not None:
                m = self.substrate.current_m
                b = self.substrate.current_b
            else:
                m = self.substrate_params.get('m', 0.01)
                b = self.substrate_params.get('b', 1.0)
            
            # Get current substrate type (handle trainer's random substrate mode)
            substrate_type = self.substrate_type
            if hasattr(self.substrate, 'current_type') and self.substrate.current_type is not None:
                substrate_type = self.substrate.current_type
            
            # Compute gradient based on substrate type
            if substrate_type == 'linear':
                # Linear: f(x) = m*x + b → f'(x) = m
                gradient = abs(m)
            
            elif substrate_type == 'exponential':
                # Exponential: f(x) = b * exp(m*x) → f'(x) = b * m * exp(m*x)
                import numpy as np
                gradient = abs(b * m * np.exp(m * x_position))
                
                # Optional: cap gradient for exponential substrates to prevent explosion
                if self._dist_gradient_cap is not None:
                    gradient = min(gradient, self._dist_gradient_cap)
            
            else:
                # Fallback for unknown types
                gradient = 1.0
            
            # Ensure gradient is positive and non-zero
            gradient = max(gradient, 1e-6)
            
            return gradient
            
        except Exception as e:
            # Fallback to neutral gradient if computation fails
            print(f"⚠️ Substrate gradient computation failed: {e}")
            return 1.0

    def _calculate_reward(self, prev_state, new_state, actions):
        """
        Calculate reward based on composition: Delete > Distance > Termination.
        
        Core reward components:
        - Delete reward (Priority 1: proper deletion compliance)
        - Distance reward (Priority 2: centroid movement toward goal)
        - Termination reward (Priority 3: sparse terminal signals, always 0 during episode)
        
        NOTE: Spawn reward calculation function (_calculate_spawn_reward) is preserved 
        for potential future use but is NOT included in reward composition.
        
        Special modes (delete-only, centroid-only) still work.
        Priority: Delete > Distance > Termination
        """
        # Initialize reward components
        num_nodes = new_state['num_nodes']
        
        # === CALCULATE CORE REWARD COMPONENTS ===
        
        # PRIORITY 1: Delete reward (proper deletion compliance)
        delete_reward = self._calculate_delete_reward(prev_state, new_state, actions)
        
        # NOTE: Spawn reward calculation function preserved but NOT used in composition
        # spawn_reward = self._calculate_spawn_reward(prev_state, new_state, actions)
        
        # PRIORITY 2: Distance reward (centroid movement toward goal)
        distance_reward = self._calculate_distance_reward(prev_state, new_state)
        
        # PRIORITY 3: Termination reward (sparse terminal signal, always 0 during episode)
        # Will be set to -1 or +1 only at termination in step() method
        termination_reward = 0.0
        
        # === REWARD COMPOSITION ===
        # Two paths: (1) Special modes (selective components), (2) Default (all three components)
        
        # Check which special modes are enabled
        has_delete_mode = self.simple_delete_only_mode
        has_centroid_mode = self.centroid_distance_only_mode
        has_spawn_mode = self.simple_spawn_only_mode
        
        # Count how many special modes are enabled
        num_special_modes = sum([has_delete_mode, has_centroid_mode, has_spawn_mode])
        
        if num_special_modes > 0:
            # === SPECIAL MODES: Use ONLY enabled components (no weighting in special modes) ===
            mode_reward = 0.0
            
            # Priority 1: Delete (if enabled)
            if has_delete_mode:
                mode_reward += delete_reward
            
            # Priority 2: Distance (if enabled)
            if has_centroid_mode:
                mode_reward += distance_reward
            
            # NOTE: spawn_mode preserved for backward compatibility but spawn_reward NOT used
            
            # Set total reward from special mode composition
            total_reward = mode_reward
            
            # Zero out components NOT in special modes (for logging clarity)
            if not has_delete_mode:
                delete_reward = 0.0
            if not has_centroid_mode:
                distance_reward = 0.0
        else:
            # === DEFAULT MODE: Weighted composition (Delete > Distance > Termination) ===
            # Apply environment-level weights to make priority explicit in task reward
            total_reward = (
                self._w_delete * float(delete_reward) +
                self._w_distance * float(distance_reward) +
                self._w_termination * float(termination_reward)
            )
        
        # === BUILD REWARD BREAKDOWN ===
        # Create reward breakdown with all components in refactored system
        reward_breakdown = {
            'total_reward': total_reward,
            'delete_reward': delete_reward,
            'distance_reward': distance_reward,
            'termination_reward': termination_reward,  # Always included (0 during episode, ±1 at termination)
            'num_nodes': num_nodes,
            'empty_graph_recovery_penalty': 0.0  # Set later if recovery occurred
        }
        
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

    def _calculate_spawn_reward(self, prev_state, new_state, actions):
        """
        Calculate reward for durotaxis-based spawning.
        
        Unified behavior (same magnitude for both modes):
        - Reward += spawn_success_reward if ΔI >= delta_intensity
        - Penalty -= spawn_failure_penalty if ΔI < delta_intensity
        - Uses same values from spawn_rewards config regardless of mode
        - Includes PBRS shaping term when enabled
        
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
        
        # DEBUG: Count new nodes and check flag values
        num_new_nodes = 0
        all_new_node_flags = []
        
        # Use new_node flag to identify newly spawned nodes
        # The new_node flag is the last feature in the node feature vector
        for node_idx in range(new_num_nodes):
            if node_idx < len(new_node_features):
                node_feature_vector = new_node_features[node_idx]
                
                # Check if this is a newly spawned node (new_node flag = 1.0)
                # Node features: [x, y, intensity, in_deg, out_deg, centrality, centroid_dist, boundary, new_node, age, stagnation]
                # new_node is at index 8 (0-indexed), NOT the last feature
                if len(node_feature_vector) > 8:
                    new_node_flag = node_feature_vector[8].item()  # Index 8 is new_node flag
                    all_new_node_flags.append(new_node_flag)
                    
                    if new_node_flag == 1.0:  # This is a newly spawned node
                        num_new_nodes += 1
                        # Get substrate intensity (3rd feature, index 2)
                        if len(node_feature_vector) > 2:
                            new_node_intensity = node_feature_vector[2].item()
                            new_node_pos = node_feature_vector[:2]  # x, y coordinates
                            
                            # REFACTORED: Boundary checks removed in simplified system
                            # Focus purely on intensity-based spawning
                            
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
                                
                                # Use unified reward values (same magnitude for both modes)
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
        
        # DEBUG: Print spawn detection and reward
        if not hasattr(self, '_debug_spawn_count'):
            self._debug_spawn_count = 0
            self._debug_no_spawn_count = 0
        
        # Add PBRS shaping term for simple_spawn_only_mode
        if self.simple_spawn_only_mode and self._pbrs_spawn_enabled and self._pbrs_spawn_coeff != 0.0:
            phi_prev = self._phi_spawn_potential(prev_state)
            phi_new = self._phi_spawn_potential(new_state)
            pbrs_shaping = self._pbrs_gamma * phi_new - phi_prev
            spawn_reward += self._pbrs_spawn_coeff * pbrs_shaping
        
        return spawn_reward
    
    # ============================================================================
    # POTENTIAL-BASED REWARD SHAPING (PBRS) - HELPER FUNCTIONS
    # ============================================================================
    
    def _phi_delete_potential(self, state):
        """
        Compute potential function Phi(s) for delete reward shaping.
        
        Uses only current state (Markov property):
        - pending_marked(s): count of nodes with to_delete=1 that still exist
        - safe_unmarked(s): count of nodes with to_delete=0 that still exist
        
        Phi(s) = -w_pending * pending_marked(s) + w_safe * safe_unmarked(s)
        
        Deleting a marked node reduces pending_marked → Phi increases → positive shaping
        Deleting an unmarked node reduces safe_unmarked → Phi decreases → negative shaping
        
        Args:
            state: State dict containing 'to_delete' flags and node existence info
            
        Returns:
            float: Potential value Phi(s)
        """
        try:
            if state['num_nodes'] == 0:
                return 0.0
            
            # Get to_delete flags from state
            to_delete = state.get('to_delete', None)
            if to_delete is None:
                return 0.0
            
            # Convert to numpy for device-agnostic computation
            if hasattr(to_delete, 'detach'):
                to_delete_np = to_delete.detach().cpu().numpy()
            else:
                to_delete_np = np.asarray(to_delete)
            
            to_delete_np = to_delete_np.astype(np.float32)
            
            # All nodes in state exist (by definition)
            pending_marked = float((to_delete_np > 0.5).sum())
            safe_unmarked = float((to_delete_np <= 0.5).sum())
            
            phi = -self._pbrs_delete_w_pending * pending_marked + self._pbrs_delete_w_safe * safe_unmarked
            return float(phi)
            
        except Exception as e:
            # Fail gracefully
            return 0.0
    
    def _phi_centroid_distance_potential(self, state):
        """
        Compute potential function Phi(s) for centroid distance shaping.
        
        Uses only current state (Markov property):
        - centroid_x(s): Current x-coordinate of graph centroid
        - goal_x: Target x-coordinate (rightmost substrate boundary)
        
        Phi(s) = -scale * (goal_x - centroid_x(s))
        
        Moving right reduces distance → Phi increases → positive shaping
        Moving left increases distance → Phi decreases → negative shaping
        
        Args:
            state: State dict containing centroid_x and goal_x
            
        Returns:
            float: Potential value Phi(s)
        """
        try:
            if state['num_nodes'] == 0:
                return 0.0
            
            # Prefer explicit field, else derive from graph_features
            if 'centroid_x' in state and state['centroid_x'] is not None:
                centroid_x = float(state['centroid_x'])
            else:
                gf = state.get('graph_features', None)
                if isinstance(gf, torch.Tensor) and gf.numel() >= 4:
                    centroid_x = float(gf[3].item())
                elif isinstance(gf, (list, tuple)) and len(gf) >= 4:
                    centroid_x = float(gf[3])
                else:
                    return 0.0
            
            goal_x = float(getattr(self, 'goal_x', self.substrate.width - 1))
            
            # Distance to goal (negative potential, so moving closer increases Phi)
            distance_to_goal = goal_x - centroid_x
            phi = -self._pbrs_centroid_scale * distance_to_goal
            
            return float(phi)
            
        except Exception:
            # Fail gracefully
            return 0.0
    
    def _phi_spawn_potential(self, state):
        """
        Compute potential function Phi(s) for spawn reward shaping.
        
        Uses only current state (Markov property):
        - spawnable_nodes(s): count of nodes on high-intensity substrate
        - high-intensity: substrate_intensity >= spawn intensity threshold
        
        Phi(s) = w_spawnable * spawnable_nodes(s)
        
        Spawning on high-intensity substrate increases spawnable count → Phi increases → positive shaping
        Spawning on low-intensity substrate doesn't increase count → no Phi change → no shaping bonus
        
        Args:
            state: State dict containing node intensities
            
        Returns:
            float: Potential value Phi(s)
        """
        try:
            if state['num_nodes'] == 0:
                return 0.0
            
            # Get intensities from state
            intensities = state.get('intensity', None)
            if intensities is None:
                return 0.0
            
            # Convert to numpy for device-agnostic computation
            if hasattr(intensities, 'detach'):
                intensities_np = intensities.detach().cpu().numpy()
            else:
                intensities_np = np.asarray(intensities)
            
            intensities_np = intensities_np.astype(np.float32)
            
            # Count nodes with high substrate intensity (spawnable candidates)
            # Use self.delta_intensity as threshold (same as spawn success criterion)
            spawnable_count = float((intensities_np >= self.delta_intensity).sum())
            
            phi = self._pbrs_spawn_w_spawnable * spawnable_count
            return float(phi)
            
        except Exception as e:
            # Fail gracefully
            return 0.0
    
    # ============================================================================
    # REWARD CALCULATION FUNCTIONS
    # ============================================================================
    
    def _calculate_distance_reward(self, prev_state, new_state):
        """
        Calculate distance reward based on centroid movement toward goal.
        
        Uses substrate-aware tanh normalization with PBRS when enabled.
        Supports both delta-based (potential-based) and static distance calculation.
        
        Args:
            prev_state: Previous state dict containing topology
            new_state: Current state dict containing topology
            
        Returns:
            float: Distance reward, typically in range (-1, 1) with substrate-aware mode
        """
        num_nodes = new_state['num_nodes']
        centroid_x = 0.0
        distance_reward = 0.0
        
        # Extract centroid x position from new state
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
            # Delta distance: Δx = (cx_t - cx_{t-1})
            delta_x = centroid_x - self._prev_centroid_x
            
            if self._dist_substrate_aware:
                # Substrate-aware scaling: compute local gradient at midpoint
                x_loc = (self._prev_centroid_x + centroid_x) / 2.0
                g = self._compute_substrate_gradient(x_loc)
                
                # Convert raw distance to substrate-aware signal: s = g(x) * Δx
                s = g * delta_x
                
                # Optional: add proper PBRS shaping BEFORE normalization (preserves optimal policy)
                if self._pbrs_centroid_enabled and self._pbrs_centroid_coeff != 0.0:
                    phi_prev = self._phi_centroid_distance_potential(prev_state)
                    phi_new = self._phi_centroid_distance_potential(new_state)
                    pbrs_term = self._pbrs_centroid_coeff * (self._pbrs_gamma * phi_new - phi_prev)
                    s += pbrs_term
                
                # Compute scale constant c based on expected good step size
                # c = tanh_scale * g * target_dx, where tanh_scale ≈ 1.4722 = atanh(0.9)
                # This ensures tanh(g * target_dx / c) ≈ 0.9
                c = self._dist_tanh_scale * g * self._dist_target_dx
                
                if self._dist_use_tanh:
                    # Apply tanh squashing: r = tanh(s / c) ∈ (-1, 1)
                    distance_reward = np.tanh(s / c) if c > 0 else 0.0
                else:
                    # Softsign alternative: r = s / (1 + |s|) ∈ (-1, 1)
                    distance_reward = s / (1.0 + abs(s))
            else:
                # Original scaling (no substrate awareness)
                distance_reward = self.dm_dist_scale * (delta_x / self.goal_x)
                
                # Optional: add proper PBRS shaping on top (preserves optimal policy)
                # Note: When substrate_aware disabled, PBRS is added after (no normalization)
                if self._pbrs_centroid_enabled and self._pbrs_centroid_coeff != 0.0:
                    phi_prev = self._phi_centroid_distance_potential(prev_state)
                    phi_new = self._phi_centroid_distance_potential(new_state)
                    distance_reward += self._pbrs_centroid_coeff * (self._pbrs_gamma * phi_new - phi_prev)
        else:
            # Fallback: static distance penalty
            if self.goal_x > 0:
                distance_reward = -(self.goal_x - centroid_x) / self.goal_x
            else:
                distance_reward = 0.0
        
        # Update previous centroid for next step
        self._prev_centroid_x = centroid_x
        
        return distance_reward

    def _calculate_delete_reward(self, prev_state, new_state, actions):
        """
        Calculate reward/penalty based on deletion compliance with to_delete flags.
        Scaled to [-1, 1] range by dividing by num_nodes.
        
        Logic (4 Rules): 
        - RULE 1: If a node from previous topology was marked to_delete=1 AND no longer exists: +delete_reward
        - RULE 2: If a node from previous topology was marked to_delete=1 BUT still exists: -delete_reward (persistence penalty)
        - RULE 3: If a node from previous topology was NOT marked (to_delete=0) BUT was deleted: -delete_reward (improper deletion)
        - RULE 4: If a node from previous topology was NOT marked (to_delete=0) AND still exists: 0 (neutral - no reward or penalty)
        
        Rationale for RULE 4 = 0:
        - Prevents exploitation where agent keeps unmarked nodes to farm per-step rewards
        - Maintains [-1, 1] range: worst case all nodes violate rules (-1), best case all comply (+1)
        - Focuses reward signal on active decisions (mark+delete or mark+persist) rather than passive persistence
        
        Scaling: Total raw reward is divided by prev_num_nodes to ensure output in [-1, 1]
        
        Tagging Strategy:
        - At step t, if node's intensity < avg(all intensities), tag as to_delete=1 at step t+delta_time
        - delta_time is configurable (default: 3)
        - Only triggered when num_nodes > max_critical_nodes (signals need for deletion)
        
        Args:
            prev_state: Previous state dict containing topology
            new_state: Current state dict containing topology  
            actions: Actions taken this step
            
        Returns:
            float: Delete reward in range [-1, 1] (positive for proper behavior, negative for violations)
        """
        # Need previous state with persistent_id and to_delete data  
        if 'persistent_id' not in prev_state or prev_state['persistent_id'] is None:
            return 0.0
        
        if 'to_delete' not in prev_state or prev_state['to_delete'] is None:
            return 0.0
            
        prev_num_nodes = prev_state['num_nodes']
        if prev_num_nodes == 0:
            return 0.0
        
        # Use the cloned data from prev_state (captured at the time of state extraction)
        prev_to_delete_flags = prev_state['to_delete']
        prev_persistent_ids = prev_state['persistent_id']
        
        # Debug: Count marked nodes in prev_state
        num_marked_prev = (prev_to_delete_flags > 0.5).sum().item()
        
        # Get current persistent IDs
        if new_state['num_nodes'] > 0 and 'persistent_id' in new_state:
            current_persistent_ids = set(new_state['persistent_id'].cpu().tolist())
        else:
            current_persistent_ids = set()
        
        # Debug: Count marked nodes in new_state  
        if new_state['num_nodes'] > 0 and 'to_delete' in new_state:
            num_marked_new = (new_state['to_delete'] > 0.5).sum().item()
        else:
            num_marked_new = 0
        
        # Debug logging every 50 steps to verify marking
        if self.current_step % 50 == 0 and (num_marked_prev > 0 or num_marked_new > 0):
            print(f"🔍 Delete Reward Debug (Step {self.current_step}): prev_marked={num_marked_prev}/{prev_num_nodes}, new_marked={num_marked_new}/{new_state['num_nodes']}")
        
        # Accumulate raw reward (will be scaled by num_nodes at the end)
        raw_delete_reward = 0.0
        
        # Check each node from previous state
        for i, to_delete_flag in enumerate(prev_to_delete_flags):
            prev_persistent_id = prev_persistent_ids[i].item()
            node_was_deleted = prev_persistent_id not in current_persistent_ids
            
            if to_delete_flag.item() > 0.5:  # Node was marked for deletion
                if node_was_deleted:
                    # RULE 1: Node was marked for deletion and was actually deleted - reward
                    raw_delete_reward += self.delete_proper_reward
                else:
                    # RULE 2: Node was marked for deletion but still exists - penalty
                    raw_delete_reward -= self.delete_persistence_penalty
            else:  # Node was NOT marked for deletion (to_delete=0)
                if node_was_deleted:
                    # RULE 3: Node was NOT marked but was deleted anyway - penalty
                    raw_delete_reward -= self.delete_improper_penalty
                else:
                    # RULE 4: Node was NOT marked and still exists - NO REWARD
                    # This prevents exploitation where agent keeps unmarked nodes for per-step reward
                    # Setting to 0 maintains [-1, 1] range while removing passive reward farming
                    pass  # No reward or penalty (neutral)
        
        # ============================================================================
        # POTENTIAL-BASED REWARD SHAPING (PBRS) - Preserves Optimal Policy
        # ============================================================================
        # Apply PBRS to raw reward BEFORE scaling
        # PBRS term: F(s,a,s') = gamma*Phi(s') - Phi(s)
        # This biases learning toward compliant deletions without changing optimal policy
        if self._pbrs_delete_enabled and self._pbrs_delete_coeff != 0.0:
            phi_prev = self._phi_delete_potential(prev_state)
            phi_new = self._phi_delete_potential(new_state)
            pbrs_shaping = self._pbrs_gamma * phi_new - phi_prev
            # Add PBRS to raw reward BEFORE scaling
            raw_delete_reward += self._pbrs_delete_coeff * pbrs_shaping
        
        # Scale by num_nodes to get value in approximately [-1, 1] range
        # Base reward range: [-1, 1]
        # PBRS can extend this to approximately [-1.125, 1.125] with default config
        # (shaping_coeff=0.1, w_pending=1.0, w_safe=0.25, gamma=0.99)
        # This small exceedance (±0.125) is intentional to allow PBRS guidance
        delete_reward = raw_delete_reward / prev_num_nodes
        
        return delete_reward

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
    
    def _heuristically_mark_nodes_for_deletion(self, debug=False):
        """
        Mark nodes for deletion based on the two rules from physical experiments.
        
        Rule 1: Only apply deletion logic if num_nodes > max_critical_nodes
        Rule 2: Mark the rightmost nodes whose intensity at t-delta_time was
                below the current average intensity
        
        This implements the domain-specific deletion strategy observed in lab
        experiments, where the organism prunes inefficient parts of its network.
        
        Args:
            debug: If True, print debug information about why marking didn't occur
        """
        # Rule 1: Only apply deletion logic if the number of nodes exceeds the critical threshold
        if self.topology.graph.num_nodes() <= self.max_critical_nodes:
            if debug:
                print(f"   🔍 Mark check: nodes={self.topology.graph.num_nodes()} <= max_critical={self.max_critical_nodes}, skipping")
            return

        # Ensure we have a past topology to compare against
        if self.dequeued_topology is None:
            if debug:
                print(f"   🔍 Mark check: dequeued_topology is None")
            return
        
        # Check if dequeued topology has the required data
        if not isinstance(self.dequeued_topology, dict):
            if debug:
                print(f"   🔍 Mark check: dequeued_topology is not a dict")
            return
        
        if 'persistent_id' not in self.dequeued_topology or 'node_features' not in self.dequeued_topology:
            if debug:
                print(f"   🔍 Mark check: missing keys in dequeued_topology: {list(self.dequeued_topology.keys())}")
            return

        # Rule 2: Identify nodes whose intensity at t-delta_time was less than current average
        
        # Get current graph info
        current_state = self.state_extractor.get_state_features(
            include_substrate=False,
            node_age=self._node_age,
            node_stagnation=self._node_stagnation
        )
        if current_state['num_nodes'] == 0:
            if debug:
                print(f"   🔍 Mark check: current_state has 0 nodes")
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
            if debug:
                print(f"   🔍 Mark check: no candidates (all {len(prev_pids)} prev nodes have intensity >= {current_avg_intensity:.3f})")
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
            if debug:
                print(f"   🔍 Mark check: {len(candidate_pids)} candidates found but NONE exist in current graph (all deleted)")
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
        
        # Debug logging to verify marking is working
        if nodes_marked_count > 0:
            total_marked = self._count_nodes_marked_for_deletion()
            print(f"🏷️  Marked {nodes_marked_count} nodes for deletion (total marked: {total_marked}/{self.topology.graph.num_nodes()}, excess: {excess_nodes})")

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
        # Will be initialized after topology reset below
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
        # Only includes components used in refactored system
        self._reward_components_template = {
            'total_reward': 0.0,
            'delete_reward': 0.0,
            'distance_reward': 0.0,
            'num_nodes': 0,
            'empty_graph_recovery_penalty': 0.0,
            'termination_reward': 0.0
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
        
        # Initialize previous centroid for distance reward calculation
        if state['num_nodes'] > 0:
            try:
                graph_features = state.get('graph_features')
                if graph_features is not None:
                    if isinstance(graph_features, torch.Tensor):
                        self._prev_centroid_x = graph_features[3].item()  # Index 3 is centroid_x
                    else:
                        self._prev_centroid_x = graph_features[3]
            except (IndexError, TypeError):
                # Fallback: calculate centroid from node positions
                node_features = state.get('node_features', [])
                if len(node_features) > 0:
                    x_positions = [node[0].item() if isinstance(node[0], torch.Tensor) else node[0] 
                                 for node in node_features]
                    self._prev_centroid_x = sum(x_positions) / len(x_positions)
        
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
                self.topology.graph.num_nodes(), dtype=torch.float32, device=self.topology.device
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
        # Clear edge index cache
        self._edge_index_cache = None
        
        if hasattr(self.topology, 'close'):
            self.topology.close()



