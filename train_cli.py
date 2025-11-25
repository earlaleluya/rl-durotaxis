#!/usr/bin/env python3
"""
Command-line interface wrapper for train.py with argument parsing

This script provides command-line argument support for the Delete Ratio DurotaxisTrainer.
It parses arguments and passes them as overrides to the trainer.

Delete Ratio Architecture:
- Single global continuous action: [delete_ratio, gamma, alpha, noise, theta]
- No discrete per-node actions
- Two-stage training supported

Usage:
    python train_cli.py --help
    python train_cli.py --pretrained-weights imagenet --sem-enabled
    python train_cli.py --seed 42 --experiment delete_ratio_baseline
"""

import argparse
import sys
import torch
import numpy as np
import random

# Import the training module
from train import DurotaxisTrainer


def parse_args():
    """Parse command-line arguments for training configuration"""
    parser = argparse.ArgumentParser(
        description='Train Durotaxis RL Agent with Delete Ratio Actor-Critic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config (delete ratio architecture)
  python train_cli.py
  
  # Override total episodes and learning rate
  python train_cli.py --total-episodes 2000 --learning-rate 0.0005
  
  # Use ImageNet pretrained weights with SEM enabled
  python train_cli.py --pretrained-weights imagenet --sem-enabled
  
  # Disable SEM and use custom experiment name
  python train_cli.py --no-sem --experiment my_ablation_test
  
  # Run specific seed for reproducibility
  python train_cli.py --seed 42 --experiment delete_ratio_seed42
  
  # SEM ablation configuration
  python train_cli.py --pretrained-weights imagenet --sem-enabled --experiment sem_enabled --seed 1
  
  # Baseline (no SEM)
  python train_cli.py --pretrained-weights imagenet --no-sem --experiment baseline --seed 1
  
  # Simple delete-only mode (Rule 0, 1, 2) with termination rewards
  python train_cli.py --simple-delete-only --include-termination-rewards --experiment delete_only_with_term
  
  # Centroid distance-only mode (pure distance learning)
  python train_cli.py --centroid-distance-only --experiment pure_distance
  
  # Combined mode (distance + delete penalties)
  python train_cli.py --simple-delete-only --centroid-distance-only --experiment combined_mode
  
  # Normal mode with all reward components
  python train_cli.py --no-simple-delete-only --no-centroid-distance-only --experiment normal_mode
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file (default: config.yaml)')
    
    # Experiment tracking
    parser.add_argument('--experiment', type=str, default=None,
                        help='Experiment name for logging and checkpointing')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Training parameters
    parser.add_argument('--total-episodes', type=int, default=None,
                        help='Total number of training episodes')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum steps per episode')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate for optimizer')
    
    # Architecture configuration
    parser.add_argument('--pretrained-weights', type=str, choices=['imagenet', 'random'], default=None,
                        help='Pretrained weights for Actor/Critic ResNet backbone (imagenet or random)')
    parser.add_argument('--sem-enabled', action='store_true', default=None,
                        help='Enable Simplicial Embedding (SEM) in encoder')
    parser.add_argument('--no-sem', dest='sem_enabled', action='store_false',
                        help='Disable Simplicial Embedding (SEM)')
    
    # Environment parameters
    parser.add_argument('--substrate-type', type=str, choices=['linear', 'exponential', 'random'], default=None,
                        help='Substrate type for environment')
    parser.add_argument('--substrate-width', type=float, default=None,
                        help='Substrate width (default from config.yaml)')
    parser.add_argument('--substrate-height', type=float, default=None,
                        help='Substrate height (default from config.yaml)')
    parser.add_argument('--substrate-m', type=float, default=None,
                        help='Substrate gradient parameter m (for linear/step types)')
    parser.add_argument('--substrate-b', type=float, default=None,
                        help='Substrate offset parameter b (for linear/step types)')
    parser.add_argument('--init-nodes', type=int, default=None,
                        help='Initial number of nodes')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Maximum critical nodes allowed')
    
    # Reward mode configuration
    parser.add_argument('--simple-delete-only', action='store_true', default=None,
                        help='Enable simple delete-only reward mode (Rule 0, 1, 2 only)')
    parser.add_argument('--no-simple-delete-only', dest='simple_delete_only', action='store_false',
                        help='Disable simple delete-only reward mode')
    parser.add_argument('--centroid-distance-only', action='store_true', default=None,
                        help='Enable centroid distance-only reward mode (pure distance learning)')
    parser.add_argument('--no-centroid-distance-only', dest='centroid_distance_only', action='store_false',
                        help='Disable centroid distance-only reward mode')
    parser.add_argument('--include-termination-rewards', action='store_true', default=None,
                        help='Include termination rewards in special modes (default: False)')
    parser.add_argument('--no-include-termination-rewards', dest='include_termination_rewards', action='store_false',
                        help='Exclude termination rewards in special modes')
    
    # Checkpoint and logging
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save training results')
    parser.add_argument('--checkpoint-every', type=int, default=None,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--log-every', type=int, default=None,
                        help='Log progress every N episodes')
    
    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to checkpoint file for resuming')
    
    args = parser.parse_args()
    return args


def apply_architecture_overrides(trainer, architecture_overrides):
    """
    Apply architecture-specific overrides to the trainer's network.
    
    This is a workaround to modify nested configuration after initialization.
    Note: These changes won't affect the already-initialized network,
    but are documented for future reference.
    """
    if not architecture_overrides:
        return
    
    print("\n🔧 Architecture Overrides Applied (for next initialization):")
    for key, value in architecture_overrides.items():
        print(f"   {key}: {value}")


def main():
    """Run the training with command-line argument support"""
    args = parse_args()
    
    print("🎯 Multi-Component Durotaxis Training (CLI Mode)")
    print("=" * 70)
    
    # Set random seed if specified
    if args.seed is not None:
        print(f"🎲 Setting random seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Config file: {args.config}")
    if args.experiment:
        print(f"   Experiment: {args.experiment}")
    if args.seed is not None:
        print(f"   Seed: {args.seed}")
    
    # Build overrides dictionary from command-line arguments
    overrides = {}
    
    # Training parameters
    if args.total_episodes is not None:
        overrides['total_episodes'] = args.total_episodes
        print(f"   Total episodes: {args.total_episodes}")
    if args.max_steps is not None:
        overrides['max_steps'] = args.max_steps
        print(f"   Max steps: {args.max_steps}")
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
        print(f"   Learning rate: {args.learning_rate}")
    
    # Environment parameters
    if args.substrate_type is not None:
        overrides['substrate_type'] = args.substrate_type
        print(f"   Substrate type: {args.substrate_type}")
    
    # Substrate dimensions
    if args.substrate_width is not None or args.substrate_height is not None:
        substrate_size = []
        if args.substrate_width is not None:
            substrate_size.append(args.substrate_width)
            print(f"   Substrate width: {args.substrate_width}")
        else:
            substrate_size.append(None)  # Use config default
        
        if args.substrate_height is not None:
            substrate_size.append(args.substrate_height)
            print(f"   Substrate height: {args.substrate_height}")
        else:
            substrate_size.append(None)  # Use config default
        
        # Only override if at least one dimension is specified
        if args.substrate_width is not None or args.substrate_height is not None:
            overrides['substrate_size'] = substrate_size
    
    # Substrate parameters (m, b)
    if args.substrate_m is not None or args.substrate_b is not None:
        substrate_params = {}
        if args.substrate_m is not None:
            substrate_params['m'] = args.substrate_m
            print(f"   Substrate m: {args.substrate_m}")
        if args.substrate_b is not None:
            substrate_params['b'] = args.substrate_b
            print(f"   Substrate b: {args.substrate_b}")
        
        if substrate_params:
            overrides['substrate_params'] = substrate_params
    
    if args.init_nodes is not None:
        # Map to the correct config key
        print(f"   ⚠️  --init-nodes override requires modifying environment config directly")
        print(f"      Please set environment.init_num_nodes in config.yaml")
    if args.max_nodes is not None:
        # Map to the correct config key
        print(f"   ⚠️  --max-nodes override requires modifying environment config directly")
        print(f"      Please set environment.max_critical_nodes in config.yaml")
    
    # Reward mode configuration
    if args.simple_delete_only is not None:
        overrides['simple_delete_only_mode'] = args.simple_delete_only
        print(f"   Simple delete-only mode: {args.simple_delete_only}")
    if args.centroid_distance_only is not None:
        overrides['centroid_distance_only_mode'] = args.centroid_distance_only
        print(f"   Centroid distance-only mode: {args.centroid_distance_only}")
    if args.include_termination_rewards is not None:
        overrides['include_termination_rewards'] = args.include_termination_rewards
        print(f"   Include termination rewards: {args.include_termination_rewards}")
    
    # Logging parameters
    if args.save_dir is not None:
        overrides['save_dir'] = args.save_dir
        print(f"   Save directory: {args.save_dir}")
    if args.checkpoint_every is not None:
        overrides['checkpoint_every'] = args.checkpoint_every
        print(f"   Checkpoint every: {args.checkpoint_every} episodes")
    if args.log_every is not None:
        overrides['log_every'] = args.log_every
        print(f"   Log every: {args.log_every} episodes")
    
    # Architecture-specific parameters
    # NOTE: These require modifying the config file before creating the network
    # We'll display a warning and instructions
    architecture_changes = []
    
    if args.pretrained_weights is not None:
        architecture_changes.append(f"actor_critic.pretrained_weights: '{args.pretrained_weights}'")
        print(f"   Pretrained weights: {args.pretrained_weights}")
    
    if args.sem_enabled is not None:
        architecture_changes.append(f"encoder.simplicial_embedding.enabled: {str(args.sem_enabled).lower()}")
        print(f"   SEM enabled: {args.sem_enabled}")
    
    # Experiment name
    if args.experiment:
        overrides['experiment_name'] = args.experiment
    
    # Resume training
    if args.resume:
        if args.checkpoint_path:
            overrides['resume_from_checkpoint'] = args.checkpoint_path
            print(f"   Resuming from: {args.checkpoint_path}")
        else:
            print(f"   ⚠️  --resume requires --checkpoint-path")
            sys.exit(1)
    
    # Display architecture change instructions
    if architecture_changes:
        print(f"\n⚠️  Architecture Configuration Required:")
        print(f"   The following changes need to be made to {args.config}:")
        for change in architecture_changes:
            print(f"      • {change}")
        print(f"\n   For ablation studies, use pre-configured files or modify config.yaml")
        print(f"   Then run: python train_cli.py --config {args.config} [other args]")
        
        # Ask for confirmation
        response = input("\n   Continue with current config.yaml settings? [y/N]: ")
        if response.lower() != 'y':
            print("   Exiting. Please update config.yaml and try again.")
            sys.exit(0)
    
    print("=" * 70 + "\n")
    
    # Create trainer with overrides
    trainer = DurotaxisTrainer(
        config_path=args.config,
        **overrides
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
