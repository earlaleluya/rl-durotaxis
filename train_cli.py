#!/usr/bin/env python3
"""
Command-line interface wrapper for train.py with argument parsing

This script provides command-line argument support for the DurotaxisTrainer.
It parses arguments and passes them as overrides to the trainer.

Usage:
    python train_cli.py --help
    python train_cli.py --pretrained-weights imagenet --wsa-enabled
    python train_cli.py --seed 42 --experiment baseline_seed42
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
        description='Train Durotaxis RL Agent with Hybrid Actor-Critic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python train_cli.py
  
  # Override total episodes and learning rate
  python train_cli.py --total-episodes 2000 --learning-rate 0.0005
  
  # Use random pretrained weights with WSA enabled
  python train_cli.py --pretrained-weights random --wsa-enabled
  
  # Disable SEM and use custom experiment name
  python train_cli.py --no-sem --experiment my_ablation_test
  
  # Run specific seed for reproducibility
  python train_cli.py --seed 42 --experiment baseline_seed42
  
  # Full ablation configuration (d1)
  python train_cli.py --pretrained-weights imagenet --wsa-enabled --sem-enabled --experiment d1_full_stack --seed 1
  
  # Baseline ablation configuration (a1)
  python train_cli.py --pretrained-weights imagenet --no-wsa --no-sem --experiment a1_baseline --seed 1
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
                        help='Pretrained weights for Actor/Critic (imagenet or random)')
    parser.add_argument('--wsa-enabled', action='store_true', default=None,
                        help='Enable Weight Sharing Attention (WSA)')
    parser.add_argument('--no-wsa', dest='wsa_enabled', action='store_false',
                        help='Disable Weight Sharing Attention (WSA)')
    parser.add_argument('--sem-enabled', action='store_true', default=None,
                        help='Enable Simplicial Embedding (SEM)')
    parser.add_argument('--no-sem', dest='sem_enabled', action='store_false',
                        help='Disable Simplicial Embedding (SEM)')
    
    # Environment parameters
    parser.add_argument('--substrate-type', type=str, choices=['linear', 'exponential', 'random'], default=None,
                        help='Substrate type for environment')
    parser.add_argument('--init-nodes', type=int, default=None,
                        help='Initial number of nodes')
    parser.add_argument('--max-nodes', type=int, default=None,
                        help='Maximum critical nodes allowed')
    
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
    if args.init_nodes is not None:
        # Map to the correct config key
        print(f"   ⚠️  --init-nodes override requires modifying environment config directly")
        print(f"      Please set environment.init_num_nodes in config.yaml")
    if args.max_nodes is not None:
        # Map to the correct config key
        print(f"   ⚠️  --max-nodes override requires modifying environment config directly")
        print(f"      Please set environment.max_critical_nodes in config.yaml")
    
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
    
    if args.wsa_enabled is not None:
        architecture_changes.append(f"actor_critic.wsa.enabled: {str(args.wsa_enabled).lower()}")
        print(f"   WSA enabled: {args.wsa_enabled}")
    
    if args.sem_enabled is not None:
        architecture_changes.append(f"actor_critic.simplicial_embedding.enabled: {str(args.sem_enabled).lower()}")
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
