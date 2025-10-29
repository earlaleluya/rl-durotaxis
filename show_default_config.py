"""
Test script to display the current default configuration
"""

import yaml
from config_loader import ConfigLoader

def display_default_config():
    """Display the current default configuration from config.yaml"""
    
    print("=" * 80)
    print("📋 DEFAULT CONFIGURATION SUMMARY (Delete Ratio Architecture)")
    print("=" * 80)
    
    # Load configuration
    config_loader = ConfigLoader('config.yaml')
    
    # Get actor-critic config
    actor_critic_config = config_loader.get_actor_critic_config()
    
    print("\n🏗️  ARCHITECTURE CONFIGURATION:")
    print("-" * 80)
    
    pretrained_weights = actor_critic_config.get('pretrained_weights', 'not set')
    print(f"  Pretrained Weights: {pretrained_weights}")
    print(f"  Action Space: Delete Ratio (5D continuous)")
    print(f"    - [delete_ratio, gamma, alpha, noise, theta]")
    
    # Get encoder config for SEM
    encoder_config = config_loader.get_encoder_config()
    sem_config = encoder_config.get('simplicial_embedding', {})
    sem_enabled = sem_config.get('enabled', False)
    print(f"  SEM Enabled: {sem_enabled}")
    if sem_enabled:
        print(f"    - Num Groups: {sem_config.get('num_groups', 'not set')}")
        print(f"    - Temperature: {sem_config.get('temperature', 'not set')}")
    
    # Determine ablation configuration (simplified for delete ratio)
    print("\n🎯 ABLATION CONFIGURATION:")
    print("-" * 80)
    
    if not sem_enabled:
        if pretrained_weights == 'imagenet':
            config_name = "baseline"
            description = "Baseline - ImageNet + No SEM"
        else:
            config_name = "random_baseline"
            description = "Baseline - Random Weights + No SEM"
    else:
        if pretrained_weights == 'imagenet':
            config_name = "sem_enhanced"
            description = "SEM Enhanced - ImageNet + SEM"
        else:
            config_name = "random_sem"
            description = "SEM Enhanced - Random Weights + SEM"
    
    print(f"  Configuration: {config_name}")
    print(f"  Description: {description}")
    
    # Get training config
    trainer_config = config_loader.get_trainer_config()
    
    print("\n📚 TRAINING PARAMETERS:")
    print("-" * 80)
    print(f"  Total Episodes: {trainer_config.get('total_episodes', 'not set')}")
    print(f"  Max Steps: {trainer_config.get('max_steps', 'not set')}")
    print(f"  Learning Rate: {trainer_config.get('learning_rate', 'not set')}")
    print(f"  Save Directory: {trainer_config.get('save_dir', 'not set')}")
    print(f"  Log Every: {trainer_config.get('log_every', 'not set')} episodes")
    print(f"  Checkpoint Every: {trainer_config.get('checkpoint_every', 'disabled')}")
    
    # Get environment config
    env_config = config_loader.get_environment_config()
    
    print("\n🌍 ENVIRONMENT PARAMETERS:")
    print("-" * 80)
    print(f"  Substrate Size: {env_config.get('substrate_size', 'not set')}")
    print(f"  Substrate Type: {env_config.get('substrate_type', 'not set')}")
    print(f"  Initial Nodes: {env_config.get('init_num_nodes', 'not set')}")
    print(f"  Max Nodes: {env_config.get('max_critical_nodes', 'not set')}")
    print(f"  Max Steps: {env_config.get('max_steps', 'not set')}")
    
    # Get encoder config
    encoder_config = config_loader.get_encoder_config()
    
    print("\n🔧 ENCODER CONFIGURATION:")
    print("-" * 80)
    print(f"  Output Dimension: {encoder_config.get('out_dim', 'not set')}")
    print(f"  Number of Layers: {encoder_config.get('num_layers', 'not set')}")
    
    # Get algorithm config
    algorithm_config = config_loader.get_algorithm_config()
    
    print("\n⚙️  ALGORITHM PARAMETERS:")
    print("-" * 80)
    print(f"  Gamma (discount): {algorithm_config.get('gamma', 'not set')}")
    print(f"  GAE Lambda: {algorithm_config.get('gae_lambda', 'not set')}")
    print(f"  PPO Epochs: {algorithm_config.get('ppo_epochs', 'not set')}")
    print(f"  Clip Epsilon: {algorithm_config.get('clip_epsilon', 'not set')}")
    
    # Get system config
    system_config = config_loader.config.get('system', {})
    
    print("\n💻 SYSTEM CONFIGURATION:")
    print("-" * 80)
    print(f"  Device: {system_config.get('device', 'not set')}")
    print(f"  Seed: {system_config.get('seed', 'not set (random)')}")
    
    print("\n" + "=" * 80)
    print("✅ Configuration loaded from: config.yaml")
    print("=" * 80 + "\n")
    
    # Summary
    print("📝 QUICK SUMMARY:")
    print(f"  • Running 'python train.py' will use configuration {config_name}")
    print(f"  • Architecture: Delete Ratio (5D continuous action)")
    print(f"  • Pretrained: {pretrained_weights}, SEM: {sem_enabled}")
    print(f"  • Training for {trainer_config.get('total_episodes')} episodes")
    print(f"  • Max {env_config.get('max_steps')} steps per episode")
    print("")


if __name__ == '__main__':
    display_default_config()
