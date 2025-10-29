#!/usr/bin/env python3
"""
Verification script to check if ResNet freezing is properly configured.
This script will:
1. Load the config
2. Initialize the network
3. Print parameter counts (total vs trainable)
4. Print optimizer groups and LRs
5. Verify conv1 preservation status
"""

import torch
from config_loader import ConfigLoader
from encoder import GraphInputEncoder
from actor_critic import HybridActorCritic

def main():
    print("\n" + "="*70)
    print("🔍 VERIFYING RESNET FREEZING CONFIGURATION")
    print("="*70 + "\n")
    
    # Load config
    config_loader = ConfigLoader('config.yaml')
    ac_config = config_loader.get_actor_critic_config()
    
    # Print config settings
    print("📋 Configuration Settings:")
    print(f"   Pretrained weights: {ac_config.get('pretrained_weights', 'imagenet')}")
    
    backbone_cfg = ac_config.get('backbone', {})
    input_adapter = backbone_cfg.get('input_adapter', 'repeat3')
    freeze_mode = backbone_cfg.get('freeze_mode', 'none')
    backbone_lr = backbone_cfg.get('backbone_lr', 1e-4)
    head_lr = backbone_cfg.get('head_lr', 3e-4)
    
    print(f"   Input adapter: {input_adapter}")
    print(f"   Freeze mode: {freeze_mode}")
    print(f"   Backbone LR: {backbone_lr:.6f}")
    print(f"   Head LR: {head_lr:.6f}")
    print()
    
    # Initialize encoder
    encoder_config = config_loader.get_encoder_config()
    encoder = GraphInputEncoder(
        out_dim=encoder_config.get('out_dim', 128),
        num_layers=encoder_config.get('num_layers', 4)
    )
    
    # Initialize HybridActorCritic (this will trigger the parameter info print)
    network = HybridActorCritic(
        encoder=encoder,
        config_path='config.yaml'
    )
    
    print("\n✅ Verification Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
