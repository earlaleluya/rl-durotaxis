#!/usr/bin/env python3
"""
Minimal test to isolate memory corruption issue
"""
import sys
sys.path.insert(0, '.')

import torch
from durotaxis_env import DurotaxisEnv

print("Testing minimal training loop for memory corruption...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Force CPU to avoid CUDA-related corruption
import os
os.environ['FORCE_CPU'] = '1'

from device import get_device
device = get_device()
print(f"Using device: {device}")

# Create environment
env = DurotaxisEnv(config_path='config.yaml', device=device)
print(f"Environment created")
print(f"  Env device: {env.device}")
print(f"  Topology device: {env.topology.device}")
print(f"  Encoder device: {next(env.observation_encoder.parameters()).device}")

# Test reset
obs, info = env.reset()
print(f"\nAfter reset:")
print(f"  Observation shape: {obs.shape}")
print(f"  Graph device: {env.topology.graph.device}")
print(f"  Graph ndata devices:")
for key in env.topology.graph.ndata.keys():
    print(f"    {key}: {env.topology.graph.ndata[key].device}")

# Test a few steps
for step in range(5):
    obs, reward_dict, terminated, truncated, info = env.step(0)
    print(f"\nStep {step+1}:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Num nodes: {env.topology.graph.num_nodes()}")
    print(f"  Reward: {reward_dict.get('total_reward', 0.0):.3f}")
    print(f"  Graph device: {env.topology.graph.device}")
    
    if terminated or truncated:
        print(f"  Episode ended")
        break

print("\n✅ Minimal test completed successfully!")
