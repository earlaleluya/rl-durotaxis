#!/usr/bin/env python3
"""Quick test to verify delete ratio environment works"""
import numpy as np
from durotaxis_env import DurotaxisEnv

print("Creating environment...")
env = DurotaxisEnv(
    config_path='config.yaml',
    substrate_type='linear',
    init_num_nodes=10,
    max_critical_nodes=50,
    max_steps=20,
    enable_visualization=False
)

print("Resetting environment...")
obs, info = env.reset()
print(f"✅ Reset successful! Initial nodes: {env.topology.graph.number_of_nodes()}")

done = False
step = 0
while not done and step < 5:
    # Random action: [delete_ratio, gamma, alpha, noise, theta]
    action = np.array([
        np.random.uniform(0.1, 0.3),  # delete_ratio
        np.random.uniform(0.7, 0.9),  # gamma
        np.random.uniform(0.6, 0.8),  # alpha
        np.random.uniform(0.05, 0.15), # noise
        np.random.uniform(0.0, np.pi/4)  # theta
    ])
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Extract reward
    if isinstance(reward, dict):
        reward_scalar = reward.get('total_reward', 0.0)
    else:
        reward_scalar = reward
    
    step += 1
    print(f"Step {step}: Nodes={env.topology.graph.number_of_nodes()}, Reward={reward_scalar:.2f}")

print(f"\n✅ Test successful! Ran {step} steps")
