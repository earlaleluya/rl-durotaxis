#!/usr/bin/env python3
"""
Test script to debug Actor network output values during training initialization.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from actor_critic import HybridActorCritic
from encoder import GraphInputEncoder
from config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader("config.yaml")
encoder_config = config_loader.get_encoder_config()

# Create encoder and actor-critic network
encoder = GraphInputEncoder(**encoder_config)
network = HybridActorCritic(encoder=encoder, config_path="config.yaml")

print("=" * 80)
print("TESTING ACTOR OUTPUT WITH RANDOM GRAPH")
print("=" * 80)

# Create a small random graph for testing
num_nodes = 10
node_dim = encoder_config.get('node_feature_dim', 8)
graph_dim = encoder_config.get('graph_feature_dim', 4)
edge_dim = encoder_config.get('edge_feature_dim', 2)

# Generate random state
state_dict = {
    'node_features': torch.randn(num_nodes, node_dim),
    'graph_features': torch.randn(graph_dim),
    'edge_attr': torch.randn(20, edge_dim),  # Random edges
    'edge_index': torch.randint(0, num_nodes, (2, 20)),
    'num_nodes': num_nodes,
    'num_edges': 20
}

# Forward pass (non-deterministic, so we sample)
print("\n1. Testing Actor Forward Pass...")
with torch.no_grad():
    output = network(state_dict, deterministic=False)
    
    print(f"\nOutput keys: {output.keys()}")
    print(f"\nContinuous mu (bounded): {output['continuous_mu']}")
    print(f"Continuous std: {output['continuous_std']}")
    print(f"Continuous actions (sampled): {output['continuous_actions']}")
    print(f"\nAction values breakdown:")
    print(f"  delete_ratio: {output['continuous_actions'][0].item():.4f}")
    print(f"  gamma: {output['continuous_actions'][1].item():.4f}")
    print(f"  alpha: {output['continuous_actions'][2].item():.4f}")
    print(f"  noise: {output['continuous_actions'][3].item():.4f}")
    print(f"  theta: {output['continuous_actions'][4].item():.4f}")

# Test deterministic mode
print("\n2. Testing Deterministic Mode...")
with torch.no_grad():
    output_det = network(state_dict, deterministic=True)
    print(f"Deterministic actions: {output_det['continuous_actions']}")

# Test multiple samples to see if there's variation
print("\n3. Testing Multiple Samples (should show variation)...")
samples = []
for i in range(5):
    with torch.no_grad():
        output = network(state_dict, deterministic=False)
        samples.append(output['continuous_actions'])
        print(f"Sample {i+1}: {output['continuous_actions']}")

# Check if samples are all the same (which would indicate a problem)
all_same = all(torch.allclose(samples[0], s) for s in samples[1:])
print(f"\nAll samples identical: {all_same}")
if all_same:
    print("⚠️  WARNING: All samples are identical! This suggests:")
    print("   - std is too small (near zero)")
    print("   - or Actor is not producing variation")

# Check the raw mu and logstd from Actor
print("\n4. Inspecting Raw Actor Output...")
with torch.no_grad():
    encoder_out = network.encoder(
        graph_features=state_dict['graph_features'],
        node_features=state_dict['node_features'],
        edge_features=state_dict['edge_attr'],
        edge_index=state_dict['edge_index']
    )
    graph_token = encoder_out[0]
    node_tokens = encoder_out[1:]
    
    continuous_mu, continuous_logstd = network.actor(node_tokens, graph_token)
    print(f"Raw mu: {continuous_mu}")
    print(f"Raw logstd: {continuous_logstd}")
    print(f"Raw std (exp of logstd): {torch.exp(continuous_logstd)}")
    
    # Check if mu or logstd are zeros
    if torch.allclose(continuous_mu, torch.zeros_like(continuous_mu)):
        print("⚠️  WARNING: Raw mu is all zeros!")
    if torch.allclose(continuous_logstd, torch.zeros_like(continuous_logstd)):
        print("⚠️  WARNING: Raw logstd is all zeros! (std=1.0)")

print("\n" + "=" * 80)
print("Test complete. Check for warnings above.")
print("=" * 80)
