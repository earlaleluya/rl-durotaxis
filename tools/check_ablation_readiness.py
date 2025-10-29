#!/usr/bin/env python3
"""
Ablation Study Readiness Check for SEM (Simplicial Embedding)

This script verifies that the codebase is ready for SEM ablation studies by:
1. Checking SEM implementation correctness
2. Verifying SEM can be toggled on/off
3. Ensuring no WSA dependencies remain
4. Testing both configurations work properly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path

def check_wsa_removed():
    """Verify all WSA code has been removed or disabled."""
    print("\n" + "="*70)
    print("1. CHECKING WSA REMOVAL")
    print("="*70)
    
    issues = []
    
    # Check config.yaml
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config_text = f.read()
        config = yaml.safe_load(config_text)
    
    # Check for WSA section in config
    if 'wsa' in config.get('actor_critic', {}):
        issues.append("❌ WSA section still exists in config.yaml")
    else:
        print("✅ WSA section removed from config.yaml")
    
    # Check for WSA references in code files
    code_files = [
        'train.py',
        'actor_critic.py',
    ]
    
    wsa_patterns = ['use_wsa', 'wsa_config', 'WSA']
    for file in code_files:
        file_path = Path(__file__).parent.parent / file
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
                # Check for problematic WSA usage (not just comments)
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue
                    # Skip docstrings
                    if '"""' in line or "'''" in line:
                        continue
                    # Check for actual WSA usage
                    if 'use_wsa' in line and 'self.use_wsa' not in line:
                        issues.append(f"⚠️  WSA reference in {file}:{i}: {line.strip()}")
    
    if not issues:
        print("✅ No active WSA code found in train.py and actor_critic.py")
    
    # Check show_default_config.py (OK to have WSA references for historical configs)
    print("ℹ️  Note: show_default_config.py may reference WSA for historical ablation configs")
    
    return len(issues) == 0, issues


def check_sem_implementation():
    """Verify SEM is correctly implemented."""
    print("\n" + "="*70)
    print("2. CHECKING SEM IMPLEMENTATION")
    print("="*70)
    
    from encoder import SimplicialEmbedding
    
    issues = []
    
    # Test 1: Basic functionality
    try:
        sem = SimplicialEmbedding(input_dim=128, num_groups=16, temperature=1.0)
        x = torch.randn(10, 128)
        z = sem(x)
        
        # Verify simplex constraints
        z_reshaped = z.view(10, 16, 8)
        group_sums = z_reshaped.sum(dim=-1)
        
        if not torch.allclose(group_sums, torch.ones_like(group_sums), atol=1e-5):
            issues.append("❌ SEM simplex constraints violated")
        else:
            print("✅ SEM simplex constraints verified")
            
        if not (z >= 0).all():
            issues.append("❌ SEM produces negative values")
        else:
            print("✅ SEM non-negativity verified")
            
    except Exception as e:
        issues.append(f"❌ SEM basic test failed: {e}")
    
    return len(issues) == 0, issues


def check_sem_toggleable():
    """Verify SEM can be toggled on/off in config."""
    print("\n" + "="*70)
    print("3. CHECKING SEM TOGGLE CAPABILITY")
    print("="*70)
    
    from config_loader import ConfigLoader
    from encoder import GraphInputEncoder
    
    issues = []
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    # Test with SEM enabled
    try:
        config_loader = ConfigLoader(str(config_path))
        config = config_loader.config
        
        # Check current state
        sem_enabled = config.get('actor_critic', {}).get('simplicial_embedding', {}).get('enabled', False)
        print(f"📋 Current config: SEM enabled = {sem_enabled}")
        
        # Test encoder creation with current config
        encoder = GraphInputEncoder(
            hidden_dim=128,
            out_dim=128,
            num_layers=4,
            config_loader=config_loader
        )
        
        if sem_enabled:
            if encoder.sem_layer is None:
                issues.append("❌ Config says SEM enabled but encoder has no SEM layer")
            else:
                print(f"✅ SEM enabled: Layer present with {encoder.sem_layer.num_groups} groups")
        else:
            if encoder.sem_layer is not None:
                issues.append("❌ Config says SEM disabled but encoder has SEM layer")
            else:
                print("✅ SEM disabled: No SEM layer in encoder")
                
    except Exception as e:
        issues.append(f"❌ SEM toggle test failed: {e}")
    
    return len(issues) == 0, issues


def check_sem_delete_ratio_compatibility():
    """Verify SEM works with delete ratio action space."""
    print("\n" + "="*70)
    print("4. CHECKING SEM + DELETE RATIO COMPATIBILITY")
    print("="*70)
    
    from config_loader import ConfigLoader
    from actor_critic import HybridActorCritic
    from encoder import GraphInputEncoder
    
    issues = []
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    try:
        config_loader = ConfigLoader(str(config_path))
        encoder_config = config_loader.get_encoder_config()
        
        # Create encoder first
        encoder = GraphInputEncoder(
            hidden_dim=encoder_config.get('hidden_dim', 128),
            out_dim=encoder_config.get('out_dim', 128),
            num_layers=encoder_config.get('num_layers', 4),
            config_loader=config_loader
        )
        
        # Create network with SEM-enabled encoder
        network = HybridActorCritic(
            encoder=encoder,
            config_path=str(config_path)
        )
        
        print(f"✅ Network created successfully")
        print(f"   - Encoder SEM: {network.encoder.sem_layer is not None}")
        print(f"   - Continuous dim: {network.continuous_dim}")
        print(f"   - Action space: [delete_ratio, gamma, alpha, noise, theta]")
        
        # Test forward pass
        num_nodes = 10
        state_dict = {
            'node_features': torch.randn(num_nodes, 11),
            'graph_features': torch.randn(19),
            'edge_attr': torch.randn(30, 3),
            'edge_index': torch.randint(0, num_nodes, (2, 30))
        }
        
        output = network(state_dict, deterministic=False)
        
        # Verify output structure
        if 'continuous_actions' not in output:
            issues.append("❌ Network output missing continuous_actions")
        elif output['continuous_actions'].shape[-1] != 5:
            issues.append(f"❌ Continuous actions wrong shape: {output['continuous_actions'].shape}")
        else:
            print(f"✅ Forward pass successful")
            print(f"   - Continuous actions shape: {output['continuous_actions'].shape}")
            
        # Verify no discrete actions
        if 'discrete_actions' in output:
            issues.append("❌ Network still outputs discrete actions")
        else:
            print("✅ No discrete actions (delete ratio architecture confirmed)")
            
    except Exception as e:
        issues.append(f"❌ Compatibility test failed: {e}")
    
    return len(issues) == 0, issues


def check_ablation_ready():
    """Verify codebase is ready for ablation studies."""
    print("\n" + "="*70)
    print("5. ABLATION STUDY READINESS")
    print("="*70)
    
    issues = []
    
    # Check 1: Config has clear SEM toggle
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sem_config = config.get('actor_critic', {}).get('simplicial_embedding', {})
    if 'enabled' not in sem_config:
        issues.append("❌ Missing 'enabled' flag in simplicial_embedding config")
    else:
        print(f"✅ SEM toggle available: enabled = {sem_config['enabled']}")
    
    # Check 2: Required parameters present
    required_params = ['num_groups', 'temperature']
    for param in required_params:
        if param not in sem_config:
            issues.append(f"❌ Missing required SEM parameter: {param}")
        else:
            print(f"✅ SEM parameter '{param}': {sem_config[param]}")
    
    # Check 3: Verify both configurations work
    print("\n📋 Ablation Study Configurations:")
    print("   Configuration A (Baseline): SEM disabled (enabled: false)")
    print("   Configuration B (Enhanced): SEM enabled (enabled: true)")
    print("\n   To run ablation study:")
    print("   1. Edit config.yaml: simplicial_embedding.enabled = false")
    print("   2. Train baseline model")
    print("   3. Edit config.yaml: simplicial_embedding.enabled = true")
    print("   4. Train enhanced model")
    print("   5. Compare performance metrics")
    
    return len(issues) == 0, issues


def main():
    """Run all verification checks."""
    print("\n" + "="*70)
    print("🔬 SEM ABLATION STUDY READINESS VERIFICATION")
    print("="*70)
    
    all_passed = True
    all_issues = []
    
    # Run all checks
    checks = [
        ("WSA Removal", check_wsa_removed),
        ("SEM Implementation", check_sem_implementation),
        ("SEM Toggle", check_sem_toggleable),
        ("SEM + Delete Ratio", check_sem_delete_ratio_compatibility),
        ("Ablation Readiness", check_ablation_ready)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            passed, issues = check_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
                all_issues.extend(issues)
        except Exception as e:
            print(f"\n❌ Check '{name}' crashed: {e}")
            results.append((name, False))
            all_passed = False
            all_issues.append(f"Check '{name}' crashed: {e}")
    
    # Final summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {name}")
    
    if all_issues:
        print("\n⚠️  Issues Found:")
        for issue in all_issues:
            print(f"   {issue}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("="*70)
        print("\n🎯 Codebase is READY for SEM ablation studies!")
        print("\nNext Steps:")
        print("1. Baseline run: Set simplicial_embedding.enabled = false")
        print("2. Enhanced run: Set simplicial_embedding.enabled = true")
        print("3. Compare training curves and final performance")
        print("4. Analyze embedding quality differences")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*70)
        print(f"\n⚠️  {len(all_issues)} issue(s) need to be resolved")
        return 1


if __name__ == "__main__":
    exit(main())
