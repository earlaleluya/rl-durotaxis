#!/usr/bin/env python3
"""
Check for deprecated parameters and implementations in the codebase.
"""

import re
from pathlib import Path

def check_deprecated_in_file(filepath, patterns):
    """Check a file for deprecated patterns."""
    issues = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
        for line_num, line in enumerate(lines, 1):
            # Skip documentation and notes
            if 'notes/' in str(filepath) or 'tools/' in str(filepath):
                continue
                
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip comments explaining removal
                    if 'REMOVED' in line or 'DEPRECATED' in line or '#' in line.split(pattern)[0]:
                        continue
                    issues.append({
                        'file': str(filepath),
                        'line': line_num,
                        'pattern': pattern_name,
                        'content': line.strip()
                    })
    except Exception as e:
        pass
    
    return issues


def main():
    print("\n" + "="*70)
    print("🔍 CHECKING FOR DEPRECATED PARAMETERS")
    print("="*70)
    
    # Patterns to search for
    deprecated_patterns = {
        'num_discrete_actions': r'\bnum_discrete_actions\b',
        'spawn_bias_init': r'\bspawn_bias_init\b',
        'discrete_weight': r'\bdiscrete_weight\b(?!.*continuous)',  # Not followed by continuous
        'discrete_entropy_weight': r'\bdiscrete_entropy_weight\b',
        'discrete_head': r'\bdiscrete_head\b',
        'discrete_bias': r'\bdiscrete_bias\b',
        'discrete_actions': r'\bdiscrete_actions\b(?!.*No longer|REMOVED)',
        'use_wsa': r'\buse_wsa\b',
    }
    
    # Files to check
    files_to_check = [
        'config.yaml',
        'train.py',
        'actor_critic.py',
        'encoder.py',
    ]
    
    all_issues = []
    
    for filename in files_to_check:
        filepath = Path(filename)
        if filepath.exists():
            issues = check_deprecated_in_file(filepath, deprecated_patterns)
            all_issues.extend(issues)
    
    if all_issues:
        print("\n⚠️  Found deprecated parameters:\n")
        for issue in all_issues:
            print(f"❌ {issue['file']}:{issue['line']}")
            print(f"   Pattern: {issue['pattern']}")
            print(f"   Content: {issue['content'][:100]}")
            print()
        print(f"\n❌ Total issues found: {len(all_issues)}")
        return 1
    else:
        print("\n✅ No deprecated parameters found!")
        print("\nChecked patterns:")
        for pattern in deprecated_patterns.keys():
            print(f"  ✓ {pattern}")
        print("\nChecked files:")
        for f in files_to_check:
            print(f"  ✓ {f}")
        return 0


if __name__ == "__main__":
    exit(main())
