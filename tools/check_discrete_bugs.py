#!/usr/bin/env python3
"""
Bug checker for delete ratio refactoring - finds remaining discrete action references
"""

import re
import sys

print("="*60)
print("CHECKING FOR DISCRETE ACTION BUGS")
print("="*60)

bugs_found = []

# Read train.py
with open('train.py', 'r') as f:
    lines = f.readlines()

# Check for problematic patterns
patterns = [
    (r"discrete_head", "References to discrete_head (should be removed)"),
    (r"discrete_bias", "References to discrete_bias (should be removed)"),
    (r"discrete_weight", "References to discrete_weight (should be removed or updated)"),
    (r"discrete_grad_norm", "References to discrete gradient norms"),
    (r"discrete_loss", "References to discrete loss (should be removed)"),
    (r"\.get\('discrete'", "Dictionary access to 'discrete' key"),
    (r"'discrete_actions'", "String literal 'discrete_actions'"),
    (r"'discrete_log_probs'", "String literal 'discrete_log_probs'"),
    (r"\[0, 4\]", "Continuous action shape [0, 4] should be [0, 5]"),
]

for line_num, line in enumerate(lines, 1):
    for pattern, description in patterns:
        if re.search(pattern, line):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            
            bugs_found.append({
                'line': line_num,
                'pattern': pattern,
                'description': description,
                'code': line.strip()
            })

# Report findings
print(f"\n📊 Found {len(bugs_found)} potential discrete action bugs:\n")

# Group by line number to avoid duplicates
seen_lines = set()
for bug in bugs_found:
    if bug['line'] not in seen_lines:
        seen_lines.add(bug['line'])
        print(f"Line {bug['line']}: {bug['description']}")
        print(f"  Code: {bug['code'][:100]}")
        print()

if bugs_found:
    print("⚠️  BUGS FOUND - Need fixes!")
    print("\nPriority fixes:")
    print("1. Remove discrete_head and discrete_bias references in optimizer")
    print("2. Update compute_adaptive_gradient_scaling (no longer needs discrete_loss)")
    print("3. Remove discrete_weight from policy_loss_weights")
    print("4. Fix continuous action shape [0,4] → [0,5]")
    print("5. Remove discrete_grad_norm_ema tracking")
    sys.exit(1)
else:
    print("✅ No discrete action bugs found!")
    sys.exit(0)
