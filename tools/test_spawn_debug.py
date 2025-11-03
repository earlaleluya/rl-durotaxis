"""
Minimal test to debug spawn multiplicity issue.
"""
import os
os.environ['FORCE_CPU'] = '1'

from topology import Topology
from substrate import Substrate

# Create minimal topology
substrate = Substrate(size=(100, 100))
substrate.create('linear', m=0.05, b=1.0)
topo = Topology(substrate=substrate, flush_delay=5, verbose=False)

print(f"=== Initial state ===")
print(f"N={topo.graph.num_nodes()}")

print(f"\n=== Step 1 (act) ===")
topo.act()
print(f"After act: N={topo.graph.num_nodes()}")

print(f"\n=== Step 2 (act) ===")
topo.act()
print(f"After act: N={topo.graph.num_nodes()}")

print(f"\n=== Step 3 (act) ===")
topo.act()
print(f"After act: N={topo.graph.num_nodes()}")

print(f"\n=== Step 4 (act) ===")
topo.act()
n = topo.graph.num_nodes()
e = topo.graph.num_edges()
if n > 0:
    print(f"After act: N={n} E={e} (E/N ratio = {e/n:.2f})")
else:
    print(f"After act: N={n} E={e} (graph collapsed!)")

# Run 10 more steps to see if edges accumulate
for i in range(5, 15):
    if topo.graph.num_nodes() == 0:
        print(f"\n=== Graph collapsed, stopping test ===")
        break
    print(f"\n=== Step {i} (act) ===")
    topo.act()
    n = topo.graph.num_nodes()
    e = topo.graph.num_edges()
    if n > 0:
        print(f"After act: N={n} E={e} (E/N ratio = {e/n:.2f})")
    else:
        print(f"After act: N={n} E={e} (graph collapsed!)")
