"""Quick test of edge explosion fix"""
import os
os.environ['FORCE_CPU'] = '1'

from topology import Topology
from substrate import Substrate

substrate = Substrate(size=(100, 100))
substrate.create('linear', m=0.05, b=1.0)
topo = Topology(substrate=substrate, flush_delay=5, verbose=False)

print("Testing edge explosion fix (max 2 successor inheritance)")
print(f"Initial: N={topo.graph.num_nodes()} E={topo.graph.num_edges()}")

for step in range(1, 11):
    topo.act()
    n = topo.graph.num_nodes()
    e = topo.graph.num_edges()
    ratio = e/n if n > 0 else 0
    print(f"Step {step}: N={n:3d} E={e:4d} (E/N={ratio:.2f})")
    if n == 0:
        break
