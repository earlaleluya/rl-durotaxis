"""
Demonstrate the bug that occurs WITHOUT persistent_id tracking.
This shows what WOULD happen if we used node_ids directly.
"""
import os
os.environ['FORCE_CPU'] = '1'

from topology import Topology
from substrate import Substrate

# Create topology
substrate = Substrate(size=(100, 100))
substrate.create('linear', m=0.05, b=1.0)
topo = Topology(substrate=substrate, flush_delay=5, verbose=False)

print("=== Initial State ===")
print(f"N={topo.graph.num_nodes()}")
print(f"Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

# Simulate OLD BUGGY behavior: use node_ids directly
print(f"\n=== BUGGY Approach (using node_ids directly) ===")
print(f"Policy decides: spawn from [1, 3], delete [2, 4]")

spawn_targets = [1, 3]
delete_targets = [2, 4]  # These will be WRONG after spawning!

print(f"\nExecuting spawns...")
for node_id in spawn_targets:
    print(f"  Spawn from node_id={node_id}")
    topo.spawn(node_id, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0)
    print(f"    After: N={topo.graph.num_nodes()}, persistent_ids={topo.graph.ndata['persistent_id'].tolist()}")

print(f"\nExecuting deletes (using STALE node_ids)...")
delete_targets_sorted = sorted(delete_targets, reverse=True)
for node_id in delete_targets_sorted:
    if node_id < topo.graph.num_nodes():
        pid_at_position = topo.graph.ndata['persistent_id'][node_id].item()
        print(f"  Delete node_id={node_id} (which NOW has persistent_id={pid_at_position})")
        topo.delete(node_id)
        print(f"    After: N={topo.graph.num_nodes()}, persistent_ids={topo.graph.ndata['persistent_id'].tolist()}")

print(f"\n=== Result ===")
print(f"Final persistent_ids: {topo.graph.ndata['persistent_id'].tolist()}")
print(f"\n❌ BUG: We intended to delete persistent_ids [2, 4]")
print(f"       But we actually deleted persistent_ids that were AT POSITIONS 2 and 4 AFTER spawning!")
print(f"       This deleted THE WRONG NODES because indices shifted during spawning.")
