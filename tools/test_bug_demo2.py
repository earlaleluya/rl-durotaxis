"""
Better demonstration of the index shifting bug.
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
print(f"Nodes: {list(range(topo.graph.num_nodes()))}")
print(f"Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

# Case that shows the bug clearly:
# - Spawn from node 0 (will insert at beginning, shifting ALL indices)
# - Then try to delete node 3 (but indices have shifted!)

print(f"\n=== BUGGY Scenario ===")
print(f"Policy decides: spawn from [0], delete [3]")
print(f"Expected: spawn adds new node, delete removes persistent_id=3")

# Execute spawn from node 0
print(f"\n1. Spawn from node_id=0")
topo.spawn(0, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0)
print(f"   After spawn: N={topo.graph.num_nodes()}")
print(f"   Node indices: {list(range(topo.graph.num_nodes()))}")
print(f"   Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")
print(f"   ⚠️  Notice: new node was inserted, but persistent_id=3 is now at a DIFFERENT position!")

# Now if we delete using the ORIGINAL node_id=3 (BUGGY approach)
original_delete_target = 3
print(f"\n2. Delete node_id={original_delete_target} (BUGGY: using original node_id)")
pid_at_position = topo.graph.ndata['persistent_id'][original_delete_target].item()
print(f"   Position {original_delete_target} now contains persistent_id={pid_at_position}")
topo.delete(original_delete_target)
print(f"   After delete: N={topo.graph.num_nodes()}")
print(f"   Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

print(f"\n=== Bug Analysis ===")
print(f"❌ We wanted to delete persistent_id=3")
print(f"   But we actually deleted persistent_id={pid_at_position}")
print(f"   This happened because spawning shifted all indices!")

# Show the CORRECT approach
print(f"\n\n=== CORRECT Approach (using persistent_id) ===")
topo2 = Topology(substrate=substrate, flush_delay=5, verbose=False)
print(f"Initial: {topo2.graph.ndata['persistent_id'].tolist()}")

# Convert to persistent IDs BEFORE spawning
target_spawn = 0
target_delete = 3
spawn_pid = topo2.node_id_to_persistent_id(target_spawn)
delete_pid = topo2.node_id_to_persistent_id(target_delete)
print(f"Will spawn from persistent_id={spawn_pid}, delete persistent_id={delete_pid}")

# Spawn
current_node_id = topo2.persistent_id_to_node_id(spawn_pid)
topo2.spawn(current_node_id, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0)
print(f"After spawn: {topo2.graph.ndata['persistent_id'].tolist()}")

# Delete (look up current position of persistent_id)
current_node_id = topo2.persistent_id_to_node_id(delete_pid)
print(f"persistent_id={delete_pid} is now at node_id={current_node_id}")
topo2.delete(current_node_id)
print(f"After delete: {topo2.graph.ndata['persistent_id'].tolist()}")
print(f"✅ Correctly deleted persistent_id={delete_pid}!")
