"""
Test persistent_id tracking to ensure spawn/delete use correct nodes.
"""
import os
os.environ['FORCE_CPU'] = '1'

from topology import Topology
from substrate import Substrate
import torch

# Create topology
substrate = Substrate(size=(100, 100))
substrate.create('linear', m=0.05, b=1.0)
topo = Topology(substrate=substrate, flush_delay=5, verbose=False)

print("=== Initial State ===")
print(f"N={topo.graph.num_nodes()}")
print(f"Node IDs: {list(range(topo.graph.num_nodes()))}")
print(f"Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

# Simulate what policy does: decide actions on current state
node_ids = list(range(topo.graph.num_nodes()))
print(f"\n=== Policy Decision ===")
print(f"Decide to spawn from nodes: [1, 3]")
print(f"Decide to delete nodes: [2, 4]")

# Get persistent IDs for these nodes
spawn_targets = [1, 3]
delete_targets = [2, 4]

spawn_persistent_ids = {topo.node_id_to_persistent_id(nid): nid for nid in spawn_targets}
delete_persistent_ids = {topo.node_id_to_persistent_id(nid): nid for nid in delete_targets}

print(f"\nSpawn targets (node_id -> persistent_id):")
for nid in spawn_targets:
    pid = topo.node_id_to_persistent_id(nid)
    print(f"  node_id={nid} -> persistent_id={pid}")

print(f"\nDelete targets (node_id -> persistent_id):")
for nid in delete_targets:
    pid = topo.node_id_to_persistent_id(nid)
    print(f"  node_id={nid} -> persistent_id={pid}")

# Execute spawns first (this will shift indices!)
print(f"\n=== Executing Spawns ===")
for persistent_id, original_node_id in spawn_persistent_ids.items():
    current_node_id = topo.persistent_id_to_node_id(persistent_id)
    print(f"Spawning from persistent_id={persistent_id} (original_node_id={original_node_id}, current_node_id={current_node_id})")
    if current_node_id is not None:
        topo.spawn(current_node_id, gamma=5.0, alpha=2.0, noise=0.5, theta=0.0)
        print(f"  After spawn: N={topo.graph.num_nodes()}")
        print(f"  Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

# Now execute deletes using persistent IDs
print(f"\n=== Executing Deletes (using persistent IDs) ===")
delete_items = []
for persistent_id, original_node_id in delete_persistent_ids.items():
    current_node_id = topo.persistent_id_to_node_id(persistent_id)
    if current_node_id is not None:
        delete_items.append((current_node_id, persistent_id, original_node_id))

# Sort by current_node_id in reverse
delete_items.sort(reverse=True, key=lambda x: x[0])

for current_node_id, persistent_id, original_node_id in delete_items:
    print(f"Deleting persistent_id={persistent_id} (original_node_id={original_node_id}, current_node_id={current_node_id})")
    topo.delete(current_node_id)
    print(f"  After delete: N={topo.graph.num_nodes()}")
    print(f"  Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

print(f"\n=== Final State ===")
print(f"N={topo.graph.num_nodes()}")
print(f"Persistent IDs: {topo.graph.ndata['persistent_id'].tolist()}")

# Verify we deleted the correct nodes
print(f"\n=== Verification ===")
print(f"Expected to delete persistent_ids: {list(delete_persistent_ids.keys())}")
remaining_pids = topo.graph.ndata['persistent_id'].tolist()
print(f"Remaining persistent_ids: {remaining_pids}")
deleted_correctly = all(pid not in remaining_pids for pid in delete_persistent_ids.keys())
print(f"Deleted correct nodes: {'✅ YES' if deleted_correctly else '❌ NO'}")
