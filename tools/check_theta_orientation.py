#!/usr/bin/env python3
import math
import os
import sys
import torch

# Ensure project root is on sys.path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from topology import Topology
from substrate import Substrate


def check_theta_mapping(gamma=20.0, alpha=0.0, noise=0.0):
    # Create substrate and topology
    substrate = Substrate((600, 400))  # width x height
    substrate.create('linear', m=0.0, b=1.0)

    topo = Topology(substrate=substrate)
    topo.reset(init_num_nodes=1)

    # Center the only node to avoid boundary clamping
    device = topo.graph.ndata['pos'].device
    center = torch.tensor([substrate.width / 2.0, substrate.height / 2.0], dtype=torch.float32, device=device)
    topo.graph.ndata['pos'][0] = center

    # Helper to perform one spawn and compute displacement
    def one(theta):
        # Reset to one node centered each time
        topo.reset(init_num_nodes=1)
        topo.graph.ndata['pos'][0] = center
        parent = topo.graph.ndata['pos'][0].detach().cpu().numpy().copy()
        new_id = topo.spawn(0, gamma=gamma, alpha=alpha, noise=noise, theta=theta)
        new_pos = topo.graph.ndata['pos'][new_id].detach().cpu().numpy().copy()
        dx, dy = new_pos[0] - parent[0], new_pos[1] - parent[1]
        return dx, dy, new_id

    tests = [
        ("0", 0.0),
        ("pi/2", math.pi/2),
        ("pi", math.pi),
        ("-pi/2", -math.pi/2),
        ("3pi/2", 3*math.pi/2),
    ]

    print("Theta orientation probe (expected: 0→+x, pi/2→+y, pi→-x, -pi/2→-y)")
    for name, ang in tests:
        dx, dy, new_id = one(ang)
        print(f"theta={name:>6s} rad: dx={dx:+.3f}, dy={dy:+.3f}, new_id={new_id}")


if __name__ == "__main__":
    check_theta_mapping()
