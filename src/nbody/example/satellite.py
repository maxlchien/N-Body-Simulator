from __future__ import annotations

from pathlib import Path

import numpy as np

from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.preprocess import read_simulation_config

config_path = Path(__file__).parent.parent / "example" / "satellite.yaml"
# NOTE: ALL DISTANCES ARE IN SI UNITS

bodies, sim_params = read_simulation_config(config_path)

propagator = EulerPropagator(bodies, sim_params)
states = propagator.propagate()


positions = states[:, :, 0:2]

ids = [b.id for b in bodies]
i_collider = ids.index("Collider")

collisions = []
num_bodies = len(bodies)

for i in range(num_bodies):
    if i == i_collider:
        continue  # skip Collider itself
    delta = positions[:, i_collider] - positions[:, i]
    distances = np.linalg.norm(delta, axis=1)
    threshold = bodies[i_collider].radius + bodies[i].radius
    collision_steps = np.where(distances < threshold)[0]
    for step in collision_steps:
        collisions.append((step, ids[i], distances[step]))

if collisions:
    print("COLLISIONS DETECTED WITH COLLIDER:")
    for step, sat_id, dist in collisions:
        print(f"Step {step}: Collider ↔ {sat_id}, distance = {dist:.3f} m")
else:
    print("No collisions detected with Collider.")
