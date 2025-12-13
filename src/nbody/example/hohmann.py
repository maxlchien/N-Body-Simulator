from __future__ import annotations

from pathlib import Path

import numpy as np

from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.preprocess import read_simulation_config

config_path = Path(__file__).parent.parent / "example" / "example_hohmann.yaml"
# NOTE: DISTANCES ARE IN AU, MASS IN M_SUN, TIME IN YRS

bodies, sim_params = read_simulation_config(config_path)

propagator = EulerPropagator(bodies, sim_params)
states = propagator.propagate()


positions = states[:, :, 0:2]

ids = [b.id for b in bodies]
i_probe = ids.index("Probe")
i_mars = ids.index("Mars")

delta = positions[:, i_probe] - positions[:, i_mars]
distances = np.linalg.norm(delta, axis=1)

# arrival criterion:
ARRIVAL_RADIUS = 0.01

min_dist = distances.min()
min_step = distances.argmin()
arrived = bool(np.any(distances < ARRIVAL_RADIUS))

arrived = bool(np.any(distances < ARRIVAL_RADIUS))

# Print results
print("\nEarth → Mars Hohmann transfer result")
print("----------------------------------")
print(f"Minimum distance to Mars: {min_dist:.6g} AU")
print(f"Occurred at step: {min_step}")
print(f"Arrival threshold: {ARRIVAL_RADIUS} AU")

if arrived:
    first_hit = np.where(distances < ARRIVAL_RADIUS)[0][0]
    print(f"STATUS: ARRIVED (first entry at step {first_hit})")
else:
    print("STATUS: DID NOT ARRIVE")
