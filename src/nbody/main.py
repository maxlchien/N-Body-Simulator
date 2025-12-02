from __future__ import annotations

import time

import numpy as np
from body import Body
from EulerPropagator import EulerPropagator

USE_NUMBA = (
    False  # not helpful until the number of bodies is large due to compilation overhead
)


def main():
    # Dummy bodies for replacement/showing use case
    bodies = [
        Body(
            id="Earth",
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            mass=5.97e24,
            radius=1,
        ),
        Body(
            id="Moon",
            position=np.array([384_400_000.0, 0.0]),
            velocity=np.array([0.0, 1022.0]),
            mass=7.35e22,
            radius=1,
        ),
    ]

    # Simulation parameters sample
    params = {
        "G": 6.6743e-11,  # real gravitational constant
        "dt": 60.0,  # 1 minute timestep
        "t_total": 3600.0,  # 1 hour total for testing
    }

    # Create propagator and run
    propagator = EulerPropagator(bodies, params)

    if USE_NUMBA:
        propagator.propagate()  # get compilation out of the way, this will get overwritten with the same data

    # time the calculation portion
    start_time = time.perf_counter_ns()
    propagator.propagate()
    end_time = time.perf_counter_ns()
    total_ns = end_time - start_time

    filename = propagator.write_results(
        header_options=[f"USE_NUMBA={USE_NUMBA}", f"Nanoseconds taken={total_ns}"]
    )
    print(f"Simulation complete. Results written to {filename}")


if __name__ == "__main__":
    main()
