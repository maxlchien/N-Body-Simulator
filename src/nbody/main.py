from __future__ import annotations

import numpy as np
from body import Body
from EulerPropagator import EulerPropagator


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
    propagator.propagate()
    filename = propagator.write_results()
    print(f"Simulation complete. Results written to {filename}")


if __name__ == "__main__":
    main()
