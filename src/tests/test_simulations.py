from __future__ import annotations

from pathlib import Path

import pytest

from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.preprocess import read_simulation_config


def test_two_body_orbit():
    """Test a simple two-body circular orbit using a known YAML config."""

    config_path = Path(__file__).parent.parent / "nbody" / "config" / "test1.yaml"

    # Load bodies and simulation parameters
    bodies, sim_params = read_simulation_config(config_path)

    # Run the Euler propagator
    propagator = EulerPropagator(bodies, sim_params)
    states = propagator.propagate()

    # Extract positions at start and end
    pos_start = states[0, :, 0:2]
    pos_end = states[-1, :, 0:2]

    # Since the simulation is one full period, the positions should return to (approximately) initial
    for i in range(len(bodies)):
        assert pos_start[i][0] == pytest.approx(pos_end[i][0], rel=1e-5)
        assert pos_start[i][1] == pytest.approx(pos_end[i][1], rel=1e-5)
