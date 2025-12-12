from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.preprocess import read_simulation_config


def test_three_body_equilateral_triangle_orbit_euler():
    """Test a symmetric three-body equilateral triangle orbit using a known YAML config."""

    config_path = Path(__file__).parent.parent / "nbody" / "config" / "3body_euler.yaml"

    bodies, sim_params = read_simulation_config(config_path)

    propagator = EulerPropagator(bodies, sim_params)
    states = propagator.propagate()

    pos_start = states[0, :, 0:2]
    pos_1_3 = states[states.shape[0] // 3, :, 0:2]
    pos_2_3 = states[(2 * states.shape[0]) // 3, :, 0:2]
    pos_end = states[-1, :, 0:2]

    #  initial configuration checks
    m = np.array([b.mass for b in bodies], dtype=float)
    com_start = (pos_start * m[:, None]).sum(axis=0) / m.sum()
    assert com_start[0] == pytest.approx(0.0, abs=1e-6)
    assert com_start[1] == pytest.approx(0.0, abs=1e-6)

    r_start = np.linalg.norm(pos_start, axis=1)
    for i in range(len(bodies)):
        assert r_start[i] == pytest.approx(1.0, abs=1e-6)

    def dists(p):
        d01 = np.linalg.norm(p[0] - p[1])
        d02 = np.linalg.norm(p[0] - p[2])
        d12 = np.linalg.norm(p[1] - p[2])
        return d01, d02, d12

    d01_0, d02_0, d12_0 = dists(pos_start)
    assert d01_0 == pytest.approx(np.sqrt(3.0), abs=1e-6)
    assert d02_0 == pytest.approx(np.sqrt(3.0), abs=1e-6)
    assert d12_0 == pytest.approx(np.sqrt(3.0), abs=1e-6)

    # --- one full period: return near start
    for i in range(len(bodies)):
        assert pos_start[i][0] == pytest.approx(pos_end[i][0], abs=5e-2)
        assert pos_start[i][1] == pytest.approx(pos_end[i][1], abs=5e-2)

    # checkpoints: bodies "swap" vertices
    # At T/3: A->B_start, B->C_start, C->A_start
    assert pos_1_3[0][0] == pytest.approx(pos_start[1][0], abs=5e-2)
    assert pos_1_3[0][1] == pytest.approx(pos_start[1][1], abs=5e-2)
    assert pos_1_3[1][0] == pytest.approx(pos_start[2][0], abs=5e-2)
    assert pos_1_3[1][1] == pytest.approx(pos_start[2][1], abs=5e-2)
    assert pos_1_3[2][0] == pytest.approx(pos_start[0][0], abs=5e-2)
    assert pos_1_3[2][1] == pytest.approx(pos_start[0][1], abs=5e-2)

    # At 2T/3: A->C_start, B->A_start, C->B_start
    assert pos_2_3[0][0] == pytest.approx(pos_start[2][0], abs=5e-2)
    assert pos_2_3[0][1] == pytest.approx(pos_start[2][1], abs=5e-2)
    assert pos_2_3[1][0] == pytest.approx(pos_start[0][0], abs=5e-2)
    assert pos_2_3[1][1] == pytest.approx(pos_start[0][1], abs=5e-2)
    assert pos_2_3[2][0] == pytest.approx(pos_start[1][0], abs=5e-2)
    assert pos_2_3[2][1] == pytest.approx(pos_start[1][1], abs=5e-2)

    # Shape stays approximately equilateral at checkpoints
    d01_1, d02_1, d12_1 = dists(pos_1_3)
    assert d01_1 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d02_1 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d12_1 == pytest.approx(np.sqrt(3.0), abs=5e-2)

    d01_2, d02_2, d12_2 = dists(pos_2_3)
    assert d01_2 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d02_2 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d12_2 == pytest.approx(np.sqrt(3.0), abs=5e-2)


def test_three_body_equilateral_triangle_orbit_barneshut():
    """Test a symmetric three-body equilateral triangle orbit using a known YAML config."""

    config_path = (
        Path(__file__).parent.parent / "nbody" / "config" / "3body_barneshut.yaml"
    )

    bodies, sim_params = read_simulation_config(config_path)

    propagator = BarnesHutPropagator(bodies, sim_params)
    states = propagator.propagate()

    pos_start = states[0, :, 0:2]
    pos_1_3 = states[states.shape[0] // 3, :, 0:2]
    pos_2_3 = states[(2 * states.shape[0]) // 3, :, 0:2]
    pos_end = states[-1, :, 0:2]

    #  initial configuration checks
    m = np.array([b.mass for b in bodies], dtype=float)
    com_start = (pos_start * m[:, None]).sum(axis=0) / m.sum()
    assert com_start[0] == pytest.approx(0.0, abs=1e-6)
    assert com_start[1] == pytest.approx(0.0, abs=1e-6)

    r_start = np.linalg.norm(pos_start, axis=1)
    for i in range(len(bodies)):
        assert r_start[i] == pytest.approx(1.0, abs=1e-6)

    def dists(p):
        d01 = np.linalg.norm(p[0] - p[1])
        d02 = np.linalg.norm(p[0] - p[2])
        d12 = np.linalg.norm(p[1] - p[2])
        return d01, d02, d12

    d01_0, d02_0, d12_0 = dists(pos_start)
    assert d01_0 == pytest.approx(np.sqrt(3.0), abs=1e-6)
    assert d02_0 == pytest.approx(np.sqrt(3.0), abs=1e-6)
    assert d12_0 == pytest.approx(np.sqrt(3.0), abs=1e-6)

    # --- one full period: return near start
    for i in range(len(bodies)):
        assert pos_start[i][0] == pytest.approx(pos_end[i][0], abs=5e-2)
        assert pos_start[i][1] == pytest.approx(pos_end[i][1], abs=5e-2)

    # checkpoints: bodies "swap" vertices
    # At T/3: A->B_start, B->C_start, C->A_start
    assert pos_1_3[0][0] == pytest.approx(pos_start[1][0], abs=5e-2)
    assert pos_1_3[0][1] == pytest.approx(pos_start[1][1], abs=5e-2)
    assert pos_1_3[1][0] == pytest.approx(pos_start[2][0], abs=5e-2)
    assert pos_1_3[1][1] == pytest.approx(pos_start[2][1], abs=5e-2)
    assert pos_1_3[2][0] == pytest.approx(pos_start[0][0], abs=5e-2)
    assert pos_1_3[2][1] == pytest.approx(pos_start[0][1], abs=5e-2)

    # At 2T/3: A->C_start, B->A_start, C->B_start
    assert pos_2_3[0][0] == pytest.approx(pos_start[2][0], abs=5e-2)
    assert pos_2_3[0][1] == pytest.approx(pos_start[2][1], abs=5e-2)
    assert pos_2_3[1][0] == pytest.approx(pos_start[0][0], abs=5e-2)
    assert pos_2_3[1][1] == pytest.approx(pos_start[0][1], abs=5e-2)
    assert pos_2_3[2][0] == pytest.approx(pos_start[1][0], abs=5e-2)
    assert pos_2_3[2][1] == pytest.approx(pos_start[1][1], abs=5e-2)

    # Shape stays approximately equilateral at checkpoints
    d01_1, d02_1, d12_1 = dists(pos_1_3)
    assert d01_1 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d02_1 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d12_1 == pytest.approx(np.sqrt(3.0), abs=5e-2)

    d01_2, d02_2, d12_2 = dists(pos_2_3)
    assert d01_2 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d02_2 == pytest.approx(np.sqrt(3.0), abs=5e-2)
    assert d12_2 == pytest.approx(np.sqrt(3.0), abs=5e-2)
