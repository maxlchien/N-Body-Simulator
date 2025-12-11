from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import yaml

from nbody.model.body import Body
from nbody.utility import preprocess
from nbody.utility.preprocess import read_simulation_config


def test_read_simulation_config(tmp_path: Path):
    yaml_content = textwrap.dedent(
        """
        simulation:
          name: "Earth-Moon Simulation"
          engine: "euler"
          dt: 3600
          t_total: 2592000
          output_dir: "results"

        bodies:
          list:
            - id: "Earth"
              mass: 5.972e24
              radius: 6371000.0
              position: [0.0, 0.0]
              velocity: [0.0, 0.0]
            - id: "Moon"
              mass: 7.348e22
              radius: 1737000.0
              position: [384400000.0, 0.0]
              velocity: [0.0, 1022.0]

        preprocess:
          normalize_positions: false
          normalize_velocities: false
        """
    )

    cfg_path = tmp_path / "earth_moon.yaml"
    cfg_path.write_text(yaml_content)

    bodies, sim_params = read_simulation_config(str(cfg_path))

    assert sim_params["name"] == "Earth-Moon Simulation"
    assert sim_params["engine"] == "euler"
    assert sim_params["dt"] == 3600
    assert sim_params["t_total"] == 2_592_000
    assert sim_params["output_dir"] == "results"

    assert "normalize_positions" not in sim_params
    assert "normalize_velocities" not in sim_params

    assert len(bodies) == 2
    assert all(isinstance(b, Body) for b in bodies)

    earth, moon = bodies

    assert earth.id == "Earth"
    assert earth.mass == pytest.approx(5.972e24)
    assert earth.radius == pytest.approx(6_371_000.0)
    np.testing.assert_allclose(earth.position, np.array([0.0, 0.0]))
    np.testing.assert_allclose(earth.velocity, np.array([0.0, 0.0]))
    assert earth.position.shape == (2,)
    assert earth.velocity.shape == (2,)

    assert moon.id == "Moon"
    assert moon.mass == pytest.approx(7.348e22)
    assert moon.radius == pytest.approx(1_737_000.0)
    np.testing.assert_allclose(moon.position, np.array([384_400_000.0, 0.0]))
    np.testing.assert_allclose(moon.velocity, np.array([0.0, 1022.0]))
    assert moon.position.shape == (2,)
    assert moon.velocity.shape == (2,)


def test_generate_bodies(monkeypatch, tmp_path: Path):
    yaml_content = textwrap.dedent(
        """
        simulation:
          name: "Earth-Moon Simulation"
          engine: "euler"
          dt: 3600
          t_total: 2592000
          n_bodies: 3
        """
    )
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml_content)

    generated = [
        Body(
            id=f"body_{i}",
            mass=1.0,
            radius=1.0,
            position=np.zeros(2),
            velocity=np.zeros(2),
        )
        for i in range(3)
    ]

    def fake_generate_random_bodies(n):
        assert n == 3
        return generated

    monkeypatch.setattr(
        preprocess, "generate_random_bodies", fake_generate_random_bodies
    )

    bodies, sim_params = read_simulation_config(str(cfg_path))

    assert bodies is generated
    assert sim_params["n_bodies"] == 3
    assert sim_params["engine"] == "euler"
    assert sim_params["dt"] == 3600
    assert sim_params["t_total"] == 2_592_000


def test_filenotfounderror(tmp_path: Path):
    non_existing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        read_simulation_config(str(non_existing))


def test_yamlerror(tmp_path: Path):
    bad_yaml = "jn0oijinpib: [ "
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(bad_yaml)

    with pytest.raises(yaml.YAMLError) as excinfo:
        read_simulation_config(str(cfg_path))

    assert isinstance(excinfo.value, yaml.YAMLError)
