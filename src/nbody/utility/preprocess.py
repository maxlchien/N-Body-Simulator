from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from nbody.model.body import Body
from nbody.utility.generate_bodies import generate_random_bodies

# The following import is commented out because the module has not been developed yet.
# from generate_bodies import generate_random_bodies


def read_simulation_config(filename: str) -> tuple[list[Body], dict[str, Any]]:
    """
    Read simulation config and body data from a YAML file.
    If no bodies are provided, generate random ones using an external generator that will be
    implemented soon.

    Parameters
    ----------
    filename : str
        Path to the YAML configuration file containing the simulation setup.

    Returns
    -------
    bodies : list of Body instances
        A list of Body objects initialized with parameters from the YAML file.
    sim_params : dict of {str: Any}
        A dictionary containing parameters such as
        timestep, duration, or number of bodies.

    Raises
    ------
    FileNotFoundError
        If the YAML file cannot be found.
    yaml.YAMLError
        If the YAML file cannot be parsed.
    KeyError
        If YAML file is missing Body fields.

    Examples
    --------
    Read configuration and print summary:

    >>> from preprocess import read_simulation_config
    >>> bodies, sim_params = read_simulation_config("testfile.yaml")
    >>> print(sim_params)
    {'timestep': 0.01, 'duration': 10.0}
    >>> for body in bodies:
    ...     print(body.id, body.mass)

    Notes
    -----
    The YAML file must define either:
      - a `bodies` list with physical parameters for each object, or
      - a `simulation` block with `n_bodies` for automatic generation
        (when `generate_random_bodies` is implemented).
    """

    with open(filename) as f:
        data = yaml.safe_load(f) or {}

    sim_params = data.get("simulation", {})
    bodies_data = data.get("bodies", {}).get("list", None)

    if not bodies_data:
        n_bodies = sim_params.get("n_bodies", 5)
        print(f"No bodies in input file, generating {n_bodies} random ones.")
        bodies = generate_random_bodies(n_bodies)
    else:
        bodies = [
            Body(
                id=b.get("id", f"Body{i}"),
                mass=float(b["mass"]),
                radius=float(b["radius"]),
                position=np.array(b["position"], dtype=float),
                velocity=np.array(b["velocity"], dtype=float),
            )
            for i, b in enumerate(bodies_data)
        ]

    return bodies, sim_params


if __name__ == "__main__":
    config_file = Path("config/simulation.yaml")
    bodies, sim_params = read_simulation_config(config_file)
    print("Simulation parameters:")
    print(sim_params)
    print("\nBodies:")
    for body in bodies:
        print(body)
