from __future__ import annotations

from typing import Any

import yaml
from body import Body

# the following class is commented out because the module has not been developed yet
# from generate_bodies import generate_random_bodies


def read_simulation_config(filename: str) -> tuple[list[Body], dict[str, Any]]:
    """Read bodies from yaml configuration file. If no bodies are provided,
    generate random ones using an external generator."""

    with open(filename) as f:
        data = yaml.safe_load(f) or {}

    sim_params = data.get("simulation", {})

    bodies_data = data.get("bodies", None)

    if not bodies_data:
        # In case no bodies are provided: generate random ones
        n_bodies = sim_params.get("n_bodies", 5)
        print(f"No bodies in input file, generating {n_bodies} random ones.")
        # comment this out when module actually gets developed
        # bodies = generate_random_bodies(n_bodies)
    else:
        # Bodies provided: construct from YAML
        bodies = [
            Body(
                name=b.get("id", f"Body{i}"),
                mass=b["mass"],
                radius=b["radius"],
                position=b["pos_arr"],
                velocity=b["v_arr"],
            )
            for i, b in enumerate(bodies_data)
        ]

    return bodies, sim_params


if __name__ == "__main__":
    bodies, sim_params = read_simulation_config("simulation.yaml")
    print("Simulation parameters:")
    print(sim_params)
    print("\nBodies:")
    for body in bodies:
        print(body)
