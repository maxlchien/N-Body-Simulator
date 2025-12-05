from __future__ import annotations

from pathlib import Path

from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.preprocess import read_simulation_config

USE_NUMBA = (
    False  # not helpful until the number of bodies is large due to compilation overhead
)


def main():
    # Path to simulation YAML
    config_path = Path(__file__).parent / "config" / "simulation.yaml"

    # Load bodies and simulation parameters from YAML
    bodies, sim_params = read_simulation_config(config_path)

    # Simulation settings
    engine_type = sim_params.get("engine", "euler")
    output_dir_cfg = sim_params.get("output_dir", "results")
    output_dir = Path(__file__).parent / output_dir_cfg
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check required fields
    if sim_params.get("timestep") is None or sim_params.get("duration") is None:
        msg = "Missing 'timestep' or 'duration' in simulation parameters"
        raise ValueError(msg)

    # Select propagator
    if engine_type == "euler":
        propagator = EulerPropagator(bodies, sim_params, output_dir)
    elif engine_type == "barnes_hut":
        propagator = BarnesHutPropagator(bodies, sim_params, output_dir)
    else:
        msg = f"Unknown engine: {engine_type}"
        raise ValueError(msg)

    # Run simulation
    propagator.propagate()
    propagator.write_results()

    print(f"Simulation complete. Results saved in {output_dir}")


if __name__ == "__main__":
    main()
