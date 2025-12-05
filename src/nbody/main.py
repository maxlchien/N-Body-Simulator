from __future__ import annotations

import argparse
from pathlib import Path

from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.preprocess import read_simulation_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--numba", action="store_true")
    parser.add_argument("-i", "--input", default="config/simulation.yaml")
    args = parser.parse_args()

    visualize = args.visualize
    input_file = args.input
    USE_NUMBA = args.numba
    # Path to simulation YAML
    config_path = Path(__file__).parent / input_file

    # Load bodies and simulation parameters from YAML
    bodies, sim_params = read_simulation_config(config_path)
    # Simulation settings
    engine_type = sim_params.get("engine", "euler")
    output_dir_cfg = sim_params.get("output_dir", "results")
    output_dir = Path(__file__).parent / output_dir_cfg
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check required fields
    if sim_params.get("dt") is None or sim_params.get("t_total") is None:
        msg = "Missing 'dt' or 't_total' in simulation parameters"
        raise ValueError(msg)

    # Select propagator
    if engine_type == "euler":
        propagator = EulerPropagator(
            bodies, sim_params, output_dir, use_numba=USE_NUMBA
        )
    elif engine_type == "barnes_hut":
        if USE_NUMBA:
            msg = "Option --numba is incompatible engine type barnes_hut"
            raise ValueError(msg)
        propagator = BarnesHutPropagator(bodies, sim_params, output_dir)
    else:
        msg = f"Unknown engine: {engine_type}"
        raise ValueError(msg)

    # Run simulation
    propagator.propagate()
    propagator.write_results()

    print(f"Simulation complete. Results saved in {output_dir}")

    if visualize:
        ...
        #### integrate visualization module here


if __name__ == "__main__":
    main()
