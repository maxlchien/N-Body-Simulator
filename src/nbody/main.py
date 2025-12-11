from __future__ import annotations

import argparse
from pathlib import Path

from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.engine.euler_propagator import EulerPropagator
from nbody.utility.plotter import plotter
from nbody.utility.preprocess import read_simulation_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--colors",
        type=str,
        default=None,
        nargs="+",
        help="List of colors for plotting each body",
    )
    parser.add_argument(
        "--plotname",
        type=str,
        default="results/nbody_animation.mp4",
        help="Name of the output video file (must end with .mp4)",
    )
    parser.add_argument("--numba", action="store_true")
    parser.add_argument("-i", "--input", default="config/simulation.yaml")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        help="Directory to save output results",
    )
    args = parser.parse_args()

    visualize = args.visualize
    input_file = args.input
    USE_NUMBA = args.numba
    # Path to simulation YAML
    config_path = Path(input_file)
    # Path to output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load bodies and simulation parameters from YAML
    bodies, sim_params = read_simulation_config(config_path)
    # Simulation settings
    engine_type = sim_params.get("engine", "euler")
    if output_dir is not None:
        pass
    if output_dir is None:
        output_dir_cfg = sim_params.get("output_dir", "results")
        output_dir = Path(output_dir_cfg)
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
    filename = propagator.write_results()

    print(f"Simulation complete. Results saved in {output_dir}")

    if visualize:
        plotter(filename, args.colors, args.plotname)


if __name__ == "__main__":
    main()
