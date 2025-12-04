import yaml
from pathlib import Path

from nbody.engine.euler_propagator import EulerPropagator
from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.model.body import Body
from nbody.utility.preprocess import read_simulation_config


def main():
    # Path to simulation YAML
    config_path = Path(__file__).parent / "config" / "simulation.yaml"

    # Load bodies and simulation parameters from YAML
    bodies, sim_params = read_simulation_config(config_path)

    # Simulation settings
    timestep = sim_params.get("timestep")
    duration = sim_params.get("duration")
    engine_type = sim_params.get("engine", "euler")
    output_dir_cfg = sim_params.get("output_dir", "results")
    output_dir = Path(__file__).parent / output_dir_cfg
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check required fields
    if timestep is None or duration is None:
        raise ValueError("Missing 'timestep' or 'duration' in simulation parameters")

    # Select propagator
    if engine_type == "euler":
        propagator = EulerPropagator(bodies, sim_params, output_dir)
    elif engine_type == "barnes_hut":
        propagator = BarnesHutPropagator(bodies, sim_params)
    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    # Run simulation
    propagator.propagate()
    propagator.write_results()

    print(f"Simulation complete. Results saved in {output_dir}")


if __name__ == "__main__":
    main()
