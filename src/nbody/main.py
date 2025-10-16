from EulerPropagator import EulerPropagator
from setup import Body

def main():
    # Dummy bodies for replacement/showing use case
    bodies = [
        Body(name="Earth", x=0.0, y=0.0, vx=0.0, vy=0.0, mass=5.97e24),
        Body(name="Moon", x=384_400_000.0, y=0.0, vx=0.0, vy=1022.0, mass=7.35e22)
    ]

    # Simulation parameters sample
    params = {
        "G": 6.6743e-11,  # real gravitational constant
        "dt": 60.0,       # 1 minute timestep
        "t_total": 3600.0 # 1 hour total for testing
    }

    # Create propagator and run
    propagator = EulerPropagator(bodies, params)
    states = propagator.propagate()
    filename = propagator.write_results()
    print(f"Simulation complete. Results written to {filename}")

if __name__ == "__main__":
    main()
