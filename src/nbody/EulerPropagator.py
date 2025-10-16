import numpy as np
import pandas as pd
from datetime import datetime
from setup import Body

class EulerPropagator:

    def __init__(self, bodies, params, output_name="nbody_results"):
        """
        Initialize the propagator with body objects and simulation parameters.

        Args:
            bodies (list): A list of Body objects with initial states.
            params (dict): Dictionary containing keys 'G', 'dt', and 't_total'.
            output_name (str, optional): Base name for the output CSV file. Defaults to 'nbody_results'.

        Returns:
            None
        """
        self.bodies = bodies
        self.G = params.get("G", 6.6743e-11)
        self.dt = params.get("dt", 1.0)
        self.t_total = params.get("t_total", 1000.0)
        self.num_steps = int(self.t_total / self.dt)
        self.output_name = output_name

        self.n_bodies = len(bodies)
        self.state_vector_size = 4  # x, y, vx, vy

        # 3D array to store states: time × body × state_vector
        self.states = np.zeros((self.num_steps, self.n_bodies, self.state_vector_size))

    def compute_accelerations(self, positions):
        """
        Compute accelerations on all bodies due to mutual gravitational attraction.

        Args:
            positions (np.ndarray): Nx2 array of x and y positions for N bodies.

        Returns:
            np.ndarray: Nx2 array of accelerations for each body in x and y directions.
        """
        acc = np.zeros_like(positions)
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec)
                    if r > 0:
                        acc[i] += self.G * self.bodies[j].mass * r_vec / (r**3)
        return acc

    def propagate(self):
        """
        Propagate the system forward in time using the Euler (forward difference) method.

        Returns:
            np.ndarray: 3D array of shape (num_steps, n_bodies, 4) containing x, y, vx, vy.
        """
        positions = np.array([[b.x, b.y] for b in self.bodies])
        velocities = np.array([[b.vx, b.vy] for b in self.bodies])
        dt = self.dt

        for step in range(self.num_steps):
            acc = self.compute_accelerations(positions)
            velocities += acc * dt
            positions += velocities * dt
            self.states[step, :, 0:2] = positions
            self.states[step, :, 2:4] = velocities

        return self.states



    def write_results(self):
        dt = self.dt
        num_steps = self.num_steps

        param_row = [f"G={self.G}", f"dt={self.dt}", f"t_total={self.t_total}"]
        headers = ["step", "time", "body", "x", "y", "vx", "vy"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_name}_{timestamp}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(param_row)  # First row: simulation parameters
            writer.writerow(headers)    # Second row: headers

            for step in range(num_steps):
                for b in range(self.n_bodies):
                    row = [
                        step,
                        step * dt,
                        self.bodies[b].name,
                        self.states[step, b, 0],
                        self.states[step, b, 1],
                        self.states[step, b, 2],
                        self.states[step, b, 3]
                    ]
                    writer.writerow(row)

        print(f"Results written to {filename}")
        return filename

def run_simulation(bodies, params, output_name="nbody_results"):
    """
    Run an N-body simulation using Euler propagation and write results to CSV.

    Args:
        bodies (list): A list of Body objects.
        params (dict): Dictionary of parameters including G, dt, and t_total.
        output_name (str, optional): Base name for CSV output.

    Returns:
        np.ndarray: 3D array of shape (num_steps, n_bodies, 4) containing x, y, vx, vy.
    """
    propagator = EulerPropagator(bodies, params, output_name=output_name)
    states = propagator.propagate()
    propagator.write_results()
    return states
