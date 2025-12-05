from __future__ import annotations

import os
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from nbody.model.body import Body


class EulerPropagator:
    bodies: list[Body]
    G: float
    dt: float
    t_total: float
    num_steps: int
    output_name: str
    n_bodies: int
    state_vector_size: int
    states: np.ndarray

    def __init__(self, bodies, params, output_dir="results", output_name="nbody_results"):
        """
        Initialize the EulerPropagator with bodies and simulation parameters.

        Parameters
        ----------
        bodies : list of Body
            List of Body objects containing initial positions, velocities, mass, and name.
        params : dict
            Simulation parameters. Expected keys:
                'G' : float
                    Gravitational constant. Default is 6.6743e-11.
                'dt' : float
                    Time step for numerical integration. Default is 1.0
                't_total' : float
                    Total simulation time. Default is 1000.0
        output_name : str, optional
            Base name for the output CSV file. Timestamp will be appended.

        Returns
        -------
        None
        """
        self.bodies = bodies
        self.G = params.get("G", 6.6743e-11)
        self.dt = params.get("dt", 1.0)
        self.t_total = params.get("t_total", 1000.0)
        self.num_steps = int(self.t_total / self.dt)
        self.output_dir = output_dir
        self.output_name = output_name

        self.n_bodies = len(bodies)
        self.state_vector_size = 6  # x, y, vx, vy, ax, ay

        # 3D array to store states: time × body × state_vector
        self.states = np.zeros((self.num_steps, self.n_bodies, self.state_vector_size))

    def compute_accelerations(self, positions):
        """
        Compute gravitational accelerations on all bodies at the current positions due to mutual attraction.

        Parameters
        ----------
        positions : np.ndarray, shape (N, 2)
            Positions of all N bodies in 2D space (x, y).

        Returns
        -------
        acc : np.ndarray, shape (N, 2)
            Accelerations (ax, ay) for each body.

        Raises
        ------
        ValueError
            If `positions` is not shape (N, 2).

        Notes
        -----
        Uses Newtonian point-mass gravity.
        """

        acc = np.zeros((self.n_bodies, 2), dtype=float)
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i == j:
                    continue
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                if r > 0.0:
                    acc[i] += self.G * (self.bodies[j].mass) * r_vec / (r**3)
        return acc


    def propagate(self):
        """
        Propagate the system forward in time using forward-Euler integration.

        Returns
        -------
        states : np.ndarray, shape (num_steps, n_bodies, 6)
            Simulated state history. For each time step and each body:
            [x, y, vx, vy, ax, ay].

        Notes
        -----
        This modifies internal state arrays in-place.
        """
        # force float dtype explicitly
        positions = np.array([b.position for b in self.bodies], dtype=float)   # shape (N,2)
        velocities = np.array([b.velocity for b in self.bodies], dtype=float)  # shape (N,2)
        dt = float(self.dt)

        # pre-allocate states (already done in __init__)
        # store initial accelerations
        acc = self.compute_accelerations(positions)

        # store initial state
        self.states[0, :, 0:2] = positions
        self.states[0, :, 2:4] = velocities
        self.states[0, :, 4:6] = acc

        # --- Velocity Verlet integrator (symplectic, good for orbits) ---
        for step in range(1, self.num_steps):
            # x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt^2
            positions = positions + velocities * dt + 0.5 * acc * (dt**2)

            # compute new acceleration at x_{n+1}
            new_acc = self.compute_accelerations(positions)

            # v_{n+1} = v_n + 0.5*(a_n + a_{n+1})*dt
            velocities = velocities + 0.5 * (acc + new_acc) * dt

            # advance acceleration
            acc = new_acc

            # store state for this step
            self.states[step, :, 0:2] = positions
            self.states[step, :, 2:4] = velocities
            self.states[step, :, 4:6] = acc

        return self.states

    def write_results(self):
        """
        Write the state history to a timestamped CSV file.

        Returns
        -------
        filename : str
            Name of the written CSV file.

        Notes
        -----
        CSV format:
            row 0 : simulation parameters (G, dt, t_total)
            row 1 : column headers
            subsequent rows : step,time,body,x,y,vx,vy,ax,ay

        Examples
        --------
        >>> fname = propagator.write_results()
        """
        dt = self.dt
        num_steps = self.num_steps

        param_row = [f"G={self.G}", f"dt={self.dt}", f"t_total={self.t_total}"]
        headers = ["Iteration", "Time"] + [
            f"Body {i} {prop} ({axis})"
            for i in range(1, self.n_bodies + 1)
            for prop in ("position", "velocity", "acceleration")
            for axis in ("x", "y")
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        filename = Path(self.output_dir) / f"{self.output_name}_{timestamp}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(param_row)  # First row: simulation parameters
            writer.writerow(headers)  # Second row: headers

            for step in range(num_steps):
                row = [
                    step,
                    step * dt,
                ]
                for b in range(self.n_bodies):
                    row.extend(self.states[step, b, :])
                writer.writerow(row)

        print(f"Results written to {filename}")
        return filename


def run_simulation(bodies, params, output_name="nbody_results"):
    """
    Run an N-body Euler simulation and write results to CSV.

    Parameters
    ----------
    bodies : list of Body
        List of Body objects with initial state.
    params : dict
        Dictionary with simulation parameters (G, dt, t_total).
    output_name : str, optional
        Base filename for output CSV.

    Returns
    -------
    states : np.ndarray, shape (num_steps, n_bodies, 6)
        Final propagated state history.

    Examples
    --------
    > states = run_simulation([Body(...), Body(...)], {'dt':0.01,'t_total':10})
    """
    propagator = EulerPropagator(bodies, params, output_name=output_name)
    states = propagator.propagate()
    propagator.write_results()
    return states
