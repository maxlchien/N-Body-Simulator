from __future__ import annotations

import csv
import time
from datetime import datetime

import numba
import numpy as np

from nbody.model.body import Body, Body_nb, convert_to_body_nb

# put numba methods outside class


@numba.jit(nopython=True, parallel=True)
def _compute_accelerations_numba(
    positions: np.ndarray, bodies: list[Body], G: float
) -> np.ndarray:
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
    acc = np.zeros_like(positions)
    n_bodies = len(bodies)
    for i in numba.prange(n_bodies):
        for j in range(
            n_bodies
        ):  # can't parallelize this one as easily because we need to sum over it
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                if r > 0:
                    acc[i] += G * bodies[j].mass * r_vec / (r**3)
    return acc


@numba.jit(nopython=True)
def _get_positions_and_velocities(bodies: list[Body]) -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((len(bodies), 2), dtype=np.float64)
    velocities = np.zeros((len(bodies), 2), dtype=np.float64)
    for i in range(len(bodies)):
        positions[i] = bodies[i].position
        velocities[i] = bodies[i].velocity
    return positions, velocities


@numba.jit(nopython=True)
def _propagate_numba(
    bodies: list[Body], dt: float, num_steps: int, state_vector_size: int, G: float
) -> np.ndarray:
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
    positions, velocities = _get_positions_and_velocities(bodies)

    n_bodies = len(bodies)
    states = np.zeros((num_steps, n_bodies, state_vector_size))

    for step in range(num_steps):
        acc = _compute_accelerations_numba(positions, bodies, G)
        velocities += acc * dt
        positions += velocities * dt
        states[step, :, 0:2] = positions
        states[step, :, 2:4] = velocities
        states[step, :, 4:6] = acc
    return states


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
    use_numba: bool

    def __init__(
        self,
        bodies: list[Body] | list[Body_nb],
        params: dict,
        output_dir: str = "results",
        output_name: str = "nbody_results",
        use_numba=False,
    ):
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
        if use_numba:
            self.bodies = [convert_to_body_nb(body) for body in bodies]
        else:
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
        self.use_numba = use_numba

    def _compute_accelerations_nonumba(
        self,
        positions: np.ndarray,
        start_ns: int | None = None,
        timeout_ns: int | None = None,
    ) -> np.ndarray:
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
        acc = np.zeros_like(positions)
        for i in range(self.n_bodies):
            for j in range(self.n_bodies):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r = np.linalg.norm(r_vec)
                    if r > 0:
                        acc[i] += self.G * self.bodies[j].mass * r_vec / (r**3)
            if (
                timeout_ns is not None
                and time.perf_counter_ns() - start_ns > timeout_ns
            ):
                msg = f"Timed out after {timeout_ns} ns"
                raise TimeoutError(msg)
        return acc

    def _propagate_nonumba(self, timeout_ns: int | None = None) -> np.ndarray:
        """
        Propagate the system forward in time using forward-Euler integration.

        Returns
        -------
        states : np.ndarray, shape (num_steps, n_bodies, 6)
            Simulated state history. For each time step and each body:
            [x, y, vx, vy, ax, ay].
        """
        start_time = time.perf_counter_ns()
        positions = np.array([b.position for b in self.bodies])
        velocities = np.array([b.velocity for b in self.bodies])
        dt = self.dt

        states = np.zeros((self.num_steps, self.n_bodies, self.state_vector_size))

        for step in range(self.num_steps):
            acc = self._compute_accelerations_nonumba(positions, start_time, timeout_ns)
            velocities += acc * dt
            positions += velocities * dt
            states[step, :, 0:2] = positions
            states[step, :, 2:4] = velocities
            states[step, :, 4:6] = acc

        return states

    def propagate(self, timeout_ns: int | None = None):
        if self.use_numba:
            bodies = self.bodies
            dt = self.dt
            num_steps = self.num_steps
            state_vector_size = self.state_vector_size
            G = self.G
            state = _propagate_numba(bodies, dt, num_steps, state_vector_size, G)
        else:
            state = self._propagate_nonumba(timeout_ns)
        self.state = state

    def write_results(self, header_options=None):
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

        if header_options is None:
            header_options = []
        param_row = [
            f"G={self.G}",
            f"dt={self.dt}",
            f"t_total={self.t_total}",
            *header_options,
        ]
        headers = ["Iteration", "Time"] + [
            f"Body {i} {prop} ({axis})"
            for i in range(1, self.n_bodies + 1)
            for prop in ("position", "velocity", "acceleration")
            for axis in ("x", "y")
        ]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.output_dir / f"{self.output_name}_{timestamp}.csv"

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
    propagator.propagate()
    propagator.write_results()
    return propagator.states
