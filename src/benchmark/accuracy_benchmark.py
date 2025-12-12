from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.engine.euler_propagator import EulerPropagator
from nbody.model.body import Body

TIMEOUT = 2e11


def get_error_trace(
    bodies: list[Body],
    sim_params: dict,
    engine_type: type,
    ref: np.ndarray,
    theta: int | None = None,
) -> np.ndarray:
    sim_params = sim_params.copy()
    bodies = deepcopy(bodies)  # because barnes-hut modifies the bodies
    sim_params["theta"] = theta
    if engine_type is EulerPropagator:
        engine = EulerPropagator(bodies, sim_params)
    elif engine_type is BarnesHutPropagator:
        engine = BarnesHutPropagator(bodies, sim_params)
    else:
        msg = f"Engine type not recognized: {engine_type}"
        raise ValueError(msg)
    simulated_states = engine.propagate()[:, :, :2]  # get position only
    errors = simulated_states - ref
    return np.array([np.mean(np.linalg.norm(err, axis=1), axis=0) for err in errors])


matrix = [
    (EulerPropagator, None),
    (BarnesHutPropagator, 0),
    (BarnesHutPropagator, 0.5),
    (BarnesHutPropagator, 1),
    (BarnesHutPropagator, 1.5),
]

sim_params = {"dt": 0.01, "t_total": 12.5663706144, "G": 1.0}
num_steps = int(sim_params.get("t_total") / sim_params.get("dt"))

num_bodies = 5
ref = np.zeros((num_steps, num_bodies, 2))
omega = 2 * np.pi * np.linspace(0, 1, num_steps)
for i in range(num_bodies):
    ref[:, i, 0] = np.cos(omega + 2 * np.pi * i / num_bodies)
    ref[:, i, 1] = np.sin(omega + 2 * np.pi * i / num_bodies)
print("reference trace computed")

angular = np.sqrt(
    np.sum([1 / 4 / (np.sin(np.pi * i / num_bodies)) for i in range(1, num_bodies)])
)
print("forces computed")
bodies = [
    Body(
        "Body {i}",
        1.0,
        1.0,
        position=np.array(
            [np.cos(2 * np.pi * i / num_bodies), np.sin(2 * np.pi * i / num_bodies)]
        ),
        velocity=angular
        * np.array(
            [
                np.cos(2 * np.pi * i / num_bodies + np.pi / 2),
                np.sin(2 * np.pi * i / num_bodies + np.pi / 2),
            ]
        ),
    )
    for i in range(num_bodies)
]
print("bodies constructed")

errors = [
    get_error_trace(bodies, sim_params, engine_type, ref, theta)
    for engine_type, theta in matrix
]
time = np.arange(1, num_steps + 1) * sim_params.get("dt")

for i, config in enumerate(matrix):
    engine_type, theta = config
    error = errors[i]
    label = "Euler" if engine_type is EulerPropagator else f"Barnes-Hut, theta={theta}"
    plt.plot(time, error, label=label)
plt.xlabel("Time into simulation (seconds)")
plt.ylabel("Absolute Deviation from Analytic Solution")
plt.title("Propagation Step Accuracy Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("src/benchmark/accuracy_benchmark.png")
