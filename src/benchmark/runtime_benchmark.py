from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from nbody.engine.barnes_hut import BarnesHutPropagator
from nbody.engine.euler_propagator import EulerPropagator
from nbody.model.body import Body, Body_nb, convert_to_body_nb
from nbody.utility.generate_bodies import generate_random_bodies

TIMEOUT = 2e11


def test_euler(bodies: list[Body] | list[Body_nb], USE_NUMBA: bool):
    # Simulation parameters sample
    params = {
        "G": 6.6743e-11,  # real gravitational constant
        "dt": 60.0,  # 1 minute timestep
        "t_total": 3600.0,  # 1 hour total for testing
    }

    # Create propagator and run
    propagator = EulerPropagator(bodies, params, use_numba=USE_NUMBA)

    if USE_NUMBA:
        propagator.propagate()  # get compilation out of the way, this will get overwritten with the same data

    # time the calculation portion
    start_time = time.perf_counter_ns()
    try:
        propagator.propagate(timeout_ns=TIMEOUT)
    except TimeoutError:
        return np.nan
    end_time = time.perf_counter_ns()
    return end_time - start_time


def test_barnes_hut(bodies: list[Body] | list[Body_nb], theta: float):
    # Simulation parameters sample
    params = {
        "G": 6.6743e-11,  # real gravitational constant
        "dt": 60.0,  # 1 minute timestep
        "theta": theta,
        "t_total": 3600.0,  # 1 hour total for testing
    }

    # Create propagator and run
    propagator = BarnesHutPropagator(bodies, params)
    # time the calculation portion
    start_time = time.perf_counter_ns()
    try:
        propagator.propagate(timeout_ns=TIMEOUT)
    except TimeoutError:
        return np.nan
    end_time = time.perf_counter_ns()
    return end_time - start_time


# test_numbers = [1, 2, 5, 10, 25, 50, 100, 250]
test_numbers = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
numba_times = []
euler_times = []
barnes1_times = []
barnes5_times = []
barnes2_times = []

all_bodies = generate_random_bodies(max(test_numbers))
numba_bodies = [convert_to_body_nb(body) for body in all_bodies]
euler_TIMED_OUT = False
barnes1_TIMED_OUT = False
barnes5_TIMED_OUT = False
barnes2_TIMED_OUT = False
for n_bodies in test_numbers:
    bodies = all_bodies[:n_bodies]
    nb_bodies = numba_bodies[:n_bodies]

    numba_time = test_euler(nb_bodies, True)
    numba_times.append(numba_time)

    if euler_TIMED_OUT:
        euler_times.append(np.nan)
    else:
        euler_time = test_euler(bodies, False)
        euler_times.append(euler_time)
        if not np.isfinite(euler_time):
            euler_TIMED_OUT = True

    if barnes1_TIMED_OUT:
        barnes1_times.append(np.nan)
    else:
        barnes1_time = test_barnes_hut(bodies, theta=1)
        barnes1_times.append(barnes1_time)
        if not np.isfinite(barnes1_time):
            barnes1_TIMED_OUT = True
    if barnes5_TIMED_OUT:
        barnes5_times.append(np.nan)
    else:
        barnes5_time = test_barnes_hut(bodies, theta=0.5)
        barnes5_times.append(barnes5_time)
        if not np.isfinite(barnes5_time):
            barnes5_TIMED_OUT = True
    if barnes2_TIMED_OUT:
        barnes2_times.append(np.nan)
    else:
        barnes2_time = test_barnes_hut(bodies, theta=0.2)
        barnes2_times.append(barnes2_time)
        if not np.isfinite(barnes2_time):
            barnes2_TIMED_OUT = True
    print(
        f"Simulated {n_bodies} bodies in {numba_time} ns with numba, {euler_time} ns without numba, {barnes1_time} ns with Barnes-Hut (theta)=1, {barnes5_time} ns with Barnes-Hut (theta=0.5), {barnes2_time} ns with Barnes-Hut (theta=0.2)"
    )

test_numbers = np.array(test_numbers)
# convert to ms
numba_times = np.array(numba_times) / 1e6
euler_times = np.array(euler_times) / 1e6
barnes1_times = np.array(barnes1_times) / 1e6
barnes5_times = np.array(barnes5_times) / 1e6
barnes2_times = np.array(barnes2_times) / 1e6

euler_nanmask = np.isfinite(euler_times)
barnes1_nanmask = np.isfinite(barnes1_times)
barnes5_nanmask = np.isfinite(barnes5_times)
barnes2_nanmask = np.isfinite(barnes2_times)

# get scaling laws

# numba law

numba_regression = LinearRegression()
numba_regression.fit(
    np.log(test_numbers).reshape(-1, 1),
    np.log(numba_times),
)

numba_coeff, numba_pow = numba_regression.intercept_, *numba_regression.coef_

# no numba law

euler_regression = LinearRegression()
euler_regression.fit(
    np.log(test_numbers[euler_nanmask]).reshape(-1, 1),
    np.log(euler_times[euler_nanmask]),
)

euler_coeff, euler_pow = euler_regression.intercept_, *euler_regression.coef_

# barnes-hut law

barnes1_regression = LinearRegression()
barnes1_regression.fit(
    np.log(test_numbers[barnes1_nanmask]).reshape(-1, 1),
    np.log(barnes1_times[barnes1_nanmask]),
)

barnes1_coeff, barnes1_pow = barnes1_regression.intercept_, *barnes1_regression.coef_

barnes5_regression = LinearRegression()
barnes5_regression.fit(
    np.log(test_numbers[barnes5_nanmask]).reshape(-1, 1),
    np.log(barnes5_times[barnes5_nanmask]),
)

barnes5_coeff, barnes5_pow = barnes5_regression.intercept_, *barnes5_regression.coef_

barnes2_regression = LinearRegression()
barnes2_regression.fit(
    np.log(test_numbers[barnes2_nanmask]).reshape(-1, 1),
    np.log(barnes2_times[barnes2_nanmask]),
)

barnes2_coeff, barnes2_pow = barnes2_regression.intercept_, *barnes2_regression.coef_

#######

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(test_numbers, numba_times, label="Euler w/ Numba")
axes[0].scatter(test_numbers, euler_times, label="Euler w/o Numba")
axes[0].scatter(test_numbers, barnes1_times, label="Barnes-Hut (theta=1)")
axes[0].scatter(test_numbers, barnes5_times, label="Barnes-Hut (theta=0.5)")
axes[0].scatter(test_numbers, barnes2_times, label="Barnes-Hut (theta=0.2)")

dense = np.logspace(
    np.log(min(test_numbers)), np.log(max(test_numbers)), 1000, base=np.e
)
numba_interp = np.exp(numba_regression.predict(np.log(dense).reshape(-1, 1)))
# numba_interp = numba_overhead + numba_coeff * dense**numba_pow

euler_dense = np.logspace(
    np.log(min(test_numbers[euler_nanmask])),
    np.log(max(test_numbers[euler_nanmask])),
    1000,
    base=np.e,
)
euler_interp = np.exp(euler_regression.predict(np.log(euler_dense).reshape(-1, 1)))
# euler_interp = euler_overhead + euler_coeff * euler_dense ** euler_pow

barnes1_dense = np.logspace(
    np.log(min(test_numbers[barnes1_nanmask])),
    np.log(max(test_numbers[barnes1_nanmask])),
    1000,
    base=np.e,
)
barnes1_interp = np.exp(
    barnes1_regression.predict(np.log(barnes1_dense).reshape(-1, 1))
)
# barnes_interp = polylogarithmic_with_poly(barnes_dense, bh_a, bh_b, bh_overhead)

barnes5_dense = np.logspace(
    np.log(min(test_numbers[barnes5_nanmask])),
    np.log(max(test_numbers[barnes5_nanmask])),
    1000,
    base=np.e,
)
barnes5_interp = np.exp(
    barnes5_regression.predict(np.log(barnes5_dense).reshape(-1, 1))
)

barnes2_dense = np.logspace(
    np.log(min(test_numbers[barnes2_nanmask])),
    np.log(max(test_numbers[barnes2_nanmask])),
    1000,
    base=np.e,
)
barnes2_interp = np.exp(
    barnes2_regression.predict(np.log(barnes2_dense).reshape(-1, 1))
)


axes[0].plot(
    dense,
    numba_interp,
    label="Euler law with Numba",
)
axes[0].plot(
    euler_dense,
    euler_interp,
    label="Euler law without Numba",
)
axes[0].plot(
    barnes1_dense,
    barnes1_interp,
    label="Barnes-Hut law (theta=1)",
)
axes[0].plot(
    barnes5_dense,
    barnes5_interp,
    label="Barnes-Hut law (theta=0.5)",
)
axes[0].plot(
    barnes2_dense,
    barnes2_interp,
    label="Barnes-Hut law (theta=0.2)",
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(left=1)
axes[0].set_xlabel(
    f"Number of Elements (log scale)\nRuntimes over {int(TIMEOUT / 1e9) // 60} min {int(TIMEOUT / 1e9) % 60} seconds are not displayed"
)
axes[0].set_ylabel("Runtime in ms (log scale)")
axes[0].legend()
axes[1].text(
    0.5,
    0.5,
    f"Euler law with Numba:\n{np.exp(numba_coeff):.3f}*bodies^{numba_pow:.3f}\n\n \
    Euler law without Numba:\n{np.exp(euler_coeff):.3f}*bodies^{euler_pow:.3f}\n\n \
    Barnes-Hut law (theta=0.2):\n{np.exp(barnes2_coeff):.3f}bodies^{barnes2_pow:.3f}\n\n \
    Barnes-Hut law (theta=0.5):\n{np.exp(barnes5_coeff):.3f}bodies^{barnes5_pow:.3f}\n\n \
    Barnes-Hut law (theta=1):\n{np.exp(barnes1_coeff):.3f}bodies^{barnes1_pow:.3f}",
    horizontalalignment="center",
    verticalalignment="center",
)
axes[1].axis("off")
plt.suptitle("Propagation Step Runtime Scaling Comparison")
plt.tight_layout()
plt.savefig("src/benchmark/runtime_benchmark.png")
