from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from nbody.body import Body, Body_nb, convert_to_body_nb
from nbody.EulerPropagator import EulerPropagator
from nbody.generate_bodies import generate_random_bodies


def test(bodies: list[Body] | list[Body_nb], USE_NUMBA: bool):
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
        propagator.propagate(timeout_ns=1e11)
    except TimeoutError:
        return np.nan
    end_time = time.perf_counter_ns()
    return end_time - start_time


test_numbers = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
numba_times = []
normal_times = []

all_bodies = generate_random_bodies(max(test_numbers))
numba_bodies = [convert_to_body_nb(body) for body in all_bodies]
NORMAL_TIMED_OUT = False
for n_bodies in test_numbers:
    bodies = all_bodies[:n_bodies]
    nb_bodies = numba_bodies[:n_bodies]
    numba_time = test(nb_bodies, True)
    numba_times.append(numba_time)
    if NORMAL_TIMED_OUT:
        normal_times.append(np.nan)
    else:
        normal_time = test(bodies, False)
        normal_times.append(normal_time)
        if not np.isfinite(normal_time):
            NORMAL_TIMED_OUT = True
    print(
        f"Simulated {n_bodies} bodies in {numba_time} ns with numba, {normal_time} ns without numba"
    )

test_numbers = np.array(test_numbers)
numba_times = np.array(numba_times)
normal_times = np.array(normal_times)

# get scaling laws
normal_model = LinearRegression()
nanmask = np.isfinite(normal_times)
normal_model.fit(
    np.log(test_numbers[nanmask]).reshape(-1, 1), np.log(normal_times[nanmask])
)
normal_pow = normal_model.coef_[0]  # time = bodies^p
normal_coeff = normal_model.intercept_  # the constant coefficient


def power_law_with_constant(x, a, p, b):
    return a * (x**p) + b


popt, _ = curve_fit(
    power_law_with_constant, test_numbers, numba_times, p0=[1e8, 1.5, 4e8]
)

# numba_model = LinearRegression()
# numba_model.fit(np.log(test_numbers).reshape(-1, 1), np.log(numba_times))
# numba_pow = numba_model.coef_[0] # time = bodies^p
# numba_coeff = numba_model.intercept_ # the constant coefficient
numba_coeff, numba_pow, numba_intercept = popt


plt.scatter(test_numbers, numba_times, label="With Numba")
plt.scatter(test_numbers, normal_times, label="Without Numba")

dense = np.logspace(
    np.log(min(test_numbers)), np.log(max(test_numbers)), 1000, base=np.e
)
numba_interp = numba_intercept + numba_coeff * dense**numba_pow
dense_notimeout = np.logspace(
    np.log(min(test_numbers[nanmask])),
    np.log(max(test_numbers[nanmask])),
    1000,
    base=np.e,
)
normal_interp = np.exp(np.log(dense_notimeout) * normal_pow + normal_coeff)
plt.plot(
    dense,
    numba_interp,
    label=f"Numba power law: {numba_intercept}+{numba_coeff:.2f}*bodies^{numba_pow:.2f}",
)
plt.plot(
    dense_notimeout,
    normal_interp,
    label=f"Normal power law: {np.exp(normal_coeff):.2f}*bodies^{normal_pow:.2f}",
)
plt.xscale("log")
plt.yscale("log")
plt.gca().set_xlim(left=1)
plt.gca().set_ylim(bottom=1)
plt.xlabel(
    "Number of Elements (log scale)\nRuntimes over 100 seconds are not displayed"
)
plt.ylabel("Runtime in ns (log scale)")
plt.legend()
plt.title("Propagation Step Runtime Scaling Comparison")
plt.tight_layout()
plt.savefig("numba_comparison.png")
