from __future__ import annotations

import numpy as np

from nbody.model.body import Body


def generate_random_bodies(
    n_bodies: int,
    mass_range: tuple[float, float] = (1.0e20, 1.0e25),
    radius_range: tuple[float, float] = (1.0e3, 1.0e7),
    position_box: float = 1.0e9,
    velocity_dispersion: float = 0.0,
    seed: int | None = None,
) -> list[Body]:
    """
    Generate a list of random bodies for an N-body simulation.

    Each body is assigned a random mass, radius, position, and velocity.
    Positions are sampled uniformly in a cube centered at the origin with
    coordinates in the range ``[-position_box, position_box]`` for each axis.
    Velocities are sampled from a normal distribution with zero mean and
    standard deviation given by ``velocity_dispersion`` for each component.

    Parameters
    ----------
    n_bodies : int
        Number of bodies to generate.
    mass_range : tuple of float, optional
        Lower and upper bounds for the mass of each body, sampled from
        a uniform distribution. Defaults to ``(1.0e20, 1.0e25)``.
    radius_range : tuple of float, optional
        Lower and upper bounds for the radius of each body, sampled from
        a uniform distribution. Defaults to ``(1.0e3, 1.0e7)``.
    position_box : float, optional
        Half-width of the cubic region in which positions are sampled.
        Positions are drawn uniformly from
        ``[-position_box, position_box]`` along each axis.
        Defaults to ``1.0e9``.
    velocity_dispersion : float, optional
        Standard deviation of the Gaussian distribution used to sample
        each velocity component. If set to 0.0, all velocities will be
        exactly zero. Defaults to ``0.0``.
    seed : int or None, optional
        Seed for the random number generator. If ``None``, a new
        generator with a random state is used. Defaults to ``None``.

    Returns
    -------
    bodies : list of Body
        A list containing n_bodies randomly generated :class:`Body`
        instances.

    Examples
    --------
    Generate five bodies with default parameter ranges:

    >>> bodies = generate_random_bodies(5)
    >>> for body in bodies:
    ...     print(body.id, body.mass, body.position)

    Use a custom random seed and non-zero velocity dispersion:

    >>> bodies = generate_random_bodies(
    ...     3,
    ...     mass_range=(1.0e22, 1.0e23),
    ...     radius_range=(1.0e5, 1.0e6),
    ...     position_box=1.0e8,
    ...     velocity_dispersion=1.0e3,
    ...     seed=42,
    ... )
    """
    rng = np.random.default_rng(seed)

    masses = rng.uniform(mass_range[0], mass_range[1], size=n_bodies)
    radii = rng.uniform(radius_range[0], radius_range[1], size=n_bodies)

    positions = rng.uniform(low=-position_box, high=position_box, size=(n_bodies, 2))

    if velocity_dispersion > 0.0:
        velocities = rng.normal(loc=0.0, scale=velocity_dispersion, size=(n_bodies, 2))
    else:
        velocities = np.zeros((n_bodies, 2))

    bodies: list[Body] = []
    for i in range(n_bodies):
        bodies.append(
            Body(
                id=f"Body{i}",
                mass=float(masses[i]),
                radius=float(radii[i]),
                position=positions[i],
                velocity=velocities[i],
            )
        )

    return bodies
