from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BodyValidator:
    """
    This class validates that a Body object has correct attributes.

    Methods
    -------
    validate(id, mass, radius, position, velocity)
        Validate the provided body parameters and raise an error if
        tests do not pass.
    """

    @staticmethod
    def validate(
        id: str, mass: float, radius: float, position: np.ndarray, velocity: np.ndarray
    ) -> None:
        """
        Method that performs the validation.

        Parameters
        ----------
        id : str
            Body ID.
        mass : float
            Mass of the body (>0).
        radius : float
            Radius of the body (>0).
        position : numpy.ndarray
            3-element array representing the Cartesian coordinates (x, y, z)
            of the body's position.
        velocity : numpy.ndarray
            3-element array representing the velocity components (vx, vy, vz)
            of the body.

        Raises
        ------
        TypeError
            If `id` is not a string, or arrays are not NumPy arrays.
        ValueError
            If `mass` or `radius` are not positive, or arrays have invalid shape.
        """
        if not isinstance(id, str):
            raise TypeError("id must be a string.")
        if not isinstance(mass, (int, float)) or mass <= 0:
            raise ValueError("mass must be positive.")
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be positive.")
        if not isinstance(position, np.ndarray) or position.shape != (2,):
            raise ValueError("position must be a numpy array of shape (2,).")
        if not isinstance(velocity, np.ndarray) or velocity.shape != (2,):
            raise ValueError("velocity must be a numpy array of shape (2,).")


@dataclass
class Body:
    """
    Represents a celestial or simulated object in an N-body system.

    The `Body` class stores all the essential physical attributes
    required for time evolution in an N-body simulation, including
    mass, radius, position, and velocity. Upon initialization,
    parameter validation is automatically performed via `BodyValidator`.

    Parameters
    ----------
    id : str
        Unique identifier for the body.
    mass : float
        Mass of the body (must be positive).
    radius : float
        Radius of the body (must be positive).
    position : numpy.ndarray
        2-element array representing the Cartesian coordinates (x, y)
        of the body's position.
    velocity : numpy.ndarray
        3-element array representing the velocity components (vx, vy)
        of the body.

    Raises
    ------
    TypeError
        If `id` is not a string, or arrays are not numpy arrays.
    ValueError
        If `mass` or `radius` are not positive, or arrays have invalid shape.

    Examples
    --------
    >>> import numpy as np
    >>> from body import Body
    >>> position = np.array([1.0, 0.0])
    >>> vellocity = np.array([0.0, 1.0])
    >>> earth = Body(id="Earth", mass=5.97e24, radius=6.371e6,
    ...              position=position, velocity=velocity)
    """

    id: str
    mass: float
    radius: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        """Run post-initialization validation checks."""
        BodyValidator.validate(
            self.id, self.mass, self.radius, self.position, self.velocity
        )
