from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numba import types
from numba.experimental import jitclass

USE_NUMBA = "USE_NUMBA" in os.environ

"""
    Represents a celestial or simulated object in an N-body system.

    Uses an interpreted dataclass structure through the Body class, or a numba
    jitclass structure through the Body_nb class. If passing to EulerPropagator,
    the objects may be passed as Body instances and will be converted if the appropriate
    flag is set.
"""


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
        Body.validate(self.id, self.mass, self.radius, self.position, self.velocity)

    @staticmethod
    def validate(
        id: str, mass: float, radius: float, position: np.ndarray, velocity: np.ndarray
    ) -> None:
        """
        Validates that a Body object has correct attributes.

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


spec = [
    ("id", types.unicode_type),
    ("mass", types.float64),
    ("radius", types.float64),
    ("position", types.float64[::1]),
    ("velocity", types.float64[:]),
]


@jitclass(spec)
class Body_nb:
    """
    Represents a celestial or simulated object in an N-body system using a numba jitclass structure.

    The `Body_nb` class stores all the essential physical attributes
    required for time evolution in an N-body simulation, including
    mass, radius, position, and velocity. Upon initialization,
    parameter validation is automatically performed.

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
        2-element array representing the velocity components (vx, vy)
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
    >>> from body import Body_nb
    >>> position = np.array([1.0, 0.0])
    >>> vellocity = np.array([0.0, 1.0])
    >>> earth = Body_nb(id="Earth", mass=5.97e24, radius=6.371e6,
    ...              position=position, velocity=velocity)
    """

    def __init__(
        self,
        id: str,
        mass: float,
        radius: float,
        position: np.ndarray,
        velocity: np.ndarray,
    ):
        """Run initialization validation checks and set instance variables."""
        self.validate(mass, radius, position, velocity)
        self.id = id
        self.mass = mass
        self.radius = radius
        self.position = position
        self.velocity = velocity

    def validate(
        self,
        mass: float,
        radius: float,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        """
        Validates that a Body object has correct attributes.

        Parameters
        ----------
        id : str
            Body ID.
        mass : float
            Mass of the body (>0).
        radius : float
            Radius of the body (>0).
        position : numpy.ndarray
            2-element array representing the Cartesian coordinates (x, y)
            of the body's position.
        velocity : numpy.ndarray
            2-element array representing the velocity components (vx, vy)
            of the body.

        Raises
        ------
        TypeError
            If `id` is not a string, or arrays are not NumPy arrays.
        ValueError
            If `mass` or `radius` are not positive, or arrays have invalid shape.
        """
        if mass <= 0:
            raise ValueError("mass must be positive.")
        if radius <= 0:
            raise ValueError("radius must be positive.")
        if position.shape != (2,):
            raise ValueError("position must be a numpy array of shape (2,).")
        if velocity.shape != (2,):
            raise ValueError("velocity must be a numpy array of shape (2,).")


def convert_to_body_nb(body: Body | Body_nb):
    if type(body) is Body_nb:
        return body
    return Body_nb(body.id, body.mass, body.radius, body.position, body.velocity)
