from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Body:
    """A class in which to store objects that will
    be time-evolved through our N-body simulator."""

    id: str
    mass: float
    radius: float
    position: np.ndarray
    velocity: np.ndarray
