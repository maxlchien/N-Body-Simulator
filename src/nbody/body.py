from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Body:
    """
    Max's placeholder Body class. Can be replaced if `barnes_hut.py` is updated to account for changes.
    """

    mass: float
    position: np.ndarray
    velocity: np.ndarray
