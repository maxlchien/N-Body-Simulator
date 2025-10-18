from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Body:
    """A class in which to store objects that will
    be time-evolved through our N-body simulator."""

    id: str
    mass: float
    radius: float
    pos_arr: list[float]
    v_arr: list[float]
