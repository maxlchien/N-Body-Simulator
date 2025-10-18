from dataclasses import dataclass
from typing import List

@dataclass
class Body:
    '''A class in which to store objects that will
    be time-evolved through our N-body simulator.'''
    id: str
    mass: float
    radius: float
    pos_arr: List[float]
    v_arr: List[float]