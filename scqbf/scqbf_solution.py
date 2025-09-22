# scqbf/scqbf_solution.py

from dataclasses import dataclass
from typing import List, Set

@dataclass
class ScQbfSolution:
    elements: List[int]
    _last_objfun_val: float = 0