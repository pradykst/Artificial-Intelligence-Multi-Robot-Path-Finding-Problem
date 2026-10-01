from math import hypot
from typing import Callable

from .grid import Cell

Heuristic = Callable[[Cell, Cell], float]


def manhattan(cell: Cell, goal: Cell) -> float:
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def euclidean(cell: Cell, goal: Cell) -> float:
    return hypot(cell[0] - goal[0], cell[1] - goal[1])


def chebyshev(cell: Cell, goal: Cell) -> float:
    return max(abs(cell[0] - goal[0]), abs(cell[1] - goal[1]))


def zero(cell: Cell, goal: Cell) -> float:
    return 0.0


REQUIRED_HEURISTICS: dict[str, Heuristic] = {
    "Manhattan": manhattan,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev,
}
HEURISTICS: dict[str, Heuristic] = {**REQUIRED_HEURISTICS, "Zero / Dijkstra": zero}
