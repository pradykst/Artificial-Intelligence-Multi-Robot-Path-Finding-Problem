from dataclasses import dataclass
from random import Random
from typing import Iterator

Cell = tuple[int, int]


@dataclass(frozen=True)
class Grid:
    """Coordinates are (x, y), with (0, 0) at the top left."""

    width: int
    height: int
    obstacles: frozenset[Cell]
    start: Cell
    goal: Cell

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Grid dimensions must be positive.")
        object.__setattr__(self, "obstacles", frozenset(self.obstacles))
        if any(not self.in_bounds(cell) for cell in self.obstacles):
            raise ValueError("Obstacles must be inside the grid.")
        if not self.is_free(self.start) or not self.is_free(self.goal):
            raise ValueError("Start and goal must be free cells inside the grid.")
        if self.start == self.goal:
            raise ValueError("Grid start and goal must be distinct.")

    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors(self, cell: Cell) -> Iterator[Cell]:
        """Yield free neighbors in fixed up, down, left, right order."""
        x, y = cell
        for neighbor in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if self.is_free(neighbor):
                yield neighbor

    @classmethod
    def random(cls, width: int, height: int, obstacle_probability: float, seed: int) -> "Grid":
        """Reserve two random endpoints; sample other cells independently."""
        if width <= 0 or height <= 0 or width * height < 2:
            raise ValueError("A random grid needs positive dimensions and at least two cells.")
        if not 0 <= obstacle_probability <= 1:
            raise ValueError("Obstacle probability must be between 0 and 1.")
        rng = Random(seed)
        cells = [(x, y) for y in range(height) for x in range(width)]
        start, goal = rng.sample(cells, 2)
        obstacles = frozenset(
            cell for cell in cells
            if cell not in (start, goal) and rng.random() < obstacle_probability
        )
        return cls(width, height, obstacles, start, goal)
