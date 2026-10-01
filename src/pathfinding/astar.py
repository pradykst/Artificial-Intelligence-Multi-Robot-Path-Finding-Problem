from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from time import perf_counter_ns

from .grid import Cell, Grid
from .heuristics import Heuristic, manhattan
from .metrics import SearchResult


@dataclass(frozen=True)
class SearchEvent:
    current: Cell | None
    opened: tuple[Cell, ...]


class AStarSearch:
    """Advance one valid expansion per step; pauses between steps are not timed.

    Equal f-scores use insertion order. The open dictionary counts unique live
    states independently of stale heap entries. Improved states may be reopened.
    Optional endpoints let the search API handle an already-reached goal.
    """

    def __init__(
        self, grid: Grid, heuristic: Heuristic = manhattan, *,
        start: Cell | None = None, goal: Cell | None = None,
    ) -> None:
        began = perf_counter_ns()
        self.grid = grid
        self.heuristic = heuristic
        self.start = grid.start if start is None else start
        self.goal = grid.goal if goal is None else goal
        if not grid.is_free(self.start) or not grid.is_free(self.goal):
            raise ValueError("Search endpoints must be free cells inside the grid.")
        self._order = count()
        self._heap = [(heuristic(self.start, self.goal), next(self._order), 0, self.start)]
        self._g = {self.start: 0}
        self._open = {self.start: 0}
        self._parents: dict[Cell, Cell] = {}
        self.expanded_nodes = 0
        self.generated_nodes = 0
        self.peak_frontier_size = 1
        self.result: SearchResult | None = None
        self._elapsed_ns = perf_counter_ns() - began

    @property
    def frontier_size(self) -> int:
        return len(self._open)

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ns / 1_000_000

    def _path(self, cell: Cell) -> tuple[Cell, ...]:
        path = [cell]
        while cell in self._parents:
            cell = self._parents[cell]
            path.append(cell)
        return tuple(reversed(path))

    def step(self) -> SearchEvent:
        """Return frontier changes after one expansion or a terminal empty step.

        A valid goal pop counts as an expansion, but generates no successors.
        The initial start insertion is excluded from generated_nodes.
        """
        if self.result is not None:
            raise RuntimeError("Search has finished.")
        began = perf_counter_ns()
        current = None
        opened: list[Cell] = []
        path: tuple[Cell, ...] = ()
        finished = False
        while self._heap:
            _, _, cost, cell = heappop(self._heap)
            if self._open.get(cell) != cost:
                continue
            current = cell
            del self._open[cell]
            self.expanded_nodes += 1
            if cell == self.goal:
                path = self._path(cell)
                finished = True
                break
            for neighbor in self.grid.neighbors(cell):
                candidate = cost + 1
                if candidate < self._g.get(neighbor, float("inf")):
                    self._g[neighbor] = candidate
                    self._parents[neighbor] = cell
                    self._open[neighbor] = candidate
                    heappush(self._heap, (
                        candidate + self.heuristic(neighbor, self.goal),
                        next(self._order), candidate, neighbor,
                    ))
                    self.generated_nodes += 1
                    opened.append(neighbor)
                    self.peak_frontier_size = max(self.peak_frontier_size, len(self._open))
            break
        if not self._open:
            finished = True
        self._elapsed_ns += perf_counter_ns() - began
        if finished:
            self.result = SearchResult(
                path=path, path_cost=len(path) - 1 if path else None,
                expanded_nodes=self.expanded_nodes, generated_nodes=self.generated_nodes,
                peak_frontier_size=self.peak_frontier_size, elapsed_ms=self.elapsed_ms,
                found=bool(path),
            )
        return SearchEvent(current, tuple(opened))


def astar(
    grid: Grid, heuristic: Heuristic = manhattan, *,
    start: Cell | None = None, goal: Cell | None = None,
) -> SearchResult:
    """Run the same step engine to completion without animation."""
    search = AStarSearch(grid, heuristic, start=start, goal=goal)
    while search.result is None:
        search.step()
    return search.result
