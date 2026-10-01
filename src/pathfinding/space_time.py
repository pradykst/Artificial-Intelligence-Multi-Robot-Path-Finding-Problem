from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from time import perf_counter_ns

from .grid import Cell, Grid
from .heuristics import Heuristic, manhattan
from .reservations import ReservationTable

TimedState = tuple[Cell, int]


@dataclass(frozen=True)
class SpaceTimeResult:
    found: bool
    path: tuple[Cell, ...]
    path_cost: int | None
    expanded_states: int
    generated_states: int
    peak_frontier_size: int
    elapsed_ms: float
    failure_reason: str | None


class SpaceTimeSearch:
    """Bounded A* with up/down/left/right/WAIT actions and FIFO f-score ties."""

    def __init__(
        self, grid: Grid, start: Cell, goal: Cell, reservations: ReservationTable,
        horizon: int, heuristic: Heuristic = manhattan, max_expansions: int = 100_000,
    ) -> None:
        began = perf_counter_ns()
        if horizon < 0 or max_expansions < 1:
            raise ValueError("Horizon must be nonnegative and expansion limit positive.")
        if not grid.is_free(start) or not grid.is_free(goal):
            raise ValueError("Search endpoints must be free cells.")
        self.grid, self.start, self.goal = grid, start, goal
        self.reservations, self.horizon, self.heuristic = reservations, horizon, heuristic
        self.max_expansions = max_expansions
        self._order = count()
        initial = (start, 0)
        self._heap = [(heuristic(start, goal), next(self._order), 0, initial)]
        self._g = {initial: 0}
        self._open = {initial: 0}
        self._parents: dict[TimedState, TimedState] = {}
        self.expanded_states = self.generated_states = 0
        self.peak_frontier_size = 1
        self._cutoff = False
        self.result: SpaceTimeResult | None = None
        self._elapsed_ns = perf_counter_ns() - began

    def _path(self, state: TimedState) -> tuple[Cell, ...]:
        cells = [state[0]]
        while state in self._parents:
            state = self._parents[state]
            cells.append(state[0])
        return tuple(reversed(cells))

    def step(self) -> None:
        if self.result is not None:
            return
        began = perf_counter_ns()
        path: tuple[Cell, ...] = ()
        reason = None
        if self.reservations.vertex_reserved(self.start, 0):
            reason = "start_reserved"
        elif self.goal in self.reservations.terminal_goals:
            reason = "goal_reserved"
        elif self.start == self.goal and not self.reservations.can_hold_goal(self.goal, 0):
            reason = "goal_reserved"
        elif self.expanded_states >= self.max_expansions:
            reason = "expansion_limit"
        else:
            while self._heap:
                _, _, cost, state = heappop(self._heap)
                if self._open.get(state) != cost:
                    continue
                del self._open[state]
                self.expanded_states += 1
                cell, timestep = state
                if cell == self.goal and self.reservations.can_hold_goal(cell, timestep):
                    path = self._path(state)
                    break
                for neighbor in (*self.grid.neighbors(cell), cell):
                    arrival = timestep + 1
                    if not self.reservations.allows(cell, neighbor, arrival):
                        continue
                    if neighbor == self.goal and not self.reservations.can_hold_goal(neighbor, arrival):
                        continue
                    if arrival > self.horizon:
                        self._cutoff = True
                        continue
                    successor = (neighbor, arrival)
                    candidate = cost + 1
                    if candidate < self._g.get(successor, float("inf")):
                        self._g[successor] = candidate
                        self._parents[successor] = state
                        self._open[successor] = candidate
                        heappush(self._heap, (candidate + self.heuristic(neighbor, self.goal),
                                             next(self._order), candidate, successor))
                        self.generated_states += 1
                        self.peak_frontier_size = max(self.peak_frontier_size, len(self._open))
                break
            if not path and not self._open:
                reason = "horizon_exhausted" if self._cutoff else "reservation_blocked"
        self._elapsed_ns += perf_counter_ns() - began
        if path or reason:
            self.result = SpaceTimeResult(bool(path), path, len(path) - 1 if path else None,
                                          self.expanded_states, self.generated_states, self.peak_frontier_size,
                                          self._elapsed_ns / 1_000_000, reason)


def space_time_astar(
    grid: Grid, start: Cell, goal: Cell, reservations: ReservationTable, horizon: int,
    heuristic: Heuristic = manhattan, max_expansions: int = 100_000,
) -> SpaceTimeResult:
    search = SpaceTimeSearch(grid, start, goal, reservations, horizon, heuristic, max_expansions)
    while search.result is None:
        search.step()
    return search.result
