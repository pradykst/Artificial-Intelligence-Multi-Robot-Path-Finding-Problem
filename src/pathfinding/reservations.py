from dataclasses import dataclass, field
from typing import Sequence

from .grid import Cell


@dataclass(frozen=True)
class ReservationSnapshot:
    timestep: int
    vertices: frozenset[Cell]
    terminal_goals: frozenset[Cell]
    edges: tuple[tuple[Cell, Cell], ...]


@dataclass
class ReservationTable:
    """Edges are indexed by arrival time; terminal goals persist indefinitely."""

    vertices: set[tuple[Cell, int]] = field(default_factory=set)
    edges: set[tuple[Cell, Cell, int]] = field(default_factory=set)
    terminal_goals: dict[Cell, int] = field(default_factory=dict)

    def at_time(self, timestep: int) -> ReservationSnapshot:
        if timestep < 0:
            raise ValueError("Reservation timestep must be nonnegative.")
        terminal = frozenset(cell for cell, arrival in self.terminal_goals.items() if timestep >= arrival)
        vertices = frozenset(cell for cell, time in self.vertices if time == timestep) | terminal
        edges = tuple(sorted((source, target) for source, target, time in self.edges if time == timestep))
        return ReservationSnapshot(timestep, vertices, terminal, edges)

    def vertex_reserved(self, cell: Cell, timestep: int) -> bool:
        arrival = self.terminal_goals.get(cell)
        return (cell, timestep) in self.vertices or (arrival is not None and timestep >= arrival)

    def edge_reserved(self, source: Cell, target: Cell, arrival: int) -> bool:
        return (source, target, arrival) in self.edges

    def allows(self, source: Cell, target: Cell, arrival: int) -> bool:
        return not self.vertex_reserved(target, arrival) and not self.edge_reserved(target, source, arrival)

    def can_hold_goal(self, goal: Cell, arrival: int) -> bool:
        """An arriving robot cannot leave its goal to dodge a future reservation."""
        return goal not in self.terminal_goals and not any(
            cell == goal and timestep >= arrival for cell, timestep in self.vertices
        )

    def reserve_path(self, path: Sequence[Cell]) -> None:
        if not path:
            raise ValueError("Cannot reserve an empty path.")
        for timestep, cell in enumerate(path):
            self.vertices.add((cell, timestep))
            if timestep:
                self.edges.add((path[timestep - 1], cell, timestep))
        self.terminal_goals[path[-1]] = min(self.terminal_goals.get(path[-1], len(path) - 1), len(path) - 1)
