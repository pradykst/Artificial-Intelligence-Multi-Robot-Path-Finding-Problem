from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Mapping, Sequence

from .grid import Cell

Paths = Mapping[int, Sequence[Cell]]


@dataclass(frozen=True)
class Collision:
    """Edge events use the arrival timestep: the swap occurs from t-1 to t."""

    timestep: int
    kind: Literal["vertex", "edge"]
    agent_ids: tuple[int, ...]
    cells: tuple[Cell, ...]


def position_at(path: Sequence[Cell], timestep: int) -> Cell:
    """Extend a nonempty path by keeping the robot at its last cell."""
    if not path:
        raise ValueError("A trajectory must contain at least its start cell.")
    if timestep < 0:
        raise ValueError("Timestep must be nonnegative.")
    return path[min(timestep, len(path) - 1)]


def collisions_at(paths: Paths, timestep: int) -> tuple[Collision, ...]:
    positions = {agent_id: position_at(paths[agent_id], timestep) for agent_id in sorted(paths)}
    occupants: dict[Cell, list[int]] = defaultdict(list)
    for agent_id, cell in positions.items():
        occupants[cell].append(agent_id)
    events = [Collision(timestep, "vertex", tuple(ids), (cell,))
              for cell, ids in sorted(occupants.items()) if len(ids) > 1]
    if timestep > 0:
        previous = {agent_id: position_at(paths[agent_id], timestep - 1) for agent_id in positions}
        for first, second in combinations(positions, 2):
            if (previous[first] != positions[first]
                    and previous[first] == positions[second]
                    and previous[second] == positions[first]):
                events.append(Collision(timestep, "edge", (first, second), (previous[first], positions[first])))
    return tuple(events)


def detect_collisions(paths: Paths) -> tuple[Collision, ...]:
    """Count through the final arrival, including t=0; group multiway vertices."""
    if any(not path for path in paths.values()):
        raise ValueError("Cannot inspect an empty trajectory.")
    horizon = max((len(path) - 1 for path in paths.values()), default=0)
    return tuple(event for timestep in range(horizon + 1) for event in collisions_at(paths, timestep))
