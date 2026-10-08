"""Immutable, receiver-addressed messages for synchronous peer negotiation."""
from dataclasses import dataclass
from typing import Literal

from .collisions import Collision
from .grid import Cell

MessageKind = Literal["PATH_PROPOSAL", "UPDATED_PATH", "CONFLICT_NOTICE", "YIELD_DECISION",
                      "AGREEMENT", "PLANNING_FAILURE"]
PriorityKey = tuple[int, int, int]
VersionVector = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Message:
    kind: MessageKind
    sender: int
    receiver: int
    round: int
    version: int
    path: tuple[Cell, ...] = ()
    priority: PriorityKey | None = None
    conflicts: tuple[Collision, ...] = ()
    winner: int | None = None
    versions: VersionVector = ()
    reason: str | None = None


def pairwise_winner(first: PriorityKey, second: PriorityKey) -> int:
    """Smaller (-initial bottleneck exposure, -initial cost, ID) wins."""
    if first[2] == second[2]:
        raise ValueError("A pair must contain distinct agent IDs.")
    return min(first, second)[2]
