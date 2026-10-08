from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

from .grid import Cell
from .collisions import Collision
from .reservations import ReservationSnapshot
from .space_time import SearchObservation, TimedState

if TYPE_CHECKING:
    from .cooperative import CooperativePlanner


@dataclass(frozen=True)
class PlannerEvent:
    sequence: int
    kind: str
    message: str
    agent_id: int | None = None
    attempt: int = 0
    state: TimedState | None = None
    old_order: tuple[int, ...] = ()
    new_order: tuple[int, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class PriorityRow:
    agent_id: int
    conflict_load: int | None
    bottleneck_exposure: int | None
    path_cost: int | None
    initial_rank: int | None
    current_rank: int | None


@dataclass(frozen=True)
class PlanningSnapshot:
    phase: str
    agent_id: int | None
    attempt: int
    order: tuple[int, ...]
    priorities: tuple[PriorityRow, ...]
    independent_paths: tuple[tuple[int, tuple[Cell, ...]], ...]
    paths: tuple[tuple[int, tuple[Cell, ...]], ...]
    current: TimedState | None
    frontier: tuple[TimedState, ...]
    search: SearchObservation | None
    reservations: ReservationSnapshot
    predicted_conflicts: tuple[Collision, ...]


def priority_rows(planner: "CooperativePlanner") -> tuple[PriorityRow, ...]:
    costs = {plan.agent.agent_id: plan.result.path_cost for plan in planner.independent.plans}
    initial = planner.attempted_orders[0] if planner.attempted_orders else planner.order
    ranks = {agent_id: index + 1 for index, agent_id in enumerate(planner.order)}
    initial_ranks = {agent_id: index + 1 for index, agent_id in enumerate(initial)}
    analysis = planner.analysis
    return tuple(PriorityRow(agent.agent_id,
                             analysis.conflict_load[agent.agent_id] if analysis else None,
                             analysis.bottleneck_exposure[agent.agent_id] if analysis else None,
                             costs.get(agent.agent_id), initial_ranks.get(agent.agent_id), ranks.get(agent.agent_id))
                 for agent in planner.agents)


def rank_badges(rows: tuple[PriorityRow, ...]) -> dict[int, int]:
    return {row.agent_id: row.current_rank for row in rows if row.current_rank is not None}


class EventLog:
    def __init__(self, capacity: int = 300) -> None:
        if capacity < 1:
            raise ValueError("Event capacity must be positive.")
        self.lines: deque[str] = deque(maxlen=capacity)

    def clear(self) -> None:
        self.lines.clear()

    def append(self, event: PlannerEvent, detailed: bool = False) -> None:
        if detailed or event.kind not in ("Search", "Rejected"):
            self.lines.append(f"[{event.kind}] {event.message}")


def advance_planning(planner: "CooperativePlanner", detailed: bool, budget: int = 200) -> bool:
    """Bound work per UI callback; return whether one playback unit completed."""
    if planner.done:
        return True
    before = (len(planner.independent.plans), planner.analysis is not None,
              planner.current_agent_id, len(planner.completed_paths),
              len(planner.attempted_orders), planner.horizon_retries)
    began = perf_counter()
    for _ in range(1 if detailed else budget):
        planner.step()
        after = (len(planner.independent.plans), planner.analysis is not None,
                 planner.current_agent_id, len(planner.completed_paths),
                 len(planner.attempted_orders), planner.horizon_retries)
        if detailed or planner.done or before != after:
            return True
        if perf_counter() - began >= 0.008:
            break
    return False
