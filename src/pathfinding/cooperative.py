from dataclasses import dataclass, replace
from collections import deque
from itertools import combinations
from time import perf_counter_ns
from typing import Sequence

from .agents import Agent
from .collisions import detect_collisions
from .grid import Cell, Grid
from .multi_agent import AgentPlan, IndependentPlanner, compute_metrics
from .planning_view import PlannerEvent, PlanningSnapshot, priority_rows
from .reservations import ReservationTable
from .space_time import SpaceTimeSearch


@dataclass(frozen=True)
class PriorityAnalysis:
    conflict_load: dict[int, int]
    bottleneck_exposure: dict[int, int]
    independent_path_cost: dict[int, int | None]
    conflict_graph: dict[int, tuple[int, ...]]
    priority_order: tuple[int, ...]


def analyze_priorities(grid: Grid, plans: Sequence[AgentPlan]) -> PriorityAnalysis:
    paths = {plan.agent.agent_id: plan.path for plan in plans if plan.result.found}
    loads = {plan.agent.agent_id: 0 for plan in plans}
    graph: dict[int, set[int]] = {agent_id: set() for agent_id in loads}
    for event in detect_collisions(paths):
        for agent_id in event.agent_ids:
            loads[agent_id] += 1
        for first, second in combinations(event.agent_ids, 2):
            graph[first].add(second)
            graph[second].add(first)
    bottlenecks = {plan.agent.agent_id: sum(sum(1 for _ in grid.neighbors(cell)) <= 2 for cell in plan.path)
                   for plan in plans}
    costs = {plan.agent.agent_id: plan.result.path_cost for plan in plans}
    order = tuple(sorted(loads, key=lambda agent_id: (-loads[agent_id], -bottlenecks[agent_id],
                                                     -(costs[agent_id] or 0), agent_id)))
    return PriorityAnalysis(loads, bottlenecks, costs,
                            {agent_id: tuple(sorted(neighbors)) for agent_id, neighbors in graph.items()}, order)


def next_promotion(order: tuple[int, ...], failed: int, seen: set[tuple[int, ...]]) -> tuple[tuple[int, ...], int] | None:
    """Try one place earlier first, then larger promotions to skip visited orders."""
    index = order.index(failed)
    for target in range(index - 1, -1, -1):
        candidate = list(order)
        candidate.insert(target, candidate.pop(index))
        if tuple(candidate) not in seen:
            return tuple(candidate), index - target
    return None


@dataclass(frozen=True)
class CooperativeResult:
    found: bool
    individual_planning_success: bool
    paths: dict[int, tuple[Cell, ...]]
    initial_priority_order: tuple[int, ...]
    final_priority_order: tuple[int, ...]
    attempted_orders: tuple[tuple[int, ...], ...]
    planning_attempts: int
    priority_promotions: int
    analysis: PriorityAnalysis
    sum_of_costs: int | None
    makespan: int | None
    wait_actions: int | None
    expanded_states: int
    generated_states: int
    peak_frontier_size: int
    elapsed_ms: float
    vertex_collisions: int | None
    edge_collisions: int | None
    failure_reason: str | None
    search_limit_reached: bool
    horizon_retries: int
    maximum_horizon_used: int
    horizon_limit: int


class CooperativePlanner:
    """Shared incremental engine for fixed and conflict-guided priorities.

    Horizon starts at each independent cost and doubles up to F + max(L),
    where F is the free-cell count. The explicit space-time expansion budget
    covers all horizon retries and priority attempts together.
    """

    def __init__(
        self, grid: Grid, agents: Sequence[Agent], *, adaptive: bool = True,
        max_horizon: int | None = None, max_priority_retries: int | None = None,
        max_expansions: int = 100_000,
        observe: bool = False, event_capacity: int = 256,
        initial_priority: str | None = None,
    ) -> None:
        began = perf_counter_ns()
        if max_horizon is not None and max_horizon < 0:
            raise ValueError("Maximum horizon must be nonnegative.")
        if max_priority_retries is not None and max_priority_retries < 0:
            raise ValueError("Priority retries must be nonnegative.")
        if max_expansions < 1:
            raise ValueError("Expansion budget must be positive.")
        if initial_priority is None:
            initial_priority = "conflict" if adaptive else "fixed"
        if initial_priority not in ("fixed", "conflict"):
            raise ValueError("Initial priority must be 'fixed' or 'conflict'.")
        self.initial_priority = initial_priority
        self.grid, self.agents = grid, tuple(agents)
        if event_capacity < 1:
            raise ValueError("Event capacity must be positive.")
        self.observe = observe
        self._events: deque[PlannerEvent] = deque(maxlen=event_capacity)
        self._event_sequence = 0
        self.last_search_observation = None
        self.predicted_conflicts = ()
        self.independent = IndependentPlanner(grid, agents)
        self.adaptive = adaptive
        self.max_horizon = max_horizon
        self.max_priority_retries = 2 * len(agents) if max_priority_retries is None else max_priority_retries
        self.max_expansions = max_expansions
        self.analysis: PriorityAnalysis | None = None
        self.order: tuple[int, ...] = ()
        self.attempted_orders: list[tuple[int, ...]] = []
        self.priority_promotions = self.horizon_retries = self.maximum_horizon_used = 0
        self._agent_by_id = {agent.agent_id: agent for agent in agents}
        self._index = 0
        self._horizon = 0
        self._reservations = ReservationTable()
        self._paths: dict[int, tuple[Cell, ...]] = {}
        self._search: SpaceTimeSearch | None = None
        self._expanded = self._generated = self._peak = self._space_expanded = 0
        self.result: CooperativeResult | None = None
        self._elapsed_ns = perf_counter_ns() - began
        self.phase = "Analyzing independent paths"

    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def current_agent_id(self) -> int | None:
        if self.done:
            return None
        if self.analysis is None:
            return self.agents[len(self.independent.plans)].agent_id if not self.independent.done else None
        return self.order[self._index] if self._search is not None and self._index < len(self.order) else None

    @property
    def completed_paths(self) -> tuple[tuple[int, tuple[Cell, ...]], ...]:
        return tuple(self._paths.items())

    def drain_events(self) -> tuple[PlannerEvent, ...]:
        events = tuple(self._events)
        self._events.clear()
        return events

    def _emit(self, kind: str, message: str, agent_id: int | None = None, **details) -> None:
        if self.observe:
            self._event_sequence += 1
            self._events.append(PlannerEvent(self._event_sequence, kind, message, agent_id,
                                             len(self.attempted_orders), **details))

    def snapshot(self, timestep: int | None = None) -> PlanningSnapshot:
        observation = self.last_search_observation
        current = observation.expanded if observation else None
        time = timestep if timestep is not None else (current[1] + 1 if current else 0)
        if self.done:
            phase = "Complete" if self.result.found else "Failed"
        elif self.analysis is None:
            phase = "Ranking" if self.independent.done else "Analyzing"
        else:
            phase = "Ranking" if self._index == 0 and self._search is None and not self._paths else "Planning"
        return PlanningSnapshot(phase, self.current_agent_id, len(self.attempted_orders), self.order,
                                priority_rows(self),
                                tuple((plan.agent.agent_id, plan.path) for plan in self.independent.plans),
                                self.completed_paths, current, self._search.frontier() if self._search else (),
                                observation, self._reservations.at_time(time), self.predicted_conflicts)

    def step(self) -> None:
        if self.done:
            return
        began = perf_counter_ns()
        self._advance()
        self._elapsed_ns += perf_counter_ns() - began
        if self.result is not None:
            self.result = replace(self.result, elapsed_ms=self._elapsed_ns / 1_000_000)

    def _advance(self) -> None:
        if self.analysis is None:
            if not self.independent.done:
                self.independent.step()
                if self.independent.search is None:
                    plan = self.independent.plans[-1]
                    self._emit("Analysis", f"Agent {plan.agent.agent_id} independent path {'found' if plan.result.found else 'unreachable'}; cost {plan.result.path_cost}.", plan.agent.agent_id)
                return
            plans = self.independent.plans
            self.analysis = analyze_priorities(self.grid, plans)
            self._expanded = sum(plan.result.expanded_nodes for plan in plans)
            self._generated = sum(plan.result.generated_nodes for plan in plans)
            self._peak = max(plan.result.peak_frontier_size for plan in plans)
            longest = max((cost or 0 for cost in self.analysis.independent_path_cost.values()), default=0)
            if self.max_horizon is None:
                self.max_horizon = self.grid.width * self.grid.height - len(self.grid.obstacles) + longest
            self.order = self.analysis.priority_order if self.initial_priority == "conflict" else tuple(sorted(self._agent_by_id))
            if self.observe:
                self._emit("Analysis", "Independent paths calculated; calculating lexicographic priority (-C, -B, -L, ID).")
                conflicts = detect_collisions({plan.agent.agent_id: plan.path for plan in plans if plan.result.found})
                self.predicted_conflicts = conflicts
                self._emit("Conflict", f"{len(conflicts)} predicted collision events; showing at most 20.")
                for event in conflicts[:20]:
                    self._emit("Conflict", f"Agents {event.agent_ids}: predicted {event.kind} conflict at t={event.timestep}, cells={event.cells}.")
                for agent_id in self.order:
                    self._emit("Priority", f"Agent {agent_id}: C={self.analysis.conflict_load[agent_id]}, B={self.analysis.bottleneck_exposure[agent_id]}, L={self.analysis.independent_path_cost[agent_id]}.", agent_id)
                self._emit("Priority", f"Initial order: {self.order}.", new_order=self.order)
            if any(not plan.result.found for plan in plans):
                self._finish("individual_path_unreachable")
                return
            self._start_attempt()
            return
        if self._index == len(self.order):
            self._finish(None)
            return
        agent = self._agent_by_id[self.order[self._index]]
        if self._search is None:
            remaining = self.max_expansions - self._space_expanded
            if remaining <= 0:
                self._finish("expansion_limit", limited=True)
                return
            self._search = SpaceTimeSearch(self.grid, agent.start, agent.goal, self._reservations,
                                           self._horizon, max_expansions=remaining, observe=self.observe)
            self.last_search_observation = None
            self._emit("Planning", f"Attempt {len(self.attempted_orders)}, planning Agent {agent.agent_id}; horizon {self._horizon}.", agent.agent_id)
            self.maximum_horizon_used = max(self.maximum_horizon_used, self._horizon)
        self.phase = f"Planning Agent {agent.agent_id}; attempt {len(self.attempted_orders)}; horizon {self._horizon}"
        self._search.step()
        if self.observe:
            self.last_search_observation = self._search.last_observation
            observation = self.last_search_observation
            if observation.expanded is not None:
                self._emit("Search", f"Agent {agent.agent_id} expanded {observation.expanded}.", agent.agent_id, state=observation.expanded)
            for rejected in observation.rejected:
                self._emit("Rejected", f"Agent {agent.agent_id}: {rejected.source} -> {rejected.target}, arrival t={rejected.arrival}: {rejected.reason}.", agent.agent_id, reason=rejected.reason)
        result = self._search.result
        if result is None:
            return
        self._expanded += result.expanded_states
        self._space_expanded += result.expanded_states
        self._generated += result.generated_states
        self._peak = max(self._peak, result.peak_frontier_size)
        self._search = None
        if result.found:
            self._paths[agent.agent_id] = result.path
            self._reservations.reserve_path(result.path)
            self._emit("Reservation", f"Agent {agent.agent_id} path found (cost {result.path_cost}) and reserved; goal held from t={result.path_cost}.", agent.agent_id)
            self._index += 1
            if self._index < len(self.order):
                self._horizon = self._initial_horizon(self.order[self._index])
        elif result.failure_reason == "expansion_limit":
            self._emit("Failure", f"Agent {agent.agent_id}: expansion_limit; finite search budget, not proof of infeasibility.", agent.agent_id, reason=result.failure_reason)
            self._finish("expansion_limit", limited=True)
        elif result.failure_reason == "horizon_exhausted" and self._horizon < self.max_horizon:
            self._horizon = min(self.max_horizon, max(1, self._horizon * 2))
            self.horizon_retries += 1
            self._emit("Horizon", f"Agent {agent.agent_id}: horizon exhausted; retry with horizon {self._horizon}.", agent.agent_id, reason=result.failure_reason)
        else:
            self._adapt(agent.agent_id, result.failure_reason)

    def _initial_horizon(self, agent_id: int) -> int:
        return min(self.max_horizon, self.analysis.independent_path_cost[agent_id])

    def _start_attempt(self) -> None:
        self.attempted_orders.append(self.order)
        self._reservations = ReservationTable()
        self._paths = {}
        self._index = 0
        self._horizon = self._initial_horizon(self.order[0])
        self._search = None
        self.last_search_observation = None
        self._emit("Planning", f"Attempt {len(self.attempted_orders)}: reservations cleared before replanning; order {self.order}.", new_order=self.order)

    def _adapt(self, failed: int, reason: str) -> None:
        self._emit("Failure", f"Agent {failed}: {reason}; bounded prioritized search failed, not a proof that no joint solution exists.", failed, reason=reason)
        limited = reason == "horizon_exhausted"
        if not self.adaptive:
            policy = "fixed_priority" if self.initial_priority == "fixed" else "initial_conflict_priority"
            self._finish(f"{policy}_failed: Agent {failed}, {reason}", limited)
            return
        if len(self.attempted_orders) - 1 >= self.max_priority_retries:
            self._finish(f"priority_retry_limit: Agent {failed}, {reason}", True)
            return
        candidate = next_promotion(self.order, failed, set(self.attempted_orders))
        if candidate is None:
            self._finish(f"no_new_priority_order: Agent {failed}, {reason}", limited)
            return
        old_order = self.order
        self.order, distance = candidate
        self.priority_promotions += distance
        self._emit("Promotion", f"Agent {failed}: {reason}; {old_order} -> {self.order}; next attempt {len(self.attempted_orders) + 1}.", failed,
                   old_order=old_order, new_order=self.order, reason=reason)
        self._start_attempt()

    def _finish(self, failure: str | None, limited: bool = False) -> None:
        success = failure is None
        events = detect_collisions(self._paths) if success else None
        if events:
            raise RuntimeError(f"Cooperative planning correctness failure: {events}")
        metrics = compute_metrics(self._paths, events) if success else None
        self.phase = "Complete" if success else failure
        self.result = CooperativeResult(
            found=success, individual_planning_success=all(plan.result.found for plan in self.independent.plans),
            paths=dict(self._paths) if success else {},
            initial_priority_order=self.attempted_orders[0] if self.attempted_orders else self.order,
            final_priority_order=self.order, attempted_orders=tuple(self.attempted_orders),
            planning_attempts=len(self.attempted_orders), priority_promotions=self.priority_promotions,
            analysis=self.analysis, sum_of_costs=metrics.sum_of_costs if metrics else None,
            makespan=metrics.makespan if metrics else None, wait_actions=metrics.wait_actions if metrics else None,
            expanded_states=self._expanded, generated_states=self._generated, peak_frontier_size=self._peak,
            elapsed_ms=0.0, vertex_collisions=0 if success else None, edge_collisions=0 if success else None,
            failure_reason=failure, search_limit_reached=limited, horizon_retries=self.horizon_retries,
            maximum_horizon_used=self.maximum_horizon_used, horizon_limit=self.max_horizon,
        )
        self._emit("Ready" if success else "Failed",
                   "All coordinated paths ready for joint execution." if success else f"No complete solution: {failure}; finite limits/priorities do not prove infeasibility.",
                   reason=failure)


def cooperative_plan(grid: Grid, agents: Sequence[Agent], **options) -> CooperativeResult:
    planner = CooperativePlanner(grid, agents, **options)
    while not planner.done:
        planner.step()
    return planner.result
