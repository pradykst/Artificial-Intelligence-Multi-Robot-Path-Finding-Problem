from dataclasses import dataclass
from typing import Sequence

from .agents import Agent
from .astar import AStarSearch
from .collisions import Collision, Paths, collisions_at, detect_collisions, position_at
from .grid import Cell, Grid
from .heuristics import Heuristic, manhattan
from .metrics import SearchResult


@dataclass(frozen=True)
class AgentPlan:
    agent: Agent
    result: SearchResult

    @property
    def path(self) -> tuple[Cell, ...]:
        return self.result.path


class IndependentPlanner:
    """Incremental ordinary A*: every robot ignores every other robot."""

    def __init__(self, grid: Grid, agents: Sequence[Agent], heuristic: Heuristic = manhattan) -> None:
        if not agents or len({agent.agent_id for agent in agents}) != len(agents):
            raise ValueError("Provide at least one agent with unique agent IDs.")
        if any(not grid.is_free(cell) for agent in agents for cell in (agent.start, agent.goal)):
            raise ValueError("All agent endpoints must be free cells inside the grid.")
        self.grid = grid
        self.agents = tuple(agents)
        self.heuristic = heuristic
        self.plans: list[AgentPlan] = []
        self.search: AStarSearch | None = None

    @property
    def done(self) -> bool:
        return len(self.plans) == len(self.agents)

    def step(self) -> None:
        """Perform at most one expansion, allowing the UI to schedule planning."""
        if self.done:
            return
        agent = self.agents[len(self.plans)]
        if self.search is None:
            self.search = AStarSearch(self.grid, self.heuristic, start=agent.start, goal=agent.goal)
        self.search.step()
        if self.search.result is not None:
            self.plans.append(AgentPlan(agent, self.search.result))
            self.search = None


def independent_astar(grid: Grid, agents: Sequence[Agent], heuristic: Heuristic = manhattan) -> tuple[AgentPlan, ...]:
    planner = IndependentPlanner(grid, agents, heuristic)
    while not planner.done:
        planner.step()
    return tuple(planner.plans)


@dataclass(frozen=True)
class MultiAgentMetrics:
    individual_costs: dict[int, int]
    sum_of_costs: int
    makespan: int
    vertex_collisions: int
    edge_collisions: int
    total_collisions: int
    wait_actions: int = 0


def arrival_cost(path: Sequence[Cell]) -> int:
    """First goal arrival excludes any explicitly padded holding at that goal."""
    if not path:
        raise ValueError("Path costs are undefined for an unreachable agent.")
    arrival = path.index(path[-1])
    if any(cell != path[-1] for cell in path[arrival:]):
        raise ValueError("A robot must remain at its goal after first arrival.")
    return arrival


def compute_metrics(paths: Paths, collisions: Sequence[Collision] | None = None) -> MultiAgentMetrics:
    """Use complete trajectories; optionally count only an observed event prefix."""
    if any(not path for path in paths.values()):
        raise ValueError("Path costs are undefined for an unreachable agent.")
    costs = {agent_id: arrival_cost(path) for agent_id, path in paths.items()}
    waits = sum(path[t] == path[t - 1] for agent_id, path in paths.items() for t in range(1, costs[agent_id] + 1))
    events = detect_collisions(paths) if collisions is None else collisions
    vertices = sum(event.kind == "vertex" for event in events)
    edges = sum(event.kind == "edge" for event in events)
    return MultiAgentMetrics(costs, sum(costs.values()), max(costs.values(), default=0), vertices, edges, vertices + edges, waits)


class Simulation:
    """Execute supplied trajectories concurrently without modifying any path."""

    def __init__(self, paths: Paths) -> None:
        if not paths or any(not path for path in paths.values()):
            raise ValueError("Simulation requires a nonempty path for every agent.")
        self.paths = {agent_id: tuple(path) for agent_id, path in paths.items()}
        self.timestep = 0
        self.makespan = max(len(path) - 1 for path in self.paths.values())
        self.current_collisions = collisions_at(self.paths, 0)
        self.collisions = list(self.current_collisions)

    @property
    def done(self) -> bool:
        return self.timestep == self.makespan

    @property
    def positions(self) -> dict[int, Cell]:
        return {agent_id: position_at(path, self.timestep) for agent_id, path in self.paths.items()}

    @property
    def metrics(self) -> MultiAgentMetrics:
        return compute_metrics(self.paths, self.collisions)

    def step(self) -> tuple[Collision, ...]:
        if self.done:
            return ()
        self.timestep += 1
        self.current_collisions = collisions_at(self.paths, self.timestep)
        self.collisions.extend(self.current_collisions)
        return self.current_collisions
