from dataclasses import dataclass, field

from .agents import Agent, random_agents
from .astar import AStarSearch
from .cooperative import CooperativePlanner
from .decentralized import DCN, DecentralizedPlanner
from .grid import Grid
from .multi_agent import IndependentPlanner, Simulation
from .planning_view import EventLog, PlanningSnapshot


@dataclass
class DemoState:
    """Display-independent mode changes discard all obsolete planning state."""

    grid: Grid | None = None
    multi_agent: bool = False
    agents: tuple[Agent, ...] = ()
    search: AStarSearch | None = None
    planner: IndependentPlanner | None = None
    simulation: Simulation | None = None
    algorithm: str = "Independent A*"
    cooperative: CooperativePlanner | None = None
    decentralized: DecentralizedPlanner | None = None
    planning_snapshot: PlanningSnapshot | None = None
    planning_log: EventLog = field(default_factory=EventLog)

    def reset(self) -> None:
        self.search = None
        self.planner = None
        self.simulation = None
        self.cooperative = None
        self.decentralized = None
        self.planning_snapshot = None
        self.planning_log.clear()

    def set_algorithm(self, algorithm: str) -> None:
        if algorithm not in ("Independent A*", "CG-ST-A*", DCN):
            raise ValueError("Unknown multi-agent algorithm.")
        self.reset()
        self.algorithm = algorithm

    def set_mode(self, enabled: bool, count: int, seed: int) -> None:
        if self.grid is None:
            raise ValueError("Generate a grid first.")
        agents = random_agents(self.grid, count, seed) if enabled else ()
        self.reset()
        self.multi_agent = enabled
        self.agents = agents

    def set_grid(self, grid: Grid, count: int, seed: int) -> None:
        agents = random_agents(grid, count, seed) if self.multi_agent else ()
        self.reset()
        self.grid = grid
        self.agents = agents

    def randomize_agents(self, count: int, seed: int) -> None:
        if not self.multi_agent or self.grid is None:
            raise ValueError("Enable multi-agent mode on an existing grid first.")
        agents = random_agents(self.grid, count, seed)
        self.reset()
        self.agents = agents
