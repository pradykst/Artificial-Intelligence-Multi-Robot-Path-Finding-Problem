from .agents import Agent, random_agents
from .grid import Grid


def conflict_demo() -> tuple[Grid, tuple[Agent, ...]]:
    """Two perpendicular corridor routes meet at t=2 under Independent A*."""
    obstacles = frozenset((x, y) for y in range(5) for x in range(5) if x != 2 and y != 2)
    grid = Grid(5, 5, obstacles, (0, 2), (4, 2))
    agents = (Agent(1, (0, 2), (4, 2)), Agent(2, (2, 0), (2, 4)))
    return grid, agents


def promotion_demo() -> tuple[Grid, tuple[Agent, ...]]:
    """Historical s0-p2-a8-t0004: seed 156, 10x10, p=0.3, eight agents."""
    grid = Grid.random(10, 10, 0.3, 156)
    return grid, random_agents(grid, 8, 156)
