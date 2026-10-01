from dataclasses import dataclass
from random import Random

from .grid import Cell, Grid


@dataclass(frozen=True)
class Agent:
    agent_id: int
    start: Cell
    goal: Cell


def random_agents(grid: Grid, count: int, seed: int) -> tuple[Agent, ...]:
    """Choose disjoint endpoint pairs within free-space connected components.

    All 2*count endpoints are distinct. Pair selection is deterministic, but is
    not a uniform sample of all possible reachable endpoint assignments.
    """
    if count < 1:
        raise ValueError("At least one agent is required.")
    unseen = {(x, y) for y in range(grid.height) for x in range(grid.width) if grid.is_free((x, y))}
    components: list[list[Cell]] = []
    for cell in sorted(unseen):
        if cell not in unseen:
            continue
        unseen.remove(cell)
        pending = [cell]
        component = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbor in grid.neighbors(current):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        if len(component) >= 2:
            components.append(sorted(component))
    capacity = sum(len(component) // 2 for component in components)
    if capacity < count:
        raise ValueError(f"This map supports only {capacity} disjoint reachable endpoint pairs; reduce agents or generate another map.")
    rng = Random(seed)
    pairs = []
    for component in components:
        rng.shuffle(component)
        pairs.extend(zip(component[::2], component[1::2]))
    rng.shuffle(pairs)
    return tuple(Agent(index + 1, start, goal) for index, (start, goal) in enumerate(pairs[:count]))
