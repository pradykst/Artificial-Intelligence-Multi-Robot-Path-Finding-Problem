from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathfinding.agents import Agent
from pathfinding.grid import Grid
from pathfinding.multi_agent import Simulation, independent_astar


def main() -> None:
    cases = [
        ("collision-free", Grid(3, 2, frozenset(), (0, 0), (2, 0)),
         (Agent(1, (0, 0), (2, 0)), Agent(2, (0, 1), (2, 1))), (0, 0)),
        ("vertex", Grid(3, 3, frozenset(), (0, 1), (2, 1)),
         (Agent(1, (0, 1), (2, 1)), Agent(2, (1, 0), (1, 2))), (1, 0)),
        ("edge", Grid(4, 1, frozenset(), (0, 0), (2, 0)),
         (Agent(1, (0, 0), (2, 0)), Agent(2, (3, 0), (1, 0))), (0, 1)),
    ]
    for name, grid, agents, expected in cases:
        plans = independent_astar(grid, agents)
        simulation = Simulation({plan.agent.agent_id: plan.path for plan in plans})
        while not simulation.done:
            simulation.step()
        metrics = simulation.metrics
        assert (metrics.vertex_collisions, metrics.edge_collisions) == expected
        assert all(simulation.positions[agent.agent_id] == agent.goal for agent in agents)
        print(f"{name}: vertex={metrics.vertex_collisions}, edge={metrics.edge_collisions}, "
              f"total={metrics.total_collisions}, sum_of_costs={metrics.sum_of_costs}, makespan={metrics.makespan}")
        for event in simulation.collisions:
            print(f"  t={event.timestep}: {event.kind}, agents={event.agent_ids}, cells={event.cells}")


if __name__ == "__main__":
    main()
