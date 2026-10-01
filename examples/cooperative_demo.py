from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathfinding.cooperative import cooperative_plan
from pathfinding.demos import conflict_demo
from pathfinding.multi_agent import compute_metrics, independent_astar


def main() -> None:
    grid, agents = conflict_demo()
    independent = independent_astar(grid, agents)
    metrics = compute_metrics({plan.agent.agent_id: plan.path for plan in independent})
    coordinated = cooperative_plan(grid, agents)
    assert metrics.vertex_collisions == 1
    assert coordinated.found and coordinated.vertex_collisions == coordinated.edge_collisions == 0
    assert coordinated.wait_actions >= 1
    print(f"Same 5x5 corridor map, same {len(agents)} agents and endpoints.")
    print(f"Independent A*: vertex={metrics.vertex_collisions}, edge={metrics.edge_collisions}, "
          f"sum_of_costs={metrics.sum_of_costs}, makespan={metrics.makespan}")
    print(f"CG-ST-A*: vertex={coordinated.vertex_collisions}, edge={coordinated.edge_collisions}, "
          f"sum_of_costs={coordinated.sum_of_costs}, makespan={coordinated.makespan}, waits={coordinated.wait_actions}")
    print(f"Priority={coordinated.final_priority_order}, attempts={coordinated.planning_attempts}, "
          f"promotions={coordinated.priority_promotions}")


if __name__ == "__main__":
    main()
