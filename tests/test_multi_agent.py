import pytest

from pathfinding.agents import Agent, random_agents
from pathfinding.astar import AStarSearch, astar
from pathfinding.collisions import Collision, collisions_at, detect_collisions, position_at
from pathfinding.demo_state import DemoState
from pathfinding.grid import Grid
from pathfinding.heuristics import REQUIRED_HEURISTICS
from pathfinding.multi_agent import IndependentPlanner, Simulation, compute_metrics, independent_astar


def test_collision_free_paths():
    paths = {1: ((0, 0), (1, 0), (2, 0)), 2: ((0, 1), (1, 1), (2, 1))}
    assert detect_collisions(paths) == ()


def test_known_vertex():
    paths = {1: ((0, 1), (1, 1), (2, 1)), 2: ((1, 0), (1, 1), (1, 2))}
    assert detect_collisions(paths) == (Collision(1, "vertex", (1, 2), ((1, 1),)),)


def test_known_edge_swap():
    paths = {1: ((0, 0), (1, 0), (2, 0)), 2: ((3, 0), (2, 0), (1, 0))}
    assert detect_collisions(paths) == (Collision(2, "edge", (1, 2), ((1, 0), (2, 0))),)


def test_collision_after_goal_arrival():
    paths = {1: ((0, 0), (1, 0)), 2: ((2, 1), (2, 0), (1, 0), (0, 0))}
    assert position_at(paths[1], 20) == (1, 0)
    assert detect_collisions(paths) == (Collision(2, "vertex", (1, 2), ((1, 0),)),)
    simulation = Simulation(paths)
    for _ in range(3):
        simulation.step()
    assert simulation.positions == {1: (1, 0), 2: (0, 0)}


def test_three_way_vertex_is_one_event_with_every_agent():
    paths = {3: ((1, 0), (1, 1)), 1: ((0, 1), (1, 1)), 2: ((2, 1), (1, 1))}
    assert detect_collisions(paths) == (Collision(1, "vertex", (1, 2, 3), ((1, 1),)),)
    assert compute_metrics(paths).vertex_collisions == 1


def test_shared_stationary_cell_is_not_an_edge_collision():
    paths = {1: ((1, 1),), 2: ((1, 1), (1, 1))}
    assert [event.kind for event in detect_collisions(paths)] == ["vertex", "vertex"]


@pytest.mark.parametrize("heuristic", REQUIRED_HEURISTICS.values(), ids=REQUIRED_HEURISTICS)
def test_independent_paths_match_canonical_astar(heuristic):
    grid = Grid(5, 4, frozenset({(2, 0), (2, 1), (2, 2)}), (0, 0), (4, 0))
    agents = (Agent(1, (0, 0), (4, 0)), Agent(2, (4, 1), (0, 1)))
    plans = independent_astar(grid, agents, heuristic)
    for plan in plans:
        reference = astar(grid, heuristic, start=plan.agent.start, goal=plan.agent.goal)
        assert plan.path == reference.path
        assert plan.result.path_cost == reference.path_cost
        assert plan.path[0] == plan.agent.start and plan.path[-1] == plan.agent.goal
        assert all(grid.is_free(cell) for cell in plan.path)
        assert all(second in tuple(grid.neighbors(first)) for first, second in zip(plan.path, plan.path[1:]))


def test_baseline_keeps_colliding_plans_and_continues():
    grid = Grid(3, 3, frozenset(), (0, 1), (2, 1))
    plans = independent_astar(grid, (Agent(1, (0, 1), (2, 1)), Agent(2, (1, 0), (1, 2))))
    simulation = Simulation({plan.agent.agent_id: plan.path for plan in plans})
    first = simulation.step()
    assert first[0].kind == "vertex" and simulation.timestep == 1
    assert not simulation.done
    simulation.step()
    assert simulation.done and simulation.timestep == 2
    assert simulation.metrics.total_collisions == 1
    assert simulation.positions == {plan.agent.agent_id: plan.agent.goal for plan in plans}
    assert simulation.step() == () and simulation.timestep == 2


def test_random_agents_are_unique_reachable_and_deterministic():
    grid = Grid.random(15, 12, 0.3, 42)
    first = random_agents(grid, 10, 99)
    assert first == random_agents(grid, 10, 99)
    assert first != random_agents(grid, 10, 100)
    endpoints = [cell for agent in first for cell in (agent.start, agent.goal)]
    assert len(set(endpoints)) == 20
    assert all(grid.is_free(cell) for cell in endpoints)
    assert all(plan.result.found for plan in independent_astar(grid, first))


def test_initialization_uses_multiple_components():
    grid = Grid(5, 1, frozenset({(2, 0)}), (0, 0), (1, 0))
    agents = random_agents(grid, 2, 1)
    assert all(plan.result.found for plan in independent_astar(grid, agents))
    with pytest.raises(ValueError, match="only 2"):
        random_agents(grid, 3, 1)
    with pytest.raises(ValueError):
        random_agents(grid, 0, 1)


def test_makespan_sum_costs_and_collision_metrics():
    paths = {1: ((0, 0), (1, 0)), 2: ((2, 1), (2, 0), (1, 0), (0, 0))}
    metrics = compute_metrics(paths)
    assert metrics.individual_costs == {1: 1, 2: 3}
    assert metrics.sum_of_costs == 4
    assert metrics.makespan == 3
    assert (metrics.vertex_collisions, metrics.edge_collisions, metrics.total_collisions) == (1, 0, 1)
    assert compute_metrics(paths, ()).total_collisions == 0
    swapped = compute_metrics({1: ((0, 0), (1, 0)), 2: ((1, 0), (0, 0))})
    assert (swapped.vertex_collisions, swapped.edge_collisions, swapped.total_collisions) == (0, 1, 1)


def test_unreachable_plan_is_explicit_and_cannot_be_simulated():
    grid = Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0))
    plans = independent_astar(grid, (Agent(1, grid.start, grid.goal),))
    assert not plans[0].result.found and plans[0].result.path_cost is None
    with pytest.raises(ValueError):
        Simulation({1: plans[0].path})
    with pytest.raises(ValueError):
        compute_metrics({1: ()})


def test_mode_changes_clear_state_without_a_display():
    grid = Grid.random(10, 10, 0.2, 42)
    state = DemoState(grid=grid)
    state.search = AStarSearch(grid)
    state.search.step()
    state.set_mode(True, 3, 10)
    assert state.grid is grid and state.search is None and len(state.agents) == 3
    state.planner = IndependentPlanner(grid, state.agents)
    while not state.planner.done:
        state.planner.step()
    state.simulation = Simulation({plan.agent.agent_id: plan.path for plan in state.planner.plans})
    state.simulation.step()
    state.set_mode(False, 0, 0)
    assert state.grid is grid and not state.multi_agent and state.agents == ()
    assert state.search is state.planner is state.simulation is None
    state.set_mode(True, 3, 10)
    assert state.agents == random_agents(grid, 3, 10)
    state.randomize_agents(2, 11)
    assert state.agents == random_agents(grid, 2, 11)
    state.set_grid(Grid.random(8, 8, 0.1, 2), 2, 2)
    assert state.grid != grid and state.simulation is None


def test_failed_mode_change_is_atomic():
    grid = Grid(2, 1, frozenset(), (0, 0), (1, 0))
    state = DemoState(grid=grid, search=AStarSearch(grid))
    original = state.search
    with pytest.raises(ValueError):
        state.set_mode(True, 2, 1)
    assert not state.multi_agent and state.grid is grid and state.search is original


def test_duplicate_ids_and_blocked_endpoints_rejected():
    grid = Grid(3, 2, frozenset({(1, 0)}), (0, 0), (2, 0))
    with pytest.raises(ValueError, match="unique"):
        independent_astar(grid, (Agent(1, (0, 0), (2, 0)), Agent(1, (2, 0), (0, 0))))
    with pytest.raises(ValueError, match="endpoints"):
        independent_astar(grid, (Agent(1, (1, 0), (2, 0)),))


def test_empty_or_negative_path_queries_rejected():
    with pytest.raises(ValueError):
        position_at((), 0)
    with pytest.raises(ValueError):
        position_at(((0, 0),), -1)
    with pytest.raises(ValueError):
        detect_collisions({1: ()})


def test_initial_collisions_and_finished_simulation_do_not_double_count():
    simulation = Simulation({1: ((0, 0),), 2: ((0, 0),)})
    assert simulation.done
    assert simulation.metrics.total_collisions == 1
    assert simulation.step() == ()
    assert simulation.metrics.total_collisions == 1
