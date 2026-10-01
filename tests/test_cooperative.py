from dataclasses import replace

import pytest

from pathfinding.agents import Agent, random_agents
from pathfinding.collisions import Collision, detect_collisions
from pathfinding.cooperative import analyze_priorities, cooperative_plan, next_promotion
from pathfinding.demo_state import DemoState
from pathfinding.demos import conflict_demo
from pathfinding.grid import Grid
from pathfinding.multi_agent import compute_metrics, independent_astar
from pathfinding.reservations import ReservationTable
from pathfinding.space_time import space_time_astar


def test_reserve_entire_path_vertices_edges_wait_and_terminal():
    table = ReservationTable()
    table.reserve_path(((0, 0), (1, 0), (1, 0), (2, 0)))
    assert table.vertex_reserved((0, 0), 0)
    assert table.vertex_reserved((1, 0), 1) and table.vertex_reserved((1, 0), 2)
    assert table.edge_reserved((0, 0), (1, 0), 1)
    assert table.edge_reserved((1, 0), (1, 0), 2)
    assert not table.allows((1, 0), (0, 0), 1)
    assert table.vertex_reserved((2, 0), 3) and table.vertex_reserved((2, 0), 1_000_000)
    assert not table.vertex_reserved((2, 0), 2)
    assert not table.vertex_reserved((0, 0), 1)
    assert not table.can_hold_goal((2, 0), 0)
    assert not table.can_hold_goal((1, 0), 1)
    assert table.can_hold_goal((1, 0), 3)
    assert len(table.vertices) == 4


def test_space_time_without_reservations():
    grid = Grid(4, 3, frozenset(), (0, 0), (3, 2))
    result = space_time_astar(grid, grid.start, grid.goal, ReservationTable(), 5)
    assert result.found and result.path_cost == 5
    assert result.path[0] == grid.start and result.path[-1] == grid.goal
    assert result.expanded_states > 0 and result.elapsed_ms >= 0


def test_wait_is_required_by_reserved_vertex():
    grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
    table = ReservationTable(vertices={((1, 0), 1)})
    result = space_time_astar(grid, grid.start, grid.goal, table, 4)
    assert result.path == ((0, 0), (0, 0), (1, 0), (2, 0))
    assert result.path_cost == 3
    assert all(not table.vertex_reserved(cell, t) for t, cell in enumerate(result.path))


def test_avoids_edge_swap_with_detour():
    grid = Grid(2, 2, frozenset(), (0, 0), (1, 0))
    prior = ((1, 0), (0, 0), (0, 1))
    table = ReservationTable()
    table.reserve_path(prior)
    result = space_time_astar(grid, grid.start, grid.goal, table, 6)
    assert result.found and result.path[1] != (1, 0)
    assert detect_collisions({1: prior, 2: result.path}) == ()


def test_space_time_obstacles():
    grid = Grid(3, 3, frozenset({(1, 0), (1, 1)}), (0, 0), (2, 0))
    result = space_time_astar(grid, grid.start, grid.goal, ReservationTable(), 6)
    assert result.found and result.path_cost == 6
    assert all(grid.is_free(cell) for cell in result.path)


def test_horizon_and_expansion_limits_are_explicit():
    grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
    result = space_time_astar(grid, grid.start, grid.goal, ReservationTable(), 1)
    assert not result.found and result.failure_reason == "horizon_exhausted"
    assert result.path == () and result.path_cost is None
    result = space_time_astar(grid, grid.start, grid.goal, ReservationTable(), 5, max_expansions=1)
    assert not result.found and result.failure_reason == "expansion_limit"
    assert result.expanded_states == 1


def test_terminal_goal_and_start_reservations():
    grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
    table = ReservationTable()
    table.reserve_path(((1, 0), (2, 0)))
    result = space_time_astar(grid, grid.start, grid.goal, table, 10)
    assert not result.found and result.failure_reason == "goal_reserved"
    table = ReservationTable(vertices={(grid.start, 0)})
    assert space_time_astar(grid, grid.start, grid.goal, table, 10).failure_reason == "start_reserved"


def test_goal_cannot_be_reached_then_abandoned_before_future_reservation():
    grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
    table = ReservationTable(vertices={(grid.goal, 5)})
    result = space_time_astar(grid, grid.start, grid.goal, table, 10)
    assert result.found and result.path_cost == 6
    assert grid.goal not in result.path[:-1]
    assert not space_time_astar(grid, grid.start, grid.goal, table, 4).found


def test_already_at_goal_is_zero_cost_only_if_safe_forever():
    grid = Grid(2, 1, frozenset(), (0, 0), (1, 0))
    assert space_time_astar(grid, grid.start, grid.start, ReservationTable(), 0).path_cost == 0
    table = ReservationTable(vertices={(grid.start, 3)})
    assert not space_time_astar(grid, grid.start, grid.start, table, 10).found


def test_known_priority_analysis_and_id_tie_break():
    grid, agents = conflict_demo()
    plans = independent_astar(grid, tuple(reversed(agents)))
    analysis = analyze_priorities(grid, plans)
    assert analysis.conflict_load == {1: 1, 2: 1}
    assert analysis.bottleneck_exposure == {1: 4, 2: 4}
    assert analysis.independent_path_cost == {1: 4, 2: 4}
    assert analysis.conflict_graph == {1: (2,), 2: (1,)}
    assert analysis.priority_order == (1, 2)
    assert analysis == analyze_priorities(grid, plans)


def test_three_way_conflict_load_counts_events_not_pairs():
    grid = Grid(3, 3, frozenset(), (0, 1), (2, 1))
    agents = (Agent(1, (0, 1), (2, 1)), Agent(2, (1, 0), (1, 2)), Agent(3, (2, 1), (0, 1)))
    analysis = analyze_priorities(grid, independent_astar(grid, agents))
    assert analysis.conflict_load == {1: 1, 2: 1, 3: 1}
    assert analysis.conflict_graph[1] == (2, 3)


def test_priority_is_exact_lexicographic_rule():
    grid = Grid.random(8, 7, 0.25, 12)
    agents = random_agents(grid, 5, 12)
    analysis = analyze_priorities(grid, independent_astar(grid, agents))
    expected = tuple(sorted((agent.agent_id for agent in agents), key=lambda i: (
        -analysis.conflict_load[i], -analysis.bottleneck_exposure[i], -analysis.independent_path_cost[i], i)))
    assert analysis.priority_order == expected


def test_crossing_solution_has_waits_correct_goals_and_no_collisions():
    grid, agents = conflict_demo()
    result = cooperative_plan(grid, agents)
    assert result.found and result.vertex_collisions == result.edge_collisions == 0
    assert result.wait_actions == 1 and result.sum_of_costs == 9 and result.makespan == 5
    assert result.horizon_retries >= 1
    assert result.horizon_limit == 9 + 4
    assert detect_collisions(result.paths) == ()
    for agent in agents:
        path = result.paths[agent.agent_id]
        assert path[0] == agent.start and path[-1] == agent.goal
        assert all(grid.is_free(cell) for cell in path)
        assert all(a == b or b in tuple(grid.neighbors(a)) for a, b in zip(path, path[1:]))


def test_edge_conflict_resolved_when_alternate_route_exists():
    grid = Grid(4, 2, frozenset(), (0, 0), (2, 0))
    agents = (Agent(1, (0, 0), (2, 0)), Agent(2, (3, 0), (1, 0)))
    plain = independent_astar(grid, agents)
    assert compute_metrics({p.agent.agent_id: p.path for p in plain}).edge_collisions == 1
    coordinated = cooperative_plan(grid, agents)
    assert coordinated.found and not detect_collisions(coordinated.paths)


def test_real_priority_promotion_recovers_solution():
    grid = Grid.random(5, 4, 0.2, 4)
    agents = random_agents(grid, 3, 4)
    result = cooperative_plan(grid, agents, max_expansions=5000)
    assert result.found and result.initial_priority_order == (3, 2, 1)
    assert result.final_priority_order == (2, 3, 1)
    assert result.planning_attempts == 2 and result.priority_promotions == 1
    assert len(set(result.attempted_orders)) == result.planning_attempts


def test_promotion_skips_visited_orders_and_terminates():
    order = (1, 2, 3)
    seen = {order, (1, 3, 2)}
    assert next_promotion(order, 3, seen) == ((3, 1, 2), 2)
    seen.add((3, 1, 2))
    assert next_promotion(order, 3, seen) is None
    assert next_promotion(order, 1, set()) is None


def test_impossible_corridor_terminates_without_repeated_orders():
    grid = Grid(4, 1, frozenset(), (0, 0), (2, 0))
    agents = (Agent(1, (0, 0), (2, 0)), Agent(2, (3, 0), (1, 0)))
    result = cooperative_plan(grid, agents, max_horizon=10, max_priority_retries=3, max_expansions=500)
    assert not result.found and result.failure_reason
    assert result.paths == {} and result.sum_of_costs is None and result.vertex_collisions is None
    assert result.planning_attempts <= 4
    assert len(result.attempted_orders) == len(set(result.attempted_orders))


@pytest.mark.parametrize("limits", [{"max_horizon": 0}, {"max_expansions": 1}])
def test_cooperative_search_limit_is_exposed(limits):
    grid, agents = conflict_demo()
    result = cooperative_plan(grid, agents, **limits)
    assert not result.found and result.search_limit_reached


def test_unreachable_individual_fails_before_priority_planning():
    grid = Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0))
    result = cooperative_plan(grid, (Agent(1, grid.start, grid.goal),))
    assert not result.found and not result.individual_planning_success
    assert result.planning_attempts == 0 and result.failure_reason == "individual_path_unreachable"


def test_fixed_priority_reuses_engine_without_promotions():
    grid, agents = conflict_demo()
    result = cooperative_plan(grid, tuple(reversed(agents)), adaptive=False)
    assert result.found and result.initial_priority_order == (1, 2)
    assert result.priority_promotions == 0 and result.planning_attempts == 1


def test_costs_and_waits_exclude_explicit_goal_padding():
    paths = {1: ((0, 0), (0, 0), (1, 0), (1, 0), (1, 0)), 2: ((0, 1), (1, 1), (2, 1))}
    metrics = compute_metrics(paths)
    assert metrics.individual_costs == {1: 2, 2: 2}
    assert metrics.sum_of_costs == 4 and metrics.makespan == 2 and metrics.wait_actions == 1


def test_existing_detector_is_the_final_correctness_gate(monkeypatch):
    import pathfinding.cooperative as module

    calls = []
    real_detector = module.detect_collisions

    def detector(paths):
        calls.append(paths)
        if len(calls) == 2:
            return (Collision(1, "vertex", (1, 2), ((2, 2),)),)
        return real_detector(paths)

    monkeypatch.setattr(module, "detect_collisions", detector)
    grid, agents = conflict_demo()
    with pytest.raises(RuntimeError, match="correctness failure"):
        module.cooperative_plan(grid, agents)
    assert len(calls) == 2


def test_algorithm_switch_preserves_scenario_and_clears_state():
    grid, agents = conflict_demo()
    state = DemoState(grid=grid, agents=agents, multi_agent=True)
    state.set_algorithm("CG-ST-A*")
    assert state.grid is grid and state.agents is agents
    assert state.planner is state.cooperative is state.simulation is None
    state.set_algorithm("Independent A*")
    assert state.grid is grid and state.agents is agents
    state.set_mode(False, 0, 0)
    assert state.grid is grid and state.agents == () and not state.multi_agent


def test_random_successes_are_valid_and_deterministic():
    successes = 0
    for seed in range(15):
        grid = Grid.random(6, 5, 0.2, seed)
        agents = random_agents(grid, 3, seed)
        first = cooperative_plan(grid, agents, max_expansions=3000)
        second = cooperative_plan(grid, agents, max_expansions=3000)
        assert replace(first, elapsed_ms=0) == replace(second, elapsed_ms=0)
        if first.found:
            successes += 1
            assert detect_collisions(first.paths) == ()
            for agent in agents:
                assert first.paths[agent.agent_id][-1] == agent.goal
    assert successes > 0
