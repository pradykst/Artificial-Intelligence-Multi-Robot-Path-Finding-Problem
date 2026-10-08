from dataclasses import FrozenInstanceError, replace

import pytest

from pathfinding.agents import Agent, random_agents
from pathfinding.collisions import detect_collisions
from pathfinding.cooperative import CooperativePlanner, cooperative_plan
from pathfinding.demo_state import DemoState
from pathfinding.demos import conflict_demo, promotion_demo
from pathfinding.grid import Grid
from pathfinding.planning_view import EventLog, PlannerEvent, advance_planning, priority_rows, rank_badges
from pathfinding.reservations import ReservationTable
from pathfinding.space_time import SpaceTimeSearch


def finish(planner):
    events = []
    while not planner.done:
        planner.step()
        events.extend(planner.drain_events())
    return events


def test_progressive_analysis_and_exact_priority_values():
    grid, agents = conflict_demo()
    planner = CooperativePlanner(grid, agents, observe=True)
    assert all(row.path_cost is None for row in priority_rows(planner))
    while len(planner.independent.plans) < 1:
        planner.step()
    rows = priority_rows(planner)
    assert rows[0].path_cost == 4 and rows[1].path_cost is None
    assert rows[0].conflict_load is rows[0].current_rank is None
    while planner.analysis is None:
        planner.step()
    rows = priority_rows(planner)
    assert [(r.conflict_load, r.bottleneck_exposure, r.path_cost) for r in rows] == [(1, 4, 4), (1, 4, 4)]
    assert rank_badges(rows) == {1: 1, 2: 2}
    assert planner.snapshot().phase == "Ranking"
    assert planner.snapshot().predicted_conflicts[0].timestep == 2
    with pytest.raises(FrozenInstanceError):
        rows[0].path_cost = 99


def test_actual_event_order_and_reservation_completion():
    grid, agents = conflict_demo()
    planner = CooperativePlanner(grid, agents, observe=True)
    events = finish(planner)
    kinds = [event.kind for event in events]
    assert kinds.count("Analysis") == 3
    assert kinds.index("Priority") < kinds.index("Planning") < kinds.index("Search") < kinds.index("Reservation") < kinds.index("Ready")
    reserved = [event.agent_id for event in events if event.kind == "Reservation"]
    assert tuple(reserved) == planner.result.final_priority_order
    assert any(event.kind == "Rejected" and "reservation" in event.reason for event in events)
    assert any(event.kind == "Horizon" for event in events)
    assert all(event.state is not None for event in events if event.kind == "Search")
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert detect_collisions(planner.result.paths) == ()


def test_real_historical_promotion_updates_ranks_and_clears_reservations():
    grid, agents = promotion_demo()
    planner = CooperativePlanner(grid, agents, observe=True)
    promoted = None
    while not planner.done:
        planner.step()
        for event in planner.drain_events():
            if event.kind == "Promotion":
                promoted = event
                snapshot = planner.snapshot()
                assert snapshot.paths == () and snapshot.reservations.vertices == frozenset()
                assert snapshot.reservations.edges == () and snapshot.reservations.terminal_goals == frozenset()
                badges = rank_badges(snapshot.priorities)
                assert badges[8] == 4 and badges[5] == 5
                assert next(row for row in snapshot.priorities if row.agent_id == 8).initial_rank == 5
    assert promoted is not None and promoted.agent_id == 8 and promoted.reason
    assert promoted.old_order == (4, 2, 6, 5, 8, 3, 1, 7)
    assert promoted.new_order == (4, 2, 6, 8, 5, 3, 1, 7)
    assert planner.result.found and planner.result.priority_promotions == 1 and planner.result.planning_attempts == 2
    assert planner.result.expanded_states == 6480 and planner.result.sum_of_costs == 48


@pytest.mark.parametrize("scenario", ["conflict", "promotion", "unreachable", "limited", "random"])
def test_observations_and_playback_leave_paths_and_all_deterministic_metrics_unchanged(scenario):
    options = {}
    if scenario == "promotion":
        grid, agents = promotion_demo()
    elif scenario == "unreachable":
        grid = Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0))
        agents = (Agent(1, grid.start, grid.goal),)
    elif scenario == "random":
        grid = Grid.random(20, 20, .2, 42)
        agents = random_agents(grid, 6, 42)
    else:
        grid, agents = conflict_demo()
        if scenario == "limited":
            options = {"max_expansions": 1}
    normal = cooperative_plan(grid, agents, **options)
    detailed = CooperativePlanner(grid, agents, observe=True, **options)
    finish(detailed)
    agent_level = CooperativePlanner(grid, agents, observe=True, **options)
    while not agent_level.done:
        advance_planning(agent_level, False)
        agent_level.drain_events()
    assert replace(normal, elapsed_ms=0) == replace(detailed.result, elapsed_ms=0) == replace(agent_level.result, elapsed_ms=0)


def test_space_time_reports_actual_state_and_exact_vertex_and_edge_rejections():
    grid = Grid(3, 2, frozenset(), (0, 0), (2, 0))
    table = ReservationTable(vertices={((1, 0), 1)}, edges={((0, 1), (0, 0), 1)})
    search = SpaceTimeSearch(grid, grid.start, grid.goal, table, 8, observe=True)
    search.step()
    observation = search.last_observation
    assert observation.expanded == ((0, 0), 0)
    rejections = {(r.target, r.arrival, r.reason) for r in observation.rejected}
    assert ((1, 0), 1, "vertex reservation") in rejections
    assert ((0, 1), 1, "edge-swap reservation") in rejections
    assert ((0, 0), 1) in observation.opened  # Actual WAIT successor.
    assert len(search.frontier(1)) == 1


def test_time_specific_reservations_and_persistent_goal_hold_are_immutable():
    table = ReservationTable()
    table.reserve_path(((0, 0), (1, 0), (2, 0)))
    assert table.at_time(0).vertices == frozenset({(0, 0)})
    assert table.at_time(1).vertices == frozenset({(1, 0)})
    assert table.at_time(1).edges == (((0, 0), (1, 0)),)
    assert table.at_time(2).terminal_goals == frozenset({(2, 0)})
    assert table.at_time(100).vertices == frozenset({(2, 0)})
    assert table.at_time(100).edges == ()
    snapshot = table.at_time(1)
    table.reserve_path(((1, 1), (2, 1)))
    assert snapshot.vertices == frozenset({(1, 0)})


def test_planner_and_console_buffers_are_bounded_and_summary_filters_search():
    grid, agents = promotion_demo()
    planner = CooperativePlanner(grid, agents, observe=True, event_capacity=7)
    while not planner.done:
        planner.step()
    events = planner.drain_events()
    assert len(events) == 7 and events[-1].kind == "Ready"
    assert planner.drain_events() == ()
    log = EventLog(3)
    for index in range(10):
        log.append(PlannerEvent(index, "Priority", str(index)))
    assert list(log.lines) == ["[Priority] 7", "[Priority] 8", "[Priority] 9"]
    log.append(PlannerEvent(11, "Search", "expanded"))
    assert list(log.lines)[-1] == "[Priority] 9"
    log.append(PlannerEvent(12, "Search", "expanded"), detailed=True)
    assert len(log.lines) == 3 and log.lines[-1] == "[Search] expanded"
    log.clear()
    assert not log.lines


def test_detailed_step_is_one_engine_step_and_agent_mode_is_bounded():
    grid, agents = conflict_demo()
    planner = CooperativePlanner(grid, agents, observe=True)
    assert advance_planning(planner, True)
    assert planner.independent.search.expanded_nodes == 1
    assert not planner.independent.plans
    assert not advance_planning(planner, False, budget=1)
    assert planner.independent.search.expanded_nodes == 2


def test_reset_preserves_scenario_and_clears_obsolete_explanation():
    grid, agents = conflict_demo()
    state = DemoState(grid=grid, agents=agents, multi_agent=True, algorithm="CG-ST-A*")
    state.cooperative = CooperativePlanner(grid, agents, observe=True)
    state.cooperative.step()
    state.planning_snapshot = state.cooperative.snapshot()
    state.planning_log.append(PlannerEvent(1, "Analysis", "started"))
    state.reset()
    assert state.grid is grid and state.agents is agents
    assert state.planning_snapshot is state.cooperative is None and not state.planning_log.lines
