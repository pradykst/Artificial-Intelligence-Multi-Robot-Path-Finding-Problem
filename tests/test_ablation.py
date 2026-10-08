from dataclasses import replace
import csv

import pytest

from pathfinding import experiments_multi as experiments
from pathfinding.analysis_ablation import ablation_comparisons, summarize_ablation
from pathfinding.analysis_multi import FIXED, INDEPENDENT, NO_PROMOTION, PROPOSED, summarize_methods
from pathfinding.collisions import detect_collisions
from pathfinding.cooperative import CooperativePlanner, cooperative_plan
from pathfinding.demos import conflict_demo, promotion_demo
from pathfinding.reservations import ReservationTable
from pathfinding.space_time import SpaceTimeSearch


def observed(grid, agents, **options):
    planner = CooperativePlanner(grid, agents, observe=True, **options)
    events = []
    while not planner.done:
        planner.step()
        events.extend(planner.drain_events())
        if planner._search:
            assert type(planner._search) is SpaceTimeSearch
            assert type(planner._search.reservations) is ReservationTable
    return planner.result, events


def test_controlled_initial_policies_and_real_promotion():
    grid, agents = promotion_demo()
    fixed = cooperative_plan(grid, tuple(reversed(agents)), adaptive=False)
    no, no_events = observed(grid, agents, adaptive=False, initial_priority="conflict")
    full, full_events = observed(grid, agents)
    assert fixed.initial_priority_order == tuple(range(1, 9))
    assert no.initial_priority_order == full.initial_priority_order == full.analysis.priority_order
    assert no.analysis == full.analysis
    assert no.initial_priority_order == (4, 2, 6, 5, 8, 3, 1, 7)
    assert not no.found and no.priority_promotions == 0 and no.planning_attempts == 1
    assert no.attempted_orders == (no.initial_priority_order,)
    assert no.final_priority_order == no.initial_priority_order
    assert not any(e.kind == "Promotion" for e in no_events)
    assert full.found and full.priority_promotions == 1 and full.planning_attempts == 2
    assert any(e.kind == "Promotion" and e.agent_id == 8 for e in full_events)
    assert not detect_collisions(full.paths)
    assert full.vertex_collisions == full.edge_collisions == 0
    assert no.sum_of_costs is no.makespan is no.wait_actions is None
    assert no.vertex_collisions is no.edge_collisions is None
    assert no.paths == {}
    # Same analysis, horizons, search decisions and reservations until promotion.
    kinds = {"Analysis", "Priority", "Conflict", "Search", "Rejected", "Reservation", "Horizon", "Failure", "Planning"}
    assert [e for e in no_events if e.kind in kinds] == [e for e in full_events if e.kind in kinds and e.attempt <= 1]
    assert no.horizon_limit == full.horizon_limit


@pytest.mark.parametrize("adaptive,policy", [(False, "fixed"), (True, "conflict")])
def test_existing_defaults_match_explicit_policy(adaptive, policy):
    for make in (conflict_demo, promotion_demo):
        grid, agents = make()
        old = cooperative_plan(grid, agents, adaptive=adaptive)
        explicit = cooperative_plan(grid, agents, adaptive=adaptive, initial_priority=policy)
        assert replace(old, elapsed_ms=0) == replace(explicit, elapsed_ms=0)


@pytest.mark.parametrize("make", [conflict_demo, promotion_demo])
def test_no_promotion_observation_preserves_results(make):
    grid, agents = make()
    normal = cooperative_plan(grid, agents, adaptive=False, initial_priority="conflict")
    logged, _ = observed(grid, agents, adaptive=False, initial_priority="conflict")
    assert replace(normal, elapsed_ms=0) == replace(logged, elapsed_ms=0)


def test_four_method_benchmark_pairing_observation_off_and_missing_method(monkeypatch, tmp_path):
    scenarios = experiments.generate_scenarios(3, [(6, 6)], [.2], [2], 42)
    actual = experiments.cooperative_plan
    calls = []

    def recording(grid, agents, **options):
        calls.append((grid, agents, options))
        assert not options.get("observe", False)
        return actual(grid, agents, **options)

    monkeypatch.setattr(experiments, "cooperative_plan", recording)
    rows = experiments.run_benchmark(scenarios, True, include_no_promotion=True)
    assert len(rows) == 12 and len(calls) == 9
    for index, scenario in enumerate(scenarios):
        assert all(g is scenario.grid and a is scenario.agents for g, a, _ in calls[index * 3:index * 3 + 3])
        group = [r for r in rows if r["scenario_id"] == scenario.scenario_id]
        assert {r["scenario_hash"] for r in group} == {experiments.scenario_hash(scenario)}
        no = next(r for r in group if r["algorithm"] == NO_PROMOTION)
        full = next(r for r in group if r["algorithm"] == PROPOSED)
        assert no["initial_priority_order"] == full["initial_priority_order"]
        assert no["initial_priority_order"] == no["final_priority_order"]
        assert no["priority_promotions"] == 0
    damaged = [r for r in rows if r["algorithm"] != NO_PROMOTION]
    with pytest.raises(ValueError, match="Missing algorithm"):
        ablation_comparisons(damaged)
    with pytest.raises(ValueError):
        experiments.save_results(scenarios, damaged, tmp_path, {"include_fixed": True, "include_no_promotion": True})
    assert not list(tmp_path.iterdir())
    from pathfinding import final_plots
    monkeypatch.setattr(final_plots, "plot_q2", lambda rows, output: [])
    experiments.save_results(scenarios, rows, tmp_path, {"include_no_promotion": True})
    with (tmp_path / "raw_results.csv").open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 12
    assert {r["algorithm"] for r in saved} == {INDEPENDENT, FIXED, NO_PROMOTION, PROPOSED}
    for scenario in scenarios:
        group = [r for r in saved if r["scenario_id"] == scenario.scenario_id]
        assert {r["seed"] for r in group} == {str(scenario.seed)}
        assert {r["scenario_hash"] for r in group} == {experiments.scenario_hash(scenario)}
    assert (tmp_path / "ablation_comparisons.csv").exists()
    assert (tmp_path / "ablation_summary.txt").exists()


def test_ablation_both_success_population_and_missing_values():
    scenarios = experiments.generate_scenarios(4, [(5, 5)], [.2], [2], 42)
    rows = experiments.run_benchmark(scenarios, True, include_no_promotion=True, max_horizon=0)
    outcomes = ((True, True), (True, False), (False, True), (False, False))
    for index, scenario in enumerate(scenarios):
        for method, success in zip((FIXED, NO_PROMOTION), outcomes[index]):
            row = next(r for r in rows if r["scenario_id"] == scenario.scenario_id and r["algorithm"] == method)
            row["coordinated_planning_success"] = success
            row["sum_of_costs"] = (10 if method == FIXED else 12) if success else None
            row["elapsed_ms"] = 2 if method == FIXED else 3
    pairs = ablation_comparisons(list(reversed(rows)))
    summary = next(r for r in summarize_ablation(pairs) if r["group_dimension"] == "all" and r["mechanism"] == "initial_order")
    assert summary["scenario_count"] == 4
    assert [summary[f"{key}_count"] for key in ("both_success", "first_only_success", "second_only_success", "both_fail")] == [1, 1, 1, 1]
    assert summary["both_success_delta_soc_n"] == 1 and summary["both_success_delta_soc_mean"] == 2
    assert summary["both_success_delta_runtime_ms_n"] == 1 and summary["both_success_delta_runtime_ms_mean"] == 1
    assert summary["both_success_delta_waits_n"] == 0 and summary["both_success_delta_waits_mean"] is None
    for p in pairs:
        if p["outcome"] != "both succeed":
            assert all(p[f"delta_{short}"] is None for short in ("soc", "expanded", "generated", "runtime_ms"))


def test_failed_attempt_and_collision_totals_have_defined_populations():
    grid, agents = promotion_demo()
    scenario = experiments.Scenario("promotion", 156, 0, .3, 1, grid, agents)
    rows = experiments.run_benchmark([scenario], True, include_no_promotion=True)
    full = next(r for r in rows if r["algorithm"] == PROPOSED)
    no = next(r for r in rows if r["algorithm"] == NO_PROMOTION)
    assert full["failed_planning_attempts"] == no["failed_planning_attempts"] == 1
    overall = {r["algorithm"]: r for r in summarize_methods(rows) if r["group_dimension"] == "all"}
    assert overall[PROPOSED]["failed_planning_attempts_total"] == 1
    assert overall[PROPOSED]["vertex_collisions_total"] == 0
    assert overall[NO_PROMOTION]["vertex_collisions_n"] == 0
    assert overall[NO_PROMOTION]["vertex_collisions_total"] is None


def test_invalid_initial_policy_rejected():
    grid, agents = conflict_demo()
    with pytest.raises(ValueError, match="Initial priority"):
        CooperativePlanner(grid, agents, initial_priority="random")


def test_seeded_no_promotion_solutions_obey_static_movement_and_goal_rules():
    scenarios = experiments.generate_scenarios(6, [(8, 8)], [.3], [4], 123)
    successes = 0
    for scenario in scenarios:
        no = cooperative_plan(scenario.grid, scenario.agents, adaptive=False, initial_priority="conflict")
        full = cooperative_plan(scenario.grid, scenario.agents)
        assert no.initial_priority_order == full.initial_priority_order
        assert no.priority_promotions == 0 and len(set(no.attempted_orders)) == len(no.attempted_orders)
        if not no.found:
            assert no.sum_of_costs is no.makespan is no.wait_actions is None
            continue
        successes += 1
        assert replace(no, elapsed_ms=0) == replace(full, elapsed_ms=0)
        assert not detect_collisions(no.paths)
        for agent in scenario.agents:
            path = no.paths[agent.agent_id]
            assert path[0] == agent.start and path[-1] == agent.goal
            assert all(scenario.grid.is_free(cell) for cell in path)
            assert all(a == b or b in tuple(scenario.grid.neighbors(a)) for a, b in zip(path, path[1:]))
            assert agent.goal not in path[:-1]
    assert successes > 0
