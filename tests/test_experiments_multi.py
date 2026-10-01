from collections import defaultdict
import csv
import json

import pytest

from pathfinding import experiments_multi as experiments


def test_paired_benchmark_uses_exact_same_scenario(monkeypatch):
    scenarios = experiments.generate_scenarios(2, [(6, 6)], [0.2], [2, 3], 42)
    original = experiments.run_algorithm
    calls = []

    def recording(scenario, algorithm, **limits):
        calls.append((scenario, algorithm))
        return original(scenario, algorithm, **limits)

    monkeypatch.setattr(experiments, "run_algorithm", recording)
    rows = experiments.run_benchmark(scenarios, include_fixed=True, max_expansions=3000)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["scenario_id"]].append(row)
    assert len(rows) == 12
    for index, scenario in enumerate(scenarios):
        group = grouped[scenario.scenario_id]
        assert {row["algorithm"] for row in group} == {experiments.INDEPENDENT, experiments.FIXED, experiments.PROPOSED}
        assert {row["scenario_hash"] for row in group} == {experiments.scenario_hash(scenario)}
        assert all(item[0] is scenario for item in calls[index * 3:index * 3 + 3])


def test_scenarios_and_results_reproducible_except_runtime():
    args = (2, [(6, 5)], [0.1, 0.3], [2, 3], 13)
    first = experiments.generate_scenarios(*args)
    assert first == experiments.generate_scenarios(*args)
    assert first != experiments.generate_scenarios(2, [(6, 5)], [0.1, 0.3], [2, 3], 14)
    assert len({scenario.scenario_id for scenario in first}) == len(first)
    rows = experiments.run_benchmark(first, max_expansions=3000)
    repeated = experiments.run_benchmark(first, max_expansions=3000)
    for row in rows + repeated:
        row.pop("elapsed_ms")
    assert rows == repeated


def test_failures_have_blank_costs_and_explicit_denominators(tmp_path):
    scenarios = experiments.generate_scenarios(2, [(5, 5)], [0.2], [2], 42)
    rows = experiments.run_benchmark(scenarios, include_fixed=True, max_horizon=0)
    for row in rows:
        if row["algorithm"] != experiments.INDEPENDENT:
            assert not row["coordinated_planning_success"] and not row["collision_free"]
            assert row["sum_of_costs"] is row["makespan"] is row["wait_actions"] is None
            assert row["total_collisions"] is row["cost_overhead_percent"] is None
            assert row["failure_reason"]
    summary = experiments.summarize(rows)
    proposed = next(row for row in summary if row["algorithm"] == experiments.PROPOSED)
    assert proposed["collision_free_count"] == 0 and proposed["collision_free_denominator"] == 2
    assert proposed["sum_of_costs_count"] == 0 and proposed["sum_of_costs_mean"] is None
    experiments.save_results(scenarios, rows, tmp_path, {"max_horizon": 0})
    with (tmp_path / "raw_results.csv").open(newline="", encoding="utf-8") as stream:
        saved = list(csv.DictReader(stream))
    assert all(row["sum_of_costs"] == "" for row in saved if row["algorithm"] != experiments.INDEPENDENT)
    manifest = json.loads((tmp_path / "scenarios.json").read_text())
    assert len(manifest["scenarios"]) == 2
    assert len(list(tmp_path.glob("comparison_*.png"))) == 1


def test_sampling_does_not_filter_collision_free_scenarios():
    scenarios = experiments.generate_scenarios(12, [(10, 10)], [0.1], [2], 42)
    rows = experiments.run_benchmark(scenarios, max_expansions=1000)
    baseline = [row for row in rows if row["algorithm"] == experiments.INDEPENDENT]
    assert len(baseline) == 12
    assert any(row["collision_free"] for row in baseline)


def test_cost_overhead_is_paired_with_independent_lower_bound():
    from pathfinding.demos import conflict_demo

    grid, agents = conflict_demo()
    scenario = experiments.Scenario("test", 0, 0, 0, 1, grid, agents)
    rows = experiments.run_benchmark([scenario])
    proposed = next(row for row in rows if row["algorithm"] == experiments.PROPOSED)
    assert proposed["cost_overhead_percent"] == pytest.approx(12.5)


def test_generation_failure_has_bounded_attempts(monkeypatch):
    def impossible(*args):
        raise ValueError("insufficient reachable pairs")

    monkeypatch.setattr(experiments, "random_agents", impossible)
    with pytest.raises(ValueError, match="after 2 attempts"):
        experiments.generate_scenarios(1, [(5, 5)], [0.5], [2], 1, max_attempts=2)
