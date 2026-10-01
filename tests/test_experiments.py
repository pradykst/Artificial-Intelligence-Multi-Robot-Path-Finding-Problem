from collections import defaultdict
import json

import pytest

from pathfinding import experiments
from pathfinding.heuristics import REQUIRED_HEURISTICS


def test_benchmark_uses_identical_maps(monkeypatch):
    instances = experiments.generate_instances(5, [(8, 7)], [0.25], 123)
    calls = []
    real_astar = experiments.astar

    def recording_astar(grid, heuristic):
        calls.append(grid)
        return real_astar(grid, heuristic)

    monkeypatch.setattr(experiments, "astar", recording_astar)
    rows = experiments.run_benchmark(instances)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["instance_id"]].append(row)
    assert len(rows) == 15
    for index, instance in enumerate(instances):
        group = grouped[instance.instance_id]
        assert {row["heuristic"] for row in group} == set(REQUIRED_HEURISTICS)
        assert {row["grid_hash"] for row in group} == {experiments.grid_hash(instance.grid)}
        assert len({row["path_cost"] for row in group}) == 1
        assert all(grid is instance.grid for grid in calls[index * 3:index * 3 + 3])


def test_dataset_reproducibility_and_reachability():
    args = (5, [(6, 5), (7, 6)], [0.2, 0.4], 9)
    first = experiments.generate_instances(*args)
    assert first == experiments.generate_instances(*args)
    assert len(first) == 20
    assert len({item.instance_id for item in first}) == 20
    assert all(experiments.reachable(item.grid) for item in first)
    assert len({item.seed for item in first}) == 20
    assert sum(item.attempts - 1 for item in first) > 0


def test_generation_retry_limit(monkeypatch):
    monkeypatch.setattr(experiments, "reachable", lambda grid: False)
    with pytest.raises(ValueError, match="after 3 attempts"):
        experiments.generate_instances(1, [(5, 5)], [0.5], 1, max_attempts=3)


@pytest.mark.parametrize("trials,sizes,probabilities", [
    (0, [(5, 5)], [0.2]), (1, [], [0.2]), (1, [(1, 1)], [0.2]),
    (1, [(5, 5)], [-0.2]), (1, [(5, 5)], [float("nan")]),
])
def test_invalid_dataset_parameters(trials, sizes, probabilities):
    with pytest.raises(ValueError):
        experiments.generate_instances(trials, sizes, probabilities, 1)


def test_summary_and_saved_results(tmp_path):
    instances = experiments.generate_instances(3, [(5, 5)], [0.2], 1)
    rows = experiments.run_benchmark(instances, include_zero=True)
    summary = experiments.summarize(rows)
    assert len(summary) == 4
    for row in summary:
        assert row["runs"] == 3 and row["found_count"] == 3
        assert row["path_cost_min"] <= row["path_cost_mean"] <= row["path_cost_max"]
        assert row["elapsed_ms_stdev"] >= 0
    figures = experiments.save_results(instances, rows, tmp_path, {"base_seed": 1})
    assert len(figures) == 1 and figures[0].stat().st_size > 0
    assert (tmp_path / "raw_results.csv").read_text().count("\n") == 13
    saved = json.loads((tmp_path / "instances.json").read_text())
    assert saved["path_costs_verified_equal"]
    assert saved["instances"][0]["grid_hash"] == experiments.grid_hash(instances[0].grid)


def test_mismatched_costs_are_rejected(monkeypatch):
    from dataclasses import replace

    instances = experiments.generate_instances(1, [(5, 5)], [0], 1)
    real_astar = experiments.astar

    def wrong_cost(grid, heuristic):
        result = real_astar(grid, heuristic)
        if heuristic is REQUIRED_HEURISTICS["Euclidean"]:
            return replace(result, path_cost=result.path_cost + 1)
        return result

    monkeypatch.setattr(experiments, "astar", wrong_cost)
    with pytest.raises(RuntimeError, match="disagree"):
        experiments.run_benchmark(instances)
