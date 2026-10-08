from collections import defaultdict
from copy import deepcopy
import csv
import json

import pytest

from pathfinding import experiments
from pathfinding.grid import Grid
from pathfinding.heuristics import HEURISTICS, REQUIRED_HEURISTICS


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


def test_five_trials_pair_all_four_heuristics_and_preserve_effective_seeds():
    instances = experiments.generate_instances(5, [(8, 7)], [0.3], 42)
    assert len({item.seed for item in instances}) == 5
    # This particular sample varies; different seeds do not guarantee unique maps.
    assert len({experiments.grid_hash(item.grid) for item in instances}) == 5
    consumed = 0
    for item in instances:
        consumed += item.attempts
        assert item.seed == 42 + consumed - 1
        assert item.grid == Grid.random(8, 7, 0.3, item.seed)
    rows = experiments.run_benchmark(instances, include_zero=True)
    groups = experiments.validate_paired_rows(rows, list(HEURISTICS))
    assert len(rows) == 20 and len(groups) == 5
    for item in instances:
        group = groups[item.instance_id]
        assert set(group) == set(HEURISTICS)
        assert len({r["path_cost"] for r in group.values()}) == 1
        assert {r["seed"] for r in group.values()} == {item.seed}
        assert all(r["found"] for r in group.values())
    assert [r["run_order"] for r in rows] == list(range(4)) * 5
    assert len({rows[i * 4]["heuristic"] for i in range(4)}) == 4


def test_summary_aggregates_scenarios_and_retains_missing_values():
    instances = experiments.generate_instances(3, [(5, 5)], [0.2], 1)
    rows = [r for r in experiments.run_benchmark(instances) if r["heuristic"] == "Manhattan"]
    for row, expanded, cost, found in zip(rows, (2, 4, 12), (1, 3, None), (True, True, False)):
        row.update(expanded_nodes=expanded, path_cost=cost, found=found, elapsed_ms=None)
    summary = experiments.summarize(rows)[0]
    assert summary["runs"] == 3 and summary["found_count"] == 2
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["expanded_nodes_count"] == 3
    assert summary["expanded_nodes_mean"] == 6 and summary["expanded_nodes_median"] == 4
    assert summary["expanded_nodes_stdev"] == pytest.approx(28 ** 0.5)
    assert summary["path_cost_count"] == 2 and summary["path_cost_mean"] == 2
    assert summary["elapsed_ms_count"] == 0 and summary["elapsed_ms_mean"] is None
    assert summary["elapsed_ms_stdev"] is None
    assert experiments.metric_statistics([None, float("nan")])["count"] == 0
    assert experiments.metric_statistics([7])["stdev"] == 0


def test_pairs_align_by_id_and_report_a_minus_b_with_missing_metrics():
    instances = experiments.generate_instances(3, [(5, 5)], [0.2], 1)
    rows = experiments.run_benchmark(instances, include_zero=True)
    pairs = experiments.paired_comparisons(list(reversed(rows)))
    assert len(pairs) == 18  # Six heuristic pairs for each of three scenarios.
    lookup = {(r["instance_id"], r["heuristic"]): r for r in rows}
    for pair in pairs:
        a = lookup[pair["instance_id"], pair["heuristic_a"]]
        b = lookup[pair["instance_id"], pair["heuristic_b"]]
        assert pair["delta_expanded_nodes"] == a["expanded_nodes"] - b["expanded_nodes"]
    summary = experiments.summarize_pairs(pairs)
    assert len(summary) == 6 and all(r["delta_expanded_nodes_count"] == 3 for r in summary)
    rows[0]["expanded_nodes"] = None
    pairs = experiments.paired_comparisons(rows)
    assert sum(p["delta_expanded_nodes"] is None for p in pairs) == 3
    assert sum(r["delta_expanded_nodes_count"] for r in experiments.summarize_pairs(pairs)) == 15


@pytest.mark.parametrize("damage", ["missing", "missing_entire_method", "duplicate", "hash", "seed", "trial", "endpoint", "cost", "manifest", "empty"])
def test_save_rejects_invalid_pairs_before_writing(tmp_path, damage):
    instances = experiments.generate_instances(2, [(5, 5)], [0.2], 1)
    rows = deepcopy(experiments.run_benchmark(instances, include_zero=True))
    if damage == "missing":
        rows.pop()
    elif damage == "missing_entire_method":
        rows = [r for r in rows if r["heuristic"] != "Zero / Dijkstra"]
    elif damage == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif damage == "manifest":
        for row in rows:
            row["seed"] += 1
    elif damage == "empty":
        rows = []
    else:
        field = {"hash": "grid_hash", "seed": "seed", "trial": "trial", "endpoint": "start_x", "cost": "path_cost"}[damage]
        rows[0][field] = "wrong" if damage == "hash" else rows[0][field] + 1
    with pytest.raises(ValueError):
        experiments.save_results(instances, rows, tmp_path, {"include_zero": True})
    assert list(tmp_path.iterdir()) == []


def test_csv_metadata_matches_saved_scenarios_and_paired_statistics(tmp_path):
    instances = experiments.generate_instances(5, [(6, 5)], [0.25], 42)
    rows = experiments.run_benchmark(instances, include_zero=True)
    experiments.save_results(instances, rows, tmp_path, {"base_seed": 42, "include_zero": True})
    with (tmp_path / "raw_results.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    manifest = json.loads((tmp_path / "instances.json").read_text())
    lookup = {r["instance_id"]: r for r in manifest["instances"]}
    assert len(saved) == 20 and len(lookup) == 5
    for row in saved:
        item = lookup[row["instance_id"]]
        assert row["grid_hash"] == item["grid_hash"]
        assert int(row["seed"]) == item["seed"] and int(row["trial"]) == item["trial"]
        assert float(row["obstacle_probability"]) == item["obstacle_probability"] == 0.25
        assert (int(row["width"]), int(row["height"])) == (6, 5)
        assert (int(row["start_x"]), int(row["start_y"])) == tuple(item["start"])
        assert (int(row["goal_x"]), int(row["goal_y"])) == tuple(item["goal"])
    with (tmp_path / "paired_comparisons.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 30
    with (tmp_path / "paired_summary.csv").open(newline="") as stream:
        assert all(int(r["delta_expanded_nodes_count"]) == 5 for r in csv.DictReader(stream))


def test_cli_includes_dijkstra_by_default(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(experiments, "save_results", lambda instances, rows, output, config: saved.append((rows, config)) or [])
    experiments.main(["--trials", "1", "--sizes", "5x5", "--probabilities", "0.2", "--output", str(tmp_path)])
    rows, config = saved[0]
    assert config["include_zero"] and {r["heuristic"] for r in rows} == set(HEURISTICS)
