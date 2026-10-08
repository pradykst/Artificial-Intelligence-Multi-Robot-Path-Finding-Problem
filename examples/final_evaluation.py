"""Validate final paired datasets and regenerate selected measured figures.

Run after benchmark.py and benchmark_multi.py; never edits historical results.
"""
import argparse
import csv
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathfinding import experiments as q1, experiments_multi as q2
from pathfinding.analysis_ablation import ablation_comparisons, ablation_text, summarize_ablation
from pathfinding.analysis_multi import (
    INDEPENDENT, FIXED, NO_PROMOTION, PROPOSED, paired_comparisons,
    summarize_adaptation, summarize_methods, summarize_pairs, validate_paired_rows,
)
from pathfinding.final_plots import plot_q1, plot_q2
from pathfinding.heuristics import HEURISTICS


def decode(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: decode(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def verify_table(path, expected, keys):
    actual = read_csv(path)
    index = lambda row: tuple(row[k] for k in keys)
    saved = {index(row): row for row in actual}
    assert len(saved) == len(actual) == len(expected), f"Count/duplicate mismatch: {path}"
    for row in expected:
        other = saved[index(row)]
        for field, value in row.items():
            observed = other[field]
            same = (isclose(value, observed, rel_tol=1e-12, abs_tol=1e-12)
                    if isinstance(value, (int, float)) and isinstance(observed, (int, float)) else value == observed)
            assert same, f"Statistics mismatch: {path}, {index(row)}, {field}"


def verify_seed_counter(items, base_seed, attempts):
    consumed = 0
    for item in items:
        consumed += item[attempts]
        assert item["seed"] == base_seed + consumed - 1
    assert len({item["seed"] for item in items}) == len(items)


def validate_q1(directory):
    rows = read_csv(directory / "raw_results.csv")
    manifest = json.loads((directory / "instances.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    groups = q1.validate_paired_rows(rows, HEURISTICS)
    assert len(groups) == 400 and len(rows) == 1600
    assert config["trials"] == 100 and config["base_seed"] == 123
    assert config["sizes"] == [[20, 20], [40, 30]] and config["probabilities"] == [.2, .3]
    verify_seed_counter(manifest["instances"], 123, "attempts")
    regenerated = q1.generate_instances(config["trials"], config["sizes"], config["probabilities"],
                                        config["base_seed"], config["max_attempts"])
    assert len(regenerated) == len(manifest["instances"])
    for instance, saved in zip(regenerated, manifest["instances"]):
        record = {"instance_id": instance.instance_id, "seed": instance.seed, "trial": instance.trial,
                "obstacle_probability": instance.obstacle_probability, "attempts": instance.attempts,
                  "grid_hash": q1.grid_hash(instance.grid), **q1.grid_record(instance.grid)}
        assert json.loads(json.dumps(record)) == saved
        group = groups[instance.instance_id]
        assert all(r["found"] for r in group.values())
        assert len({r["path_cost"] for r in group.values()}) == 1
        for r in group.values():
            grid = instance.grid
            identity = (instance.instance_id, q1.grid_hash(grid), instance.trial, instance.seed,
                        grid.width, grid.height, instance.obstacle_probability, len(grid.obstacles),
                        instance.attempts, *grid.start, *grid.goal)
            assert tuple(r[field] for field in q1.IDENTITY) == identity
    verify_table(directory / "summary.csv", q1.summarize(rows), ("width", "height", "obstacle_probability", "heuristic"))
    pairs = q1.paired_comparisons(rows)
    verify_table(directory / "paired_comparisons.csv", pairs, ("instance_id", "heuristic_a", "heuristic_b"))
    verify_table(directory / "paired_summary.csv", q1.summarize_pairs(pairs),
                 ("width", "height", "obstacle_probability", "heuristic_a", "heuristic_b"))
    return rows, dict(scenarios=len(groups), runs=len(rows), distinct_seeds=len(groups),
                      distinct_fingerprints=len({r["grid_hash"] for r in rows}),
                      seed_min=min(r["seed"] for r in rows), seed_max=max(r["seed"] for r in rows),
                      rejected_candidates=manifest["rejected_candidates"], equal_optimal_costs=True,
                      reproducible_manifest=True, statistics_verified=True)


def validate_q2(directory, historical):
    rows = read_csv(directory / "raw_results.csv")
    manifest = json.loads((directory / "scenarios.json").read_text(encoding="utf-8"))
    config = manifest["configuration"]
    methods = (INDEPENDENT, FIXED, NO_PROMOTION, PROPOSED)
    groups = validate_paired_rows(rows, methods)
    assert len(groups) == 360 and len(rows) == 1440
    assert config["trials"] == 10 and config["base_seed"] == 42
    assert config["sizes"] == [[10, 10], [20, 20], [30, 30]]
    assert config["probabilities"] == [.1, .2, .3] and config["agents"] == [2, 4, 6, 8]
    verify_seed_counter(manifest["scenarios"], 42, "generation_attempts")
    regenerated = q2.generate_scenarios(config["trials"], config["sizes"], config["probabilities"], config["agents"],
                                        config["base_seed"], config["max_generation_attempts"])
    assert len(regenerated) == len(manifest["scenarios"]) == 360
    for scenario, saved in zip(regenerated, manifest["scenarios"]):
        record = {"scenario_id": scenario.scenario_id, "seed": scenario.seed, "trial": scenario.trial,
                "obstacle_probability": scenario.obstacle_probability, "generation_attempts": scenario.generation_attempts,
                  "scenario_hash": q2.scenario_hash(scenario), **q2.scenario_record(scenario)}
        assert json.loads(json.dumps(record)) == saved
        group = groups[scenario.scenario_id]
        for r in group.values():
            for field in ("scenario_id", "scenario_hash", "seed", "trial", "generation_attempts", "width", "height", "obstacle_probability"):
                assert r[field] == saved[field]
            assert r["number_of_agents"] == len(saved["agents"])
            if r["algorithm"] != INDEPENDENT:
                if r["coordinated_planning_success"]:
                    assert r["vertex_collisions"] == r["edge_collisions"] == 0
                    assert r["sum_of_costs"] is not None and r["makespan"] is not None
                else:
                    assert r["sum_of_costs"] is r["makespan"] is r["wait_actions"] is None
        no, full = group[NO_PROMOTION], group[PROPOSED]
        assert no["initial_priority_order"] == full["initial_priority_order"]
        assert no["initial_priority_order"] == no["final_priority_order"] and no["priority_promotions"] == 0
    old_manifest = json.loads((historical / "scenarios.json").read_text(encoding="utf-8"))
    assert manifest["scenarios"] == old_manifest["scenarios"], "Historical scenario manifest changed"
    old = read_csv(historical / "raw_results.csv")
    assert len(old) == 1080
    for previous in old:
        current = groups[previous["scenario_id"]][previous["algorithm"]]
        for field, value in previous.items():
            if field not in ("elapsed_ms", "run_order"):
                assert current[field] == value, f"Historical regression: {previous['scenario_id']}, {previous['algorithm']}, {field}"
    pairs = paired_comparisons(rows)
    ablation = ablation_comparisons(rows)
    tables = (
        ("summary.csv", q2.summarize(rows), ("width", "height", "obstacle_probability", "number_of_agents", "algorithm")),
        ("stratified_summary.csv", summarize_methods(rows), ("group_dimension", "group_value", "algorithm")),
        ("subset_summary.csv", summarize_methods(rows, subset_only=True), ("group_dimension", "group_value", "algorithm")),
        ("paired_comparisons.csv", pairs, ("scenario_id",)),
        ("paired_summary.csv", summarize_pairs(pairs), ("group_dimension", "group_value")),
        ("adaptation_summary.csv", summarize_adaptation(rows), ("group_dimension", "group_value")),
        ("ablation_comparisons.csv", ablation, ("scenario_id", "mechanism")),
        ("ablation_summary.csv", summarize_ablation(ablation), ("mechanism", "group_dimension", "group_value")),
    )
    for name, expected, keys in tables:
        verify_table(directory / name, expected, keys)
    return rows, dict(scenarios=len(groups), runs=len(rows), distinct_seeds=len(groups),
                      distinct_fingerprints=len({r["scenario_hash"] for r in rows}),
                      seed_min=min(r["seed"] for r in rows), seed_max=max(r["seed"] for r in rows),
                      rejected_candidates=manifest["rejected_generation_candidates"],
                      original_method_rows_matching_history=len(old),
                      historical_excluded_fields=["elapsed_ms", "run_order"],
                      reproducible_manifest=True, statistics_verified=True,
                      successes={method: sum(r["coordinated_planning_success"] for r in rows if r["algorithm"] == method) for method in methods},
                      ablation=[r for r in summarize_ablation(ablation) if r["group_dimension"] == "all"])


def history_hashes(directory):
    return {str(path.relative_to(directory)): sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q1", type=Path, default=Path("results/question1/final_100"))
    parser.add_argument("--q2", type=Path, default=Path("results/question2/final_ablation"))
    parser.add_argument("--historical", type=Path, default=Path("results/question2/full"))
    parser.add_argument("--history-snapshot", type=Path, default=Path("results/phase3a_validation/history_sha256.json"))
    parser.add_argument("--assets", type=Path, default=Path("docs/figures"))
    parser.add_argument("--snapshot-history", action="store_true", help="Save historical file checksums before running experiments")
    args = parser.parse_args()
    if args.snapshot_history:
        assert not args.history_snapshot.exists(), "History snapshot already exists"
        args.history_snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.history_snapshot.write_text(json.dumps(history_hashes(args.historical), indent=2) + "\n", encoding="utf-8")
        print(f"Historical checksums: {args.history_snapshot}")
        return
    expected = json.loads(args.history_snapshot.read_text(encoding="utf-8"))
    assert history_hashes(args.historical) == expected, "Historical files modified"
    rows1, report1 = validate_q1(args.q1)
    rows2, report2 = validate_q2(args.q2, args.historical)
    (args.q2 / "ablation_summary.txt").write_text(ablation_text(ablation_comparisons(rows2)), encoding="utf-8")
    plot_q1(rows1, args.q1)
    plot_q2(rows2, args.q2)
    args.assets.mkdir(parents=True, exist_ok=True)
    selected = ((args.q1, "q1_expansions"), (args.q1, "q1_paired_expansions"),
                (args.q2, "q2_overall_success"), (args.q2, "q2_success_agents"),
                (args.q2, "q2_ablation"), (args.q2, "q2_conflict_subsets"))
    for directory, name in selected:
        shutil.copy2(directory / f"{name}.png", args.assets / f"{name}.png")
    for directory, report in ((args.q1, report1), (args.q2, report2)):
        report["historical_files_unchanged"] = True
        (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    assert history_hashes(args.historical) == expected
    print(f"Selected measured figures: {args.assets.resolve()}")


if __name__ == "__main__":
    main()
