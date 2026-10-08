import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from itertools import combinations
from math import isnan
from pathlib import Path
import platform
from statistics import mean, median, stdev
from typing import Sequence

from .astar import astar
from .grid import Grid
from .heuristics import HEURISTICS, REQUIRED_HEURISTICS

METRICS = ("path_cost", "expanded_nodes", "generated_nodes", "peak_frontier_size", "elapsed_ms")
IDENTITY = ("instance_id", "grid_hash", "trial", "seed", "width", "height",
            "obstacle_probability", "obstacle_count", "generation_attempts",
            "start_x", "start_y", "goal_x", "goal_y")


@dataclass(frozen=True)
class Instance:
    instance_id: str
    trial: int
    seed: int
    obstacle_probability: float
    attempts: int
    grid: Grid


def reachable(grid: Grid) -> bool:
    """Check connectivity during dataset preparation, outside search timing."""
    pending = [grid.start]
    seen = {grid.start}
    while pending:
        current = pending.pop()
        if current == grid.goal:
            return True
        for neighbor in grid.neighbors(current):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return False


def grid_record(grid: Grid) -> dict:
    return {
        "width": grid.width, "height": grid.height,
        "start": grid.start, "goal": grid.goal, "obstacles": sorted(grid.obstacles),
    }


def grid_hash(grid: Grid) -> str:
    encoded = json.dumps(grid_record(grid), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def generate_instances(
    trials: int, sizes: Sequence[tuple[int, int]], probabilities: Sequence[float],
    base_seed: int, max_attempts: int = 1000,
) -> list[Instance]:
    """Keep solvable draws, advancing the seed after every accepted/rejected map."""
    if trials < 1 or max_attempts < 1 or not sizes or not probabilities:
        raise ValueError("Trials and max attempts must be positive; provide sizes and probabilities.")
    if any(w <= 0 or h <= 0 or w * h < 2 for w, h in sizes):
        raise ValueError("Each grid size must have positive dimensions and at least two cells.")
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Obstacle probabilities must be between 0 and 1.")
    instances = []
    seed = base_seed
    for size_index, (width, height) in enumerate(sizes):
        for probability_index, probability in enumerate(probabilities):
            for trial in range(trials):
                for attempt in range(1, max_attempts + 1):
                    candidate_seed = seed
                    seed += 1
                    grid = Grid.random(width, height, probability, candidate_seed)
                    if reachable(grid):
                        instances.append(Instance(
                            f"s{size_index}-p{probability_index}-t{trial:04d}",
                            trial, candidate_seed, probability, attempt, grid,
                        ))
                        break
                else:
                    raise ValueError(
                        f"No solvable {width}x{height} map at p={probability} after "
                        f"{max_attempts} attempts for trial {trial}; reduce density or increase --max-attempts."
                    )
    return instances


def run_benchmark(instances: Sequence[Instance], include_zero: bool = False) -> list[dict]:
    """Run every selected heuristic on each immutable instance and verify costs."""
    if len({item.instance_id for item in instances}) != len(instances):
        raise ValueError("Instance IDs must be unique within a benchmark.")
    heuristics = HEURISTICS if include_zero else REQUIRED_HEURISTICS
    names = list(heuristics)
    rows = []
    for index, instance in enumerate(instances):
        grid = instance.grid
        fingerprint = grid_hash(grid)
        offset = index % len(names)
        order = names[offset:] + names[:offset]
        costs = set()
        for run_order, name in enumerate(order):
            result = astar(grid, heuristics[name])
            if not result.found:
                raise RuntimeError(f"Search failed on solvable instance {instance.instance_id} using {name}.")
            costs.add(result.path_cost)
            metrics = asdict(result)
            del metrics["path"]
            rows.append({
                "instance_id": instance.instance_id, "grid_hash": fingerprint,
                "trial": instance.trial, "seed": instance.seed,
                "width": grid.width, "height": grid.height,
                "obstacle_probability": instance.obstacle_probability,
                "obstacle_count": len(grid.obstacles), "generation_attempts": instance.attempts,
                "start_x": grid.start[0], "start_y": grid.start[1],
                "goal_x": grid.goal[0], "goal_y": grid.goal[1],
                "heuristic": name, "run_order": run_order, **metrics,
            })
        if len(costs) != 1:
            raise RuntimeError(f"Heuristics disagree on path cost for {instance.instance_id}: {costs}")
    return rows


def validate_paired_rows(
    rows: Sequence[dict], expected_heuristics: Sequence[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Check complete pairs and identity before reporting or saving results."""
    names = set(expected_heuristics) if expected_heuristics is not None else {r["heuristic"] for r in rows}
    groups: dict[str, dict[str, dict]] = {}
    for row in rows:
        group = groups.setdefault(row["instance_id"], {})
        name = row["heuristic"]
        if name in group:
            raise ValueError(f"Duplicate row for {row['instance_id']}: {name}")
        if group:
            reference = next(iter(group.values()))
            if any(row[field] != reference[field] for field in IDENTITY):
                raise ValueError(f"Scenario mismatch for {row['instance_id']}.")
        group[name] = row
    for instance_id, group in groups.items():
        if set(group) != names:
            raise ValueError(f"Missing or unexpected heuristic row for {instance_id}.")
    return groups


def missing(value) -> bool:
    return value is None or isinstance(value, float) and isnan(value)


def metric_statistics(values: Sequence) -> dict:
    values = [value for value in values if not missing(value)]
    return {"count": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "stdev": stdev(values) if len(values) > 1 else (0.0 if values else None),
            "min": min(values) if values else None, "max": max(values) if values else None}


def paired_comparisons(rows: Sequence[dict]) -> list[dict]:
    """Expanded-node differences are A minus B, aligned by verified instance ID."""
    pairs = []
    for group in validate_paired_rows(rows).values():
        names = [name for name in HEURISTICS if name in group]
        for name_a, name_b in combinations(names, 2):
            a, b = group[name_a], group[name_b]
            av, bv = a["expanded_nodes"], b["expanded_nodes"]
            pairs.append({**{field: a[field] for field in IDENTITY},
                          "heuristic_a": name_a, "heuristic_b": name_b,
                          "both_found": a["found"] and b["found"],
                          "expanded_nodes_a": av, "expanded_nodes_b": bv,
                          "delta_expanded_nodes": av - bv if not missing(av) and not missing(bv) else None})
    return pairs


def summarize_pairs(pairs: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    fields = ("width", "height", "obstacle_probability", "heuristic_a", "heuristic_b")
    for pair in pairs:
        groups.setdefault(tuple(pair[field] for field in fields), []).append(pair)
    return [{**dict(zip(fields, key)), "paired_trials": len(group),
             **{f"delta_expanded_nodes_{stat}": value for stat, value in
                metric_statistics([pair["delta_expanded_nodes"] for pair in group]).items()}}
            for key, group in groups.items()]


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["width"], row["height"], row["obstacle_probability"], row["heuristic"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (width, height, probability, name), group in groups.items():
        summary = {
            "width": width, "height": height, "obstacle_probability": probability,
            "heuristic": name, "runs": len(group), "found_count": sum(row["found"] for row in group),
            "success_rate": sum(row["found"] for row in group) / len(group),
        }
        for metric in METRICS:
            summary.update({f"{metric}_{stat}": value for stat, value in
                            metric_statistics([row[metric] for row in group]).items()})
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summaries: Sequence[dict], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configurations: dict[tuple, list[dict]] = {}
    for row in summaries:
        key = (row["width"], row["height"], row["obstacle_probability"])
        configurations.setdefault(key, []).append(row)
    paths = []
    for (width, height, probability), group in configurations.items():
        group.sort(key=lambda row: list(HEURISTICS).index(row["heuristic"]))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
        for axis, metric, label in zip(
            axes, ("expanded_nodes", "elapsed_ms", "peak_frontier_size"),
            ("Expanded nodes", "Search time (ms)", "Peak unique frontier size"),
        ):
            positions = list(range(len(group)))
            means = [r[f"{metric}_mean"] if r[f"{metric}_mean"] is not None else float("nan") for r in group]
            medians = [r[f"{metric}_median"] if r[f"{metric}_median"] is not None else float("nan") for r in group]
            axis.bar([x - 0.19 for x in positions], means,
                     width=0.38, label="Mean", color="#3178a8")
            axis.bar([x + 0.19 for x in positions], medians,
                     width=0.38, label="Median", color="#e0a34a")
            axis.set_xticks(positions, [r["heuristic"].replace(" / ", "\n") for r in group], rotation=20)
            axis.set_ylabel(label)
            maximum = max((r[f"{metric}_{stat}"] for r in group for stat in ("mean", "median")
                           if r[f"{metric}_{stat}"] is not None), default=0)
            axis.set_ylim(0, maximum * 1.25 if maximum else 1)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.legend(loc="upper left")
            if metric == "elapsed_ms":
                axis.set_title("Descriptive timing; tiny differences are noisy", fontsize=9)
        count_text = ", ".join(f"{r['heuristic']}: {r['runs']}" for r in group)
        fig.suptitle(f"{width} × {height}, obstacle probability {probability:g}\nPaired trial counts — {count_text}")
        filename = output / f"comparison_{width}x{height}_p{probability:g}.png"
        fig.savefig(filename, dpi=160)
        plt.close(fig)
        paths.append(filename)
    return paths


def save_results(
    instances: Sequence[Instance], rows: Sequence[dict], output: Path, configuration: dict,
) -> list[Path]:
    if not instances or not rows:
        raise ValueError("Provide nonempty instances and result rows.")
    expected = (list(HEURISTICS if configuration["include_zero"] else REQUIRED_HEURISTICS)
                if "include_zero" in configuration else None)
    groups = validate_paired_rows(rows, expected)
    instance_ids = {item.instance_id for item in instances}
    if len(instance_ids) != len(instances) or set(groups) != instance_ids:
        raise ValueError("Result instance IDs do not match the instance manifest.")
    for instance in instances:
        group = groups[instance.instance_id]
        row = next(iter(group.values()))
        grid = instance.grid
        identity = (instance.instance_id, grid_hash(grid), instance.trial, instance.seed,
                    grid.width, grid.height, instance.obstacle_probability, len(grid.obstacles),
                    instance.attempts, *grid.start, *grid.goal)
        if tuple(row[field] for field in IDENTITY) != identity:
            raise ValueError(f"Result identity does not match instance {instance.instance_id}.")
        if (any(not r["found"] or missing(r["path_cost"]) for r in group.values())
                or len({r["path_cost"] for r in group.values()}) != 1):
            raise ValueError(f"Successful equal path costs required for solvable instance {instance.instance_id}.")
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    write_csv(output / "raw_results.csv", rows)
    write_csv(output / "summary.csv", summaries)
    pairs = paired_comparisons(rows)
    if pairs:
        write_csv(output / "paired_comparisons.csv", pairs)
        write_csv(output / "paired_summary.csv", summarize_pairs(pairs))
    manifest = {
        "configuration": configuration, "python": platform.python_version(),
        "platform": platform.platform(), "path_costs_verified_equal": True,
        "accepted_instances": len(instances),
        "seed_policy": "One global candidate seed counter from base_seed; increment after every draw, including rejections.",
        "sampling_policy": "Conditional on start-goal reachability; no heuristic-specific sampling.",
        "heuristics": list(next(iter(groups.values()))) if groups else [],
        "rejected_candidates": sum(instance.attempts - 1 for instance in instances),
        "instances": [
            {"instance_id": item.instance_id, "seed": item.seed, "trial": item.trial,
             "obstacle_probability": item.obstacle_probability, "attempts": item.attempts,
             "grid_hash": grid_hash(item.grid), **grid_record(item.grid)}
            for item in instances
        ],
    }
    (output / "instances.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plots = plot_results(summaries, output)
    lines = [
        f"Solvable instances: {len(instances)}", f"Search runs: {len(rows)}",
        f"Rejected disconnected candidates: {manifest['rejected_candidates']}",
        "Verified identical path costs across all selected heuristics on every instance.",
        "Expanded nodes are the primary algorithmic performance metric.",
        "Runtime varies with machine load; use repeated trials and descriptive statistics.",
        "Summary CSV includes success rates, per-metric counts, mean, median, sample standard deviation, minimum and maximum.",
        "Paired expanded-node differences use heuristic A minus B on the same scenario.",
        "Sub-millisecond runtime differences alone do not establish performance superiority.",
    ]
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plots


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x"))
        if width <= 0 or height <= 0 or width * height < 2:
            raise ValueError
        return width, height
    except ValueError as error:
        raise argparse.ArgumentTypeError("Size must be WIDTHxHEIGHT with at least two cells.") from error


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare A* heuristics on shared seeded solvable grids.")
    parser.add_argument("--trials", type=int, default=50, help="Accepted maps per size/probability pair (default: 50)")
    parser.add_argument("--sizes", type=parse_size, nargs="+", default=[(20, 20)], metavar="WIDTHxHEIGHT")
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.25])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=1000, help="Maximum draws per accepted map")
    parser.add_argument("--include-zero", action="store_true", default=True,
                        help="Compatibility flag: Zero / Dijkstra is now included by default")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        instances = generate_instances(args.trials, args.sizes, args.probabilities, args.base_seed, args.max_attempts)
    except ValueError as error:
        parser.error(str(error))
    rows = run_benchmark(instances, args.include_zero)
    config = vars(args).copy()
    config["output"] = str(args.output)
    plots = save_results(instances, rows, args.output, config)
    print(f"Completed {len(rows)} searches on {len(instances)} shared solvable instances.")
    print("Verified equal path costs for all selected heuristics on every instance.")
    print(f"Results: {args.output.resolve()} ({len(plots)} comparison figure(s))")


if __name__ == "__main__":
    main()
