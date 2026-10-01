import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import mean, median, stdev
from typing import Sequence

from .astar import astar
from .grid import Grid
from .heuristics import HEURISTICS, REQUIRED_HEURISTICS

METRICS = ("path_cost", "expanded_nodes", "generated_nodes", "peak_frontier_size", "elapsed_ms")


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
        }
        for metric in METRICS:
            values = [row[metric] for row in group]
            summary.update({
                f"{metric}_mean": mean(values), f"{metric}_median": median(values),
                f"{metric}_stdev": stdev(values) if len(values) > 1 else 0.0,
                f"{metric}_min": min(values), f"{metric}_max": max(values),
            })
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
            axis.bar([x - 0.19 for x in positions], [r[f"{metric}_mean"] for r in group],
                     width=0.38, label="Mean", color="#3178a8")
            axis.bar([x + 0.19 for x in positions], [r[f"{metric}_median"] for r in group],
                     width=0.38, label="Median", color="#e0a34a")
            axis.set_xticks(positions, [r["heuristic"].replace(" / ", "\n") for r in group], rotation=20)
            axis.set_ylabel(label)
            maximum = max(r[f"{metric}_{stat}"] for r in group for stat in ("mean", "median"))
            axis.set_ylim(0, maximum * 1.25 if maximum else 1)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
            axis.legend(loc="upper left")
        count_text = ", ".join(f"{r['heuristic']}: {r['runs']}" for r in group)
        fig.suptitle(f"{width} × {height}, obstacle probability {probability:g}\nRuns — {count_text}")
        filename = output / f"comparison_{width}x{height}_p{probability:g}.png"
        fig.savefig(filename, dpi=160)
        plt.close(fig)
        paths.append(filename)
    return paths


def save_results(
    instances: Sequence[Instance], rows: Sequence[dict], output: Path, configuration: dict,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    write_csv(output / "raw_results.csv", rows)
    write_csv(output / "summary.csv", summaries)
    manifest = {
        "configuration": configuration, "python": platform.python_version(),
        "platform": platform.platform(), "path_costs_verified_equal": True,
        "accepted_instances": len(instances),
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
        "Summary CSV includes mean, median, sample standard deviation, minimum and maximum.",
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
    parser.add_argument("--include-zero", action="store_true", help="Also run the optional Dijkstra baseline")
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
