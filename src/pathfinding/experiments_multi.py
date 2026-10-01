import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
from statistics import mean, median, stdev
from time import perf_counter_ns
from typing import Sequence

from .agents import Agent, random_agents
from .analysis_multi import (
    FIXED, HARDNESS, IDENTITY, INDEPENDENT, PROPOSED, analysis_text, conflict_group,
    independent_hardness, paired_comparisons, summarize_adaptation, summarize_methods,
    summarize_pairs, validate_paired_rows,
)
from .analysis_plots import plot_analysis
from .cooperative import cooperative_plan
from .experiments import grid_record, parse_size, write_csv
from .grid import Grid
from .multi_agent import compute_metrics, independent_astar

METRICS = ("total_collisions", "sum_of_costs", "makespan", "wait_actions", "elapsed_ms",
           "expanded_states", "generated_states", "peak_frontier_size", "cost_overhead_percent",
           "planning_attempts", "priority_promotions")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    trial: int
    obstacle_probability: float
    generation_attempts: int
    grid: Grid
    agents: tuple[Agent, ...]


def scenario_record(scenario: Scenario) -> dict:
    return {**grid_record(scenario.grid), "agents": [asdict(agent) for agent in scenario.agents]}


def scenario_hash(scenario: Scenario) -> str:
    return sha256(json.dumps(scenario_record(scenario), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_scenarios(
    trials: int, sizes: Sequence[tuple[int, int]], probabilities: Sequence[float],
    agent_counts: Sequence[int], base_seed: int, max_attempts: int = 1000,
) -> list[Scenario]:
    """Select only for endpoint capacity/reachability, never for joint feasibility."""
    if trials < 1 or max_attempts < 1 or not sizes or not probabilities or not agent_counts:
        raise ValueError("Provide positive trials/attempts and nonempty configuration lists.")
    if any(w < 1 or h < 1 or w * h < 2 for w, h in sizes):
        raise ValueError("Grid sizes need positive dimensions and at least two cells.")
    if any(not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Obstacle probabilities must be between 0 and 1.")
    if any(count < 1 or 2 * count > w * h for count in agent_counts for w, h in sizes):
        raise ValueError("Agent counts must be positive and fit distinct endpoints in every grid size.")
    scenarios = []
    seed = base_seed
    for size_index, (width, height) in enumerate(sizes):
        for density_index, probability in enumerate(probabilities):
            for count in agent_counts:
                for trial in range(trials):
                    for attempt in range(1, max_attempts + 1):
                        candidate_seed = seed
                        seed += 1
                        grid = Grid.random(width, height, probability, candidate_seed)
                        try:
                            agents = random_agents(grid, count, candidate_seed)
                        except ValueError:
                            continue
                        scenarios.append(Scenario(f"s{size_index}-p{density_index}-a{count}-t{trial:04d}",
                                                  candidate_seed, trial, probability, attempt, grid, agents))
                        break
                    else:
                        raise ValueError(f"Unable to place {count} reachable endpoint pairs on {width}x{height}, p={probability} "
                                         f"after {max_attempts} attempts.")
    return scenarios


def run_algorithm(scenario: Scenario, algorithm: str, **limits) -> dict:
    if algorithm == INDEPENDENT:
        began = perf_counter_ns()
        plans = independent_astar(scenario.grid, scenario.agents)
        found = all(plan.result.found for plan in plans)
        metrics = compute_metrics({plan.agent.agent_id: plan.path for plan in plans}) if found else None
        collision_free = found and metrics.total_collisions == 0
        row = {
            "individual_planning_success": found, "coordinated_planning_success": collision_free,
            "collision_free": collision_free, "vertex_collisions": metrics.vertex_collisions if metrics else None,
            "edge_collisions": metrics.edge_collisions if metrics else None,
            "total_collisions": metrics.total_collisions if metrics else None,
            "sum_of_costs": metrics.sum_of_costs if metrics else None, "makespan": metrics.makespan if metrics else None,
            "wait_actions": metrics.wait_actions if metrics else None,
            "elapsed_ms": (perf_counter_ns() - began) / 1_000_000,
            "expanded_states": sum(plan.result.expanded_nodes for plan in plans),
            "generated_states": sum(plan.result.generated_nodes for plan in plans),
            "peak_frontier_size": max(plan.result.peak_frontier_size for plan in plans),
            "planning_attempts": 1, "priority_promotions": 0,
            "initial_priority_order": "[]", "final_priority_order": "[]", "attempted_orders": "[]",
            "conflict_load": "{}", "bottleneck_exposure": "{}", "conflict_graph": "{}",
            "independent_path_cost": json.dumps({plan.agent.agent_id: plan.result.path_cost for plan in plans}, sort_keys=True),
            "failure_reason": None if collision_free else ("independent_paths_collide" if found else "individual_path_unreachable"),
            "search_limit_reached": False, "horizon_retries": 0, "maximum_horizon_used": None, "horizon_limit": None,
        }
        row.update(independent_hardness({plan.agent.agent_id: plan.path for plan in plans} if found else None))
        return row
    if algorithm not in (FIXED, PROPOSED):
        raise ValueError("Unknown benchmark algorithm.")
    result = cooperative_plan(scenario.grid, scenario.agents, adaptive=algorithm == PROPOSED, **limits)
    return {
        "individual_planning_success": result.individual_planning_success,
        "coordinated_planning_success": result.found, "collision_free": result.found,
        "vertex_collisions": result.vertex_collisions, "edge_collisions": result.edge_collisions,
        "total_collisions": 0 if result.found else None, "sum_of_costs": result.sum_of_costs,
        "makespan": result.makespan, "wait_actions": result.wait_actions, "elapsed_ms": result.elapsed_ms,
        "expanded_states": result.expanded_states, "generated_states": result.generated_states,
        "peak_frontier_size": result.peak_frontier_size,
        "planning_attempts": result.planning_attempts, "priority_promotions": result.priority_promotions,
        "initial_priority_order": json.dumps(result.initial_priority_order),
        "final_priority_order": json.dumps(result.final_priority_order), "attempted_orders": json.dumps(result.attempted_orders),
        "conflict_load": json.dumps(result.analysis.conflict_load, sort_keys=True),
        "bottleneck_exposure": json.dumps(result.analysis.bottleneck_exposure, sort_keys=True),
        "conflict_graph": json.dumps(result.analysis.conflict_graph, sort_keys=True),
        "independent_path_cost": json.dumps(result.analysis.independent_path_cost, sort_keys=True),
        "failure_reason": result.failure_reason, "search_limit_reached": result.search_limit_reached,
        "horizon_retries": result.horizon_retries, "maximum_horizon_used": result.maximum_horizon_used,
        "horizon_limit": result.horizon_limit,
    }


def run_benchmark(scenarios: Sequence[Scenario], include_fixed: bool = False, **limits) -> list[dict]:
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("Scenario IDs must be unique within a benchmark.")
    algorithms = [INDEPENDENT, FIXED, PROPOSED] if include_fixed else [INDEPENDENT, PROPOSED]
    rows = []
    for index, scenario in enumerate(scenarios):
        offset = index % len(algorithms)
        order = algorithms[offset:] + algorithms[:offset]
        paired = []
        for run_order, algorithm in enumerate(order):
            row = {
                "scenario_id": scenario.scenario_id, "scenario_hash": scenario_hash(scenario),
                "seed": scenario.seed, "trial": scenario.trial, "width": scenario.grid.width, "height": scenario.grid.height,
                "obstacle_probability": scenario.obstacle_probability, "number_of_agents": len(scenario.agents),
                "generation_attempts": scenario.generation_attempts, "algorithm": algorithm, "run_order": run_order,
                **run_algorithm(scenario, algorithm, **limits),
            }
            paired.append(row)
        baseline = next(row for row in paired if row["algorithm"] == INDEPENDENT)
        for row in paired:
            row.update({field: baseline[field] for field in HARDNESS})
            cost = row["sum_of_costs"]
            base_cost = baseline["sum_of_costs"]
            row["cost_overhead_percent"] = ((cost - base_cost) / base_cost * 100
                                            if cost is not None and base_cost and baseline["individual_planning_success"] else None)
        rows.extend(paired)
    validate_paired_rows(rows)
    return rows


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["width"], row["height"], row["obstacle_probability"], row["number_of_agents"], row["algorithm"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (width, height, probability, agents, algorithm), group in groups.items():
        successes = sum(row["collision_free"] for row in group)
        summary = {
            "width": width, "height": height, "obstacle_probability": probability, "number_of_agents": agents,
            "algorithm": algorithm, "scenario_count": len(group),
            "individual_success_count": sum(row["individual_planning_success"] for row in group),
            "coordinated_success_count": sum(row["coordinated_planning_success"] for row in group),
            "collision_free_count": successes, "collision_free_denominator": len(group),
            "collision_free_rate": successes / len(group), "coordinated_failure_count": len(group) - successes,
            "no_complete_plan_count": sum(row["sum_of_costs"] is None for row in group),
            "colliding_plan_count": sum(row["sum_of_costs"] is not None and not row["collision_free"] for row in group),
        }
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary.update({f"{metric}_count": len(values),
                            f"{metric}_mean": mean(values) if values else None,
                            f"{metric}_median": median(values) if values else None,
                            f"{metric}_stdev": stdev(values) if len(values) > 1 else (0.0 if values else None),
                            f"{metric}_min": min(values) if values else None,
                            f"{metric}_max": max(values) if values else None})
        summaries.append(summary)
    return summaries


def plot_results(summaries: Sequence[dict], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configurations: dict[tuple, list[dict]] = {}
    for row in summaries:
        configurations.setdefault((row["width"], row["height"], row["obstacle_probability"]), []).append(row)
    plots = []
    colors = {INDEPENDENT: "#3178a8", FIXED: "#d17b12", PROPOSED: "#228653"}
    for (width, height, probability), group in configurations.items():
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        panels = [("collision_free_rate", "Collision-free success (%)"),
                  ("total_collisions", "Collisions (complete plans)"), ("elapsed_ms", "Planning time (ms, all runs)"),
                  ("sum_of_costs", "Sum of costs (complete plans)"), ("cost_overhead_percent", "Cost overhead (%, paired successes)"),
                  ("makespan", "Makespan (complete plans)")]
        for axis, (metric, label) in zip(axes.flat, panels):
            for algorithm in colors:
                series = sorted((row for row in group if row["algorithm"] == algorithm), key=lambda row: row["number_of_agents"])
                if not series:
                    continue
                x = [row["number_of_agents"] for row in series]
                if metric == "collision_free_rate":
                    axis.plot(x, [100 * row[metric] for row in series], "o-", color=colors[algorithm], label=algorithm)
                else:
                    for stat, style in [("mean", "-"), ("median", "--")]:
                        y = [row[f"{metric}_{stat}"] if row[f"{metric}_{stat}"] is not None else float("nan") for row in series]
                        axis.plot(x, y, marker="o" if stat == "mean" else "x", linestyle=style, color=colors[algorithm])
            axis.set_xlabel("Number of agents")
            axis.set_xticks(sorted({row["number_of_agents"] for row in group}))
            axis.set_ylabel(label)
            axis.set_ylim(bottom=0)
            if metric == "collision_free_rate":
                axis.set_ylim(0, 105)
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8, loc="lower left")
        fig.suptitle(f"{width} × {height}, obstacle probability {probability:g}\n"
                     "Solid = mean; dashed = median. Quality/collision averages exclude missing plans; success rates include failures.")
        path = output / f"comparison_{width}x{height}_p{probability:g}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots.append(path)
    return plots


def save_results(scenarios: Sequence[Scenario], rows: Sequence[dict], output: Path, configuration: dict) -> list[Path]:
    groups = validate_paired_rows(rows)
    if set(groups) != {scenario.scenario_id for scenario in scenarios}:
        raise ValueError("Result scenario IDs do not match the scenario manifest.")
    for scenario in scenarios:
        row = next(iter(groups[scenario.scenario_id].values()))
        expected = {"scenario_hash": scenario_hash(scenario), "seed": scenario.seed,
                    "width": scenario.grid.width, "height": scenario.grid.height,
                    "obstacle_probability": scenario.obstacle_probability, "number_of_agents": len(scenario.agents)}
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError(f"Result identity does not match scenario {scenario.scenario_id}.")
    output.mkdir(parents=True, exist_ok=True)
    summaries = summarize(rows)
    write_csv(output / "raw_results.csv", rows)
    write_csv(output / "summary.csv", summaries)
    manifest = {
        "configuration": configuration, "python": platform.python_version(), "platform": platform.platform(),
        "rejected_generation_candidates": sum(scenario.generation_attempts - 1 for scenario in scenarios),
        "scenarios": [{"scenario_id": scenario.scenario_id, "seed": scenario.seed, "trial": scenario.trial,
                       "obstacle_probability": scenario.obstacle_probability, "generation_attempts": scenario.generation_attempts,
                       "scenario_hash": scenario_hash(scenario), **scenario_record(scenario)} for scenario in scenarios],
    }
    (output / "scenarios.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [f"Paired scenarios: {len(scenarios)}", f"Algorithm runs: {len(rows)}"]
    for algorithm in dict.fromkeys(row["algorithm"] for row in rows):
        group = [row for row in rows if row["algorithm"] == algorithm]
        successes = sum(row["collision_free"] for row in group)
        missing = sum(row["sum_of_costs"] is None for row in group)
        lines.append(f"{algorithm}: collision-free {successes}/{len(group)} ({successes / len(group):.1%}); missing complete plans: {missing}")
    lines.extend(["Scenarios were not filtered for collisions or cooperative success.",
                  "Independent costs include colliding complete paths; cooperative quality is conditional on successful complete plans.",
                  "Runtime is machine-dependent; these descriptive statistics alone do not establish general performance superiority."])
    pairs = paired_comparisons(rows)
    lines.extend(analysis_text(rows, pairs))
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_analysis_outputs(rows, pairs, output)
    return plot_results(summaries, output) + plot_analysis(rows, pairs, output)


def write_analysis_outputs(rows: Sequence[dict], pairs: Sequence[dict], output: Path) -> None:
    unique = {row["scenario_id"]: row for row in rows}
    hardness = [{**{field: row[field] for field in (*IDENTITY, *HARDNESS)},
                 "independent_conflict_group": conflict_group(row)} for row in unique.values()]
    tables = {
        "scenario_hardness.csv": hardness,
        "paired_comparisons.csv": pairs,
        "paired_summary.csv": summarize_pairs(pairs),
        "stratified_summary.csv": summarize_methods(rows),
        "subset_summary.csv": summarize_methods(rows, subset_only=True),
        "adaptation_summary.csv": summarize_adaptation(rows),
    }
    for name, table in tables.items():
        if table:
            write_csv(output / name, table)
        else:
            fields = ["scenario_id", "success_outcome"] if name == "paired_comparisons.csv" else ["group_dimension", "group_value", "scenario_count"]
            with (output / name).open("w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(fields)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paired Question 2 comparison with Manhattan heuristics.")
    parser.add_argument("--trials", type=int, default=3, help="Trials per size/density/agent-count combination")
    parser.add_argument("--sizes", type=parse_size, nargs="+", default=[(10, 10), (20, 20)])
    parser.add_argument("--probabilities", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--agents", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--include-fixed", action="store_true")
    parser.add_argument("--max-horizon", type=int)
    parser.add_argument("--max-priority-retries", type=int)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--max-generation-attempts", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("results/question2"))
    args = parser.parse_args(argv)
    limits = {"max_horizon": args.max_horizon, "max_priority_retries": args.max_priority_retries, "max_expansions": args.max_expansions}
    if args.max_expansions < 1 or any(value is not None and value < 0 for value in (args.max_horizon, args.max_priority_retries)):
        parser.error("Expansion budget must be positive; horizon/retries must be nonnegative.")
    try:
        scenarios = generate_scenarios(args.trials, args.sizes, args.probabilities, args.agents,
                                       args.base_seed, args.max_generation_attempts)
    except ValueError as error:
        parser.error(str(error))
    print(f"Planning {len(scenarios)} paired random scenarios...", flush=True)
    rows = run_benchmark(scenarios, args.include_fixed, **limits)
    configuration = vars(args).copy()
    configuration["output"] = str(args.output)
    plots = save_results(scenarios, rows, args.output, configuration)
    print((args.output / "summary.txt").read_text(encoding="utf-8"), end="")
    print(f"Outputs: {args.output.resolve()} ({len(plots)} figures)")


if __name__ == "__main__":
    main()
