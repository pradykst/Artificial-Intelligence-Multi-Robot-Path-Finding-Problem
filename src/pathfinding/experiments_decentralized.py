"""Separate paired pilot; original Q1/Q2 benchmark files are never written."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

from .analysis_multi import FIXED, INDEPENDENT, NO_PROMOTION, PROPOSED
from .decentralized import DCN, decentralized_plan
from .experiments import parse_size, write_csv
from .experiments_multi import generate_scenarios, run_algorithm, scenario_hash, scenario_record

METRICS = ("sum_of_costs", "makespan", "wait_actions", "elapsed_ms", "expanded_states",
           "generated_states", "negotiation_rounds", "messages_sent", "messages_delivered", "local_replans")
IDENTITY = ("scenario_hash", "seed", "trial", "width", "height", "obstacle_probability", "number_of_agents")


def run_pilot(scenarios, *, include_no_promotion=False, **limits):
    methods = [INDEPENDENT, FIXED, *([NO_PROMOTION] if include_no_promotion else []), PROPOSED, DCN]
    if not scenarios or len({s.scenario_id for s in scenarios}) != len(scenarios):
        raise ValueError("Provide nonempty scenarios with unique IDs.")
    rows = []
    for index, scenario in enumerate(scenarios):
        order = methods[index % len(methods):] + methods[:index % len(methods)]
        for position, method in enumerate(order):
            identity = dict(scenario_id=scenario.scenario_id, scenario_hash=scenario_hash(scenario), seed=scenario.seed,
                trial=scenario.trial, width=scenario.grid.width, height=scenario.grid.height,
                obstacle_probability=scenario.obstacle_probability, number_of_agents=len(scenario.agents),
                generation_attempts=scenario.generation_attempts, algorithm=method, run_order=position)
            if method == DCN:
                r = decentralized_plan(scenario.grid, scenario.agents, **limits)
                row = dict(coordinated_planning_success=r.found, collision_free=r.found,
                    vertex_collisions=0 if r.found else None, edge_collisions=0 if r.found else None,
                    total_collisions=0 if r.found else None, sum_of_costs=r.sum_of_costs, makespan=r.makespan,
                    wait_actions=r.wait_actions, elapsed_ms=r.elapsed_ms, expanded_states=r.expanded_states,
                    generated_states=r.generated_states, failure_reason=r.failure_reason,
                    negotiation_rounds=r.rounds, messages_sent=r.messages_sent, messages_delivered=r.messages_delivered,
                    local_replans=r.replans, replans_by_agent=json.dumps(dict(r.replans_by_agent), sort_keys=True),
                    initial_conflicts=len(r.initial_conflicts), rejected_proposal_conflicts=len(r.final_conflicts) if not r.found else None)
            else:
                central = run_algorithm(scenario, method)
                row = {key: central[key] for key in ("coordinated_planning_success", "collision_free", "vertex_collisions",
                    "edge_collisions", "total_collisions", "sum_of_costs", "makespan", "wait_actions", "elapsed_ms",
                    "expanded_states", "generated_states", "failure_reason")}
                # Pilot quality evaluates executable coordinated solutions only, including the independent baseline.
                if not row["coordinated_planning_success"]:
                    for metric in ("sum_of_costs", "makespan", "wait_actions"):
                        row[metric] = None
                row.update(negotiation_rounds=None, messages_sent=None, messages_delivered=None,
                           local_replans=None, replans_by_agent=None, initial_conflicts=None, rejected_proposal_conflicts=None)
            rows.append({**identity, **row})
    validate_pairs(rows, methods)
    return rows


def validate_pairs(rows, methods):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scenario_id"], []).append(row)
    if not grouped:
        raise ValueError("Missing scenarios.")
    for scenario, group in grouped.items():
        if Counter(r["algorithm"] for r in group) != Counter(methods):
            raise ValueError(f"Missing/duplicate paired method for {scenario}.")
        if any(tuple(r[k] for k in IDENTITY) != tuple(group[0][k] for k in IDENTITY) for r in group):
            raise ValueError(f"Scenario inputs differ within {scenario}.")
        for row in group:
            if row["coordinated_planning_success"] and row["total_collisions"] != 0:
                raise ValueError("Successful coordinated solution contains collisions.")
            if not row["coordinated_planning_success"] and any(row[k] is not None for k in ("sum_of_costs", "makespan", "wait_actions")):
                raise ValueError("Failed coordinated solution has defined quality.")


def summarize(rows):
    summaries = []
    for method in sorted({r["algorithm"] for r in rows}):
        group = [r for r in rows if r["algorithm"] == method]
        success = sum(r["coordinated_planning_success"] for r in group)
        summary = dict(algorithm=method, sample_count=len(group), success_count=success,
                       success_rate=success / len(group), failure_reasons=json.dumps(dict(Counter(
                           r["failure_reason"] for r in group if r["failure_reason"])), sort_keys=True))
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            summary.update({metric + "_count": len(values), metric + "_mean": mean(values) if values else None,
                            metric + "_median": median(values) if values else None,
                            metric + "_std": stdev(values) if len(values) > 1 else None})
        summaries.append(summary)
    return summaries


def paired_quality(rows):
    indexed = {(r["scenario_id"], r["algorithm"]): r for r in rows}
    pairs = []
    for method in (INDEPENDENT, FIXED, NO_PROMOTION, PROPOSED):
        for scenario in sorted({r["scenario_id"] for r in rows}):
            a, b = indexed.get((scenario, method)), indexed.get((scenario, DCN))
            if a and b and a["coordinated_planning_success"] and b["coordinated_planning_success"]:
                pairs.append(dict(scenario_id=scenario, reference=method, compared=DCN,
                    **{metric + "_difference": b[metric] - a[metric]
                       for metric in ("sum_of_costs", "makespan", "wait_actions", "elapsed_ms", "expanded_states")}))
    return pairs


def plot_pilot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries = summarize(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {INDEPENDENT: "Independent", FIXED: "Fixed", PROPOSED: "CG", NO_PROMOTION: "CG no promotion", DCN: "DCN"}
    axes[0].bar([labels[r["algorithm"]] for r in summaries], [100*r["success_rate"] for r in summaries])
    axes[0].set(ylabel="Collision-free success (%)", ylim=(0, 105), title=f"Paired pilot: {len(rows)//len(summaries)} scenarios per method")
    axes[0].tick_params(axis="x", labelrotation=20)
    dcn = [r for r in rows if r["algorithm"] == DCN]
    counts = sorted({r["number_of_agents"] for r in dcn})
    axes[1].bar([str(n) for n in counts], [mean(r["messages_sent"] for r in dcn if r["number_of_agents"] == n) for n in counts])
    axes[1].set(xlabel="Agents", ylabel="Mean receiver-addressed transmissions", title="DCN communication, all trials (including failures)")
    fig.suptitle("10×10 / 20×20, p=0.1 / 0.3, 5 trials per condition" if len(dcn) == 80 else "Paired randomized pilot")
    fig.tight_layout()
    fig.savefig(output / "success_and_messages.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(10, 10), (20, 20)])
    parser.add_argument("--probabilities", nargs="+", type=float, default=[0.1, 0.3])
    parser.add_argument("--agents", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=32)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--max-horizon", type=int)
    parser.add_argument("--include-no-promotion", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/decentralized/pilot"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    # Existing experiments are evidence; reruns require a fresh output directory.
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is nonempty; choose a fresh directory to preserve results.")
    scenarios = generate_scenarios(args.trials, args.sizes, args.probabilities, args.agents, args.seed)
    rows = run_pilot(scenarios, include_no_promotion=args.include_no_promotion,
                     max_rounds=args.max_rounds, max_expansions=args.max_expansions, max_horizon=args.max_horizon)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(base_seed=args.seed, trials=args.trials, sizes=args.sizes, probabilities=args.probabilities,
                    agent_counts=args.agents, max_rounds=args.max_rounds, max_expansions=args.max_expansions,
                    max_horizon=args.max_horizon, scenario_count=len(scenarios), run_count=len(rows),
                    scenarios=[dict(scenario_id=s.scenario_id, seed=s.seed, trial=s.trial,
                                    obstacle_probability=s.obstacle_probability, generation_attempts=s.generation_attempts,
                                    scenario_hash=scenario_hash(s), scenario=scenario_record(s)) for s in scenarios])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(args.output / "raw_runs.csv", rows)
    summaries = summarize(rows)
    write_csv(args.output / "summary.csv", summaries)
    pairs = paired_quality(rows)
    if pairs:
        write_csv(args.output / "paired_quality.csv", pairs)
    if not args.no_plots:
        plot_pilot(rows, args.output)
    print(f"{len(scenarios)} scenarios; {len(rows)} paired runs; output {args.output}")
    for r in summaries:
        print(f"{r['algorithm']}: {r['success_count']}/{r['sample_count']} successes; failures {r['failure_reasons']}")
