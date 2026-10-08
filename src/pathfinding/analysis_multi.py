from collections import Counter, defaultdict
import json
from math import isnan
from statistics import mean, median, stdev
from typing import Sequence

from .collisions import Paths, detect_collisions

INDEPENDENT = "Independent A*"
FIXED = "Fixed-Priority ST-A*"
NO_PROMOTION = "Conflict-Guided ST-A* (no promotion)"
PROPOSED = "CG-ST-A*"
IDENTITY = ("scenario_id", "seed", "scenario_hash", "width", "height", "obstacle_probability", "number_of_agents")
HARDNESS = ("independent_total_collisions", "independent_vertex_collisions", "independent_edge_collisions",
            "number_of_agents_involved_in_any_conflict", "total_predicted_conflict_load", "maximum_agent_conflict_load")
PAIR_METRICS = {"soc": "sum_of_costs", "makespan": "makespan", "waits": "wait_actions",
                "runtime_ms": "elapsed_ms", "expanded": "expanded_states", "generated": "generated_states"}
SUCCESS_OUTCOMES = ("both succeed", "fixed succeeds, CG fails", "fixed fails, CG succeeds", "both fail")


def missing(value) -> bool:
    return value is None or isinstance(value, float) and isnan(value)


def independent_hardness(paths: Paths | None) -> dict:
    """Count events, with one load increment per involved agent, including multiway vertices."""
    if paths is None:
        return dict.fromkeys(HARDNESS)
    events = detect_collisions(paths)
    loads = Counter(agent_id for event in events for agent_id in event.agent_ids)
    return {
        "independent_total_collisions": len(events),
        "independent_vertex_collisions": sum(event.kind == "vertex" for event in events),
        "independent_edge_collisions": sum(event.kind == "edge" for event in events),
        "number_of_agents_involved_in_any_conflict": len(loads),
        "total_predicted_conflict_load": sum(loads.values()),
        "maximum_agent_conflict_load": max(loads.values(), default=0),
    }


def validate_paired_rows(
    rows: Sequence[dict], expected_algorithms: Sequence[str] | None = None,
) -> dict[str, dict[str, dict]]:
    """Reject duplicate/missing method rows and mismatched scenario identities."""
    groups: dict[str, dict[str, dict]] = {}
    algorithms = set(expected_algorithms) if expected_algorithms is not None else {row["algorithm"] for row in rows}
    for row in rows:
        group = groups.setdefault(row["scenario_id"], {})
        algorithm = row["algorithm"]
        if algorithm in group:
            raise ValueError(f"Duplicate row for {row['scenario_id']}: {algorithm}")
        if group:
            reference = next(iter(group.values()))
            for field in (*IDENTITY, *HARDNESS):
                if row[field] != reference[field] and not (missing(row[field]) and missing(reference[field])):
                    raise ValueError(f"Scenario mismatch for {row['scenario_id']}: {field}")
            for field in ("trial", "generation_attempts"):
                if row.get(field) != reference.get(field):
                    raise ValueError(f"Scenario mismatch for {row['scenario_id']}: {field}")
        group[algorithm] = row
    for scenario_id, group in groups.items():
        if set(group) != algorithms:
            raise ValueError(f"Missing algorithm row or unexpected method for {scenario_id}")
    return groups


def describe(values: Sequence) -> dict:
    values = [value for value in values if not missing(value)]
    return {"n": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "stdev": stdev(values) if len(values) > 1 else (0.0 if values else None),
            "min": min(values) if values else None, "max": max(values) if values else None}


def _add_statistics(row: dict, name: str, values: Sequence) -> None:
    row.update({f"{name}_{stat}": value for stat, value in describe(values).items()})


def priority_order_changed(row: dict) -> bool | None:
    initial, final = row.get("initial_priority_order"), row.get("final_priority_order")
    if initial is None or final is None:
        return None
    if isinstance(initial, str):
        initial = json.loads(initial)
    if isinstance(final, str):
        final = json.loads(final)
    return tuple(initial) != tuple(final)


def paired_comparisons(rows: Sequence[dict]) -> list[dict]:
    groups = validate_paired_rows(rows)
    paired = []
    for scenario_id in sorted(groups):
        group = groups[scenario_id]
        if FIXED not in group or PROPOSED not in group:
            continue
        fixed, cg = group[FIXED], group[PROPOSED]
        fs, cs = fixed["coordinated_planning_success"], cg["coordinated_planning_success"]
        both = fs and cs
        outcome = SUCCESS_OUTCOMES[0 if both else 1 if fs else 2 if cs else 3]
        pair = {field: fixed[field] for field in (*IDENTITY, *HARDNESS)}
        pair.update({"fixed_success": fs, "cg_success": cs, "success_outcome": outcome,
                     "cg_success_minus_fixed_success": int(cs) - int(fs),
                     "fixed_collision_free": fixed["collision_free"], "cg_collision_free": cg["collision_free"]})
        for short, metric in PAIR_METRICS.items():
            quality = short in ("soc", "makespan", "waits")
            fv = fixed[metric] if not quality or fs else None
            cv = cg[metric] if not quality or cs else None
            fv, cv = (None if missing(fv) else fv), (None if missing(cv) else cv)
            pair[f"fixed_{metric}"] = fv
            pair[f"cg_{metric}"] = cv
            delta = cv - fv if fv is not None and cv is not None and (both or not quality) else None
            pair[f"delta_{short}"] = delta
            if short in ("soc", "makespan", "waits", "expanded"):
                pair[f"{short}_outcome"] = ("CG lower" if delta < 0 else "CG higher" if delta > 0 else "equal") if both and delta is not None else None
        for prefix, method in (("fixed", fixed), ("cg", cg)):
            for field in ("planning_attempts", "priority_promotions", "initial_priority_order", "final_priority_order", "failure_reason"):
                pair[f"{prefix}_{field}"] = method[field]
        pair["cg_priority_order_changed"] = priority_order_changed(cg)
        paired.append(pair)
    return paired


def conflict_group(row: dict) -> str:
    value = row["independent_total_collisions"]
    return "unknown" if missing(value) else "zero collisions" if value == 0 else "at least one collision"


def strata(rows: Sequence[dict]):
    yield "all", "all scenarios", list(rows)
    for field, key in (("number_of_agents", lambda row: row["number_of_agents"]),
                       ("grid_size", lambda row: f"{row['width']}x{row['height']}"),
                       ("obstacle_probability", lambda row: row["obstacle_probability"])):
        groups: dict[object, list[dict]] = defaultdict(list)
        for row in rows:
            groups[key(row)].append(row)
        for value in sorted(groups):
            yield field, value, groups[value]
    labels = ["zero collisions", "at least one collision"]
    if any(conflict_group(row) == "unknown" for row in rows):
        labels.append("unknown")
    for label in labels:
        yield "independent_conflicts", label, [row for row in rows if conflict_group(row) == label]


def subsets(rows: Sequence[dict]):
    for threshold in (1, 3):
        yield "subset", f"independent_total_collisions >= {threshold}", [
            row for row in rows if not missing(row["independent_total_collisions"]) and row["independent_total_collisions"] >= threshold]
    yield "subset", "number_of_agents >= 6", [row for row in rows if row["number_of_agents"] >= 6]
    yield "subset", "obstacle_probability >= 0.3", [row for row in rows if row["obstacle_probability"] >= 0.3]


def summarize_methods(rows: Sequence[dict], *, subset_only: bool = False) -> list[dict]:
    algorithms = sorted({row["algorithm"] for row in rows})
    summaries = []
    for dimension, value, group in (subsets(rows) if subset_only else strata(rows)):
        for algorithm in algorithms:
            selected = [row for row in group if row["algorithm"] == algorithm]
            count = len(selected)
            successes = sum(row["coordinated_planning_success"] for row in selected)
            safe = sum(row["collision_free"] for row in selected)
            result = {"group_dimension": dimension, "group_value": value, "algorithm": algorithm,
                      "scenario_count": count, "coordinated_success_count": successes,
                      "coordinated_success_denominator": count, "coordinated_success_rate": successes / count if count else None,
                      "collision_free_count": safe, "collision_free_denominator": count,
                      "collision_free_rate": safe / count if count else None,
                      "coordinated_failure_count": count - successes}
            for metric in ("sum_of_costs", "makespan", "wait_actions", "expanded_states", "generated_states", "elapsed_ms",
                           "vertex_collisions", "edge_collisions", "failed_planning_attempts", "planning_attempts", "priority_promotions"):
                values = [row.get(metric) for row in selected if metric not in ("sum_of_costs", "makespan", "wait_actions")
                          or row["coordinated_planning_success"]]
                _add_statistics(result, metric, values)
            for metric in ("vertex_collisions", "edge_collisions", "failed_planning_attempts", "planning_attempts", "priority_promotions"):
                values = [row.get(metric) for row in selected if not missing(row.get(metric))]
                result[f"{metric}_total"] = sum(values) if values else None
            summaries.append(result)
    return summaries


def summarize_pairs(pairs: Sequence[dict]) -> list[dict]:
    if not pairs:
        return []
    summaries = []
    for dimension, value, group in (*strata(pairs), *subsets(pairs)):
        both = [pair for pair in group if pair["fixed_success"] and pair["cg_success"]]
        result = {"group_dimension": dimension, "group_value": value, "scenario_count": len(group),
                  "both_success_count": len(both),
                  "fixed_only_success_count": sum(pair["fixed_success"] and not pair["cg_success"] for pair in group),
                  "cg_only_success_count": sum(pair["cg_success"] and not pair["fixed_success"] for pair in group),
                  "both_fail_count": sum(not pair["fixed_success"] and not pair["cg_success"] for pair in group)}
        _add_statistics(result, "cg_success_minus_fixed_success", [pair["cg_success_minus_fixed_success"] for pair in group])
        for short in PAIR_METRICS:
            _add_statistics(result, f"delta_{short}", [pair[f"delta_{short}"] for pair in group])
            if short in ("expanded", "generated", "runtime_ms"):
                _add_statistics(result, f"both_success_delta_{short}", [pair[f"delta_{short}"] for pair in both])
            if short in ("soc", "makespan", "waits", "expanded"):
                outcomes = [pair[f"{short}_outcome"] for pair in both]
                result.update({f"{short}_classified_n": sum(outcome is not None for outcome in outcomes),
                               f"{short}_cg_lower_count": outcomes.count("CG lower"), f"{short}_equal_count": outcomes.count("equal"),
                               f"{short}_cg_higher_count": outcomes.count("CG higher")})
        summaries.append(result)
    return summaries


def summarize_adaptation(rows: Sequence[dict]) -> list[dict]:
    cg_rows = [row for row in rows if row["algorithm"] == PROPOSED]
    results = []
    for dimension, value, group in (*strata(cg_rows), *subsets(cg_rows)):
        promotions = [row["priority_promotions"] for row in group if not missing(row["priority_promotions"])]
        changed = [priority_order_changed(row) for row in group]
        result = {"group_dimension": dimension, "group_value": value, "scenario_count": len(group),
                  "zero_promotions_count": sum(value == 0 for value in promotions),
                  "one_or_more_promotions_count": sum(value >= 1 for value in promotions),
                  "order_comparison_count": sum(value is not None for value in changed),
                  "priority_order_changed_count": changed.count(True)}
        _add_statistics(result, "priority_promotions", promotions)
        _add_statistics(result, "planning_attempts", [row["planning_attempts"] for row in group])
        results.append(result)
    return results


def analysis_text(rows: Sequence[dict], pairs: Sequence[dict]) -> list[str]:
    lines = []
    for algorithm in sorted({row["algorithm"] for row in rows}):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        success = sum(row["coordinated_planning_success"] for row in selected)
        lines.append(f"{algorithm}: coordinated success {success}/{len(selected)} ({success / len(selected):.1%}).")
    activity = summarize_adaptation(rows)[0]
    lines.extend([
        f"CG promotion activity: zero={activity['zero_promotions_count']}, at least one={activity['one_or_more_promotions_count']}; "
        f"mean={activity['priority_promotions_mean']}, max={activity['priority_promotions_max']} (n={activity['priority_promotions_n']}).",
        f"CG planning attempts: mean={activity['planning_attempts_mean']}, max={activity['planning_attempts_max']} (n={activity['planning_attempts_n']}).",
        f"CG final priority differs from initial: {activity['priority_order_changed_count']}/{activity['order_comparison_count']}.",
    ])
    if activity["scenario_count"] and activity["zero_promotions_count"] == activity["scenario_count"]:
        lines.append("Priority promotion did not activate in this sample; these runs cannot establish its contribution.")
    if pairs:
        summary = summarize_pairs(pairs)[0]
        lines.append(f"Fixed vs CG: both succeed={summary['both_success_count']}, fixed-only={summary['fixed_only_success_count']}, "
                     f"CG-only={summary['cg_only_success_count']}, both fail={summary['both_fail_count']}.")
        for metric in ("soc", "makespan", "waits", "expanded"):
            lines.append(f"Paired {metric}, both successful: CG lower={summary[f'{metric}_cg_lower_count']}, "
                         f"equal={summary[f'{metric}_equal_count']}, higher={summary[f'{metric}_cg_higher_count']} "
                         f"(n={summary[f'{metric}_classified_n']}).")
        contended = [pair for pair in pairs if not missing(pair["independent_total_collisions"]) and pair["independent_total_collisions"] >= 1]
        lines.append(f"Independent-conflict subset: n={len(contended)}, Fixed successes={sum(pair['fixed_success'] for pair in contended)}, "
                     f"CG successes={sum(pair['cg_success'] for pair in contended)}.")
    else:
        lines.append("Fixed-vs-CG paired comparisons are unavailable; run with --include-fixed to collect both methods.")
    lines.extend(["Solution-quality deltas and outcome classes use both-success pairs only. Computation deltas retain failed runs when measured.",
                  "Subset analyses are descriptive and overlapping; scenario generation is unchanged.",
                  "Every mean/median in the analysis CSVs has an observation count; empty samples remain blank."])
    return lines
