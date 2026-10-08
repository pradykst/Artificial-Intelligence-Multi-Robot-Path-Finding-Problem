"""Controlled mechanism comparisons; every continuous delta uses both-success pairs."""
from collections import defaultdict
from typing import Sequence

from .analysis_multi import (
    FIXED, HARDNESS, IDENTITY, INDEPENDENT, NO_PROMOTION, PAIR_METRICS, PROPOSED,
    describe, missing, strata, subsets, validate_paired_rows,
)

COMPARISONS = (
    ("reservations", INDEPENDENT, FIXED),
    ("initial_order", FIXED, NO_PROMOTION),
    ("promotion", NO_PROMOTION, PROPOSED),
)


def ablation_comparisons(rows: Sequence[dict]) -> list[dict]:
    groups = validate_paired_rows(rows, (INDEPENDENT, FIXED, NO_PROMOTION, PROPOSED))
    pairs = []
    for scenario_id in sorted(groups):
        for mechanism, first, second in COMPARISONS:
            a, b = groups[scenario_id][first], groups[scenario_id][second]
            sa, sb = a["coordinated_planning_success"], b["coordinated_planning_success"]
            pair = {field: a[field] for field in (*IDENTITY, *HARDNESS)}
            pair.update(mechanism=mechanism, first_method=first, second_method=second,
                        first_success=sa, second_success=sb,
                        outcome="both succeed" if sa and sb else "first only" if sa else "second only" if sb else "both fail")
            for short, metric in PAIR_METRICS.items():
                av, bv = a.get(metric), b.get(metric)
                pair[f"delta_{short}"] = bv - av if sa and sb and not missing(av) and not missing(bv) else None
            pairs.append(pair)
    return pairs


def summarize_ablation(pairs: Sequence[dict]) -> list[dict]:
    methods = defaultdict(list)
    for pair in pairs:
        methods[pair["mechanism"]].append(pair)
    results = []
    for mechanism, selected in methods.items():
        for dimension, value, group in (*strata(selected), *subsets(selected)):
            result = dict(mechanism=mechanism, first_method=selected[0]["first_method"],
                          second_method=selected[0]["second_method"], group_dimension=dimension,
                          group_value=value, scenario_count=len(group),
                          both_success_count=sum(p["outcome"] == "both succeed" for p in group),
                          first_only_success_count=sum(p["outcome"] == "first only" for p in group),
                          second_only_success_count=sum(p["outcome"] == "second only" for p in group),
                          both_fail_count=sum(p["outcome"] == "both fail" for p in group))
            for short in PAIR_METRICS:
                result.update({f"both_success_delta_{short}_{stat}": observed
                               for stat, observed in describe([p[f"delta_{short}"] for p in group]).items()})
            results.append(result)
    return results


def ablation_text(pairs: Sequence[dict]) -> str:
    lines = ["All success-outcome counts use complete coordinated success.",
             "Continuous differences use second minus first, on both-success pairs only."]
    for row in summarize_ablation(pairs):
        if row["group_dimension"] != "all":
            continue
        lines.append(f"{row['first_method']} -> {row['second_method']}: n={row['scenario_count']}, "
                     f"first-only={row['first_only_success_count']}, second-only={row['second_only_success_count']}, "
                     f"both-success={row['both_success_count']}, both-fail={row['both_fail_count']}.")
        for short in PAIR_METRICS:
            prefix = f"both_success_delta_{short}"
            lines.append(f"  {short}: n={row[prefix + '_n']}, mean={row[prefix + '_mean']}, "
                         f"median={row[prefix + '_median']}, sample SD={row[prefix + '_stdev']}.")
    lines.append("Descriptive comparisons; no significance claim. Runtime is machine-dependent.")
    return "\n".join(lines) + "\n"
