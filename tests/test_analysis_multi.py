from copy import deepcopy

import pytest

from pathfinding.analysis_multi import (
    FIXED, HARDNESS, INDEPENDENT, PROPOSED, SUCCESS_OUTCOMES, analysis_text,
    independent_hardness, paired_comparisons, summarize_adaptation,
    summarize_methods, summarize_pairs, validate_paired_rows,
)
from pathfinding import experiments_multi as experiments


def scenario_rows(scenario_id="s1", *, fixed_success=True, cg_success=True, conflicts=1, agents=6, probability=0.3):
    identity = {"scenario_id": scenario_id, "seed": 42, "scenario_hash": f"hash-{scenario_id}",
                "width": 10, "height": 10, "obstacle_probability": probability, "number_of_agents": agents}
    hardness = {"independent_total_collisions": conflicts, "independent_vertex_collisions": conflicts,
                "independent_edge_collisions": 0, "number_of_agents_involved_in_any_conflict": 2 if conflicts else 0,
                "total_predicted_conflict_load": 2 * conflicts, "maximum_agent_conflict_load": conflicts}
    rows = []
    for algorithm, success, soc, span, waits, runtime, expanded, generated in (
        (INDEPENDENT, conflicts == 0, 8, 4, 0, 5, 20, 40),
        (FIXED, fixed_success, 10, 7, 0, 10, 30, 60),
        (PROPOSED, cg_success, 12, 8, 2, 8, 25, 50),
    ):
        has_paths = algorithm == INDEPENDENT or success
        rows.append({**identity, **hardness, "algorithm": algorithm,
                     "coordinated_planning_success": success, "collision_free": success,
                     "sum_of_costs": soc if has_paths else None, "makespan": span if has_paths else None,
                     "wait_actions": waits if has_paths else None, "elapsed_ms": runtime,
                     "expanded_states": expanded, "generated_states": generated,
                     "planning_attempts": 1, "priority_promotions": 0,
                     "initial_priority_order": "[1, 2, 3]", "final_priority_order": "[1, 2, 3]",
                     "failure_reason": None if success else "test failure"})
    return rows


def test_pairs_align_by_id_not_input_order():
    rows = scenario_rows("b") + scenario_rows("a", fixed_success=False)
    pairs = paired_comparisons(list(reversed(rows)))
    assert [pair["scenario_id"] for pair in pairs] == ["a", "b"]
    assert pairs[0]["success_outcome"] == "fixed fails, CG succeeds"
    assert pairs[1]["success_outcome"] == "both succeed"
    assert pairs[0]["scenario_hash"] == "hash-a"


@pytest.mark.parametrize("field,value", [("scenario_hash", "different"), ("seed", 99), ("width", 11),
                                         ("number_of_agents", 8), ("independent_total_collisions", 20)])
def test_mismatched_scenario_data_is_rejected(field, value):
    rows = scenario_rows()
    rows[-1][field] = value
    with pytest.raises(ValueError, match="Scenario mismatch"):
        paired_comparisons(rows)


def test_duplicates_and_missing_algorithm_rows_rejected():
    rows = scenario_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_paired_rows(rows + [deepcopy(rows[1])])
    with pytest.raises(ValueError, match="Missing algorithm"):
        validate_paired_rows(rows + scenario_rows("s2")[:-1])


@pytest.mark.parametrize("fixed,cg,expected,difference", [
    (True, True, SUCCESS_OUTCOMES[0], 0), (True, False, SUCCESS_OUTCOMES[1], -1),
    (False, True, SUCCESS_OUTCOMES[2], 1), (False, False, SUCCESS_OUTCOMES[3], 0),
])
def test_success_outcome_classification(fixed, cg, expected, difference):
    pair = paired_comparisons(scenario_rows(fixed_success=fixed, cg_success=cg))[0]
    assert pair["success_outcome"] == expected
    assert pair["cg_success_minus_fixed_success"] == difference


def test_paired_deltas_and_successful_outcomes():
    pair = paired_comparisons(scenario_rows())[0]
    assert pair["delta_soc"] == 2 and pair["soc_outcome"] == "CG higher"
    assert pair["delta_makespan"] == 1 and pair["makespan_outcome"] == "CG higher"
    assert pair["delta_waits"] == 2 and pair["waits_outcome"] == "CG higher"
    assert pair["delta_runtime_ms"] == -2
    assert pair["delta_expanded"] == -5 and pair["expanded_outcome"] == "CG lower"
    assert pair["delta_generated"] == -10


def test_failures_keep_quality_missing_but_keep_measured_computation():
    pair = paired_comparisons(scenario_rows(cg_success=False))[0]
    assert pair["fixed_sum_of_costs"] == 10 and pair["cg_sum_of_costs"] is None
    for name in ("soc", "makespan", "waits"):
        assert pair[f"delta_{name}"] is None and pair[f"{name}_outcome"] is None
    assert pair["delta_runtime_ms"] == -2 and pair["delta_expanded"] == -5
    assert pair["expanded_outcome"] is None


def test_nan_values_are_missing_not_zero_or_equal():
    rows = scenario_rows()
    rows[-1]["sum_of_costs"] = float("nan")
    pair = paired_comparisons(rows)[0]
    assert pair["cg_sum_of_costs"] is None and pair["delta_soc"] is None and pair["soc_outcome"] is None
    summary = summarize_pairs([pair])[0]
    assert summary["delta_soc_n"] == 0 and summary["delta_soc_mean"] is None
    assert summary["soc_classified_n"] == 0


def test_pair_summary_uses_correct_denominators_and_outcome_counts():
    rows = (scenario_rows("both") + scenario_rows("fixed", cg_success=False)
            + scenario_rows("cg", fixed_success=False) + scenario_rows("neither", fixed_success=False, cg_success=False))
    summary = summarize_pairs(paired_comparisons(rows))[0]
    assert summary["scenario_count"] == 4
    for field in ("both_success_count", "fixed_only_success_count", "cg_only_success_count", "both_fail_count"):
        assert summary[field] == 1
    assert summary["delta_soc_n"] == 1 and summary["delta_soc_mean"] == 2
    assert summary["delta_expanded_n"] == 4 and summary["both_success_delta_expanded_n"] == 1
    assert summary["soc_cg_higher_count"] == 1 and summary["expanded_cg_lower_count"] == 1


def test_hardness_counts_multiway_events_and_edge_swaps():
    paths = {1: ((0, 1), (1, 1)), 2: ((2, 1), (1, 1)), 3: ((1, 0), (1, 1)),
             4: ((5, 0), (6, 0)), 5: ((6, 0), (5, 0))}
    hardness = independent_hardness(paths)
    assert hardness == dict(zip(HARDNESS, (2, 1, 1, 5, 5, 1)))


def test_hardness_includes_conflicts_after_goal_arrival_and_unknown_paths():
    paths = {1: ((0, 0), (1, 0)), 2: ((2, 1), (2, 0), (1, 0), (0, 0))}
    assert independent_hardness(paths)["independent_total_collisions"] == 1
    assert all(value is None for value in independent_hardness(None).values())
    assert independent_hardness({1: ((0, 0), (1, 0))}) == dict.fromkeys(HARDNESS, 0)


def test_stratification_and_subsets_do_not_drop_scenarios():
    rows = scenario_rows("easy", conflicts=0, agents=2, probability=0.1) + scenario_rows("hard", conflicts=3)
    original = deepcopy(rows)
    summary = summarize_methods(rows)
    contended = next(row for row in summary if row["algorithm"] == FIXED and row["group_value"] == "at least one collision")
    assert contended["scenario_count"] == contended["sum_of_costs_n"] == 1
    assert contended["coordinated_success_rate"] == 1 and contended["elapsed_ms_n"] == 1
    assert {row["group_dimension"] for row in summary} == {"all", "number_of_agents", "grid_size", "obstacle_probability", "independent_conflicts"}
    subsets = summarize_methods(rows, subset_only=True)
    assert len(subsets) == 12
    assert all(row["scenario_count"] == 1 for row in subsets)
    assert rows == original


def test_empty_subsets_and_failed_quality_have_counts_and_blanks():
    rows = scenario_rows(conflicts=0, agents=2, probability=0.1, cg_success=False)
    subsets = summarize_methods(rows, subset_only=True)
    assert all(row["scenario_count"] == 0 and row["coordinated_success_rate"] is None for row in subsets)
    cg = next(row for row in summarize_methods(rows) if row["group_dimension"] == "all" and row["algorithm"] == PROPOSED)
    assert cg["sum_of_costs_n"] == 0 and cg["sum_of_costs_mean"] is None
    assert cg["expanded_states_n"] == 1 and cg["expanded_states_mean"] == 25


def test_promotion_activity_and_semantic_priority_comparison():
    rows = scenario_rows("a") + scenario_rows("b")
    rows[2]["final_priority_order"] = "[1,2,3]"
    rows[5].update(priority_promotions=2, planning_attempts=3, final_priority_order="[3,1,2]")
    summary = summarize_adaptation(rows)[0]
    assert summary["zero_promotions_count"] == summary["one_or_more_promotions_count"] == 1
    assert summary["priority_promotions_mean"] == 1 and summary["priority_promotions_max"] == 2
    assert summary["planning_attempts_mean"] == 2 and summary["planning_attempts_max"] == 3
    assert summary["priority_order_changed_count"] == 1 and summary["order_comparison_count"] == 2


def test_text_reports_no_promotion_activity_without_claiming_contribution():
    rows = scenario_rows()
    text = "\n".join(analysis_text(rows, paired_comparisons(rows)))
    assert "Priority promotion did not activate" in text
    assert "fixed-only=0" in text and "CG-only=0" in text and "both fail=0" in text


def test_no_fixed_method_is_unavailable_not_a_failure():
    rows = [row for row in scenario_rows() if row["algorithm"] != FIXED]
    assert paired_comparisons(rows) == []
    assert "unavailable" in "\n".join(analysis_text(rows, []))


def test_real_benchmark_repeats_hardness_and_preserves_row_identity():
    scenarios = experiments.generate_scenarios(2, [(6, 6)], [0.2], [2], 42)
    rows = experiments.run_benchmark(scenarios, include_fixed=True, max_expansions=3000)
    groups = validate_paired_rows(rows)
    for group in groups.values():
        baseline = group[INDEPENDENT]
        assert baseline["independent_total_collisions"] == baseline["total_collisions"]
        assert baseline["independent_vertex_collisions"] == baseline["vertex_collisions"]
        assert baseline["independent_edge_collisions"] == baseline["edge_collisions"]
        assert all(tuple(row[field] for field in HARDNESS) == tuple(baseline[field] for field in HARDNESS) for row in group.values())


def test_duplicate_scenario_ids_fail_before_running(monkeypatch):
    scenarios = experiments.generate_scenarios(1, [(5, 5)], [0.2], [2], 42)
    monkeypatch.setattr(experiments, "run_algorithm", lambda *args, **kwargs: pytest.fail("Should reject duplicate IDs before search"))
    with pytest.raises(ValueError, match="unique"):
        experiments.run_benchmark(scenarios * 2)


def test_analysis_csv_outputs_and_missing_fixed_header(tmp_path):
    rows = scenario_rows()
    experiments.write_analysis_outputs(rows, paired_comparisons(rows), tmp_path)
    expected = {"scenario_hardness.csv", "paired_comparisons.csv", "paired_summary.csv", "stratified_summary.csv",
                "subset_summary.csv", "adaptation_summary.csv"}
    assert {path.name for path in tmp_path.glob("*.csv")} == expected
    without_fixed = [row for row in rows if row["algorithm"] != FIXED]
    experiments.write_analysis_outputs(without_fixed, [], tmp_path)
    assert (tmp_path / "paired_comparisons.csv").read_text().count("\n") == 1
