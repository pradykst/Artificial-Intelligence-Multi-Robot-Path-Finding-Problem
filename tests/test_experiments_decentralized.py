import csv
import json

import pytest

from pathfinding.decentralized import DCN
from pathfinding.experiments_decentralized import METRICS, main, paired_quality, run_pilot, summarize, validate_pairs
from pathfinding.experiments_multi import generate_scenarios, run_algorithm, scenario_hash


def test_paired_seeded_inputs_and_existing_central_results_unchanged():
    scenarios = generate_scenarios(5, [(10, 10)], [0.3], [4], 42)
    assert len({s.seed for s in scenarios}) == len({scenario_hash(s) for s in scenarios}) == 5
    assert scenarios == generate_scenarios(5, [(10, 10)], [0.3], [4], 42)
    rows = run_pilot(scenarios)
    assert len(rows) == 20
    for scenario in scenarios:
        group = [r for r in rows if r['scenario_id'] == scenario.scenario_id]
        assert len({r['scenario_hash'] for r in group}) == len({r['seed'] for r in group}) == 1
        for row in group:
            if row['algorithm'] != DCN:
                original = run_algorithm(scenario, row['algorithm'])
                for key in ('coordinated_planning_success', 'expanded_states', 'generated_states', 'failure_reason'):
                    assert row[key] == original[key]
                assert row['messages_sent'] is row['messages_delivered'] is row['negotiation_rounds'] is None
    methods = list({r['algorithm'] for r in rows})
    with pytest.raises(ValueError, match='Missing/duplicate'):
        validate_pairs(rows[:-1], methods)
    altered = [dict(r) for r in rows]
    altered[0]['seed'] += 1
    with pytest.raises(ValueError, match='inputs differ'):
        validate_pairs(altered, methods)


def test_missing_failure_quality_and_correct_aggregation():
    scenarios = generate_scenarios(2, [(10, 10)], [0.1], [2], 11)
    rows = run_pilot(scenarios, max_rounds=1)
    summary = next(r for r in summarize(rows) if r['algorithm'] == DCN)
    assert summary['sample_count'] == 2 and summary['success_count'] == 0
    assert summary['sum_of_costs_count'] == 0 and summary['sum_of_costs_mean'] is None
    assert summary['negotiation_rounds_mean'] == summary['negotiation_rounds_median'] == 1
    assert summary['negotiation_rounds_std'] == 0
    assert not paired_quality(rows)


def test_csv_metadata_manifest_and_preserving_nonempty_outputs(tmp_path):
    output = tmp_path / 'pilot'
    args = ['--trials', '1', '--sizes', '10x10', '--probabilities', '0.1', '--agents', '2',
            '--seed', '123', '--no-plots', '--output', str(output)]
    main(args)
    manifest = json.loads((output / 'manifest.json').read_text())
    rows = list(csv.DictReader((output / 'raw_runs.csv').open()))
    assert manifest['scenario_count'] == 1 and manifest['run_count'] == len(rows) == 4
    assert {r['seed'] for r in rows} == {'123'}
    assert {r['scenario_hash'] for r in rows} == {manifest['scenarios'][0]['scenario_hash']}
    assert all(r['messages_sent'] == '' for r in rows if r['algorithm'] != DCN)
    with pytest.raises(SystemExit):
        main(args)


def test_sample_statistics_exclude_missing_values_without_zero_imputation():
    rows = []
    for value, success in ((2, True), (8, True), (None, False)):
        rows.append(dict(algorithm=DCN, coordinated_planning_success=success,
                         failure_reason=None if success else 'local_search_limit',
                         **{metric: value for metric in METRICS}))
    r = summarize(rows)[0]
    assert r['sample_count'] == 3 and r['success_count'] == 2
    assert r['success_rate'] == pytest.approx(2/3)
    assert r['sum_of_costs_count'] == 2
    assert r['sum_of_costs_mean'] == r['sum_of_costs_median'] == 5
    assert r['sum_of_costs_std'] == pytest.approx(18**0.5)  # sample, not population deviation
