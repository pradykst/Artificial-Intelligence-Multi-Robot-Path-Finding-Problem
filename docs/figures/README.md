# Selected measured figures

Source runs: `results/question1/final_100` and
`results/question2/final_ablation`. Regenerate and validate with
`python examples/final_evaluation.py`. The plotting source is
[`final_plots.py`](../../src/pathfinding/final_plots.py); raw CSVs and exact
scenario manifests remain in the run directories. These six primary PNGs are the selected
presentation assets. Full experimental outputs remain ignored by Git.

| Figure | Factual caption and source |
| --- | --- |
| [Q1 expansions](q1_expansions.png) | Mean expanded nodes, median markers and sample standard deviations for 100 paired reachable scenarios in each of four grid-size/probability conditions. Error bars describe scenario variability, not confidence intervals. Source: Q1 `raw_results.csv`. |
| [Q1 paired expansions](q1_paired_expansions.png) | Expanded nodes for each alternative minus Manhattan on the same scenario; 100 pairs per condition. Markers show mean and median; error bars show sample standard deviation. Source: Q1 `raw_results.csv` and `paired_comparisons.csv`. |
| [Q2 coordinated success](q2_overall_success.png) | Complete collision-free planning on 360 paired scenarios: Independent 176, Fixed 351, conflict-guided without promotion 354, full CG 359. Counts include unsuccessful plans. Source: Q2 `raw_results.csv`. |
| [Q2 success by agent count](q2_success_agents.png) | Coordinated success by 2, 4, 6 and 8 agents; 90 scenarios per agent count, pooled over three grid sizes and three obstacle probabilities. Source: Q2 `stratified_summary.csv`. |
| [Q2 mechanism comparisons](q2_ablation.png) | Discordant success counts over all 360 scenarios; SOC and expansion differences use second minus first on both-success pairs only (n=176, 349, 354). Diamonds show means; expansion differences use a symmetric log scale. Source: Q2 `ablation_comparisons.csv` and `ablation_summary.csv`. |
| [Q2 conflict subsets](q2_conflict_subsets.png) | For at least one independent collision (n=184), Fixed/no-promotion/full-CG successes are 175/178/183. For at least three (n=54), they are 48/50/54. Subsets overlap and do not change sampling. Source: Q2 `subset_summary.csv`. |

The separate [DCN pilot success/messages](dcn_pilot_success_messages.png) is copied unchanged from `results/decentralized/pilot/success_and_messages.png`. It summarizes 80 paired scenarios (400 runs): safe counts 29/73/77/79/77 for Independent/Fixed/no-promotion/CG/DCN, and DCN actual receiver-addressed sent messages by agent count. The plot includes failures; centralized transport is N/A. Its source is `experiments_decentralized.py`; reproduce into a fresh directory with `benchmark_decentralized.py --include-no-promotion --seed 42 --output results/decentralized/pilot_reproduction`.
