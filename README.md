# Artificial Intelligence: Single-Agent and Multi-Agent Pathfinding

CSMI17 Assignment 2 implements canonical A*, centralized conflict-guided coordination and an experimental decentralized negotiation protocol on static grids.

| Team member | Roll number |
| --- | --- |
| Mohd. Kaif | 114123050 |
| Pradyumna Kaushal | 114123065 |
| Ujjwal Sinha | 103123118 |

The [final report](docs/Assignment_2_Final_Report.pdf) and [editable source](docs/report.md) include the problem, assumptions, implementation diagrams, measured comparisons, limitations and actual UI captures. The [decentralized protocol notes](docs/decentralized_extension.md) give detailed message and termination semantics.

## Install and run

Use Python **3.11 or later**, with Tkinter and a desktop display for the UI. Matplotlib is required for benchmark figures; pytest is included for validation. ReportLab is optional and needed only to rebuild the PDF.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

On Unix, activate with `source .venv/bin/activate`. Check Tk availability with `python -m tkinter`; install your platform's Tk package if missing. Benchmark engines and core tests work without opening a window. UI tests skip when Tk cannot initialize a desktop. An editable package installation is optional: `python -m pip install -e .`.

## Movement model and methods

Coordinates are `(x, y)` from the upper-left `(0, 0)`. Free orthogonal moves cost one; there are no diagonals. Obstacles are static. Multi-agent time is discrete; WAIT costs one. Robots remain at goals after arrival. A vertex collision is simultaneous occupancy; an edge-swap collision exchanges two cells in one timestep. Spatial intersections at different times are allowed.

| Method | Implementation and role |
| --- | --- |
| Manhattan A* | Q1 estimate `abs(dx) + abs(dy)` |
| Euclidean A* | Q1 estimate `sqrt(dx*dx + dy*dy)` |
| Chebyshev A* | Q1 estimate `max(abs(dx), abs(dy))` |
| Zero / Dijkstra | Q1 estimate zero; Uniform-Cost Search baseline |
| Independent A* | Manhattan paths computed independently, then jointly collision-checked |
| Fixed-Priority ST-A* | Ascending-ID Space-Time A* with a shared reservation table; benchmark baseline |
| CG without promotion | Conflict-guided initial ordering with the same reserved searches; experiment ablation |
| Full CG-ST-A* | Conflict-guided ordering plus bounded failed-agent promotion and whole-order restart |
| DCN-ST-A* | Experimental private per-robot controllers, peer messages and local reserved replanning |

All Q1 heuristics share FIFO insertion-order ties, up/down/left/right neighbor order, stale-entry rejection and reopening after improved `g`. They are admissible and consistent under this movement model, giving equal optimal costs on solvable inputs. Tied optimal paths may differ. Unreachable results have `found=False`, empty paths and missing cost. Core endpoint overrides may coincide, yielding cost zero.

Central ranking sorts `(-C, -B, -L, ID)`: C counts predicted initial collision events involving each agent, B counts initial path positions with free-cell degree at most two, and L is independent cost. B includes endpoints and is not an articulation-point test. A multi-robot vertex event contributes once per involved agent. Reservations exclude arrival vertices, reverse edges and permanent goal holds; terminal arrival must allow all future occupancy.

Central default horizon cap is `F + max(L)`, with F free cells. Each search starts at `min(cap, L)` and doubles after horizon exhaustion, at least to one, within the cap. One shared 100,000 Space-Time expansion budget covers horizon retries and restarts; independent preprocessing is reported outside it. Full CG permits at most `2N` additional orders. It promotes the first failed agent earlier, skips tried orders, and clears paths/reservations before replanning. Promotion counts positions advanced. A successful initial order needs no promotion. Finite failure does not prove joint infeasibility.

DCN controllers own searches, reservation tables, proposals, versions and inboxes. They access peer paths only through immutable receiver-addressed messages. A neutral scheduler delivers prior-round messages and steps local searches round-robin; it assigns no priorities and computes no routes. Conflicting pairs locally compare fixed `(-initial B, -initial L, ID)` keys. A yielding controller reserves **all received superior peer paths**, replans itself and publishes a higher version. Complete matching version vectors and final collision verification precede execution. The final observer rejects unsafe plans without repairing them.

DCN defaults to 32 total rounds and 100,000 expansions **per controller**, including initial A*. Its horizon cap is F plus maximum initial advertised L. Stale proposals/notices cannot command path changes. Conservative repeated-proposal detection and other limits can stop on feasible inputs. This is sequential Python simulation of reliable synchronous all-to-all messaging, without packet loss, radio range, asynchronous transport, real network delay or hardware parallelism. Central and DCN default aggregate budgets differ.

## Interactive demonstration

Q1 opens by default. Set width, height, obstacle probability and seed. **Generate** reconstructs those settings; **Next map** increments the seed before generation. UI dimensions are capped at 100 per side; the core is not. Single-agent random maps may be unreachable.

**Run**, **Pause**, **Step**, **Reset** and the delay slider control playback. Reset preserves the scenario while clearing search/simulation explanations. Heuristic changes preserve the map and endpoints. Frontier/explored/path overlays reflect actual engine state.

Enable **Multi-Agent Mode** for 2-8 robots. The algorithm selector offers Independent, CG and **Decentralized Negotiation (Experimental)**. Fixed and no-promotion are benchmark methods. **Randomize agents** advances the seed and selects new reachable distinct endpoints on the same obstacles. Generate/Next Map regenerate obstacles and agents. Changing algorithm or heuristic does not generate an input. Coordinated modes enforce Manhattan and restore the prior Q1 heuristic when left. Invalid inputs and planning failures are reported without executing unsafe partial plans.

CG shows its actual C/B/L ranking, rank badges, active robot, reservations and bounded event log. **Agent** playback advances a planning milestone; **Detailed** advances one engine step. Summary/Detailed logging controls search-event verbosity. Reservation t selects a time-specific view, including persistent goals. During planning, bounded frontier/explored overlays explain active search. A callback yields to Tk between bounded chunks; Pause/Reset cancels pending playback.

DCN shows proposal versions, replans, local state, completed rounds, sent/delivered counts and real protocol events. Optional links depict last-barrier delivery, then disappear during execution. **Step** completes one logical negotiation round, then after successful verification one joint execution timestep. Communication rounds and movement timesteps are different. The UI's bounded history may omit earlier events in long runs.

### Reproducible presets

```powershell
python examples/cooperative_demo.py
python examples/decentralized_demo.py
```

In the UI, use **Load Conflict Demo** for hand-built 5x5 crossing corridors. Independent collides at `(2,2)` at t=2 (SOC 8, makespan 4). CG returns SOC 9, makespan 5, one WAIT and zero collisions. Switch methods on that same input. In the DCN panel, **Load Decentralized Demo** uses the same preset: robot 2 yields, publishes v2, and agreement finishes in four rounds, with 11 sent/nine delivered messages, one replan and one WAIT. Two redundant final agreements stay queued. The seed/probability controls govern subsequent generation, not the hand-built preset.

Under CG, **Load Promotion Demo (seed 156)** reconstructs 10x10, p=0.3, eight agents. Agent 8 advances rank 5 to 4: initial order `(4,2,6,5,8,3,1,7)`, final `(4,2,6,8,5,3,1,7)`. Attempt 2 yields SOC 48, makespan 11, WAIT 0, zero collisions. This demonstrates recovery, not guaranteed feasibility.

The [screenshot index](docs/screenshots/README.md) documents genuine captures. The [diagram index](docs/diagrams/README.md) identifies four implementation diagrams, supplied as SVG/PDF/PNG.

## Paired randomized experimental design

**Different trials use different seeded scenarios. Every compared method within one trial receives identical obstacles and endpoints.** Obstacle probability remains fixed within each condition. Algorithms do not receive separately randomized maps in a paired trial.

Generation uses local `random.Random(seed)`. Two distinct endpoints are selected and reserved, then other cells independently become obstacles with probability p. Realized density varies. Q1 accepts reachable start/goal pairs after connectivity checking; reported performance is **conditional on reachability**. UI generation need not enforce that policy.

Q2 samples distinct individually reachable start/goal pairs from connected free-cell components, shuffling/pairing component cells and selecting assignments. All 2N endpoints are distinct. This is not uniform over every reachable assignment. Insufficient endpoint capacity rejects a candidate; coordinated success and predicted collisions never filter an accepted scenario. Individually reachable does not imply jointly feasible.

Each experiment has one global candidate counter, starting at zero:

```text
effective seed = base seed + candidate index
```

It advances after **every** candidate, including rejected draws and across ordered conditions. Thus seed need not equal base plus trial. Changing the ordered condition list changes subsequent accepted seeds. Manifests record exact accepted seeds, attempts, obstacles and endpoints. Same seed/configuration reproduces scenarios and deterministic search outcomes; runtime varies. Different seeds normally differ but uniqueness is not guaranteed. Run order rotates between scenarios to limit fixed execution-order effects.

For individual reproduction, call `Grid.random(width, height, p, effective_seed)`; Q2 also calls `random_agents(grid, N, effective_seed)`. The saved manifest records enough input to check the deterministic SHA-256 fingerprint. Q1 CSV uses `instance_id`/`grid_hash`; Q2 uses `scenario_id`/`scenario_hash`.

### Smoke benchmarks

Use fresh directories. These are lightweight examples, not the full final runs:

```powershell
python benchmark.py --trials 5 --sizes 10x10 --probabilities 0.2 --base-seed 42 --output results/smoke_q1_new
python benchmark_multi.py --trials 1 --sizes 10x10 --probabilities 0.1 --agents 2 --base-seed 42 --include-fixed --include-no-promotion --output results/smoke_q2_new
python benchmark_decentralized.py --trials 1 --sizes 10x10 --probabilities 0.1 --agents 2 --seed 42 --include-no-promotion --no-plots --output results/smoke_dcn_new
```

Q1 includes all four heuristics by default: five scenarios give 20 searches. Q2 with the two inclusion flags gives four methods. DCN with no-promotion gives five. `--help` lists horizon, effort, generation and trial options. DCN uses `--seed`; Q1/Q2 use `--base-seed`.

### Final configurations and measured outcomes

| Dataset | Configuration | Scenarios / runs | Safe successes |
| --- | --- | --- | --- |
| Q1 | 20x20/40x30; p=0.2/0.3; 100 trials; base 123 | 400 / 1,600 | All four: 400; identical paired optimal costs |
| Central Q2 | 10x10/20x20/30x30; p=0.1/0.2/0.3; N=2/4/6/8; 10 trials; base 42 | 360 / 1,440 | Independent 176; Fixed 351; no-promotion 354; full CG 359 |
| DCN pilot | 10x10/20x20; p=0.1/0.3; N=2/4/6/8; 5 trials; base 42 | 80 / 400 | Independent 29; Fixed 73; no-promotion 77; CG 79; DCN 77 |

Stored local outputs are `results/question1/final_100`, `results/question2/final_ablation` and `results/decentralized/pilot`. Q1 has 27 rejected candidates and accepted seeds 123-549; central Q2 seeds 42-401 and pilot 42-121 have no rejections. Each has distinct fingerprints for all accepted inputs. Historical three-method Q2 output remains at `results/question2/full` (360/1,080). Its deterministic rows match the original methods in the ablation dataset, excluding timing/run order.

Central ablation on shared inputs: reservations recover 175 safe plans; initial ordering wins five/loses two (net three); promotion wins five/loses none. Six scenarios invoke promotion, five recover, 14 positions advance, and total full-CG whole-order attempts are 374. The remaining failure is seed 397 (`s2-p2-a8-t0005`): Agent 3 horizon exhaustion with no new eligible order. No infeasibility proof follows.

The DCN pilot is separate from the 360-scenario experiment. It uses 294 completed rounds, 7,931 sent/5,926 delivered messages, 108 replans and 2,005 pending entries at termination. Pending means unprocessed outgoing messages, not loss. Failures are seed 80 (local reservations blocked), 111 (repeated proposal), 117 (local search limit). On 77 both-success pairs with CG, mean DCN-minus-CG is SOC +0.026, makespan -0.195, WAIT +0.078, expanded -733.86 and time +6.83 ms; median quality differences are zero. Unequal aggregate budgets prevent architectural superiority claims.

Deliberate full reproduction is optional and can be expensive; it is **not** an installation check:

```powershell
python benchmark.py --trials 100 --sizes 20x20 40x30 --probabilities 0.2 0.3 --base-seed 123 --output results/reproduction_q1
python benchmark_multi.py --trials 10 --sizes 10x10 20x20 30x30 --probabilities 0.1 0.2 0.3 --agents 2 4 6 8 --base-seed 42 --include-fixed --include-no-promotion --output results/reproduction_q2
python benchmark_decentralized.py --trials 5 --sizes 10x10 20x20 --probabilities 0.1 0.3 --agents 2 4 6 8 --seed 42 --include-no-promotion --output results/reproduction_dcn
```

Q1/Q2 scripts can overwrite matching output filenames; DCN rejects nonempty directories. Use unused paths to preserve historical runs. Full CSVs/manifests/plots are Git-ignored and absent in a fresh clone. Selected measured figures and their [provenance](docs/figures/README.md) are committed.

### Outputs and aggregation

| Output | Contents |
| --- | --- |
| Q1 `raw_results.csv` | Every search: ID/hash, effective seed, condition/endpoints, success, cost, expanded/generated, peak frontier, time and rotated run order |
| Q1 `instances.json` | Configuration, sampling policy, rejected count and exact accepted scenarios |
| Q1 `summary.csv` | Per-condition/heuristic counts, means, medians, sample SD, min/max and success |
| Q1 paired CSVs | Expanded-node differences for each heuristic pair, A minus B as labeled; selected comparison plots use alternative minus Manhattan |
| Q2 `raw_results.csv`, `scenarios.json` | Paired planner metrics, scenario identity, bounds, initial/final orders, attempts, promotion history and exact input |
| Q2 analysis CSVs | Hardness, strata, overlapping conflict subsets, Fixed/CG pairs, adaptation and three mechanism ablations |
| DCN `raw_runs.csv`, `manifest.json` | Five-method paired inputs, safe success, quality, effort, communication, replans and failures |
| DCN `summary.csv`, `paired_quality.csv` | Available-value counts/means/medians/sample SD and DCN-minus-reference both-success differences |

Expanded counts valid pops including goals; stale heap pops are excluded. Generated counts improved successor insertions after the initial start insertion. Peak frontier is maximum unique live OPEN size, not stale heap size or total bytes. For a composite planner it is the maximum constituent-search peak, not the sum. Repeated expansions/insertions can count again.

SOC sums arrival action costs; makespan is latest arrival; WAIT counts stationary actions before first arrival, not permanent goal holding. Coordinated success requires complete collision-free paths. Failed cooperative costs/collision-quality fields are missing, not zero. Historical central raw Independent costs describe its individual trajectories even when unsafe; the pilot records coordinated quality as missing on those unsafe cases. Paired safe-quality analysis filters both methods for coordinated success.

Success rates include all accepted scenarios. Averages/medians/sample SD include only available metric values and state the count. DCN sample SD is missing for a singleton; Q1 and legacy central summaries use zero for a singleton. Empty subsets have zero observations and missing aggregates. Pairing validators reject duplicates/inconsistent metadata/missing methods rather than silently aligning unrelated rows. Both-success quality comparisons and all-run effort/time populations are explicitly separate. Conflict subsets overlap and do not change sampling.

Computation uses `perf_counter` around active algorithm work, excluding animation pauses, drawing, dataset preparation, CSV writing and plotting. Cooperative time includes preprocessing, reservations, retries and final checks; DCN includes active delivery/local detection/search/final checks. Primary benchmark processes briefly overlapped during collection, so timings reflect machine load. Sub-millisecond differences are noisy. Error bars show scenario sample SD, not confidence intervals; there are no significance or general superiority claims.

To regenerate selected primary figures **after deliberate reproduction**, also generate a historical comparison and checksum snapshot:

```powershell
python benchmark_multi.py --trials 10 --sizes 10x10 20x20 30x30 --probabilities 0.1 0.2 0.3 --agents 2 4 6 8 --base-seed 42 --include-fixed --output results/reproduction_history
python examples/final_evaluation.py --snapshot-history --historical results/reproduction_history --history-snapshot results/reproduction_history_sha256.json
python examples/final_evaluation.py --q1 results/reproduction_q1 --q2 results/reproduction_q2 --historical results/reproduction_history --history-snapshot results/reproduction_history_sha256.json --assets results/reproduction_figures
```

That helper validates manifests, pairing, Q1 optimal costs, deterministic historical rows and eight Q2 analysis tables, then writes figures/validation files into the supplied run directories. It does not rerun planners. Use fresh paths as above to preserve saved outputs. Its `validate_q1`/`validate_q2` functions perform read-only checks when imported directly.

## Report and figure generation

```powershell
python -m pip install -r requirements-report.txt
python examples/build_diagrams.py
python examples/build_report.py
```

Report source is [docs/report.md](docs/report.md); output is [docs/Assignment_2_Final_Report.pdf](docs/Assignment_2_Final_Report.pdf), 16 pages. The builder uses bundled ReportLab Vera fonts, explicit portrait/landscape page directives, configurable table widths and image-height caps, with captions kept with images. It requires only local committed assets, so PDF rebuilding works from a fresh clone without benchmark datasets. Diagram rebuilding uses Matplotlib. Diagrams are technical vector assets; screenshots are unmodified native captures.

The report cites primary A*/cooperative/decentralized planning literature and distinguishes established algorithms from coursework adaptations. It states reachability conditioning, finite limits, message semantics, failed-quality handling, pilot scope and unequal budgets.

## Tests and repository structure

```powershell
python -m pytest -q -rs
python -m compileall -q src tests examples
git diff --check
```

Final validation: 204 passed, one transient Tk initialization skip, zero failed. The skipped decentralized Run/Pause/Step/Reset UI case passed individually on retry. Core/experiment tests cover optimality against BFS, deterministic seeds, paired identities, failure/missing aggregation, reservation/goal/edge semantics, promotion, observers, local ownership, stale versions, multilateral negotiation and bounded failure. UI tests check generation, next-map/random-agent seeds, switching, preserved inputs, cancellation and safe execution. Native screenshots provide additional live evidence.

```text
main.py                       unified Tkinter application
benchmark.py                  Q1 CLI
benchmark_multi.py            centralized Q2 CLI and ablation
benchmark_decentralized.py    five-method pilot CLI
src/pathfinding/
  grid.py, astar.py, heuristics.py    grid and canonical search
  agents.py, multi_agent.py         agents, independent paths and simulation
  collisions.py                    vertex and edge-swap events
  space_time.py, reservations.py    time search and occupancy constraints
  cooperative.py                   centralized CG/Fixed/no-promotion
  decentralized*.py                controllers, messages, UI protocol panel
  experiments*.py, analysis*.py     sampling, pairing and aggregation
  final_plots.py                   measured primary figures
  visualization.py, planning_view.py, multi_agent_view.py
examples/                     demonstrations, evaluation and report builders
tests/                        existing plus focused experiment/protocol/UI tests
docs/                         editable report, PDF, diagrams, figures, screenshots
results/                      ignored local generated runs; .gitkeep tracked
```

Caches, environments and full benchmark outputs are ignored. Source, focused tests and selected submission assets are tracked. No physical robots, real networking, dynamic obstacles, completeness, joint optimality, guaranteed negotiation convergence or parallel hardware speedup are claimed. Publishing this repository and report does not perform instructor-side submission.
