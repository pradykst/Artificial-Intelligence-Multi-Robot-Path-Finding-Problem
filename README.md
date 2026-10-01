# Question 1: Single-Robot Pathfinding

A robot must reach a goal on a rectangular grid while avoiding obstacles. This
project implements A*, an interactive Tkinter demonstration, and reproducible
experiments comparing three heuristics. The same window also supports an
Independent A* multi-agent baseline and Conflict-Guided Adaptive Space-Time A*
(CG-ST-A*), the proposed conflict-guided coursework variant used in this project.

## Model and assumptions

- Coordinates are `(x, y)`, starting at the top-left cell `(0, 0)`.
- Moves are up, down, left and right, each with cost 1. No diagonal moves.
- Cells are either free or obstacles. Grid endpoints are distinct free cells.
- Random generation uses a local `random.Random(seed)`. Two distinct endpoints
  are selected uniformly, then every other cell independently becomes an obstacle
  with the supplied probability. The realized obstacle fraction can differ from
  that probability. Equal parameters and seeds reproduce the same map.
- Single-agent random maps in the UI may be unreachable. A* returns `found=False`, an empty
  path, and `path_cost=None` in that case.
- The search API accepts optional free `start` and `goal` overrides. If these
  coincide, it returns a one-cell path with cost 0.

## A* and heuristics

A* keeps the best discovered cost `g(n)` and orders its frontier by
`f(n) = g(n) + h(n)`. A better route updates the parent and inserts a new heap
entry. Stale entries are ignored. When the goal is removed from the live
frontier, parent links reconstruct an optimal path.

All heuristics use the same FIFO insertion-order tie-break for equal `f` values
and the same up/down/left/right neighbor order. There is no heuristic-specific
tie-breaking. States can reopen after a strictly improved `g` score.

For `dx = x - goal_x` and `dy = y - goal_y`:

| Heuristic | Formula |
| --- | --- |
| Manhattan | `abs(dx) + abs(dy)` |
| Euclidean | `sqrt(dx² + dy²)` |
| Chebyshev | `max(abs(dx), abs(dy))` |

Every path needs at least `abs(dx) + abs(dy)` unit moves. Obstacles can only
increase the shortest distance, so Manhattan is admissible. Euclidean and
Chebyshev are no larger than Manhattan and are also lower bounds. Each is zero
at the goal and consistent across a unit move. These properties justify A*'s
optimality. Manhattan is the strongest of these lower bounds for this model;
experimental performance is measured rather than assumed.

The optional `Zero / Dijkstra` baseline uses `h(n) = 0`. It is uniform-cost
search, not one of the three required heuristics. The core accepts independent
heuristic functions; optimality assumes an admissible heuristic that is zero at
the goal.

## Metrics

| Metric | Exact definition |
| --- | --- |
| `found` | Whether the search reached the goal. |
| `path_cost` | Number of unit movements, `len(path) - 1`; `None` on failure. |
| `expanded_nodes` | Valid states removed from the live frontier and processed. Includes the terminal goal pop, which generates no successors; counts repeated expansions if a state reopens. Stale heap pops are excluded. |
| `generated_nodes` | Successor insertions/reinsertions caused by a strictly improved `g` score. Excludes the initial start insertion. |
| `peak_frontier_size` | Maximum number of unique states currently awaiting expansion, initially 1. Stale heap entries and explored states do not contribute. |
| `elapsed_ms` | Accumulated wall-clock time from `perf_counter_ns` around search initialization and each algorithm step, including heap work, heuristic evaluation and path reconstruction. Excludes time between steps, UI painting, animation, dataset preparation, CSV writing and plotting. |

The UI also displays the current unique frontier size. Both UI and benchmark
use the same `AStarSearch` engine and timing boundaries. Fine-grained timer
instrumentation adds overhead; very short runtime measurements are noisy.
Expanded nodes are the primary algorithmic performance measure.

## Structure

```text
main.py                       Tkinter entry point
benchmark.py                  Benchmark CLI entry point
benchmark_multi.py            Separate Question 2 benchmark CLI
src/pathfinding/
    grid.py                   Immutable grid and seeded generation
    heuristics.py             Independent heuristic functions
    astar.py                  Shared step engine and batch wrapper
    metrics.py                SearchResult dataclass
    visualization.py          Tkinter controls and Canvas view
    experiments.py            Dataset generation, CSV summaries and plots
    agents.py                 Agent endpoints and reachable random placement
    multi_agent.py            Independent planning, simulation and metrics
    collisions.py             Algorithm-independent trajectory collision detection
    demo_state.py             Display-independent mode and session state
    multi_agent_view.py       Agent/path/collision Canvas rendering
    reservations.py           Timed vertices/edges and persistent goal occupancy
    space_time.py             Bounded A* on (cell, timestep) with WAIT
    cooperative.py            Priority analysis, sequential planning and adaptation
    demos.py                  Deterministic classroom conflict scenario
    experiments_multi.py      Paired Question 2 experiments and plots
    analysis_multi.py         Hardness, paired differences, strata and adaptation summaries
    analysis_plots.py         Focused paired and coordination figures
examples/multi_agent_smoke.py  Headless baseline demonstration
examples/cooperative_demo.py  Same-scenario Independent A* / CG-ST-A* demonstration
tests/                        Core, experiment and UI tests
results/                      Local generated experiment outputs
```

## Installation

Use Python 3.11 or newer. From the repository root:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then:

```sh
python -m pip install -r requirements.txt
```

Tkinter comes with standard Windows Python installations. On Linux it may
require the operating system's `python3-tk` package. Check availability with
`python -m tkinter`. A desktop display is required for the visualization.
The core search uses only the standard library; matplotlib is used for plots
and pytest for tests. An editable package install (`python -m pip install -e .`)
is optional; the root entry points work without it.

## Visualization

```sh
python main.py
```

Edit width, height, obstacle probability and seed, then click **Generate**.
Repeating the same settings reproduces the map. **Next map** increments the
seed before generating. UI dimensions are limited to 100 cells per side to
keep the display manageable; the core and benchmark do not impose that limit.

With **Multi-Agent Mode: OFF**, select a heuristic, then use **Run**, **Pause**
or **Step** (one expansion).
The delay slider controls milliseconds between expansions; lower is faster.
**Reset** clears the search on the current map. Changing the heuristic resets
the search, and **Run** after completion starts a fresh search on the same map.
Each animation step is scheduled with Tkinter's `after()`.

The legend identifies free cells, obstacles, endpoints, frontier, explored
cells, the current cell and the final path. Endpoint colors and `S`/`G` labels
take precedence over search colors. Statistics update after each step.

## Classroom Demo

1. Launch `python main.py` from the repository root.
2. In Single-Agent mode (**Multi-Agent Mode: OFF**), select a heuristic and use
   **Run**, **Pause**, **Step** and **Reset** to demonstrate A*.
3. Enable Multi-Agent mode (**Multi-Agent Mode: ON**) and click **Load Conflict Demo**.
4. Select **Independent A*** and **Run**. Observe the vertex conflict at `t=2`.
5. Switch to **CG-ST-A*** and **Run** on the same scenario. The coordinated plan
   avoids the conflict using a WAIT action. Switching planners preserves the
   obstacles and agent endpoints; do not generate another map between runs.

## Benchmarks

Default experiment: 50 solvable 20×20 maps at obstacle probability 0.25, starting
with seed 42; three heuristics and 150 total searches.

```sh
python benchmark.py
```

Small smoke run, including the optional baseline:

```sh
python benchmark.py --trials 5 --sizes 10x10 --probabilities 0.2 --base-seed 42 --include-zero --output results/smoke
```

Multiple configurations:

```sh
python benchmark.py --trials 50 --sizes 20x20 40x30 --probabilities 0.2 0.3 --base-seed 123 --include-zero --output results/comparison
```

Trials are per size/probability pair. Dataset preparation uses a connectivity
check to reject unreachable random maps. It advances the integer seed after
every candidate, including rejected candidates, and keeps drawing until the
requested number of solvable maps is reached. `--max-attempts` (default 1000)
bounds attempts per accepted map and produces an error if exhausted. This
conditions the dataset on solvability; results describe solvable maps, not the
frequency or cost of unreachable searches. A* itself is unchanged.

Each accepted immutable map is reused for all selected heuristics. Execution
order rotates between maps to reduce systematic ordering effects. Changing the
ordered configuration list changes subsequent seeds; the saved manifest records
the exact accepted seeds and maps. Reproducibility applies to maps, paths and
search counts, not wall-clock timings. Benchmarking does not launch Tkinter.

Outputs in `results/` (or the selected output directory):

- `raw_results.csv`: every search's metrics, map ID/hash, seed, endpoints,
  dimensions, probability, obstacle count, generation attempts and run order.
- `summary.csv`: run/success counts and mean, median, sample standard deviation,
  minimum and maximum of each numeric metric, grouped by configuration and
  heuristic. A single-observation standard deviation is recorded as 0.
- `instances.json`: exact maps, generation settings, rejected candidate count,
  and Python/platform information.
- `comparison_*.png`: mean and median expanded nodes, execution time and peak
  unique frontier size, separately for each grid configuration.
- `summary.txt`: run counts and the verified path-cost agreement statement.

Every run must succeed and all heuristics must agree on path cost per instance;
the program raises an error if either check fails. There is no path-cost bar
chart. CSV summaries retain the distribution of path lengths across maps.
Reusing an output directory overwrites matching filenames; use a fresh directory
for a separate experiment. Generated outputs are ignored by Git; only
`results/.gitkeep` is tracked.

## Tests

```sh
python -m pytest -q -rs
```

Tests cover known paths, detours, unreachable and already-reached goals,
admissibility, seeded generation, agreement with BFS on small random grids,
frontier/expansion accounting with stale entries and reopened states, timing
boundaries, shared benchmark instances, output files, and UI controls.
UI tests skip if Tkinter or a desktop display is unavailable. BFS is only a
test reference; experiments use a connectivity check solely for map selection.

## Question 2: Multi-Agent Extension

The same application, Canvas, obstacles, heuristic selector and animation
controls serve both modes. Start it with `python main.py` and enable
**Multi-Agent Mode: ON**. The existing Question 1 benchmark remains single-agent.
The **Independent A*** baseline runs the canonical A*
engine separately from each robot's start to its goal. Other robots are ignored
during planning. Paths respect obstacles and are individually optimal, but
**Independent A* does not coordinate robots or resolve collisions**.

**CG-ST-A*** first analyzes the independent paths, then uses ordered Space-Time
A* searches and reservations to coordinate execution. Temporal coordination
adds the challenges of shared cells, opposite movements along an edge, waiting,
and occupied goals. The method combines established planning techniques in a
specific coursework variant; space-time search, reservation tables and
prioritized planning are not claimed as research inventions.

The interactive view supports 2–8 robots for readable colors and markers. The
model and headless planner do not impose this limit. Each agent has an ID and
endpoints; planning results and simulation positions are stored separately.
Grid contains no multi-agent planning logic. Collision detection accepts paths
independently of how they were planned.

### Initialization and controls

- **Generate** uses the displayed dimensions, obstacle probability, seed and
  agent count. **Next map** increments the seed first. Both generate new obstacles.
- **Randomize agents** applies the selected agent count and increments the seed,
  while preserving the current obstacles. The displayed seed can also be edited.
- Every accepted random scenario has mutually distinct starts and goals: all
  `2 × agent_count` endpoints are different free cells. Connected components of
  free space are shuffled deterministically and paired internally, guaranteeing
  individual reachability. The distribution is not uniform over all valid
  assignments. If the map has insufficient reachable pairs, an error is shown
  and the previous scenario is retained; choose fewer robots or another map.
- Generation does not reject colliding independent plans. With the same grid,
  agent count and seed, agent initialization is reproducible. When toggling ON,
  placement uses the currently displayed seed without changing the obstacles.
- **Run**, **Pause**, **Step**, **Reset** and the delay slider are shared. Planning
  runs in scheduled batches before execution. A multi-agent **Step** completes
  any pending planning asynchronously, then advances exactly one timestep and
  pauses. Pause can interrupt planning between batches. Reset discards planning,
  execution and collision history while preserving the scenario. Run after
  completion starts a fresh plan and execution.
- Switching modes cancels scheduled animation and discards obsolete state.
  Dimensions, obstacles and single-agent endpoints are preserved. Switching OFF
  restores the single-agent view and disables multi-agent-only controls.
- Switching **Independent A*** / **CG-ST-A*** preserves the exact grid, obstacles,
  agents and endpoints. It clears the old plan and simulation without immediately
  replanning. Run or Step starts the selected planner. CG-ST-A* uses Manhattan;
  the heuristic control is disabled for it without losing the single-agent setting.
- **Load Conflict Demo** loads a fixed 5×5 cross-shaped corridor with two agents.
  Independent A* produces a vertex conflict at `t=2`. On the same scenario,
  CG-ST-A* finds a conflict-free plan with one WAIT action. Run the baseline,
  change the algorithm, then run again. This constructed case is never included
  in random benchmark data. Seed/probability fields apply to the next random
  generation, not to this preset.

### Time and collision display

Execution starts at `t=0`, with every robot at its start. Each timestep advances
all robots concurrently by one position along their planned paths. A robot
remains at its goal after arrival. Stored paths are finite; `position_at(path, t)`
extends the final position for later timesteps. Simulation uses this accessor.
Paths are never modified in response to a collision, and execution continues
until the final robot arrives.
CG-ST-A* paths may contain repeated positions: each is a WAIT action and keeps
the robot stationary for that timestep. The view shows final priority, attempts,
promotions, total planned waits and planning time, alongside costs and collisions.

Each agent uses the same color for its numbered current-position marker,
`S1`/`G1`-style endpoint labels, and planned path. Slightly offset path traces
separate overlapping routes. Robots sharing a cell have separate numbered
markers inside it. The legend maps IDs to colors.

- **Vertex collision:** two or more robots occupy the same cell at the same
  timestep. A red cell outline and the event history identify the cell, time,
  and every participating agent. A three-way conflict is one event containing
  all three IDs, rather than three pairwise events.
- **Edge-swap collision:** two robots exchange positions between `t-1` and `t`.
  It is recorded at arrival timestep `t`. Red outlines and a dashed red edge
  show the swap even though the robots end in different cells.

Highlights show conflicts in the current timestep. The scrollable event history
and cumulative counts retain earlier conflicts. Two stationary robots sharing
a cell produce vertex conflicts, not edge swaps. Consecutive timesteps sharing
a cell count as separate vertex events. Detection covers `t=0` through the
makespan inclusive, including conflicts involving robots already at their goals.
The reusable detector also supports manually supplied paths with shared endpoints;
the stricter endpoint policy applies to random scenario generation.

### Space-Time A* and reservations

Space-Time A* searches states `(cell, timestep)`. It uses a heap, best `g` scores,
parent links, stale-entry checks and insertion-order tie-breaking for equal
`f = g + h`. Actions are up, down, left, right, then WAIT, all with cost 1.
Manhattan ignores time and other robots, so it remains a lower bound.

The reservation table records:

- `(cell, t)` for vertex occupancy;
- `(source, target, arrival_t)` for an edge traversal, including WAIT edges;
- a terminal goal and its first arrival time, representing occupancy forever.

A proposed move must have an unreserved destination and no reserved reverse
edge during the same transition. A robot may first enter its goal only when it
can stay there indefinitely, including future reservations beyond the current
search horizon. It cannot enter a goal early and later leave to avoid a conflict.

### Conflict analysis and priority adaptation

For each independent path, the method records:

- `C_i`: the number of vertex/edge collision **events** involving agent `i`,
  using the existing detector through the independent makespan. Each participant
  in a three-way vertex event gets one increment.
- `B_i`: the number of path positions with at most two traversable static
  4-connected neighbors, including the start and goal. This is a simple
  topological exposure indicator, not a proof that a cell is a bottleneck.
- `L_i`: the independent shortest-path cost.

The conflict graph links two agent IDs when they share a predicted event.
Multiway events contribute all pairwise graph links, but `C_i` remains an event
count rather than the number of graph neighbors. Graphs and per-agent values
are available in results and recorded in the raw experiment CSV.

Initial priority is the exact lexicographic ordering
`(-C_i, -B_i, -L_i, agent_id)`. There are no weighting coefficients. Agents plan
sequentially in this order; each successful path is reserved before planning the
next agent. Every complete solution is checked with the existing vertex/edge
collision detector. A detected conflict raises a correctness error.

When reservations prevent an agent from obtaining a bounded plan, that first
failed agent is promoted one position earlier and the whole prioritized attempt
restarts with an empty reservation table. If that order was already tried, the
planner tries progressively earlier positions for the same failed agent. If no
new promotion exists, it stops. Attempted orders never repeat. The default
maximum is `2 × number_of_agents` priority retries beyond the first attempt;
`--max-priority-retries` overrides it. Promotions count adjacent positions moved,
so a promotion past two agents counts as two, even though it starts one retry.

**Fixed-Priority ST-A*** is an optional experiment-only ablation using the same
engine, search, horizon policy and reservations, but ascending agent IDs and no
promotion. It also pays the common independent-analysis preprocessing cost;
this comparison isolates the ordering/adaptation policy in the shared engine.

### Bounds and limitations

Let `F` be the number of free cells and `L_max` the longest independent path cost.
Each agent's first search horizon is `L_i`. When the frontier exhausts with a
valid next transition cut off by the horizon, the horizon doubles (from zero to
one when necessary), capped by **`F + L_max`**. The `F` term gives up to one
free-cell traversal's worth of extra temporal slack; it is a practical budget,
not a MAPF completeness bound. A boundary cutoff does not prove that a larger
horizon will solve the instance. `--max-horizon` sets an explicit hard maximum.

An additional default budget of **100,000 expanded space-time states** applies
across all horizon searches and priority attempts for one method/scenario.
`--max-expansions` changes it. Independent preprocessing is finite spatial A*
and is not charged to this space-time budget, but is included in reported
expanded/generated counts and time. Exhausting the expansion budget stops the
planner immediately. Priority retry, horizon, and expansion limits are recorded
alongside the failure reason; `search_limit_reached` identifies a limit-related
failure. The result also records horizon retries and the maximum horizon used.

CG-ST-A* is **not globally optimal**. Earlier paths constrain later robots, so
priority decisions may prevent finding an existing solution. The deterministic
promotion policy explores only a bounded subset of priority orders. Prioritized
planning is not complete for arbitrary MAPF instances, and these explicit bounds
introduce additional failures. A longer coordinated path can be the necessary
cost of eliminating collisions, rather than an error.

### Question 2 metrics

- **Individual path cost:** number of actions until first reaching its goal.
  This is `len(path) - 1` for an unpadded returned path.
- **Sum of costs:** sum of the individual path costs. Holding at a goal after
  arrival adds no cost.
- **WAIT actions:** repeated consecutive positions before first goal arrival.
  Waiting before arrival contributes to cost; post-goal holding does not.
- **Makespan:** maximum individual path cost, the timestep when the last robot
  reaches its goal.
- **Vertex collisions:** number of cell/timestep events with two or more agents.
- **Edge collisions:** number of unordered agent-pair swaps, counted once per
  pair and transition.
- **Total collisions:** vertex events plus edge events.
- **Individual planning success:** every independent spatial path was found.
- **Coordinated planning success / collision-free:** a complete set of paths
  was returned and its joint execution has no vertex or edge conflicts, including
  holding at goals. Independent A* can meet the first success definition while
  failing this one.
- **Cost overhead (%):** `(coordinated_SOC - independent_SOC) / independent_SOC × 100`,
  paired on the same scenario, only when a coordinated plan exists and the
  independent denominator is nonzero. Fixed-priority overhead is recorded too.
- **Expanded states:** valid popped states, including terminal goal pops, summed
  over independent preprocessing and every space-time search, including failed
  attempts and horizon retries. Single-agent states are cells; space-time states
  are `(cell, t)`, so the latter count can exceed the number of cells.
- **Generated states:** improved successor insertions/reinsertions over those
  searches; initial insertions and stale heap pops are excluded.
- **Peak frontier:** maximum unique live frontier size of any constituent search,
  including preprocessing and failed searches; it is not a sum of peaks or total
  memory consumption.
- **Elapsed planning time:** wall-clock time spent initializing and advancing
  the cooperative planner, including independent analysis, reservations, retries,
  reconstruction and final checking. Pauses and animation are excluded. The
  baseline times its batch planning and metric/collision check. These timing
  boundaries include different dispatch overhead; tiny timing differences should
  not be treated as strong evidence.

During animation, collision counts cover only elapsed timesteps; path costs and
makespan describe the complete plans. Collision counts measure conflicts in the
executed traces. Unreachable manually
specified agents produce failed SearchResults; simulation rejects empty paths.

Run the deliberate collision examples without a display:

```sh
python examples/multi_agent_smoke.py
```

This runs actual Independent A* plans for a collision-free case, a vertex
conflict, and an edge swap, checks the expected counts, and prints their metrics.
The test suite additionally covers goal holding, three-agent vertex conflicts,
endpoint uniqueness, deterministic placement, mode-state isolation without Tk,
and multi-agent UI controls. Compare the deterministic classroom scenario with:

```sh
python examples/cooperative_demo.py
```

### Final Question 2 Experiment

The completed experiment used **360 paired scenarios and 1,080 algorithm runs**:
10 trials for every combination of grid sizes 10×10, 20×20 and 30×30, obstacle
probabilities 0.1, 0.2 and 0.3, and agent counts 2, 4, 6 and 8, with base seed 42.
Every method used the same scenarios. The recorded results are:

| Method | Collision-free complete plans | Rate |
| --- | --- | --- |
| Independent A* | 176/360 | 48.89% |
| Fixed-Priority ST-A* | 351/360 | 97.50% |
| CG-ST-A* | 359/360 | 99.72% |

For the cooperative methods, these counts are coordinated planning successes.
Fixed and CG both succeeded on 351 scenarios; Fixed alone succeeded on 0,
CG alone on 8, and both failed on 1. CG priority promotion activated in 6/360
scenarios.

| Independent-conflict subset | Scenarios | Fixed success | CG success |
| --- | --- | --- | --- |
| At least one collision | 184 | 175/184 (95.11%) | 183/184 (99.46%) |
| At least three collisions | 54 | 48/54 (88.89%) | 54/54 (100%) |

Most of the improvement over Independent A* comes from space-time coordination
and reservations. Fixed-Priority ST-A* is an ablation that separates these
reservation benefits from the priority-policy benefits. Conflict-guided/adaptive
ordering provided a smaller additional robustness improvement in this experiment;
the comparison does not isolate promotion from initial ordering. CG used somewhat
greater search effort and sometimes higher SOC and wait costs.

CG-ST-A* is our proposed coursework variant of established techniques, not a
globally optimal or complete method. These observations do not establish universal
superiority, and space-time search, reservations and prioritized planning are not
our inventions. Recorded CSVs and figures are stored locally in
`results/question2/full/`; generated benchmark outputs are excluded from Git.

### Paired Question 2 experiments

Question 1 commands remain `python main.py` for the shared UI and
`python benchmark.py` for the original heuristic experiment.
Question 2 uses a separate entry point and output directory:

```sh
python benchmark_multi.py --include-fixed
```

Defaults are three trials per combination, sizes 10×10 and 20×20, obstacle
probabilities 0.10 and 0.30, agent counts 2 and 4, base seed 42: **24 scenarios**.
All methods use Manhattan so heuristic choice is controlled. With the ablation,
each scenario produces three algorithm runs.

Explicit smoke configuration:

```sh
python benchmark_multi.py --trials 3 --sizes 10x10 20x20 --probabilities 0.1 0.3 --agents 2 4 --base-seed 42 --include-fixed --output results/question2/smoke
```

Reproduce the final Question 2 experiment (runtime grows with congestion and limits):

```sh
python benchmark_multi.py --trials 10 --sizes 10x10 20x20 30x30 --probabilities 0.1 0.2 0.3 --agents 2 4 6 8 --base-seed 42 --include-fixed --output results/question2/full
```

Generation keeps seeded maps on which the existing initializer can place the
required distinct, individually reachable endpoint pairs. It advances the seed
after every candidate and records accepted seeds and rejected counts.
`--max-generation-attempts` (default 1000) bounds draws per scenario. This is
sampling conditional on endpoint capacity/reachability, not a uniform distribution
over all MAPF assignments. **No filtering uses Independent A* collisions or
cooperative success.** Difficult accepted scenarios remain in the data.

Within each scenario, every method receives the same immutable grid and agent
tuple; no map is regenerated between methods. Algorithm execution order rotates
across scenarios. Scenario IDs and hashes include the map and agent assignments.
Each cooperative method performs and times its own independent preprocessing;
it does not reuse a free, externally precomputed baseline result.

Question 2 outputs appear under `results/question2/` or the chosen subdirectory:

- `raw_results.csv`: per-scenario/method metrics, success flags, failure reasons,
  priority analysis/orders/attempt history, search bounds and paired cost overhead.
- `summary.csv`: counts, rates, and per-metric count/mean/median/sample standard
  deviation/minimum/maximum, grouped by size, density, agent count and algorithm.
  Proportions retain their success numerator and scenario denominator.
- `scenarios.json`: exact obstacles/endpoints, seeds, hashes, generation attempts,
  command settings and Python/platform metadata.
- `summary.txt`: observed collision-free counts/rates and missing complete plans.
- `comparison_*.png`: a focused six-panel figure per size/density, against agent
  count: collision-free rate, collisions, runtime, sum of costs, cost overhead,
  and makespan. Solid lines show means, dashed lines medians.

For a failed cooperative plan, cost, makespan, waits, collision counts and overhead
are **blank**, not zero. Such a row still contributes a failure to the success-rate
denominator and contributes its consumed computation to runtime/search statistics.
Quality and collision averages use complete returned plans only. Independent
costs therefore include colliding shortest paths, while cooperative costs are
conditional on success; compare these together with failure counts to avoid
survivorship bias. Per-scenario overhead is paired with the independent shortest
paths, which need not themselves be collision-free. A single observation has
sample deviation recorded as zero; empty metric groups remain blank.

Generated results are ignored by Git. Use a new output directory for a separate
run. These scripts produce descriptive observations, not automatic claims that
the proposed ordering is superior. The final assignment report is separate.

### Fixed-vs-CG analysis

Use `--include-fixed` with the existing Question 2 benchmark command to collect
paired Fixed-Priority ST-A* and CG-ST-A* outcomes. Every accepted scenario retains
exactly one raw row per selected algorithm. Duplicate rows, missing method rows,
or inconsistent IDs, seeds, map/agent hashes or configuration fields raise an
error. The saved manifest is also checked against the result identities.

The following hardness fields are calculated from the already-produced
Independent A* paths, outside its planning timer, and copied to every method's
row for that scenario:

- `independent_total_collisions`, `independent_vertex_collisions` and
  `independent_edge_collisions`: existing detector event counts.
- `number_of_agents_involved_in_any_conflict`: number of distinct participating IDs.
- `total_predicted_conflict_load`: sum of per-agent event loads; a three-way
  vertex event adds three to this total, while remaining one collision event.
- `maximum_agent_conflict_load`: largest per-agent event load, zero when no
  collisions occur. If the independent paths are incomplete, hardness is unknown
  and remains blank rather than being treated as zero.

These are descriptive variables. They do not affect scenario generation,
planning, priority ordering, reservations, horizon bounds, or scenario retention.

Additional analysis files in the selected output directory:

| File | Contents |
| --- | --- |
| `scenario_hardness.csv` | One row per scenario, with identity and hardness variables. |
| `paired_comparisons.csv` | One Fixed/CG pair per scenario; both outcomes, costs/computation, priority activity, deltas and outcome classes. |
| `paired_summary.csv` | Success-outcome counts and paired-difference statistics overall, by stratum, and in descriptive subsets. |
| `stratified_summary.csv` | Per-method success rates and metric statistics grouped separately by agent count, grid size, density, and independent conflict presence, plus the overall group. |
| `subset_summary.csv` | Per-method statistics for the four overlapping descriptive subsets below. |
| `adaptation_summary.csv` | CG promotion and attempt distributions, zero/positive-promotion counts, and initial/final-order changes, overall and by the same strata/subsets. |

Pairs align by scenario ID and verified identity, not by row order. The four
success outcomes are **both succeed**, **Fixed succeeds / CG fails**, **Fixed
fails / CG succeeds**, and **both fail**. Success means coordinated planning
success, not merely that individual shortest paths exist.

`cg_success_minus_fixed_success` is the difference between the two binary success
indicators. All continuous deltas use **CG minus Fixed**:
`delta_soc`, `delta_makespan`, `delta_waits`, `delta_runtime_ms`, `delta_expanded`,
and `delta_generated`. Negative values mean a lower CG value, not an automatic
claim of overall superiority.

SOC, makespan and wait deltas are defined only when **both methods succeed** and
both measurements exist. Their lower/equal/higher classes, and the expanded-state
outcome classes, use this same both-success population. Failures and undefined
measurements remain blank. Runtime/expanded/generated deltas remain defined on
failed runs when both computation measurements exist: failed attempts still use
resources. Their paired summaries additionally include explicitly named
`both_success_delta_*` statistics so the populations can be distinguished.

Every aggregated continuous metric has `n`, mean, median, sample standard
deviation, minimum and maximum. Empty samples retain `n=0` and blank statistics;
single-observation standard deviation follows the existing zero convention.
Proportions retain numerator and denominator counts. No significance test is
applied automatically.

Stratified/subset **quality** statistics use successful coordinated plans for
each method separately. Computation statistics include all measured runs.
Consequently, quality means can use different scenario sets when success rates
differ; the paired summaries provide the directly comparable both-success set.
This also means that Independent A* quality in these new stratified files uses
only its collision-free cases, while the original `summary.csv` continues to
include all complete independent paths, including colliding ones.

The independent-conflict stratum explicitly separates **zero collisions** from
**at least one collision**. For the latter, the summary retains Fixed and CG
success rates and mean/median SOC, makespan, waits, expanded states and runtime,
with a count beside each metric. Unknown hardness, if present, is a separate
group rather than being assigned to either category.

The four post-generation subsets are:

- independent total collisions `>= 1`;
- independent total collisions `>= 3`;
- number of agents `>= 6`;
- obstacle probability `>= 0.3`.

They overlap and must not be added together as disjoint samples. They do not
filter the main benchmark. Empty subsets remain explicit, with zero observations
and undefined rates/means.

The existing `comparison_*.png` figures are preserved. New focused figures are
created separately for each grid-size/density combination:

- `paired_*.png`: per-scenario SOC, makespan and expanded-state differences by
  agent count, using only both-success pairs, with mean/median and observation counts.
- `coordination_*.png`: Fixed/CG coordinated success rates overall and on the
  independent-conflict subset, plus CG promotions by agent count. Counts are shown
  on the axes; absent subsets are labeled explicitly.

The text summary records rates, the four success-outcome counts, priority activity,
and paired lower/equal/higher counts for SOC, makespan, waits and expanded states.
If promotion never activates, it explicitly states that these runs cannot
establish its contribution. A difference between Fixed and CG without promotions
can reflect the initial ordering, but does not demonstrate a benefit from
adaptive promotion.

Without `--include-fixed`, hardness, strata, subset and CG activity outputs still
exist. Paired CSVs contain headers only, no paired-difference figures are created,
and the text states that Fixed-vs-CG comparisons are unavailable. A method that
was not run is never classified as a failed method.
