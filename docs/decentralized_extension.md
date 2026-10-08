# Optional decentralized pathfinding extension

DCN-ST-A* (Decentralized Conflict Negotiation with Space-Time A*) is an experimental simulated coordination protocol. It extends the completed Q1/Q2 implementation and is included in the final assignment report, with its pilot evaluated separately. It demonstrates local planning and explicit communication; it is not presented as a new research invention.

## Architecture and assumptions

Each `RobotController` owns one robot's ID, start, goal, proposed path, version, inbox, known peer proposals, conflicts, agreement state, A* search, reservation table and effort counters. Controllers receive the shared immutable static grid and peer IDs. They receive other robots' trajectories only through immutable receiver-addressed messages. Controllers never inspect another controller or use the centralized cooperative planner.

`DecentralizedPlanner` is a simulation scheduler: it delivers previous-round messages, invokes each controller's local computations and enforces round barriers. Round-robin search stepping simulates synchronous concurrent decisions without OS parallelism. The scheduler neither assigns priorities nor reserves paths globally. Its initial/final collision checks and UI snapshots are observers; they do not route robots or repair plans.

The model assumes a common static four-connected unit-cost obstacle map, known peer IDs, reliable all-to-all communication and synchronous rounds. There is no radio range, packet loss, asynchronous delay, moving obstacle sensing or real network implementation. Computation is sequential Python; rounds are logical communication steps, not network latency or hardware speedup measurements.

## Messages and ordering

The frozen `Message` structure carries sender, receiver, round and sender version, plus fields relevant to its type:

| Type | Relevant payload | Use |
|---|---|---|
| PATH_PROPOSAL | Immutable path and initial local priority key | Publish the first independently computed trajectory |
| UPDATED_PATH | Revised immutable path and unchanged key | Publish a successful local replan |
| CONFLICT_NOTICE | Collision details and the two proposal versions | Identify the predicted pair conflict |
| YIELD_DECISION | Winner ID and the two proposal versions | Advertise the independently computed pair decision |
| AGREEMENT | Sorted complete `(agent ID, proposal version)` vector | Announce local compatibility with those proposals |
| PLANNING_FAILURE | Actual local failure reason | Announce inability to continue |

A proposal replaces a known proposal only when its version is strictly greater and its round is no earlier. Duplicate versions cannot overwrite a trajectory, even if their round is newer. The initial advertised key cannot change. Notices/decisions describe their version pair and are logged; local decisions are independently recomputed from the current two advertisements, so stale notices cannot command a replan. Agreement vectors must match the controller's current complete version vector.

Every broadcast creates one message per other receiver. `messages_sent` counts these actual receiver-addressed queue entries; `messages_delivered` counts calls into receiver inboxes. Messages are delivered at the next round barrier while the protocol is active. On termination, remaining outgoing messages stay queued and are not included in delivered counts. These are terminal, unprocessed messages, not simulated packet loss. For a successful run they are redundant final agreements. Failures can also leave failure notices or proposals pending. No network latency is modeled. Central methods' communication fields are missing/N/A, not zero-message performance baselines.

## Local protocol

1. Round 0: every controller runs the existing canonical Manhattan A* from its own start to goal and publishes version 1.
2. At each later round, previous messages are delivered and each controller updates its private knowledge. It checks predicted vertex conflicts, arrival-time edge swaps and permanent goal occupancy involving its own path. Sharing a spatial cell at different times is allowed.
3. Each conflicting pair independently compares `(-B, -L, agent_id)`. `B` counts positions on the initial path whose static free-cell degree is at most two, including endpoints; `L` is initial path cost. The smaller tuple wins. Keys stay fixed across revisions to avoid reversing pairwise decisions. No conflict-load ranking or global conflict graph is constructed.
4. Each robot advertises its notices and pair decisions. A robot that loses any pair locally builds a fresh reservation table from **all received superior peers**, including currently nonconflicting superior paths. Protecting them prevents a revision from introducing a conflict with an already known superior route. Lower-priority routes can still conflict and must respond in subsequent rounds.
5. The yielding controller runs existing Space-Time A* with moves/WAIT, vertex reservations, reverse-edge exclusions and permanent goal holds. It publishes a higher-version path on success. An unsuccessful replan does not make the old conflicting proposal executable.
6. A controller with complete peer information and no own conflicts broadcasts an agreement vector. It agrees only after receiving matching vectors from every peer and checking compatibility again. Every controller must agree. The scheduler then independently checks the complete returned trajectories. Any residual collision rejects the plan without repair. Only successful verified paths enter `Simulation`.

Pair decisions use received advertisements immediately; they do not require a separate acknowledgement round before replanning. Explicit decision messages expose the same deterministic decision to the other robot. New proposals invalidate cached peer agreement and require fresh version-vector agreement.

## Bounds, termination and result definitions

Defaults are 32 total logical rounds, including round 0, and 100,000 cumulative search expansions **per controller**, including its initial A*. The default local horizon cap is `F + max(initial peer/own L)`, where `F` is the number of free grid cells. Replanning starts at the smaller of its current path cost and the cap; horizon exhaustion doubles the horizon up to the cap. All retries and later replans share that controller's expansion budget. An explicit nonnegative horizon cap is supported.

The decentralized budget is per robot. Existing centralized methods retain their original shared 100,000 space-time-expansion budget and do not charge independent preprocessing to that budget. Therefore the pilot compares actual default implementations on paired inputs, **not equal aggregate computation budgets**. Effort and runtime are reported separately; a success count cannot establish architectural superiority.

Real stopping conditions are round exhaustion (`negotiation_round_limit`), unreachable initial A* (`local_path_unreachable`), cumulative effort exhaustion (`local_search_limit`), blocked/horizon-exhausted local reservations (`local_reservation_blocked:<search reason>`), or a successful replan repeating an earlier proposal (`no_progress:repeated_proposal`). Repeat detection is conservative: changed peer trajectories might make a previously used route useful again, but this implementation aborts that cycle rather than permitting indefinite oscillation. A fixed total pair ordering avoids cyclic priority preferences; it does not guarantee feasible routes or convergence. Failed runs are not proofs of joint infeasibility.

`DecentralizedResult` reports complete successful paths or an empty executable-path mapping on failure; rounds; actual queue-entry/delivery counts; replans by agent; initial and final observed conflict events; SOC, makespan and WAIT; expanded/generated states; computation time; failure/per-agent status; and up to 256 protocol events. On failure, final conflicts describe rejected proposals, possibly an incomplete set, and **not executed solution collisions**. Failed SOC/makespan/WAIT are `None`. One local replan counts once even if its search needs several horizon retries. Every valid search pop, including a goal pop, counts as expanded; generated excludes the initial insertion, matching existing primitives.

SOC sums arrival costs; makespan is the latest arrival; WAIT counts stationary actions before first goal arrival. Goals remain occupied thereafter. Computation time accumulates active scheduler calls, including local detection, messaging, search and final verification, while excluding UI pauses, drawing and playback delay. The bounded histories need not contain a complete long-run trace.

## Measured pilot

The initial pilot uses 10×10/20×20 grids, obstacle probabilities 0.1/0.3, 2/4/6/8 agents and five trials per condition: **80 scenarios, 400 runs across five methods**. Base seed 42 produced effective seeds 42–121, 80 distinct fingerprints and no rejected candidate seeds. Results are in `results/decentralized/pilot`.

The existing generator consumes a new effective seed for every candidate, including rejected endpoint-capacity/reachability samples. Every method receives the same immutable grid and agents within a trial. Obstacle probability remains fixed within its condition. Selection conditions on individually reachable, distinct start/goal pairs; it never filters for coordinated success. Different seeds normally differ but uniqueness is not guaranteed. Difficult scenarios and all failures remain included. Planner execution order rotates between scenarios.

| Method | Collision-free successes / 80 | Success rate | Mean successful SOC | Mean successful makespan | Median computation (ms), all runs |
|---|---:|---:|---:|---:|---:|
| Independent A* | 29 | 36.25% | 33.93 | 15.17 | 0.65 |
| Fixed-Priority ST-A* | 73 | 91.25% | 57.10 | 17.68 | 2.82 |
| Conflict-Guided, no promotion | 77 | 96.25% | 58.21 | 17.52 | 3.34 |
| Full CG-ST-A* | 79 | 98.75% | 59.30 | 17.70 | 3.35 |
| DCN-ST-A* | 77 | 96.25% | 58.69 | 17.30 | 7.15 |

Successful-quality averages above use different solvable subsets and should not be interpreted as paired quality improvements. The raw CSV preserves scenario IDs and fingerprints. `summary.csv` supplies counts, means, medians and sample standard deviations for each available metric; `paired_quality.csv` contains DCN minus reference differences only when both returned safe solutions. For the **77 shared successes with CG**, mean differences were SOC +0.026, makespan −0.195, WAIT +0.078, expanded states −733.86 and computation +6.83 ms; median quality differences were all zero and median time difference was +2.65 ms. This pilot does not establish a general advantage, and small runtime differences are timing-sensitive.

DCN used 2–6 total rounds (mean 3.675, median 4), 7,931 receiver-addressed messages sent (mean 99.14), 5,926 delivered (mean 74.08), and 108 local replans (mean 1.35). There were 2,005 pending messages at termination. Counts include failures. DCN mean computation over all runs was 97.85 ms versus CG's 33.47 ms; hard failures dominate these means. Communication is an explicit extra cost, not a comparison with centralized internal calls as zero-message transport.

DCN failures were seed 80 (`local_reservation_blocked:reservation_blocked`), seed 111 (`no_progress:repeated_proposal`) and seed 117 (`local_search_limit`). CG failed one search-budget case; Fixed failed seven; no-promotion failed three. All 51 unsafe independent cases retain their observed vertex/edge collisions but have undefined coordinated solution quality. The plot `success_and_messages.png` summarizes success and DCN communication without changing existing assignment figures.

## Demonstration and UI

Use the existing application: enable Multi-Agent Mode, select **Decentralized Negotiation (Experimental)**, then press **Load Decentralized Demo** in its panel. The demo reuses the deterministic 5×5 perpendicular corridors and two robots, so the exact same input can be inspected under Independent and CG without regenerating it. The preset is hand-built; obstacle probability/seed fields continue to control subsequent Generate/Next Map operations, not the preset's layout.

Independent predicts one vertex conflict at t=2, SOC 8 and makespan 4. DCN uses four total rounds; robot 2 yields, publishes version 2 with one WAIT, and both agree. Its verified solution has SOC 9, makespan 5 and zero collisions, with 11 messages sent, nine delivered and one local replan. These events and paths are generated by the actual protocol.

The separate panel shows proposal versions, local states, round/message counts, conflict notices, pair decisions, horizon retries, revised paths, agreements and failures. It displays no centralized ranking. Conflict highlights refer to the round's detection snapshot. Optional communication links show only messages delivered at the last barrier and disappear during execution. Step completes one logical negotiation round, yielding to Tk between bounded chunks, then advances one joint execution timestep. Run/Pause/Reset and the existing speed control remain available. Each callback performs at most 200 controller search steps or approximately eight milliseconds before yielding; local round setup and observational checks are finite synchronous operations rather than hard real-time tasks. Switching algorithms preserves grid, agents and seed; coordinated modes enforce Manhattan and restore the user's previous Q1 heuristic when leaving them.

## Reproduction and validation

From the repository root with the existing requirements installed:

```powershell
python main.py
python examples/cooperative_demo.py
python examples/decentralized_demo.py
python benchmark_decentralized.py --include-no-promotion --seed 42 --output results/decentralized/pilot_reproduction
python benchmark_decentralized.py --trials 1 --sizes 10x10 --probabilities 0.1 --agents 2 --seed 42 --no-plots --output results/decentralized/smoke_reproduction
python -m pytest -q -rs
python -m compileall -q src tests examples
git diff --check
```

Use a fresh output directory: the pilot script rejects nonempty directories. Reproduction produces the same scenarios and deterministic planner outcomes; elapsed times vary. The 80-scenario pilot is separate from the historical 360-scenario Q2 experiment. Pilot reproduction does not overwrite historical benchmark data or selected assignment figures. The final report integrates the pilot separately from central Q2.

Focused tests cover local ownership, immutable delivery, stale ordering, pair decisions, timed/goal conflicts, local reservations, WAIT, multilateral negotiation, cycles, bounded failures, final observer rejection, deterministic protocol results, paired CSV identity, missing failure quality, aggregation and UI controls. Existing tests remain in place. Final validation records 204 passed, one transient Tk initialization skip and zero failures; the skipped Run/Pause/Step/Reset UI case passes individually on retry. Compilation and diff checks pass. Read-only validators verify stored Q1/Q2 manifests, pairing, summaries and deterministic historical Q2 rows. The final documentation task preserves existing planners, tests and historical datasets, integrates this pilot into the report, and adds technical diagrams and a genuine DCN screenshot. No completeness, optimal joint cost, guaranteed convergence, real network latency or parallel hardware improvement is claimed.
