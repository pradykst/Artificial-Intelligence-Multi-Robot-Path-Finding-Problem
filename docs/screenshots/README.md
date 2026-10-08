# Live application evidence

These five PNGs and one JPEG are unmodified captures of the running Tkinter application.
They illustrate behavior; displayed UI planning times are not benchmark timings.

| File | Input and captured state |
| --- | --- |
| q1_astar.png | 20x20, p=0.25, seed 42, Manhattan; final optimal path cost 24, 66 expansions. |
| q2_independent_collision.png | Hand-built 5x5 conflict preset, two agents, Independent A*, t=2; vertex collision at (2,2). The retained probability/seed controls do not generate this preset. |
| q2_priority_ranking.png | Same conflict preset; computed C=1, B=4, L=4 for each agent, ID tie-break, order (1,2). Red cell is a predicted collision. |
| q2_reservations.png | Same preset; first agent's path committed, intersection reserved at arrival t=2; second agent not yet planned. |
| q2_priority_promotion.png | Promotion preset: 10x10, p=0.3, seed 156, eight agents; log records Agent 8 advancing from rank 5 to 4, reservations cleared for attempt 2. Execution complete, SOC 48, makespan 11, WAIT 0, collisions 0. |

Reproduce with `python main.py`. Run the default single-agent input; use
**Load Conflict Demo** for the next three states and **Load Promotion Demo
(seed 156)** under CG-ST-A* for the last. Use Step with Agent playback for
analysis and reservation checkpoints. Set Reservation t to 2 and enable Show
Reservations after the first committed path. The promotion image shows the
recorded event by scrolling the planning log after execution.

| Additional capture | Input and captured state |
| --- | --- |
| [q2_decentralized_negotiation.jpg](q2_decentralized_negotiation.jpg) | Same hand-built 5x5 crossing preset, DCN after R2: three completed rounds, nine sent/seven delivered, A1 v1 and A2 v2, one replan, both local agreement. Execution remains t=0 until peer-vector confirmation. The scrolled real log shows R1 notices, winner/yield decisions, local horizon retries and UPDATED_PATH. |

Select Decentralized Negotiation, load its demo, and complete three Step rounds. Scroll the protocol log to R1 to reproduce the captured explanation. A fourth round verifies agreement; completed execution has SOC 9, makespan 5, one WAIT and zero collisions. The JPEG stores native capture bytes without cropping, annotations or content edits.
