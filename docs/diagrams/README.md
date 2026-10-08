# Implementation diagrams

Generate with `python examples/build_diagrams.py` using the existing Matplotlib dependency. Each diagram is supplied as SVG (editable vector), PDF (vector export) and 300-dpi PNG (report embedding).

| Diagram | Implementation represented |
| --- | --- |
| [A* workflow](astar_workflow.svg) | Canonical live frontier, goal test, g improvement/reopening, reconstruction and unreachable return in `astar.py` |
| [Central CG workflow](centralized_cg_workflow.svg) | Independent analysis, C/B/L ranking, sequential reserved search, successful acceptance and optional bounded promotion/restart in `cooperative.py` |
| [Architecture comparison](centralized_vs_decentralized.svg) | Central shared order/table versus local controllers/tables and immutable messages, with neutral scheduler and rejecting observer in `decentralized.py` |
| [Negotiation sequence](decentralized_negotiation_sequence.svg) | Actual two-agent crossing demo: R0 proposals, R1 pair decision/local replan, R2 updated proposal/agreement, R3 vector confirmation; execution timesteps follow |

Arrows describe the current implementation, not a general completeness guarantee. Logical communication rounds precede execution and do not model network latency. The source files remain the authoritative algorithm implementation.
