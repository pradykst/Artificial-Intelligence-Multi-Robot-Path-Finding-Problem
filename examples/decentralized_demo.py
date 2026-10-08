"""Deterministic conflict negotiation on the existing crossing corridors."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathfinding.decentralized import decentralized_plan
from pathfinding.demos import conflict_demo

if __name__ == "__main__":
    result = decentralized_plan(*conflict_demo())
    assert result.found and not result.final_conflicts
    for event in result.history:
        print(f"Round {event.round} A{event.agent_id}: {event.kind}: {event.detail}")
    print(f"Success: SOC={result.sum_of_costs}; makespan={result.makespan}; WAIT={result.wait_actions}; "
          f"rounds={result.rounds}; messages={result.messages_sent} sent/{result.messages_delivered} delivered; "
          f"replans={dict(result.replans_by_agent)}")
