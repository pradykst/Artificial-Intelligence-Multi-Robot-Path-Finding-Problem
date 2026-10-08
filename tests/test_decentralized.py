from dataclasses import FrozenInstanceError, replace

import pytest

from pathfinding.agents import Agent
from pathfinding.collisions import detect_collisions
from pathfinding.decentralized import DecentralizedPlanner, RobotController, decentralized_plan
from pathfinding.decentralized_messages import Message, pairwise_winner
from pathfinding.demos import conflict_demo, promotion_demo
from pathfinding.grid import Grid


def finish(planner):
    for _ in range(200_000):
        if planner.done:
            return planner.result
        planner.step()
    pytest.fail("Protocol exceeded bounded test work")


def test_corridor_negotiation_real_messages_wait_and_agreement():
    grid, agents = conflict_demo()
    planner = DecentralizedPlanner(grid, agents)
    result = finish(planner)
    assert result.found and len(result.initial_conflicts) == 1
    assert not result.final_conflicts and not detect_collisions(result.paths)
    assert (result.sum_of_costs, result.makespan, result.wait_actions) == (9, 5, 1)
    assert result.replans_by_agent == {1: 0, 2: 1}
    assert result.messages_sent == sum(c.messages_sent for c in planner.controllers.values())
    assert result.messages_delivered == sum(c.messages_delivered for c in planner.controllers.values())
    assert result.messages_sent - result.messages_delivered == len(planner._pending)
    assert {"PATH_PROPOSAL", "CONFLICT_NOTICE", "YIELD_DECISION", "UPDATED_PATH", "AGREEMENT"} <= {
        event.kind for event in result.history}
    for controller in planner.controllers.values():
        assert controller.agreed and not controller.detect_local_conflicts()
        assert all(vector == controller.version_vector() for vector in controller.peer_agreements.values())
        assert controller.agent_id not in controller.known_peers


def test_message_immutability_addressing_and_stale_version_round_order():
    grid, agents = conflict_demo()
    controller = RobotController(grid, agents[0], (2,))
    message = Message("PATH_PROPOSAL", 2, 1, 2, 3, ((2, 0), (2, 1)), (-2, -1, 2))
    with pytest.raises(FrozenInstanceError):
        message.version = 4
    with pytest.raises(ValueError):
        controller.receive(replace(message, receiver=2))
    controller.receive(message)
    controller.consume_inbox()
    for stale in (replace(message, version=2), replace(message, round=3), replace(message, version=4, round=1)):
        controller.receive(stale)
    controller.consume_inbox()
    assert controller.known_peers[2].version == 3 and controller.stale_proposals == 3
    assert controller.messages_delivered == 4
    controller.receive(replace(message, kind="UPDATED_PATH", version=4, round=3, path=((2, 0),)))
    controller.consume_inbox()
    assert controller.known_peers[2].path == ((2, 0),)


@pytest.mark.parametrize("first,second,winner", [((-4, -5, 2), (-3, -9, 1), 2),
    ((-4, -5, 2), (-4, -6, 1), 1), ((-4, -5, 2), (-4, -5, 1), 1)])
def test_local_policy_is_symmetric_and_deterministic(first, second, winner):
    assert pairwise_winner(first, second) == pairwise_winner(second, first) == winner


def test_tables_and_searches_owned_locally_and_built_only_from_received_paths():
    grid, agents = conflict_demo()
    planner = DecentralizedPlanner(grid, agents)
    first, second = planner.controllers.values()
    assert first.search is not second.search and first.inbox is not second.inbox
    assert first.reservations is not second.reservations
    while planner.rounds == 0:
        planner.step()
    assert first.known_peers == second.known_peers == {}
    assert not first.reservations.vertices and not second.reservations.vertices
    planner.step()  # deliver published proposals, then make independent local decisions
    assert first.known_peers[2].path == second.path
    assert second.reservations.vertices and not first.reservations.vertices
    assert first.reservations is not second.reservations
    assert second.search.reservations is second.reservations
    assert second.reservations.vertex_reserved(first.agent.goal, 10_000)


@pytest.mark.parametrize("own,peer,kind", [
    (((0, 0), (1, 0), (2, 0)), ((1, 1), (1, 0), (1, 1)), "vertex"),
    (((0, 0), (1, 0)), ((1, 0), (0, 0)), "edge"),
    (((0, 0), (1, 0)), ((2, 1), (2, 0), (1, 0)), "vertex"),
])
def test_local_detection_vertex_swap_and_permanent_goal(own, peer, kind):
    grid = Grid(3, 2, frozenset(), (0, 0), (2, 1))
    c = RobotController(grid, Agent(1, own[0], own[-1]), (2,))
    c.path = own
    c.receive(Message("PATH_PROPOSAL", 2, 1, 0, 1, peer, (-2, -2, 2)))
    c.consume_inbox()
    assert kind in {event.kind for event in c.detect_local_conflicts()}


def test_time_specific_spatial_overlap_is_safe():
    grid = Grid(3, 2, frozenset(), (0, 0), (2, 1))
    c = RobotController(grid, Agent(1, (0, 0), (2, 0)), (2,))
    c.path = ((0, 0), (1, 0), (2, 0))
    c.receive(Message("PATH_PROPOSAL", 2, 1, 0, 1, ((1, 1), (1, 1), (1, 0)), (-1, -2, 2)))
    c.consume_inbox()
    assert not c.detect_local_conflicts()


def test_three_agent_negotiation_handles_multiway_vertex():
    grid = Grid(5, 5, frozenset(), (0, 2), (4, 2))
    agents = (Agent(1, (0, 2), (4, 2)), Agent(2, (2, 0), (2, 4)), Agent(3, (4, 2), (0, 2)))
    result = decentralized_plan(grid, agents)
    assert any(len(c.agent_ids) == 3 for c in result.initial_conflicts)
    assert result.found and not detect_collisions(result.paths)
    assert result.replans >= 2


def test_deadlocked_corridor_fails_without_executable_old_proposals():
    grid = Grid(3, 1, frozenset(), (0, 0), (2, 0))
    agents = (Agent(1, (0, 0), (2, 0)), Agent(2, (2, 0), (0, 0)))
    result = decentralized_plan(grid, agents)
    assert not result.found and not result.paths
    assert result.failure_reason.startswith("local_reservation_blocked")
    assert result.sum_of_costs is result.makespan is result.wait_actions is None
    assert result.final_conflicts  # observer of rejected proposals, not executed trajectories


def test_round_and_cumulative_search_budgets_are_real():
    grid, agents = conflict_demo()
    result = decentralized_plan(grid, agents, max_rounds=1)
    assert not result.found and result.rounds == 1 and result.failure_reason == "negotiation_round_limit"
    result = decentralized_plan(grid, agents, max_expansions=1)
    assert not result.found and result.failure_reason == "local_search_limit"
    assert result.expanded_states <= len(agents)
    result = decentralized_plan(grid, agents, max_horizon=0)
    assert not result.found and result.failure_reason.startswith("local_reservation_blocked")


def test_unreachable_initial_path_is_explicit_failure():
    grid = Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0))
    result = decentralized_plan(grid, (Agent(1, (0, 0), (2, 0)),))
    assert not result.found and result.failure_reason == "local_path_unreachable"


def test_repeated_conflicting_replan_is_rejected():
    grid, agents = conflict_demo()
    planner = DecentralizedPlanner(grid, agents)
    while planner.rounds == 0:
        planner.step()
    second = planner.controllers[2]
    second._seen_paths.add(((2, 0), (2, 1), (2, 1), (2, 2), (2, 3), (2, 4)))
    result = finish(planner)
    assert not result.found and result.failure_reason == "no_progress:repeated_proposal"


def test_reproduction_includes_protocol_decisions_but_excludes_runtime():
    grid, agents = promotion_demo()
    a = decentralized_plan(grid, agents)
    b = decentralized_plan(grid, agents)
    assert replace(a, elapsed_ms=0) == replace(b, elapsed_ms=0)
    assert len(a.history) <= 256


def test_final_observer_rejects_false_local_agreements_without_repair():
    grid, agents = conflict_demo()
    planner = DecentralizedPlanner(grid, agents)
    while planner.rounds == 0:
        planner.step()
    initial = {i: c.path for i, c in planner.controllers.items()}
    for c in planner.controllers.values():
        c.agreed = True  # Simulate a faulty local claim; observer is an independent safeguard.
    planner._finish(None)
    assert not planner.result.found and planner.result.failure_reason == "final_verification_failed"
    assert not planner.result.paths and planner.result.final_conflicts
    assert {i: c.path for i, c in planner.controllers.items()} == initial
