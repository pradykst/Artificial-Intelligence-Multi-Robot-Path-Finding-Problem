"""DCN-ST-A*: independent controllers with message-only peer knowledge.

The scheduler provides reliable synchronous delivery and round barriers. It
never chooses priorities or builds a shared reservation table. Its final
collision observer can reject a plan, but cannot repair one.
"""
from collections import deque
from dataclasses import dataclass, replace
from time import perf_counter_ns
from types import MappingProxyType
from typing import Mapping

from .agents import Agent
from .astar import AStarSearch
from .collisions import Collision, detect_collisions
from .decentralized_messages import Message, PriorityKey, pairwise_winner
from .grid import Cell, Grid
from .multi_agent import compute_metrics
from .reservations import ReservationTable
from .space_time import SpaceTimeSearch

DCN = "Decentralized Negotiation (Experimental)"


@dataclass(frozen=True)
class PeerProposal:
    version: int
    round: int
    path: tuple[Cell, ...]
    priority: PriorityKey


@dataclass(frozen=True)
class ProtocolEvent:
    round: int
    agent_id: int | None
    kind: str
    detail: str


@dataclass(frozen=True)
class DecentralizedResult:
    found: bool
    paths: Mapping[int, tuple[Cell, ...]]
    rounds: int
    messages_sent: int
    messages_delivered: int
    replans: int
    replans_by_agent: Mapping[int, int]
    initial_conflicts: tuple[Collision, ...]
    final_conflicts: tuple[Collision, ...]
    sum_of_costs: int | None
    makespan: int | None
    wait_actions: int | None
    expanded_states: int
    generated_states: int
    elapsed_ms: float
    failure_reason: str | None
    per_agent_status: Mapping[int, str]
    history: tuple[ProtocolEvent, ...]


class RobotController:
    """Owns one robot; peer endpoints and trajectories arrive only in messages."""

    def __init__(self, grid: Grid, agent: Agent, peer_ids: tuple[int, ...], *,
                 max_expansions: int = 100_000, max_horizon: int | None = None,
                 history_capacity: int = 256) -> None:
        self.grid, self.agent, self.peer_ids = grid, agent, peer_ids
        self.max_expansions, self.max_horizon = max_expansions, max_horizon
        self.path: tuple[Cell, ...] = ()
        self.initial_path: tuple[Cell, ...] = ()
        self.version = 0
        self.priority: PriorityKey | None = None
        self.inbox: list[Message] = []
        self.outbox: list[Message] = []
        self.known_peers: dict[int, PeerProposal] = {}
        self.peer_agreements: dict[int, tuple] = {}
        self.local_conflicts: tuple[Collision, ...] = ()
        self.reservations = ReservationTable()
        self.search = AStarSearch(grid, start=agent.start, goal=agent.goal)
        self.status = "initial_search"
        self.failure_reason: str | None = None
        self.round = 0
        self.round_complete = False
        self.agreed = False
        self.expanded_states = self.generated_states = self.replans = 0
        self.messages_sent = self.messages_delivered = self.stale_proposals = 0
        self.history: deque[ProtocolEvent] = deque(maxlen=history_capacity)
        self._seen_paths: set[tuple[Cell, ...]] = set()
        self._horizon = self._cap = 0

    @property
    def agent_id(self) -> int:
        return self.agent.agent_id

    def _event(self, kind: str, detail: str) -> None:
        self.history.append(ProtocolEvent(self.round, self.agent_id, kind, detail))

    def _broadcast(self, kind: str, **fields) -> None:
        for peer in self.peer_ids:
            self.outbox.append(Message(kind, self.agent_id, peer, self.round, self.version, **fields))
            self.messages_sent += 1
        self._event(kind, f"v{self.version}; {len(self.peer_ids)} receiver(s)" +
                    (f"; winner A{fields['winner']}" if 'winner' in fields else "") +
                    (f"; {fields['reason']}" if 'reason' in fields else ""))

    def receive(self, message: Message) -> None:
        if message.receiver != self.agent_id or message.sender not in self.peer_ids:
            raise ValueError("Message is not addressed to this controller by a known peer.")
        self.inbox.append(message)
        self.messages_delivered += 1

    def consume_inbox(self) -> None:
        for message in self.inbox:
            if message.kind in ("PATH_PROPOSAL", "UPDATED_PATH"):
                previous = self.known_peers.get(message.sender)
                # Duplicate versions cannot replace immutable trajectories, even with a later round.
                if previous and (message.version <= previous.version or message.round < previous.round):
                    self.stale_proposals += 1
                    continue
                if not message.path or message.priority is None or message.priority[2] != message.sender:
                    raise ValueError("A proposal needs a path and its sender's advertised priority.")
                if previous and message.priority != previous.priority:
                    raise ValueError("Advertised priority must remain fixed during negotiation.")
                self.known_peers[message.sender] = PeerProposal(message.version, message.round,
                                                               message.path, message.priority)
                self.peer_agreements.pop(message.sender, None)
                self.agreed = False
                self._event("RECEIVED_PATH", f"A{message.sender} v{message.version}")
            elif message.kind == "AGREEMENT":
                self.peer_agreements[message.sender] = message.versions
                self._event("RECEIVED_AGREEMENT", f"A{message.sender}")
            elif message.kind == "PLANNING_FAILURE":
                self._fail(f"peer_failure:A{message.sender}:{message.reason}", announce=False)
            elif message.kind in ("CONFLICT_NOTICE", "YIELD_DECISION"):
                self._event("RECEIVED_" + message.kind,
                            f"A{message.sender}; versions {message.versions}" +
                            (f"; winner A{message.winner}" if message.winner is not None else ""))
        self.inbox.clear()

    def version_vector(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted([(self.agent_id, self.version),
                             *((peer, proposal.version) for peer, proposal in self.known_peers.items())]))

    def detect_local_conflicts(self) -> tuple[Collision, ...]:
        if not self.path:
            return ()
        return tuple(event for event in detect_collisions({self.agent_id: self.path,
                     **{peer: proposal.path for peer, proposal in self.known_peers.items()}})
                     if self.agent_id in event.agent_ids)

    def build_local_reservations(self) -> ReservationTable:
        """Protect all communicated superior paths, including currently nonconflicting ones."""
        table = ReservationTable()
        for proposal in self.known_peers.values():
            if pairwise_winner(self.priority, proposal.priority) != self.agent_id:
                table.reserve_path(proposal.path)
        self.reservations = table
        return table

    def begin_round(self, round_number: int) -> None:
        self.round, self.round_complete = round_number, False
        self.consume_inbox()
        if self.failure_reason:
            self.round_complete = True
            return
        self.local_conflicts = self.detect_local_conflicts()
        if len(self.known_peers) != len(self.peer_ids):
            self.status = "awaiting_proposals"
            self.round_complete = True
            return
        yielding = False
        for peer in self.peer_ids:
            conflicts = tuple(c for c in self.local_conflicts if peer in c.agent_ids)
            if not conflicts:
                continue
            winner = pairwise_winner(self.priority, self.known_peers[peer].priority)
            pair_versions = tuple(sorted(((self.agent_id, self.version),
                                           (peer, self.known_peers[peer].version))))
            # Decisions are independently reproducible from the two immutable advertisements.
            self.outbox.extend((Message("CONFLICT_NOTICE", self.agent_id, peer, self.round,
                                        self.version, conflicts=conflicts, versions=pair_versions),
                                Message("YIELD_DECISION", self.agent_id, peer, self.round,
                                        self.version, winner=winner, versions=pair_versions)))
            self.messages_sent += 2
            self._event("CONFLICT_NOTICE", f"A{peer}; " + ", ".join(
                f"{c.kind} at t={c.timestep}" for c in conflicts[:8]) +
                (f"; {len(conflicts)} events total" if len(conflicts) > 8 else ""))
            self._event("YIELD_DECISION", f"A{peer}; winner A{winner}" +
                        ("; yielding" if winner != self.agent_id else "; keeping proposal"))
            yielding |= winner != self.agent_id
        if yielding:
            self.agreed = False
            self.replans += 1
            self.build_local_reservations()
            free = self.grid.width * self.grid.height - len(self.grid.obstacles)
            largest_cost = max([-self.priority[1], *(-p.priority[1] for p in self.known_peers.values())])
            self._cap = self.max_horizon if self.max_horizon is not None else free + largest_cost
            self._horizon = min(self._cap, len(self.path) - 1)
            self._start_replan()
            return
        self.status = "awaiting_peer_revision" if self.local_conflicts else "local_agreement"
        self.agreed = (not self.local_conflicts and all(
            self.peer_agreements.get(peer) == self.version_vector() for peer in self.peer_ids))
        if not self.local_conflicts:
            self._broadcast("AGREEMENT", versions=self.version_vector())
        self.round_complete = True

    def _start_replan(self) -> None:
        remaining = self.max_expansions - self.expanded_states
        if remaining <= 0:
            self._fail("local_search_limit")
            return
        self.search = SpaceTimeSearch(self.grid, self.agent.start, self.agent.goal,
                                      self.reservations, self._horizon, max_expansions=remaining)
        self.status = "local_replanning"
        self._event("REPLAN", f"horizon {self._horizon}/{self._cap}; own table; budget {remaining}")

    def _fail(self, reason: str, *, announce: bool = True) -> None:
        self.failure_reason, self.status = reason, "failed:" + reason
        self.agreed = False
        self.round_complete = True
        if announce:
            self._broadcast("PLANNING_FAILURE", reason=reason)

    def step_search(self) -> None:
        """One bounded search expansion; UI callbacks can yield between calls."""
        if self.round_complete:
            return
        if self.expanded_states >= self.max_expansions:
            self._fail("local_search_limit")
            return
        initial = isinstance(self.search, AStarSearch)
        expanded_name = "expanded_nodes" if initial else "expanded_states"
        generated_name = "generated_nodes" if initial else "generated_states"
        before_e, before_g = getattr(self.search, expanded_name), getattr(self.search, generated_name)
        self.search.step()
        self.expanded_states += getattr(self.search, expanded_name) - before_e
        self.generated_states += getattr(self.search, generated_name) - before_g
        result = self.search.result
        if result is None:
            return
        if not result.found:
            if initial:
                self._fail("local_path_unreachable")
            elif result.failure_reason == "horizon_exhausted" and self._horizon < self._cap:
                self._horizon = min(self._cap, max(1, self._horizon * 2))
                self._start_replan()
            else:
                self._fail("local_search_limit" if result.failure_reason == "expansion_limit"
                           else "local_reservation_blocked:" + result.failure_reason)
            return
        if not initial and result.path in self._seen_paths:
            self._fail("no_progress:repeated_proposal")
            return
        self.path = tuple(result.path)
        self.version += 1
        self._seen_paths.add(self.path)
        if initial:
            self.initial_path = self.path
            exposure = sum(sum(1 for _ in self.grid.neighbors(cell)) <= 2 for cell in self.path)
            self.priority = (-exposure, -(len(self.path) - 1), self.agent_id)
        self._broadcast("PATH_PROPOSAL" if initial else "UPDATED_PATH", path=self.path, priority=self.priority)
        self.status, self.round_complete = "proposal_published", True


class DecentralizedPlanner:
    """Round-robin computation, synchronous delivery, and observational validation."""

    def __init__(self, grid: Grid, agents: tuple[Agent, ...], *, max_rounds: int = 32,
                 max_expansions: int = 100_000, max_horizon: int | None = None,
                 history_capacity: int = 256) -> None:
        ids = tuple(agent.agent_id for agent in agents)
        if not agents or len(set(ids)) != len(ids):
            raise ValueError("Provide agents with distinct IDs.")
        if max_rounds < 1 or max_expansions < 1 or history_capacity < 1 or (max_horizon is not None and max_horizon < 0):
            raise ValueError("Round/search/history limits must be positive and horizon nonnegative.")
        self.controllers = {agent.agent_id: RobotController(grid, agent, tuple(i for i in ids if i != agent.agent_id),
                            max_expansions=max_expansions, max_horizon=max_horizon,
                            history_capacity=history_capacity) for agent in agents}
        self.max_rounds, self.round, self.rounds = max_rounds, 0, 0
        self.result: DecentralizedResult | None = None
        self.history: deque[ProtocolEvent] = deque(maxlen=history_capacity)
        self.last_messages: tuple[Message, ...] = ()
        self.initial_conflicts: tuple[Collision, ...] = ()
        self._pending: list[Message] = []
        self._cursor = 0
        self._elapsed_ns = 0
        self._start_next_round = False
        self._new_events: deque[ProtocolEvent] = deque(maxlen=history_capacity)

    @property
    def done(self) -> bool:
        return self.result is not None

    def drain_events(self) -> tuple[ProtocolEvent, ...]:
        events = tuple(self._new_events)
        self._new_events.clear()
        return events

    def _collect_events(self) -> None:
        # Transfer, then clear local logs: each event enters the bounded observer log once.
        for controller in self.controllers.values():
            self.history.extend(controller.history)
            self._new_events.extend(controller.history)
            controller.history.clear()

    def step(self) -> bool:
        """Return True after one complete logical round; do at most one search step."""
        if self.done:
            return True
        began = perf_counter_ns()
        if self._start_next_round:
            self.last_messages = tuple(self._pending)
            for message in self._pending:
                self.controllers[message.receiver].receive(message)
            self._pending.clear()
            for controller in self.controllers.values():
                controller.begin_round(self.round)
            self._start_next_round = False
        controllers = tuple(self.controllers.values())
        controller = controllers[self._cursor % len(controllers)]
        self._cursor += 1
        controller.step_search()
        self._collect_events()
        complete = all(c.round_complete for c in controllers)
        if complete:
            self.rounds += 1
            for c in controllers:
                self._pending.extend(c.outbox)
                c.outbox.clear()
            if self.round == 0:
                initial = {c.agent_id: c.initial_path for c in controllers if c.initial_path}
                self.initial_conflicts = detect_collisions(initial)
            failure = next((c.failure_reason for c in controllers if c.failure_reason), None)
            if failure:
                self._finish(failure)
            elif all(c.agreed for c in controllers):
                self._finish(None)
            elif self.rounds >= self.max_rounds:
                self._finish("negotiation_round_limit")
            else:
                self.round += 1
                self._start_next_round = True
        self._elapsed_ns += perf_counter_ns() - began
        if self.result is not None:
            # Final observer cost belongs to computation, UI pauses never do.
            self.result = replace(self.result, elapsed_ms=self._elapsed_ns / 1_000_000)
        return complete

    def _finish(self, failure: str | None) -> None:
        controllers = tuple(self.controllers.values())
        proposed = {c.agent_id: c.path for c in controllers if c.path}
        conflicts = detect_collisions(proposed)
        if failure is None and (len(proposed) != len(controllers) or conflicts or
                                any(c.detect_local_conflicts() or not c.agreed for c in controllers)):
            failure = "final_verification_failed"
        found = failure is None
        metrics = compute_metrics(proposed) if found else None
        self.result = DecentralizedResult(found, MappingProxyType(proposed if found else {}), self.rounds,
            sum(c.messages_sent for c in controllers), sum(c.messages_delivered for c in controllers),
            sum(c.replans for c in controllers), MappingProxyType({c.agent_id: c.replans for c in controllers}),
            self.initial_conflicts, conflicts, metrics.sum_of_costs if metrics else None,
            metrics.makespan if metrics else None, metrics.wait_actions if metrics else None,
            sum(c.expanded_states for c in controllers), sum(c.generated_states for c in controllers),
            self._elapsed_ns / 1_000_000, failure,
            MappingProxyType({c.agent_id: "agreed" if found else (c.failure_reason or "protocol_aborted:" + failure)
                              for c in controllers}), tuple(self.history))


def decentralized_plan(grid: Grid, agents: tuple[Agent, ...], **limits) -> DecentralizedResult:
    planner = DecentralizedPlanner(grid, agents, **limits)
    while not planner.done:
        planner.step()
    return planner.result
