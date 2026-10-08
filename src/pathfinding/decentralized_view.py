"""Optional negotiation panel and bounded round playback for the existing UI."""
from collections import deque
from time import perf_counter_ns
import tkinter as tk
from tkinter import ttk

from .decentralized import DCN, DecentralizedPlanner
from .demos import conflict_demo
from .multi_agent import Simulation


class DecentralizedView:
    def __init__(self, app):
        self.app = app
        self.frame = ttk.Frame(app.panes, padding=(8, 0), width=330)
        self.frame.pack_propagate(False)
        self.phase = tk.StringVar(value="Negotiation: ready")
        self.links = tk.BooleanVar(value=False)
        self.lines = deque(maxlen=300)
        ttk.Label(self.frame, text="Decentralized Negotiation (Experimental)", wraplength=310,
                  font=("Arial", 11, "bold")).pack(anchor="w")
        ttk.Label(self.frame, text="Synchronous reliable peer messages.\nEach agent owns its path and reservation table.\nPair decision: (-initial B, -initial L, ID).",
                  wraplength=310).pack(anchor="w", pady=5)
        ttk.Label(self.frame, textvariable=self.phase, wraplength=310).pack(anchor="w")
        ttk.Button(self.frame, text="Load Decentralized Demo", command=self.load_demo).pack(fill="x", pady=4)
        ttk.Checkbutton(self.frame, text="Show last-round communication links", variable=self.links,
                        command=app.draw_multi).pack(anchor="w")
        local_frame = ttk.Frame(self.frame)
        local_frame.pack(fill="x")
        self.local = ttk.Treeview(local_frame, columns=("id", "version", "replans", "status"), show="headings", height=4)
        for name, label, width in (("id", "Agent", 40), ("version", "Version", 50), ("replans", "Replans", 50), ("status", "Local state", 170)):
            self.local.heading(name, text=label)
            self.local.column(name, width=width, minwidth=width, stretch=name == "status")
        self.local.pack(side="left", fill="x", expand=True)
        local_scroll = ttk.Scrollbar(local_frame, command=self.local.yview)
        local_scroll.pack(side="right", fill="y")
        self.local.configure(yscrollcommand=local_scroll.set)
        ttk.Label(self.frame, text="Conflict highlights: predictions at round start.", wraplength=310).pack(anchor="w")
        ttk.Label(self.frame, text="Real protocol events (last 300 lines)").pack(anchor="w", pady=4)
        log = ttk.Frame(self.frame)
        log.pack(fill="both", expand=True)
        self.log = tk.Text(log, state="disabled", width=36, height=8, wrap="word", font=("Arial", 9))
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

    def reset(self):
        self.lines.clear()
        self.refresh()

    def load_demo(self):
        app = self.app
        app.pause()
        app.state.grid, app.state.agents = conflict_demo()
        app.state.multi_agent = True
        app.state.set_algorithm(DCN)
        app.algorithm.set(DCN)
        app.multi_mode.set(True)
        app.width.set("5")
        app.height.set("5")
        app.agent_count.set("2")
        app.refresh_mode_controls()
        app.reset()

    def refresh(self):
        planner = self.app.state.decentralized
        self.local.delete(*self.local.get_children())
        if planner:
            for c in planner.controllers.values():
                self.local.insert("", "end", values=(c.agent_id, c.version, c.replans, c.status))
            self.phase.set(f"Round {planner.round}; completed {planner.rounds}; "
                           f"messages {sum(c.messages_sent for c in planner.controllers.values())} sent / "
                           f"{sum(c.messages_delivered for c in planner.controllers.values())} delivered")
            if planner.result:
                self.phase.set(self.phase.get() + ("\nAll agents agreed; verified safe." if planner.result.found
                                                   else "\nFailed: " + planner.result.failure_reason))
        else:
            self.phase.set("Negotiation: ready; Step completes one logical round.")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.insert("end", "\n".join(self.lines))
        self.log.configure(state="disabled")
        self.log.see("end")

    def advance(self):
        app = self.app
        if app.grid is None:
            return
        if app.state.simulation is not None:
            app._execute_simulation()
            self.lines.append(f"Execution t={app.state.simulation.timestep}; collisions={len(app.state.simulation.current_collisions)}")
            self.refresh()
            return
        if app.state.decentralized is None:
            app.state.decentralized = DecentralizedPlanner(app.grid, app.state.agents)
        planner = app.state.decentralized
        began = perf_counter_ns()
        for _ in range(200):
            complete = planner.step()
            if complete or perf_counter_ns() - began >= 8_000_000:
                break
        for event in planner.drain_events():
            self.lines.append(f"R{event.round} A{event.agent_id} [{event.kind}] {event.detail}")
        if complete:
            app.pending_step = False
        if planner.done:
            app.pending_step = False
            if planner.result.found:
                app.state.simulation = Simulation(planner.result.paths)
                app.status = "Decentralized agreement verified; execution ready at t=0"
                self.lines.append("Verified paths ready at t=0; next Step executes one timestep.")
            else:
                app.running = False
                app.status = "Negotiation failed: " + planner.result.failure_reason
        else:
            app.status = f"Decentralized negotiation round {planner.round}"
        self.refresh()
        app.draw_multi()
        self.update_stats()

    def update_stats(self):
        app = self.app
        planner = app.state.decentralized
        result = planner.result if planner else None
        first = f"DCN-ST-A* (Experimental) | Manhattan | {app.status} | t={app.state.simulation.timestep if app.state.simulation else 0}"
        if result:
            quality = f"SOC: {result.sum_of_costs} | Makespan: {result.makespan} | WAIT actions: {result.wait_actions}" if result.found else "Solution quality: undefined (failed negotiation)"
            app.stats.set(f"{first}\n{quality} | Rounds: {result.rounds} | Messages: {result.messages_sent} sent / {result.messages_delivered} delivered\n"
                          f"Local replans: {result.replans} | Expanded: {result.expanded_states} | Computation: {result.elapsed_ms:.3f} ms (no network latency)")
        else:
            app.stats.set(first + "\nPaths are proposals until all local agreements and final collision verification pass.")

    def draw_links(self, positions):
        canvas = self.app.canvas
        canvas.delete("communication")
        planner = self.app.state.decentralized
        if not self.links.get() or not planner or self.app.state.simulation:
            return
        size, left, top = self.app.draw_geometry
        pairs = {(min(m.sender, m.receiver), max(m.sender, m.receiver)) for m in planner.last_messages}
        for first, second in pairs:
            a, b = positions[first], positions[second]
            canvas.create_line(left+(a[0]+0.5)*size, top+(a[1]+0.5)*size,
                               left+(b[0]+0.5)*size, top+(b[1]+0.5)*size,
                               fill="#517c93", dash=(3, 5), arrow="both", tags="communication")
