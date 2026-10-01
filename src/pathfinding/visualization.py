import tkinter as tk
from tkinter import messagebox, ttk

from .astar import AStarSearch
from .cooperative import CooperativePlanner
from .demo_state import DemoState
from .demos import conflict_demo
from .grid import Cell, Grid
from .heuristics import HEURISTICS
from .multi_agent import IndependentPlanner, Simulation
from .multi_agent_view import AGENT_COLORS, collision_message, draw_agents

COLORS = {
    "Free": "#ffffff", "Obstacle": "#28323c", "Start": "#239b56",
    "Goal": "#cb4335", "Frontier": "#85c1e9", "Explored": "#d5d8dc",
    "Current": "#f5b041", "Path": "#af7ac5",
}


class PathfindingApp:
    """One grid view for single-agent search and independent multi-agent execution."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Robot Pathfinding: Single / Multi-Agent")
        root.geometry("1100x850")
        root.minsize(1000, 700)
        self.state = DemoState()
        self.multi_mode = tk.BooleanVar(value=False)
        self.mode_label = tk.StringVar(value="Multi-Agent Mode: OFF")
        self.agent_count = tk.StringVar(value="4")
        self.algorithm = tk.StringVar(value="Independent A*")
        self.pending_step = False
        self.draw_geometry = (1.0, 0.0, 0.0)
        self.width = tk.StringVar(value="20")
        self.height = tk.StringVar(value="20")
        self.probability = tk.StringVar(value="0.25")
        self.seed = tk.StringVar(value="42")
        self.heuristic = tk.StringVar(value="Manhattan")
        self.delay = tk.DoubleVar(value=50)
        self.stats = tk.StringVar()
        self.running = False
        self.timer: str | None = None
        self.search: AStarSearch | None = None
        self.grid: Grid | None = None
        self.frontier: set[Cell] = set()
        self.explored: set[Cell] = set()
        self.path: set[Cell] = set()
        self.current: Cell | None = None
        self.rectangles: dict[Cell, int] = {}
        self.status = "Ready"

        modes = ttk.Frame(root, padding=(8, 8, 8, 0))
        modes.pack(fill="x")
        ttk.Checkbutton(modes, textvariable=self.mode_label, variable=self.multi_mode,
                        command=self.switch_mode).pack(side="left", padx=(0, 20))
        ttk.Label(modes, text="Agents (2–8)").pack(side="left", padx=5)
        self.agent_input = ttk.Spinbox(modes, from_=2, to=8, textvariable=self.agent_count, width=4, state="disabled")
        self.agent_input.pack(side="left")
        self.randomize_button = ttk.Button(modes, text="Randomize agents", command=self.randomize_agents, state="disabled")
        self.randomize_button.pack(side="left", padx=8)
        ttk.Label(modes, text="Algorithm").pack(side="left", padx=5)
        self.algorithm_selector = ttk.Combobox(modes, textvariable=self.algorithm, values=["Independent A*", "CG-ST-A*"],
                                               width=18, state="disabled")
        self.algorithm_selector.pack(side="left")
        self.algorithm_selector.bind("<<ComboboxSelected>>", lambda event: self.switch_algorithm())
        ttk.Button(modes, text="Load Conflict Demo", command=self.load_conflict_demo).pack(side="left", padx=8)

        inputs = ttk.Frame(root, padding=8)
        inputs.pack(fill="x")
        for column, (label, variable) in enumerate([
            ("Width", self.width), ("Height", self.height),
            ("Obstacle probability", self.probability), ("Seed", self.seed),
        ]):
            ttk.Label(inputs, text=label).grid(row=0, column=column * 2, padx=(0, 5))
            ttk.Entry(inputs, textvariable=variable, width=9).grid(row=0, column=column * 2 + 1, padx=(0, 12))
        ttk.Button(inputs, text="Generate", command=self.generate).grid(row=0, column=8)
        ttk.Button(inputs, text="Next map", command=lambda: self.generate(next_seed=True)).grid(row=0, column=9, padx=5)

        controls = ttk.Frame(root, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Heuristic").pack(side="left", padx=(0, 6))
        selector = ttk.Combobox(controls, state="readonly", textvariable=self.heuristic, values=list(HEURISTICS), width=18)
        selector.pack(side="left", padx=(0, 12))
        selector.bind("<<ComboboxSelected>>", lambda event: self.reset())
        self.heuristic_selector = selector
        for label, command in [("Run", self.run), ("Pause", self.pause), ("Step", self.step), ("Reset", self.reset)]:
            ttk.Button(controls, text=label, command=command, width=7).pack(side="left", padx=3)
        ttk.Label(controls, text="Delay (ms): fast").pack(side="left", padx=(12, 3))
        ttk.Scale(controls, from_=10, to=1000, variable=self.delay, length=130).pack(side="left")
        ttk.Label(controls, text="slow").pack(side="left", padx=3)

        self.canvas = tk.Canvas(root, background="#f4f4f4", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8)
        self.canvas.bind("<Configure>", lambda event: self.draw_grid())
        self.legend = ttk.Frame(root, padding=8)
        self.legend.pack(fill="x")
        ttk.Label(root, textvariable=self.stats, padding=(8, 0, 8, 8), justify="left", wraplength=1050).pack(fill="x")
        self.events_frame = ttk.LabelFrame(root, text="Collision history (simulation continues after conflicts)", padding=4)
        self.events = tk.Listbox(self.events_frame, height=4, foreground="#ae0000")
        self.events.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.events_frame, command=self.events.yview)
        scrollbar.pack(side="right", fill="y")
        self.events.configure(yscrollcommand=scrollbar.set)
        self.refresh_mode_controls()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.generate()

    @property
    def grid(self) -> Grid | None:
        return self.state.grid

    @grid.setter
    def grid(self, value: Grid | None) -> None:
        self.state.grid = value

    @property
    def search(self) -> AStarSearch | None:
        return self.state.search

    @search.setter
    def search(self, value: AStarSearch | None) -> None:
        self.state.search = value

    def _agent_count(self) -> int:
        count = int(self.agent_count.get())
        if not 2 <= count <= 8:
            raise ValueError("Choose between 2 and 8 agents for the interactive view.")
        return count

    def switch_mode(self) -> None:
        self.pause()
        try:
            count = self._agent_count() if self.multi_mode.get() else 0
            seed = int(self.seed.get()) if self.multi_mode.get() else 0
            self.state.set_mode(self.multi_mode.get(), count, seed)
        except ValueError as error:
            self.multi_mode.set(self.state.multi_agent)
            messagebox.showerror("Cannot switch mode", str(error), parent=self.root)
        self.refresh_mode_controls()
        self.reset()

    def refresh_mode_controls(self) -> None:
        enabled = self.state.multi_agent
        self.mode_label.set(f"Multi-Agent Mode: {'ON' if enabled else 'OFF'}")
        self.agent_input.configure(state="normal" if enabled else "disabled")
        self.randomize_button.configure(state="normal" if enabled else "disabled")
        self.algorithm_selector.configure(state="readonly" if enabled else "disabled")
        self.heuristic_selector.configure(state="disabled" if enabled and self.state.algorithm == "CG-ST-A*" else "readonly")
        if enabled:
            self.events_frame.pack(fill="x", padx=8, pady=(0, 8))
        else:
            self.events_frame.pack_forget()
        for child in self.legend.winfo_children():
            child.destroy()
        entries = [(f"Agent {agent.agent_id}", AGENT_COLORS[index]) for index, agent in enumerate(self.state.agents)] if enabled else list(COLORS.items())
        for column, (name, color) in enumerate(entries):
            tk.Label(self.legend, background=color, width=2, relief="solid", borderwidth=1).grid(row=0, column=column * 2, padx=(0, 4))
            ttk.Label(self.legend, text=name).grid(row=0, column=column * 2 + 1, padx=(0, 13))
        if enabled:
            ttk.Label(self.legend, text="S = start, G = goal; numbered circles = robots; offset lines = planned paths; red cells / dashed edges = collisions").grid(
                row=1, column=0, columnspan=16, sticky="w", pady=(5, 0))

    def randomize_agents(self) -> None:
        self.pause()
        try:
            seed = int(self.seed.get()) + 1
            self.state.randomize_agents(self._agent_count(), seed)
        except ValueError as error:
            messagebox.showerror("Cannot place agents", str(error), parent=self.root)
            return
        self.seed.set(str(seed))
        self.refresh_mode_controls()
        self.reset()

    def switch_algorithm(self) -> None:
        self.pause()
        self.state.set_algorithm(self.algorithm.get())
        self.refresh_mode_controls()
        self.reset()

    def load_conflict_demo(self) -> None:
        self.pause()
        grid, agents = conflict_demo()
        self.state.reset()
        self.state.grid, self.state.agents = grid, agents
        self.state.multi_agent = True
        self.multi_mode.set(True)
        self.width.set(str(grid.width))
        self.height.set(str(grid.height))
        self.agent_count.set(str(len(agents)))
        self.refresh_mode_controls()
        self.reset()

    def generate(self, next_seed: bool = False) -> None:
        self.pause()
        try:
            width, height = int(self.width.get()), int(self.height.get())
            if not (1 <= width <= 100 and 1 <= height <= 100):
                raise ValueError("UI dimensions must be between 1 and 100.")
            seed = int(self.seed.get()) + int(next_seed)
            grid = Grid.random(width, height, float(self.probability.get()), seed)
            count = self._agent_count() if self.state.multi_agent else 0
            self.state.set_grid(grid, count, seed)
        except ValueError as error:
            messagebox.showerror("Invalid grid settings", str(error), parent=self.root)
            return
        self.seed.set(str(seed))
        self.refresh_mode_controls()
        self.reset()

    def _cancel_timer(self) -> None:
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None

    def reset(self) -> None:
        self._cancel_timer()
        self.running = False
        self.pending_step = False
        self.state.reset()
        self.frontier.clear()
        self.explored.clear()
        self.path.clear()
        self.current = None
        self.status = "Ready"
        self.events.delete(0, "end")
        self.draw_grid()
        self.update_stats()

    def run(self) -> None:
        if self.running:
            return
        self._cancel_timer()
        self.pending_step = False
        if ((self.search is not None and self.search.result is not None)
                or (self.state.simulation is not None and self.state.simulation.done)
                or (self.state.cooperative is not None and self.state.cooperative.done and self.state.simulation is None)):
            self.reset()
        self.running = True
        self.status = "Running"
        self._tick()

    def pause(self) -> None:
        self._cancel_timer()
        self.running = False
        self.pending_step = False
        if ((self.search is not None and self.search.result is None)
                or (self.state.cooperative is not None and not self.state.cooperative.done)
                or (self.state.simulation is not None and not self.state.simulation.done)
                or (self.state.multi_agent and self.state.planner is not None
                    and (self.state.simulation is None or not self.state.simulation.done))):
            self.status = "Paused"
        self.update_stats()

    def step(self) -> None:
        self.pause()
        if self.state.multi_agent:
            self.pending_step = True
            self._tick()
        else:
            self._advance()

    def _tick(self) -> None:
        self.timer = None
        if not self.running and not self.pending_step:
            return
        self._advance()
        if self.running or self.pending_step:
            planning = self.state.multi_agent and self.state.simulation is None
            self.timer = self.root.after(1 if planning else max(1, int(self.delay.get())), self._tick)

    def _advance(self) -> None:
        if self.state.multi_agent:
            self._advance_multi()
            return
        if self.grid is None:
            return
        if self.search is None:
            self.search = AStarSearch(self.grid, HEURISTICS[self.heuristic.get()])
            self.frontier.add(self.grid.start)
        if self.search.result is not None:
            return
        previous = self.current
        event = self.search.step()
        self.current = event.current
        if event.current is not None:
            self.frontier.discard(event.current)
            self.explored.add(event.current)
        self.frontier.update(event.opened)
        self.explored.difference_update(event.opened)
        dirty = set(event.opened)
        if previous is not None:
            dirty.add(previous)
        if self.current is not None:
            dirty.add(self.current)
        if self.search.result is not None:
            self.running = False
            self.path = set(self.search.result.path)
            dirty.update(self.path)
            self.status = "Found optimal path" if self.search.result.found else "Unreachable"
        else:
            self.status = "Running" if self.running else "Paused"
        self.paint(dirty)
        self.update_stats()

    def _advance_multi(self) -> None:
        if self.state.algorithm == "CG-ST-A*":
            self._advance_cooperative()
            return
        if self.grid is None:
            return
        if self.state.planner is None:
            self.state.planner = IndependentPlanner(self.grid, self.state.agents, HEURISTICS[self.heuristic.get()])
        planner = self.state.planner
        if not planner.done:
            for _ in range(100):
                planner.step()
                if planner.done:
                    break
            if not planner.done:
                self.status = f"Planning agent {len(planner.plans) + 1}/{len(planner.agents)}"
                self.update_stats()
                return
        if self.state.simulation is None:
            if any(not plan.result.found for plan in planner.plans):
                self.status = "Planning failed: an agent's goal is unreachable"
                self.running = self.pending_step = False
                self.update_stats()
                return
            self.state.simulation = Simulation({plan.agent.agent_id: plan.path for plan in planner.plans})
        self._execute_simulation()

    def _advance_cooperative(self) -> None:
        if self.grid is None:
            return
        if self.state.cooperative is None:
            self.state.cooperative = CooperativePlanner(self.grid, self.state.agents)
        planner = self.state.cooperative
        for _ in range(100):
            if planner.done:
                break
            planner.step()
        if not planner.done:
            self.status = planner.phase
            self.update_stats()
            return
        if not planner.result.found:
            self.status = f"Planning failed: {planner.result.failure_reason}"
            self.running = self.pending_step = False
            self.update_stats()
            return
        if self.state.simulation is None:
            self.state.simulation = Simulation(planner.result.paths)
        self._execute_simulation()

    def _execute_simulation(self) -> None:
        simulation = self.state.simulation
        simulation.step()
        self.pending_step = False
        if simulation.done:
            self.running = False
            self.status = "Execution complete" if self.state.algorithm == "CG-ST-A*" else "Execution complete (collisions are not resolved)"
        else:
            self.status = "Running" if self.running else "Paused"
        while self.events.size() < len(simulation.collisions):
            self.events.insert("end", collision_message(simulation.collisions[self.events.size()]))
        self.events.yview_moveto(1)
        self.draw_multi()
        self.update_stats()

    def draw_multi(self) -> None:
        simulation = self.state.simulation
        paths = simulation.paths if simulation else {}
        positions = simulation.positions if simulation else {agent.agent_id: agent.start for agent in self.state.agents}
        collisions = simulation.current_collisions if simulation else ()
        draw_agents(self.canvas, self.state.agents, paths, positions, collisions, *self.draw_geometry)

    def color(self, cell: Cell) -> str:
        if self.grid is None:
            return COLORS["Free"]
        if self.state.multi_agent:
            return COLORS["Obstacle"] if cell in self.grid.obstacles else COLORS["Free"]
        if cell == self.grid.start:
            return COLORS["Start"]
        if cell == self.grid.goal:
            return COLORS["Goal"]
        if cell in self.grid.obstacles:
            return COLORS["Obstacle"]
        if cell in self.path:
            return COLORS["Path"]
        if cell == self.current:
            return COLORS["Current"]
        if cell in self.frontier:
            return COLORS["Frontier"]
        if cell in self.explored:
            return COLORS["Explored"]
        return COLORS["Free"]

    def paint(self, cells: set[Cell]) -> None:
        for cell in cells:
            if cell in self.rectangles:
                self.canvas.itemconfigure(self.rectangles[cell], fill=self.color(cell))

    def draw_grid(self) -> None:
        self.canvas.delete("all")
        self.rectangles.clear()
        if self.grid is None:
            return
        size = min(max(1, self.canvas.winfo_width() - 16) / self.grid.width,
                   max(1, self.canvas.winfo_height() - 16) / self.grid.height)
        left = (self.canvas.winfo_width() - size * self.grid.width) / 2
        top = (self.canvas.winfo_height() - size * self.grid.height) / 2
        self.draw_geometry = (size, left, top)
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                x0, y0 = left + x * size, top + y * size
                self.rectangles[(x, y)] = self.canvas.create_rectangle(
                    x0, y0, x0 + size, y0 + size, fill=self.color((x, y)), outline="#aab0b6",
                )
        if self.state.multi_agent:
            self.draw_multi()
        elif size >= 12:
            for cell, label in [(self.grid.start, "S"), (self.grid.goal, "G")]:
                self.canvas.create_text(left + (cell[0] + 0.5) * size, top + (cell[1] + 0.5) * size,
                                        text=label, fill="white", font=("Arial", max(8, min(16, int(size * 0.5))), "bold"))

    def update_stats(self) -> None:
        if self.state.multi_agent:
            self.update_multi_stats()
            return
        search = self.search
        cost = "—"
        if search is not None and search.result is not None:
            cost = str(search.result.path_cost) if search.result.found else "unreachable"
        self.stats.set(
            f"Heuristic: {self.heuristic.get()}    Status: {self.status}    Path cost: {cost}\n"
            f"Expanded: {search.expanded_nodes if search else 0}    "
            f"Generated: {search.generated_nodes if search else 0}    "
            f"Frontier: {search.frontier_size if search else 0} / peak {search.peak_frontier_size if search else 0}    "
            f"Search time: {search.elapsed_ms if search else 0:.3f} ms (excludes animation)"
        )

    def update_multi_stats(self) -> None:
        if self.state.algorithm == "CG-ST-A*":
            self.update_cooperative_stats()
            return
        simulation = self.state.simulation
        if simulation is None:
            self.stats.set(f"Independent A* | {self.heuristic.get()} | {self.status} | t=0 | Agents: {len(self.state.agents)}\n"
                           "Collisions: 0 | Paths and costs appear after planning. Robots do not coordinate.")
            return
        metrics = simulation.metrics
        costs = ", ".join(f"Agent {agent_id}: {cost}" for agent_id, cost in metrics.individual_costs.items())
        self.stats.set(
            f"Independent A* | {self.heuristic.get()} | {self.status} | t={simulation.timestep}\n"
            f"Collisions so far: {metrics.total_collisions} (vertex {metrics.vertex_collisions}, edge {metrics.edge_collisions}) | "
            f"Sum of costs: {metrics.sum_of_costs} | Makespan: {metrics.makespan}\nIndividual path costs — {costs}"
        )

    def update_cooperative_stats(self) -> None:
        planner = self.state.cooperative
        result = planner.result if planner else None
        simulation = self.state.simulation
        timestep = simulation.timestep if simulation else 0
        first = f"CG-ST-A* | Manhattan | {self.status} | t={timestep}"
        if result is None:
            order = planner.order if planner else ()
            priority = ' → '.join(map(str, order)) if order else 'pending independent analysis'
            self.stats.set(f"{first}\nPriority: {priority}")
            return
        priority = ' → '.join(map(str, result.final_priority_order))
        details = (f"Priority: {priority} | Attempts: {result.planning_attempts} | Promotions: {result.priority_promotions} | "
                   f"Planning time: {result.elapsed_ms:.3f} ms")
        if result.found:
            self.stats.set(f"{first}\n{details}\nSum of costs: {result.sum_of_costs} | Makespan: {result.makespan} | "
                           f"WAIT actions: {result.wait_actions} | Collisions: {simulation.metrics.total_collisions if simulation else 0}")
        else:
            self.stats.set(f"{first}\n{details}\nNo complete solution; search limit reached: {result.search_limit_reached}")

    def close(self) -> None:
        self._cancel_timer()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PathfindingApp(root)
    root.mainloop()
