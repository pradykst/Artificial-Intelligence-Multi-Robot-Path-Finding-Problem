import pytest

tk = pytest.importorskip("tkinter")

from pathfinding.grid import Grid
from pathfinding.visualization import COLORS, PathfindingApp


@pytest.fixture
def app():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk requires an available desktop display.")
    root.withdraw()
    application = PathfindingApp(root)
    root.update_idletasks()
    yield application
    application.close()


def test_run_pause_step_reset_and_final_path(app):
    app.grid = Grid(3, 3, frozenset({(1, 0), (1, 1)}), (0, 0), (2, 0))
    app.reset()
    app.run()
    assert app.running and app.timer is not None
    assert app.search.expanded_nodes == 1
    timer = app.timer
    app.pause()
    assert not app.running and app.timer is None
    assert timer not in app.root.tk.call("after", "info")
    assert app.status == "Paused"
    for _ in range(20):
        if app.search.result is not None:
            break
        app.step()
    assert app.search.result.path_cost == 6
    assert app.status == "Found optimal path"
    assert app.path == set(app.search.result.path)
    assert app.color((0, 1)) == COLORS["Path"]
    assert "Path cost: 6" in app.stats.get()
    assert app.canvas.itemcget(app.rectangles[(0, 1)], "fill") == COLORS["Path"]
    app.reset()
    assert app.search is None and app.status == "Ready"
    assert not app.path and not app.frontier and not app.explored


def test_generation_and_unreachable_status(app):
    original = app.grid
    app.generate()
    assert app.grid == original
    app.generate(next_seed=True)
    assert app.grid != original and app.seed.get() == "43"
    app.grid = Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0))
    app.reset()
    app.step()
    assert app.status == "Unreachable"
    assert app.search.result.path_cost is None


def test_scheduled_animation_and_heuristic_change(app):
    app.grid = Grid(2, 1, frozenset(), (0, 0), (1, 0))
    app.reset()
    app.delay.set(1)
    app.run()
    app.root.after(30, app.root.quit)
    app.root.mainloop()
    assert app.search.result is not None and app.search.result.found
    assert app.timer is None and not app.running
    app.heuristic.set("Euclidean")
    app.reset()
    app.step()
    assert app.search.heuristic.__name__ == "euclidean"


def test_mode_switch_cancels_animation_and_clears_state(app):
    original = app.grid
    app.run()
    timer = app.timer
    app.multi_mode.set(True)
    app.switch_mode()
    assert app.state.multi_agent and app.grid is original
    assert app.timer is None and not app.running
    assert timer not in app.root.tk.call("after", "info")
    assert app.search is None and not app.frontier and not app.explored
    assert len(app.state.agents) == 4
    app.step()
    app.pause()
    app.multi_mode.set(False)
    app.switch_mode()
    assert not app.state.multi_agent and app.state.agents == ()
    assert app.state.planner is app.state.simulation is None
    assert app.events.size() == 0 and app.grid is original
    assert app.agent_input.instate(["disabled"])
    app.step()
    assert app.search is not None and app.search.expanded_nodes == 1


def test_multi_step_collisions_reset_and_endpoint_markers(app):
    from pathfinding.agents import Agent

    app.grid = Grid(3, 3, frozenset(), (0, 1), (2, 1))
    app.agent_count.set("2")
    app.multi_mode.set(True)
    app.switch_mode()
    app.state.agents = (Agent(1, (0, 1), (2, 1)), Agent(2, (1, 0), (1, 2)))
    app.reset()
    app.step()
    assert app.state.simulation.timestep == 1
    assert app.state.simulation.metrics.vertex_collisions == 1
    assert app.events.size() == 1 and "t=1" in app.events.get(0)
    labels = {app.canvas.itemcget(item, "text") for item in app.canvas.find_withtag("multi")
              if app.canvas.type(item) == "text"}
    assert {"S1", "G1", "S2", "G2", "1", "2"} <= labels
    app.step()
    assert app.state.simulation.timestep == 2 and app.state.simulation.done
    assert app.state.simulation.metrics.total_collisions == 1
    assert "Sum of costs: 4" in app.stats.get()
    app.reset()
    assert app.state.planner is app.state.simulation is None
    assert app.events.size() == 0


def test_multi_planning_yields_to_event_loop_and_can_be_cancelled(app):
    from pathfinding.agents import Agent

    app.grid = Grid(30, 30, frozenset(), (0, 0), (29, 29))
    app.agent_count.set("2")
    app.multi_mode.set(True)
    app.switch_mode()
    app.state.agents = (Agent(1, (0, 0), (29, 29)), Agent(2, (29, 0), (0, 29)))
    app.reset()
    app.step()
    assert app.state.planner is not None and not app.state.planner.done
    assert app.state.simulation is None and app.timer is not None
    pending = app.timer
    app.run()
    assert pending not in app.root.tk.call("after", "info")
    assert app.running and not app.pending_step
    app.pause()
    assert app.timer is None and not app.pending_step
    app.reset()
    assert app.state.planner is None


def test_multi_step_finishes_planning_then_advances_exactly_once(app):
    from pathfinding.agents import Agent

    app.grid = Grid(15, 15, frozenset(), (0, 0), (14, 14))
    app.agent_count.set("2")
    app.multi_mode.set(True)
    app.switch_mode()
    app.state.agents = (Agent(1, (0, 0), (14, 14)), Agent(2, (14, 0), (0, 14)))
    app.reset()
    app.step()
    assert app.pending_step

    def finish_when_ready():
        if not app.pending_step:
            app.root.quit()
        else:
            app.root.after(5, finish_when_ready)

    timeout = app.root.after(3000, app.root.quit)
    app.root.after(5, finish_when_ready)
    app.root.mainloop()
    app.root.after_cancel(timeout)
    assert app.state.simulation is not None and app.state.simulation.timestep == 1
    assert app.timer is None and not app.running and not app.pending_step


def test_cooperative_algorithm_switch_keeps_demo_and_waits_animate(app):
    app.load_conflict_demo()
    grid, agents = app.grid, app.state.agents
    app.step()
    app.step()
    assert app.state.simulation.metrics.vertex_collisions == 1
    app.algorithm.set("CG-ST-A*")
    app.switch_algorithm()
    assert app.grid is grid and app.state.agents is agents
    assert app.state.simulation is app.state.planner is app.state.cooperative is None
    assert app.events.size() == 0
    for _ in range(100):
        if app.state.simulation is not None:
            break
        app.step()
        while app.pending_step:
            app._cancel_timer()
            app._tick()
    assert app.state.simulation.timestep == 0
    app.step()
    assert app.state.simulation.timestep == 1
    positions = [app.state.simulation.positions]
    planner = app.state.cooperative
    app.run()
    app.pause()
    assert app.state.cooperative is planner
    positions.append(app.state.simulation.positions)
    while not app.state.simulation.done:
        app.step()
        positions.append(app.state.simulation.positions)
    assert app.state.simulation.metrics.total_collisions == 0
    assert any(a[agent.agent_id] == b[agent.agent_id] != agent.goal
               for a, b in zip(positions, positions[1:]) for agent in agents)
    assert "WAIT actions: 1" in app.stats.get() and "Priority:" in app.stats.get()
    app.multi_mode.set(False)
    app.switch_mode()
    assert app.state.cooperative is None and app.state.simulation is None
    assert app.grid is grid


def test_generate_next_map_randomize_and_switches_preserve_seed_semantics(app):
    from pathfinding.agents import random_agents

    app.agent_count.set("2")
    app.multi_mode.set(True)
    app.switch_mode()
    grid, agents = app.grid, app.state.agents
    seed = int(app.seed.get())
    app.generate()
    assert app.grid == grid and app.state.agents == agents
    assert int(app.seed.get()) == seed

    app.randomize_agents()
    assert app.grid == grid and int(app.seed.get()) == seed + 1
    assert app.state.agents == random_agents(grid, 2, seed + 1)
    grid, agents = app.grid, app.state.agents
    app.heuristic.set("Euclidean")
    app.heuristic_selector.event_generate("<<ComboboxSelected>>")
    app.root.update()
    assert app.grid is grid and app.state.agents is agents
    assert int(app.seed.get()) == seed + 1 and app.search is None
    for algorithm in ("CG-ST-A*", "Independent A*"):
        app.algorithm.set(algorithm)
        app.switch_algorithm()
        assert app.grid is grid and app.state.agents is agents
        assert int(app.seed.get()) == seed + 1

    app.generate(next_seed=True)
    assert int(app.seed.get()) == seed + 2
    expected = Grid.random(int(app.width.get()), int(app.height.get()), float(app.probability.get()), seed + 2)
    assert app.grid == expected and app.state.agents == random_agents(expected, 2, seed + 2)
    grid, agents = app.grid, app.state.agents
    app.generate()
    assert app.grid == grid and app.state.agents == agents


def complete_cooperative_planning(app):
    for _ in range(1000):
        if app.state.simulation is not None or (app.state.cooperative and app.state.cooperative.done):
            return
        app.step()
        while app.pending_step:
            app._cancel_timer()
            app._tick()
    pytest.fail("Cooperative playback did not terminate")


def test_priority_table_badges_step_pause_and_reset(app):
    app.load_conflict_demo()
    app.algorithm.set("CG-ST-A*")
    app.switch_algorithm()
    grid, agents = app.grid, app.state.agents
    app.playback.set("Detailed")
    app.step()
    assert app.state.cooperative.independent.search.expanded_nodes == 1
    app.pause()
    app.root.update()
    assert app.state.cooperative.independent.search.expanded_nodes == 1
    assert app.state.simulation is None and app.timer is None
    app.playback.set("Agent")
    while app.state.cooperative.analysis is None:
        app.step()
    assert tuple(app.priority_table.item("1", "values")) == ("1", "1", "4", "4", "1", "1")
    assert tuple(app.priority_table.item("2", "values")) == ("2", "1", "4", "4", "2", "2")
    badges = {app.canvas.itemcget(item, "text") for item in app.canvas.find_withtag("priority") if app.canvas.type(item) == "text"}
    assert badges == {"P1", "P2"}
    app.show_priorities.set(False)
    app.draw_multi()
    assert not app.canvas.find_withtag("priority")
    complete_cooperative_planning(app)
    assert app.state.simulation.timestep == 0
    app.step()
    assert app.state.simulation.timestep == 1
    app.pause()
    app.root.update()
    assert app.state.simulation.timestep == 1
    app.reset()
    assert app.grid is grid and app.state.agents is agents
    assert app.state.cooperative is app.state.simulation is app.state.planning_snapshot is None
    assert not app.priority_table.get_children() and not app.state.planning_log.lines


def test_cooperative_heuristic_display_restores_q1_selection(app):
    app.heuristic.set("Zero / Dijkstra")
    app.load_conflict_demo()
    app.algorithm.set("CG-ST-A*")
    app.switch_algorithm()
    assert app.heuristic.get() == "Manhattan" and app.heuristic_selector.instate(["disabled"])
    app.multi_mode.set(False)
    app.switch_mode()
    assert app.heuristic.get() == "Zero / Dijkstra"
    assert app.heuristic_selector.instate(["readonly"])
    app.step()
    assert app.search.heuristic.__name__ == "zero"


def test_failed_cooperative_playback_stops_without_simulation(app):
    from pathfinding.cooperative import CooperativePlanner

    app.load_conflict_demo()
    app.algorithm.set("CG-ST-A*")
    app.switch_algorithm()
    app.state.cooperative = CooperativePlanner(app.grid, app.state.agents, observe=True, max_horizon=0)
    complete_cooperative_planning(app)
    assert not app.state.cooperative.result.found and app.state.simulation is None
    assert app.timer is None and not app.running and not app.pending_step
    assert "Failed" in app.planning_phase.get()
    assert "not a proof" in "\n".join(app.state.planning_log.lines) or "do not prove" in "\n".join(app.state.planning_log.lines)


@pytest.mark.parametrize("count", [2, 6, 8])
def test_moderate_grid_and_agent_counts_have_bounded_readable_panels(app, count):
    app.width.set("20")
    app.height.set("20")
    app.probability.set("0.2")
    app.agent_count.set(str(count))
    app.multi_mode.set(True)
    app.switch_mode()
    app.algorithm.set("CG-ST-A*")
    app.switch_algorithm()
    app.generate()
    complete_cooperative_planning(app)
    assert app.state.cooperative.result.found
    assert len(app.priority_table.get_children()) == count
    assert app.priority_table.cget("height") <= 4
    assert len(app.state.planning_log.lines) <= 300
    app.root.update_idletasks()
    app.show_reservations.set(True)
    app.reservation_time.set("2")
    result = app.state.cooperative.result
    app.refresh_explanation()
    assert app.state.cooperative.result is result
    assert app.state.planning_snapshot.reservations.timestep == 2


def test_promotion_demo_updates_actual_ui_ranks(app):
    app.load_promotion_demo()
    grid, agents = app.grid, app.state.agents
    complete_cooperative_planning(app)
    result = app.state.cooperative.result
    assert result.found and result.priority_promotions == 1
    assert app.priority_table.item("8", "values")[-2:] == ("5", "4")
    assert app.priority_table.item("5", "values")[-2:] == ("4", "5")
    assert app.seed.get() == "156" and app.agent_count.get() == "8"
    assert "[Promotion]" in "\n".join(app.state.planning_log.lines)
    app.algorithm.set("Independent A*")
    app.switch_algorithm()
    assert app.grid is grid and app.state.agents is agents
    assert app.state.cooperative is None and not app.canvas.find_withtag("priority")


def test_all_q1_heuristics_preserve_map_and_animate_live_frontier(app):
    from pathfinding.heuristics import HEURISTICS

    grid = app.grid
    seed = app.seed.get()
    for name, function in HEURISTICS.items():
        app.heuristic.set(name)
        app.heuristic_selector.event_generate("<<ComboboxSelected>>")
        app.root.update()
        assert app.grid is grid and app.seed.get() == seed
        assert app.search is None and not app.explored
        app.step()
        assert app.search.heuristic is function and app.search.expanded_nodes == 1
        assert app.current == grid.start and grid.start in app.explored
        assert len(app.frontier) == app.search.frontier_size
        app.pause()
        app.root.update()
        assert app.search.expanded_nodes == 1


def test_detailed_console_rejections_and_selected_persistent_reservations(app):
    app.load_conflict_demo()
    app.algorithm.set("CG-ST-A*")
    app.switch_algorithm()
    app.playback.set("Detailed")
    app.verbosity.set("Detailed")
    complete_cooperative_planning(app)
    lines = "\n".join(app.state.planning_log.lines)
    assert "[Search]" in lines and "[Rejected]" in lines and "vertex reservation" in lines
    planner = app.state.cooperative
    result = planner.result
    app.show_reservations.set(True)
    app.reservation_time.set("100")
    app.refresh_explanation()
    snapshot = app.state.planning_snapshot.reservations
    assert snapshot.timestep == 100
    assert snapshot.vertices == snapshot.terminal_goals == frozenset(a.goal for a in app.state.agents)
    assert snapshot.edges == ()
    assert planner.result is result and result.found
