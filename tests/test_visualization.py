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
    app.step()
    while app.pending_step:
        app._cancel_timer()
        app._tick()
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
