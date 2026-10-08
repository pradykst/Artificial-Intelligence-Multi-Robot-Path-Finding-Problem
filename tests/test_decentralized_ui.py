import pytest

tk = pytest.importorskip("tkinter")
from pathfinding.agents import Agent
from pathfinding.decentralized import DCN, DecentralizedPlanner
from pathfinding.grid import Grid
from pathfinding.visualization import PathfindingApp


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


def complete_step(app):
    app.step()
    for _ in range(2000):
        if not app.pending_step:
            return
        app._cancel_timer()
        app._tick()
    pytest.fail("Decentralized step did not complete")


def test_switching_keeps_scenario_and_restores_q1_heuristic(app):
    app.heuristic.set("Euclidean")
    app.load_conflict_demo()
    grid, agents, seed = app.grid, app.state.agents, app.seed.get()
    for algorithm in (DCN, "CG-ST-A*", DCN, "Independent A*"):
        app.algorithm.set(algorithm)
        app.switch_algorithm()
        assert app.grid is grid and app.state.agents is agents and app.seed.get() == seed
        assert app.state.simulation is app.state.decentralized is app.state.cooperative is None
        assert not app.canvas.find_withtag("priority")
        if algorithm == DCN:
            assert app.heuristic.get() == "Manhattan" and app.heuristic_selector.instate(["disabled"])
            assert str(app.decentralized_view.frame) in app.panes.panes()
            assert str(app.explanation) not in app.panes.panes()
    assert app.heuristic.get() == "Euclidean"


def test_demo_round_step_run_pause_revision_safe_execution_reset(app):
    app.decentralized_view.load_demo()
    grid, agents = app.grid, app.state.agents
    complete_step(app)
    planner = app.state.decentralized
    assert planner.rounds == 1 and app.state.simulation is None
    assert all(c.version == 1 for c in planner.controllers.values())
    app.run()
    app.pause()
    round_count = planner.rounds
    app.root.update()
    assert planner.rounds == round_count and app.timer is None and not app.running
    for _ in range(40):
        if planner.done:
            break
        complete_step(app)
    assert planner.result.found and app.state.simulation.timestep == 0
    assert app.decentralized_view.local.item(app.decentralized_view.local.get_children()[1], "values")[1] == "2"
    lines = "\n".join(app.decentralized_view.lines)
    assert "YIELD_DECISION" in lines and "UPDATED_PATH" in lines and "AGREEMENT" in lines
    assert not app.priority_table.get_children() and not app.canvas.find_withtag("priority")
    complete_step(app)
    assert app.state.simulation.timestep == 1
    while not app.state.simulation.done:
        complete_step(app)
    assert app.state.simulation.metrics.total_collisions == 0
    assert "WAIT actions: 1" in app.stats.get()
    app.reset()
    assert app.grid is grid and app.state.agents is agents
    assert app.state.decentralized is app.state.simulation is None and not app.decentralized_view.lines


def test_long_round_yields_and_pause_cancels_pending_step(app):
    app.decentralized_view.load_demo()
    app.grid = Grid(50, 50, frozenset(), (0, 0), (49, 49))
    app.state.agents = (Agent(1, (0, 0), (49, 49)), Agent(2, (49, 0), (0, 49)))
    app.reset()
    app.step()
    assert app.pending_step and app.timer is not None and app.state.simulation is None
    planner = app.state.decentralized
    expanded = sum(c.expanded_states for c in planner.controllers.values())
    assert 0 < expanded <= 200
    app.pause()
    app.root.update()
    assert not app.pending_step and app.timer is None
    assert sum(c.expanded_states for c in planner.controllers.values()) == expanded


def test_failed_negotiation_never_enters_simulation_and_links_are_optional(app):
    app.decentralized_view.load_demo()
    app.state.decentralized = DecentralizedPlanner(app.grid, app.state.agents, max_rounds=1)
    complete_step(app)
    assert app.state.decentralized.done and not app.state.decentralized.result.found
    assert app.state.simulation is None and not app.running and not app.pending_step
    assert "undefined" in app.stats.get()
    app.reset()
    complete_step(app)
    complete_step(app)
    assert not app.canvas.find_withtag("communication")
    app.decentralized_view.links.set(True)
    app.draw_multi()
    assert app.canvas.find_withtag("communication")
    app.algorithm.set("Independent A*")
    app.switch_algorithm()
    assert not app.canvas.find_withtag("communication")
