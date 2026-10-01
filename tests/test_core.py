from collections import deque

import pytest

from pathfinding.astar import AStarSearch, astar
from pathfinding.grid import Grid
from pathfinding.heuristics import HEURISTICS, REQUIRED_HEURISTICS


def distances(grid, start):
    costs = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in grid.neighbors(current):
            if neighbor not in costs:
                costs[neighbor] = costs[current] + 1
                queue.append(neighbor)
    return costs


@pytest.mark.parametrize("heuristic", HEURISTICS.values(), ids=HEURISTICS)
@pytest.mark.parametrize("grid, expected", [
    (Grid(2, 1, frozenset(), (0, 0), (1, 0)), 1),
    (Grid(4, 3, frozenset(), (0, 0), (3, 2)), 5),
    (Grid(3, 3, frozenset({(1, 0), (1, 1)}), (0, 0), (2, 0)), 6),
    (Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0)), None),
])
def test_known_paths(grid, expected, heuristic):
    result = astar(grid, heuristic)
    assert result.path_cost == expected
    assert result.found == (expected is not None)
    if result.found:
        assert result.path[0] == grid.start
        assert result.path[-1] == grid.goal
        assert all(b in tuple(grid.neighbors(a)) for a, b in zip(result.path, result.path[1:]))
    else:
        assert result.path == ()


def test_adjacent_metrics():
    result = astar(Grid(2, 1, frozenset(), (0, 0), (1, 0)))
    assert result.expanded_nodes == 2
    assert result.generated_nodes == 1
    assert result.peak_frontier_size == 1
    assert result.elapsed_ms >= 0


def test_already_at_goal():
    grid = Grid(2, 1, frozenset(), (0, 0), (1, 0))
    result = astar(grid, start=grid.goal)
    assert result.found and result.path_cost == 0
    assert result.path == (grid.goal,)
    assert result.expanded_nodes == 1
    assert result.generated_nodes == 0


@pytest.mark.parametrize("heuristic", REQUIRED_HEURISTICS.values(), ids=REQUIRED_HEURISTICS)
def test_heuristic_admissibility(heuristic):
    for seed in range(20):
        grid = Grid.random(6, 5, 0.25, seed)
        assert heuristic(grid.goal, grid.goal) == 0
        for cell, distance in distances(grid, grid.goal).items():
            assert 0 <= heuristic(cell, grid.goal) <= distance


def test_random_searches_match_bfs():
    for seed in range(100):
        grid = Grid.random(8, 7, 0.3, seed)
        expected = distances(grid, grid.start).get(grid.goal)
        for heuristic in HEURISTICS.values():
            assert astar(grid, heuristic).path_cost == expected


def test_random_reproducibility():
    assert Grid.random(20, 20, 0.25, 42) == Grid.random(20, 20, 0.25, 42)
    assert Grid.random(20, 20, 0.25, 42) != Grid.random(20, 20, 0.25, 43)


def test_random_endpoints_are_distinct_and_free():
    for seed in range(20):
        grid = Grid.random(5, 6, 1.0, seed)
        assert grid.start != grid.goal
        assert grid.is_free(grid.start) and grid.is_free(grid.goal)


@pytest.mark.parametrize("args", [(0, 2, 0.2, 1), (1, 1, 0.2, 1), (2, 2, -0.1, 1), (2, 2, 1.1, 1)])
def test_invalid_generation(args):
    with pytest.raises(ValueError):
        Grid.random(*args)


def test_step_engine_matches_batch_and_is_deterministic():
    grid = Grid.random(12, 10, 0.2, 91)
    search = AStarSearch(grid)
    frontier = {grid.start}
    peak = 1
    while search.result is None:
        event = search.step()
        if event.current is not None:
            frontier.remove(event.current)
        frontier.update(event.opened)
        peak = max(peak, len(frontier))
        assert search.frontier_size == len(frontier)
    batch = astar(grid)
    assert search.result.path == batch.path
    assert search.result.expanded_nodes == batch.expanded_nodes
    assert search.result.generated_nodes == batch.generated_nodes
    assert search.peak_frontier_size == peak
    with pytest.raises(RuntimeError):
        search.step()


def test_improvements_reopenings_and_stale_entries():
    from random import Random

    grid = Grid.random(10, 10, 0.2, 0)
    true_distances = distances(grid, grid.goal)
    rng = Random(0)
    estimates = {cell: rng.randint(0, cost) for cell, cost in sorted(true_distances.items())}
    search = AStarSearch(grid, lambda cell, goal: estimates.get(cell, 0))
    frontier = {grid.start}
    explored = set()
    expansions = generated = 0
    peak = 1
    improved_open = reopened = False
    while search.result is None:
        event = search.step()
        assert event.current in frontier
        frontier.remove(event.current)
        explored.add(event.current)
        expansions += 1
        improved_open |= any(cell in frontier for cell in event.opened)
        reopened |= any(cell in explored for cell in event.opened)
        generated += len(event.opened)
        frontier.update(event.opened)
        peak = max(peak, len(frontier))
        assert search.frontier_size == len(frontier)
    assert improved_open and reopened
    assert search.result.path_cost == true_distances[grid.start]
    assert search.expanded_nodes == expansions
    assert search.generated_nodes == generated
    assert search.peak_frontier_size == peak


def test_pauses_between_steps_are_excluded_from_timing(monkeypatch):
    import importlib

    module = importlib.import_module("pathfinding.astar")
    clock = iter([0, 10, 1000, 1020, 10000, 10030])
    monkeypatch.setattr(module, "perf_counter_ns", lambda: next(clock))
    search = AStarSearch(Grid(2, 1, frozenset(), (0, 0), (1, 0)))
    search.step()
    search.step()
    assert search.result.elapsed_ms == pytest.approx(60 / 1_000_000)


def test_invalid_search_endpoint():
    grid = Grid(3, 1, frozenset({(1, 0)}), (0, 0), (2, 0))
    with pytest.raises(ValueError, match="endpoints"):
        astar(grid, start=(1, 0))
