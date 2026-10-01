from collections import defaultdict
from math import ceil, sqrt
import tkinter as tk
from typing import Mapping, Sequence

from .agents import Agent
from .collisions import Collision, Paths
from .grid import Cell

AGENT_COLORS = ("#1763aa", "#b04a00", "#087b47", "#8c3da8", "#ba2862", "#6d6613", "#007f89", "#634738")


def collision_message(event: Collision) -> str:
    agents = ", ".join(str(agent_id) for agent_id in event.agent_ids)
    place = str(event.cells[0]) if event.kind == "vertex" else f"{event.cells[0]} <-> {event.cells[1]}"
    return f"Collision at t={event.timestep}: {event.kind} conflict, Agents {agents}, {place}"


def draw_agents(
    canvas: tk.Canvas, agents: Sequence[Agent], paths: Paths, positions: Mapping[int, Cell],
    collisions: Sequence[Collision], size: float, left: float, top: float,
) -> None:
    """Offset path traces and split overlapping robot markers to preserve IDs."""
    canvas.delete("multi")

    def center(cell: Cell) -> tuple[float, float]:
        return left + (cell[0] + 0.5) * size, top + (cell[1] + 0.5) * size

    colors = {agent.agent_id: AGENT_COLORS[index % len(AGENT_COLORS)] for index, agent in enumerate(agents)}
    font_size = max(6, min(12, int(size * 0.23)))
    for index, agent in enumerate(agents):
        color = colors[agent.agent_id]
        path = paths.get(agent.agent_id, ())
        offset = (index - (len(agents) - 1) / 2) * min(size * 0.055, 2.5)
        if len(path) > 1:
            points = [coordinate + offset for cell in path for coordinate in center(cell)]
            canvas.create_line(*points, fill=color, width=max(1, min(3, size * 0.065)), tags="multi")
        for cell, label, y_offset in [(agent.start, f"S{agent.agent_id}", -0.32),
                                       (agent.goal, f"G{agent.agent_id}", 0.32)]:
            x, y = center(cell)
            canvas.create_text(x, y + y_offset * size, text=label, fill=color,
                               font=("Arial", font_size, "bold"), tags="multi")

    occupants: dict[Cell, list[int]] = defaultdict(list)
    for agent_id, cell in positions.items():
        occupants[cell].append(agent_id)
    for cell, ids in occupants.items():
        x, y = center(cell)
        columns = ceil(sqrt(len(ids)))
        rows = ceil(len(ids) / columns)
        diameter = size * 0.46 / max(columns, rows)
        for index, agent_id in enumerate(sorted(ids)):
            cx = x + (index % columns - (columns - 1) / 2) * diameter
            cy = y + (index // columns - (rows - 1) / 2) * diameter
            radius = diameter * 0.48
            canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                               fill=colors[agent_id], outline="white", tags=("multi", "robot"))
            canvas.create_text(cx, cy, text=str(agent_id), fill="white",
                               font=("Arial", max(6, min(font_size, int(diameter * 0.65))), "bold"), tags=("multi", "robot"))

    for event in collisions:
        for cell in event.cells:
            x, y = center(cell)
            half = size * 0.46
            canvas.create_rectangle(x - half, y - half, x + half, y + half,
                                    outline="#e00000", width=3, tags="multi")
        if event.kind == "edge":
            canvas.create_line(*center(event.cells[0]), *center(event.cells[1]),
                               fill="#e00000", width=4, dash=(4, 3), tags="multi")
    canvas.tag_raise("robot")
