from dataclasses import dataclass

from .grid import Cell


@dataclass(frozen=True)
class SearchResult:
    path: tuple[Cell, ...]
    path_cost: int | None
    expanded_nodes: int
    generated_nodes: int
    peak_frontier_size: int
    elapsed_ms: float
    found: bool
