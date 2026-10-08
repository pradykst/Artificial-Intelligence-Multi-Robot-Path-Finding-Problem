"""Compact presentation figures drawn from measured paired rows, using Agg."""
from pathlib import Path
from statistics import mean, median, stdev

from .analysis_ablation import ablation_comparisons, summarize_ablation
from .analysis_multi import FIXED, INDEPENDENT, NO_PROMOTION, PROPOSED, summarize_methods

METHODS = (INDEPENDENT, FIXED, NO_PROMOTION, PROPOSED)
LABELS = ("Independent", "Fixed", "CG, no promotion", "Full CG")
COLORS = ("#426b95", "#cd862b", "#8765a7", "#28856b")


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def finish(plt, fig, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{name}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def configuration_groups(rows):
    keys = sorted({(r["width"], r["height"], r["obstacle_probability"]) for r in rows})
    return [(key, [r for r in rows if (r["width"], r["height"], r["obstacle_probability"]) == key]) for key in keys]


def plot_q1(rows, output: Path) -> list[Path]:
    plt = pyplot()
    from .experiments import validate_paired_rows
    from .heuristics import REQUIRED_HEURISTICS
    groups = validate_paired_rows(rows, (*REQUIRED_HEURISTICS, "Zero / Dijkstra"))
    configs = configuration_groups(rows)
    methods = (*REQUIRED_HEURISTICS, "Zero / Dijkstra")
    paths = []
    for name, metrics in (("q1_expansions", ("expanded_nodes",)),
                          ("q1_resources", ("generated_nodes", "peak_frontier_size", "elapsed_ms"))):
        fig, axes = plt.subplots(len(metrics), len(configs), figsize=(4 * len(configs), 3.5 * len(metrics)),
                                 squeeze=False, layout="constrained")
        for column, ((w, h, p), selected) in enumerate(configs):
            for row_index, metric in enumerate(metrics):
                ax = axes[row_index, column]
                values = [[r[metric] for r in selected if r["heuristic"] == method] for method in methods]
                ax.bar(range(4), [mean(v) for v in values], color=COLORS,
                       yerr=[stdev(v) if len(v) > 1 else 0 for v in values], capsize=3, alpha=.8, label="Mean ± sample SD")
                ax.scatter(range(4), [median(v) for v in values], color="black", marker="_", s=150, label="Median", zorder=4)
                ax.set_xticks(range(4), ("Manhattan", "Euclidean", "Chebyshev", "Dijkstra"), rotation=25, ha="right")
                ax.set_title(f"{w}×{h}, p={p:g}; n={len(values[0])} paired trials")
                ax.set_ylabel({"expanded_nodes": "Expanded nodes", "generated_nodes": "Generated nodes",
                               "peak_frontier_size": "Peak unique frontier", "elapsed_ms": "Search time (ms)"}[metric])
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("Q1: means, medians and between-scenario variability" if name == "q1_expansions" else
                     "Q1 resources: time is machine-dependent; small timing gaps are descriptive")
        paths.append(finish(plt, fig, output, name))
    fig, axes = plt.subplots(1, len(configs), figsize=(4 * len(configs), 4), squeeze=False, layout="constrained")
    for ax, ((w, h, p), selected) in zip(axes.flat, configs):
        ids = sorted({r["instance_id"] for r in selected})
        differences = [[groups[i][method]["expanded_nodes"] - groups[i]["Manhattan"]["expanded_nodes"]
                        for i in ids] for method in methods[1:]]
        ax.errorbar(range(3), [mean(v) for v in differences],
                    yerr=[stdev(v) if len(v) > 1 else 0 for v in differences], fmt="o", capsize=4, label="Mean ± sample SD")
        ax.scatter(range(3), [median(v) for v in differences], marker="x", label="Median")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(3), ("Euclidean", "Chebyshev", "Dijkstra"), rotation=25)
        ax.set_title(f"{w}×{h}, p={p:g}; n={len(ids)} pairs")
        ax.set_ylabel("Expanded nodes: method − Manhattan")
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Q1 paired expansion differences (same scenario in every pair)")
    paths.append(finish(plt, fig, output, "q1_paired_expansions"))
    return paths


def success_bars(ax, rows, title):
    for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        selected = [r for r in rows if r["algorithm"] == method]
        n = len(selected)
        successes = sum(r["coordinated_planning_success"] for r in selected)
        if not n:
            ax.text(index, 2, "No observations", ha="center", rotation=90)
            continue
        rate = 100 * successes / n
        ax.bar(index, rate, color=color)
        ax.text(index, rate + 1, f"{successes}/{n}", ha="center", fontsize=9)
    ax.set_xticks(range(4), LABELS, rotation=15, ha="right")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Coordinated success (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=.2)


def plot_q2(rows, output: Path) -> list[Path]:
    plt = pyplot()
    summaries = summarize_methods(rows)
    pairs = ablation_comparisons(rows)
    paths = []
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    scenario_count = len({r["scenario_id"] for r in rows})
    trial_count = len({r["trial"] for r in rows})
    sizes = ", ".join(f"{w}×{h}" for w, h in sorted({(r['width'], r['height']) for r in rows}))
    probabilities = ", ".join(str(p) for p in sorted({r['obstacle_probability'] for r in rows}))
    success_bars(ax, rows, f"Q2: {scenario_count} paired scenarios; {trial_count} trials per condition\nGrids {sizes}; p={probabilities}")
    paths.append(finish(plt, fig, output, "q2_overall_success"))
    for dimension, xlabel, name in (("number_of_agents", "Number of agents", "q2_success_agents"),
                                   ("obstacle_probability", "Obstacle probability", "q2_success_probability")):
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        for method, label, color in zip(METHODS, LABELS, COLORS):
            series = sorted((r for r in summaries if r["group_dimension"] == dimension and r["algorithm"] == method),
                            key=lambda r: r["group_value"])
            ax.plot([r["group_value"] for r in series], [100 * r["coordinated_success_rate"] for r in series],
                    "o-", label=label, color=color)
        ns = [r for r in summaries if r["group_dimension"] == dimension and r["algorithm"] == INDEPENDENT]
        ax.set_xticks([r["group_value"] for r in ns], [f"{r['group_value']}\nn={r['scenario_count']}" for r in ns])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Coordinated success (%)")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        ax.set_title(f"Q2 paired success; {trial_count} trials per condition\nGrids {sizes}; p={probabilities}")
        paths.append(finish(plt, fig, output, name))
    overall = [r for r in summarize_ablation(pairs) if r["group_dimension"] == "all"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    mechanism_labels = ("Independent → Fixed", "Fixed → CG no promotion", "CG no promotion → Full CG")
    for index, row in enumerate(overall):
        axes[0].bar(index - .18, row["first_only_success_count"], width=.36, color=COLORS[1], label="First only" if index == 0 else None)
        axes[0].bar(index + .18, row["second_only_success_count"], width=.36, color=COLORS[3], label="Second only" if index == 0 else None)
        for offset, field in ((-.18, "first_only_success_count"), (.18, "second_only_success_count")):
            axes[0].text(index + offset, row[field] + 1, str(row[field]), ha="center", fontsize=9)
        for delta_ax, short, ylabel in ((axes[1], "soc", "SOC: second − first"), (axes[2], "expanded", "Expanded states: second − first")):
            values = [p[f"delta_{short}"] for p in pairs if p["mechanism"] == row["mechanism"] and p[f"delta_{short}"] is not None]
            if values:
                delta_ax.scatter([index] * len(values), values, s=12, alpha=.25, color=COLORS[index])
                delta_ax.scatter(index, mean(values), color="black", marker="D", s=35)
            delta_ax.text(index, .98, f"n={len(values)}", ha="center", va="top", transform=delta_ax.get_xaxis_transform())
            delta_ax.axhline(0, color="black", linewidth=.8)
            delta_ax.set_ylabel(ylabel)
        axes[0].text(index, .98, f"n={row['scenario_count']}", ha="center", va="top", transform=axes[0].get_xaxis_transform())
    for ax in axes:
        ax.set_xlim(-.45, 2.45)
        ax.set_xticks(range(3), mechanism_labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Scenarios with discordant success")
    axes[0].set_ylim(0, 5 + 1.2 * max((max(r["first_only_success_count"], r["second_only_success_count"])
                                      for r in overall), default=0))
    axes[0].legend(fontsize=9)
    axes[1].set_title("Both-success pairs; diamonds = mean")
    axes[2].set_yscale("symlog", linthresh=100)
    axes[2].set_title("Both-success pairs; symmetric log scale")
    fig.suptitle("Q2 mechanism comparisons: reservations, initial ordering, promotion")
    paths.append(finish(plt, fig, output, "q2_ablation"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, threshold in zip(axes, (1, 3)):
        selected = [r for r in rows if r["independent_total_collisions"] is not None and r["independent_total_collisions"] >= threshold]
        success_bars(ax, selected, f"Independent collision events ≥ {threshold}")
    fig.suptitle("Q2 conflict subsets: descriptive, overlapping; sampling unchanged")
    paths.append(finish(plt, fig, output, "q2_conflict_subsets"))
    cg = [r for r in rows if r["algorithm"] == PROPOSED]
    counts = sorted({r["priority_promotions"] for r in cg})
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    frequencies = [sum(r["priority_promotions"] == count for r in cg) for count in counts]
    ax.bar(counts, frequencies, color=COLORS[3])
    for count, frequency in zip(counts, frequencies):
        ax.text(count, frequency * 1.05, str(frequency), ha="center")
    ax.set_yscale("log")
    ax.set_ylim(.7, max(frequencies) * 1.5)
    ax.set_xticks(counts)
    ax.set_xlabel("Priority positions promoted (sum per scenario)")
    ax.set_ylabel("Scenario count (log scale)")
    ax.set_title(f"Full CG promotion activity; n={len(cg)} paired scenarios")
    paths.append(finish(plt, fig, output, "q2_promotions"))
    return paths
