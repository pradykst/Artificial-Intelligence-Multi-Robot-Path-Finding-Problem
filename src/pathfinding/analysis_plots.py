from pathlib import Path
from typing import Sequence

from .analysis_multi import FIXED, PROPOSED, missing


def plot_analysis(rows: Sequence[dict], pairs: Sequence[dict], output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from statistics import mean, median

    def distribution(axis, data, metric, counts, label):
        means, medians, labels = [], [], []
        for count in counts:
            values = [row[metric] for row in data if row["number_of_agents"] == count and not missing(row[metric])]
            labels.append(f"{count}\n(n={len(values)})")
            offsets = [(index - (len(values) - 1) / 2) / max(len(values), 1) * 0.3 for index in range(len(values))]
            axis.scatter([count + offset for offset in offsets], values, color="#6b7280", alpha=0.45, s=15)
            means.append(mean(values) if values else float("nan"))
            medians.append(median(values) if values else float("nan"))
        axis.plot(counts, means, "D-", color="#3178a8", label="Mean", markersize=5)
        axis.plot(counts, medians, "x--", color="#d17b12", label="Median", markersize=7)
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(counts, labels)
        axis.set_xlabel("Agents (observation count)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        if not any(not missing(row[metric]) for row in data):
            axis.text(0.5, 0.5, "No eligible observations", ha="center", transform=axis.transAxes)

    def success_rates(axis, data, counts, title):
        methods = [method for method in (FIXED, PROPOSED) if any(row["algorithm"] == method for row in rows)]
        for method, color, marker, size in ((FIXED, "#d17b12", "s", 8), (PROPOSED, "#228653", "o", 4)):
            if method not in methods:
                continue
            rates = []
            for count in counts:
                group = [row for row in data if row["algorithm"] == method and row["number_of_agents"] == count]
                rates.append(100 * sum(row["coordinated_planning_success"] for row in group) / len(group) if group else float("nan"))
            axis.plot(counts, rates, marker=marker, markersize=size, color=color, label=method,
                      markerfacecolor="none" if method == FIXED else color)
        labels = [f"{count}\n(n={sum(row['algorithm'] == PROPOSED and row['number_of_agents'] == count for row in data)})" for count in counts]
        axis.set_xticks(counts, labels)
        axis.set_ylim(-5, 105)
        axis.set_title(title)
        axis.set_xlabel("Agents (scenarios per method)")
        axis.set_ylabel("Coordinated success (%)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower left")
        if not data:
            axis.text(0.5, 0.5, "No scenarios in this subset", ha="center", transform=axis.transAxes)

    figures = []
    configurations = sorted({(row["width"], row["height"], row["obstacle_probability"]) for row in rows})
    for width, height, probability in configurations:
        def matches(row):
            return (row["width"], row["height"], row["obstacle_probability"]) == (width, height, probability)

        group = [row for row in rows if matches(row)]
        counts = sorted({row["number_of_agents"] for row in group})
        contended = [row for row in group if not missing(row["independent_total_collisions"]) and row["independent_total_collisions"] >= 1]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
        success_rates(axes[0], group, counts, "All generated scenarios")
        success_rates(axes[1], contended, counts, "Independent A* had >=1 conflict")
        cg = [row for row in group if row["algorithm"] == PROPOSED]
        trial_text = "; ".join(f"{count} agents: {sum(row['number_of_agents'] == count for row in cg)}" for count in counts)
        title = f"{width} × {height}, obstacle probability {probability:g}; trials — {trial_text}"
        distribution(axes[2], cg, "priority_promotions", counts, "CG priority promotions")
        largest = max((row["priority_promotions"] for row in cg if not missing(row["priority_promotions"])), default=0)
        axes[2].set_ylim(-0.05, max(1, largest) * 1.1)
        axes[2].set_title("CG adaptation activity (all runs)")
        axes[2].legend(fontsize=8)
        fig.suptitle(f"{title}\nOutcomes and adaptation; subset counts are shown on axes")
        filename = output / f"coordination_{width}x{height}_p{probability:g}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        figures.append(filename)
        if not pairs:
            continue
        paired = [pair for pair in pairs if matches(pair) and pair["fixed_success"] and pair["cg_success"]]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), layout="constrained")
        for axis, metric, label in zip(axes, ("delta_soc", "delta_makespan", "delta_expanded"),
                                       ("SOC: CG − Fixed", "Makespan: CG − Fixed", "Expanded states: CG − Fixed")):
            distribution(axis, paired, metric, counts, label)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{title}\n"
                     "Both methods successful only; each dot is one scenario. Negative differences mean a lower CG value.")
        filename = output / f"paired_{width}x{height}_p{probability:g}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)
        figures.append(filename)
    return figures
