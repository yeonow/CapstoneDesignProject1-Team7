"""Headless plots of selected actual evaluation episodes (x right, y down)."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import numpy as np


def _number(value):
    return "N/A" if value is None else f"{value:.3f}"


def plot_path_comparison(selections, metrics, output_dir, *, target_mode, grid_size):
    """Save a 2x2 comparison; metrics pool all evaluation episodes across runs."""
    figure = Figure(figsize=(13, 13))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    names = {"q_learning": "Q-Learning", "random": "Random Search"}
    for row, algorithm in enumerate(names):
        for column, kind in enumerate(("best", "representative")):
            ax = axes[row, column]
            ax.set_title(f"{names[algorithm]} / {kind.title()} Path", pad=58)
            ax.set_xlim(-0.5, grid_size - 0.5)
            ax.set_ylim(grid_size - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.set_xticks(range(grid_size))
            ax.set_yticks(range(grid_size))
            ax.set_xticks(np.arange(-0.5, grid_size), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size), minor=True)
            ax.grid(which="minor", alpha=0.35)
            ax.set_xlabel("x (right)")
            ax.set_ylabel("y (down)")
            record = selections[algorithm][kind]
            if record is None:
                ax.text(0.5, 0.5, "No successful episode", transform=ax.transAxes,
                        ha="center", va="center")
                continue
            points = np.asarray(record["path"])
            if len(points) > 1:
                segments = np.stack((points[:-1], points[1:]), axis=1)
                lines = LineCollection(segments, cmap="viridis", linewidths=1.6, alpha=0.65)
                lines.set_array(np.arange(len(segments)))
                ax.add_collection(lines)
                delta = np.diff(points, axis=0)
                moving = np.any(delta != 0, axis=1)
                ax.quiver(points[:-1, 0][moving], points[:-1, 1][moving],
                          delta[:, 0][moving] * 0.7, delta[:, 1][moving] * 0.7,
                          angles="xy", scale_units="xy", scale=1, width=0.004,
                          color="steelblue", alpha=0.45)
                # Time labels retain order even where a route revisits a cell.
                indices = np.unique(np.linspace(0, len(points) - 1, min(8, len(points)), dtype=int))
                labels = {}
                for index in indices:
                    labels.setdefault(tuple(points[index]), []).append(str(index))
                for position, steps in labels.items():
                    ax.annotate(",".join(steps), position, fontsize=7, xytext=(3, 3),
                                textcoords="offset points")
            ax.scatter(*record["start_position"], c="green", s=90, label="Start", zorder=4)
            ax.scatter(*record["target_position"], c="red", marker="*", s=150,
                       label="Target", zorder=5)
            ax.legend(fontsize=8, loc="upper right")
            ax.text(0, 1.02,
                    f"Run {record['run']} / Episode {record['episode']} | Success: {record['success']}\n"
                    f"Start {record['start_position']} -> Target {record['target_position']}\n"
                    f"Steps {record['steps']} | Shortest {record['shortest_path_steps']} | "
                    f"Ratio {_number(record['path_ratio'])} | Blocked {record['blocked_move_count']}",
                    transform=ax.transAxes, fontsize=8)
    figure.suptitle(
        f"Actual evaluation trajectories | {target_mode} | {grid_size}x{grid_size}\n"
        "Each panel uses its own episode target; selected episodes may have different targets.\n"
        "Path colors progress from early to late; numbers indicate action steps.", fontsize=12)
    footer = ["All evaluation episodes pooled across runs (steps and distance include failures):"]
    for algorithm, name in names.items():
        metric = metrics[algorithm]
        footer.append(
            f"{name}: success {metric['success_rate']:.1%} | "
            f"mean steps (all) {metric['average_steps']:.2f} | "
            f"mean path ratio (success only, positive shortest distance) "
            f"{_number(metric['path_ratio_on_success'])} | "
            f"mean final distance (all, cells) {metric['average_final_distance']:.2f}")
    figure.text(0.04, 0.025, "\n".join(footer), fontsize=9)
    figure.subplots_adjust(top=0.81, bottom=0.14, hspace=0.55, wspace=0.26)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = tuple(directory / f"path_comparison.{extension}" for extension in ("png", "pdf"))
    for path in outputs:
        figure.savefig(path, dpi=160)
    figure.clear()
    print("Path figures saved to: " + ", ".join(str(path) for path in outputs))
    return outputs
