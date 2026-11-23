"""Headless plots of selected actual evaluation episodes (x right, y down)."""

from collections import Counter
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import patheffects
from matplotlib.collections import LineCollection
import numpy as np


def _number(value):
    return "N/A" if value is None else f"{value:.3f}"


def _draw_best_path(figure, ax, record, name, *, presentation_positions=None,
                    title=None, colorbar_ax=None, missing_label="No successful evaluation episode"):
    """Draw every observed segment, including revisits, with action time colors."""
    grid_size = 20
    ax.set(title=f"{name} Best Path", xlim=(-0.5, 19.5), ylim=(19.5, -0.5),
           xlabel="x (right)", ylabel="y (down)", aspect="equal")
    presentation = presentation_positions is not None
    ax.set_title(title or f"{name} Best Path", pad=57 if presentation else 82,
                 fontweight="bold" if presentation else None)
    if presentation:
        ax.set_facecolor("#f8fafc")
        ax.set_axisbelow(True)
        _draw_endpoints(ax, *presentation_positions)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-0.5, grid_size), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size), minor=True)
    ax.grid(which="minor", alpha=0.25)
    if record is None:
        if colorbar_ax is not None:
            colorbar_ax.set_axis_off()
        ax.text(0.5, 0.5, missing_label, transform=ax.transAxes,
                ha="center", va="center")
        return
    points = np.asarray(record["path"])
    segments = np.stack((points[:-1], points[1:]), axis=1)
    lines = LineCollection(segments, cmap="viridis", linewidths=1.8, alpha=0.8)
    lines.set_array(np.arange(1, len(points)))
    lines.set_clim(0, max(record["steps"], 1))
    ax.add_collection(lines)
    colorbar = figure.colorbar(lines, ax=ax, cax=colorbar_ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Action step (early → late)", fontsize=8)
    delta = np.diff(points, axis=0)
    moving = np.any(delta != 0, axis=1)
    origins = points[:-1][moving]
    directions = delta[moving]
    if presentation:
        origins = origins + 0.23 * directions
    ax.quiver(*origins.T, *(directions * 0.65).T, angles="xy", scale_units="xy", scale=1,
              width=0.0034 if presentation else 0.003,
              color="#15283b" if presentation else "steelblue",
              alpha=0.72 if presentation else 0.45, zorder=3 if presentation else 1)
    if presentation:
        visits = Counter(map(tuple, points[np.r_[True, moving]]))
        revisits = [position for position, count in visits.items() if count > 1]
        if revisits:
            ax.scatter(*np.asarray(revisits).T, s=38, facecolors="none", edgecolors="#34465d",
                       linewidths=0.75, alpha=0.75, zorder=2, label="Revisit")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    labels = {}
    for index in np.unique(np.linspace(0, len(points) - 1, min(8, len(points)), dtype=int)):
        labels.setdefault(tuple(points[index]), []).append(str(index))
    for position, steps in labels.items():
        dx, dy = 4, -8
        if presentation:
            dx = -4 if position[0] >= grid_size - 2 else 4
            dy = 4 if position[1] == grid_size - 1 else -8
        annotation = ax.annotate(",".join(steps), position, fontsize=7, xytext=(dx, dy),
                                 ha="right" if dx < 0 else "left",
                                 textcoords="offset points", zorder=6 if presentation else None)
        if presentation:
            annotation.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])
    if presentation:
        ax.text(0, 1.028,
                f"Run {record['run']} / Episode {record['episode']}  |  "
                f"Blocked moves {record['blocked_move_count']}\n"
                f"Steps {record['steps']}  |  Shortest {record['shortest_path_steps']}  |  "
                f"Path ratio {_number(record['path_ratio'])}",
                transform=ax.transAxes, fontsize=10, linespacing=1.5)
        return
    _draw_endpoints(ax, record["start_position"], record["target_position"])
    ax.text(0, 1.035,
            f"Target Mode: Random Target | Run {record['run']} / Episode {record['episode']}\n"
            f"Start {tuple(record['start_position'])} → Target {tuple(record['target_position'])}\n"
            f"Actual Steps: {record['steps']} | Shortest Steps: {record['shortest_path_steps']}\n"
            f"Path Ratio: {_number(record['path_ratio'])} | Blocked Moves: {record['blocked_move_count']}",
            transform=ax.transAxes, fontsize=8)


def _draw_endpoints(ax, start, target):
    ax.scatter(*start, c="green", marker="o", s=85, label="Start", zorder=4)
    ax.scatter(*target, c="red", marker="*", s=160, label="Target", zorder=5)
    ax.legend(fontsize=8, loc="upper right")


def plot_best_paths(selections, output_dir):
    """Paper experiment only: two individual PNGs and a side-by-side PNG/PDF."""
    for record in selections.values():
        if record is not None and (record["grid_size"] != 20 or record["target_mode"] != "split"):
            raise ValueError("Best-path figures require grid=20 and target_mode='split'")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    names = {"q_learning": "Q-Learning", "random": "Random Search"}
    for algorithm, name in names.items():
        figure = Figure(figsize=(7.5, 8.5))
        FigureCanvasAgg(figure)
        _draw_best_path(figure, figure.subplots(), selections[algorithm], name)
        figure.subplots_adjust(top=0.76, bottom=0.1, left=0.1, right=0.91)
        path = directory / f"{'random_search' if algorithm == 'random' else algorithm}_best.png"
        figure.savefig(path, dpi=240)
        figure.clear()
        outputs.append(path)
    figure = Figure(figsize=(14, 8.5))
    FigureCanvasAgg(figure)
    for ax, (algorithm, name) in zip(figure.subplots(1, 2), names.items()):
        _draw_best_path(figure, ax, selections[algorithm], name)
    figure.text(0.5, 0.035,
                "Best episodes selected independently; targets may differ.\n"
                "Path lengths are not a paired comparison. Numbers show sampled action steps; all coordinates are retained.",
                ha="center", fontsize=9)
    figure.subplots_adjust(top=0.77, bottom=0.12, left=0.05, right=0.95, wspace=0.27)
    for extension in ("png", "pdf"):
        path = directory / f"best_path_comparison.{extension}"
        figure.savefig(path, dpi=240)
        outputs.append(path)
    figure.clear()
    return tuple(outputs)


def plot_presentation_paths(best, paired, output_dir, *, start, target):
    """Save presentation panels with common limits and episode provenance."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    positions = (start, target)
    names = {"q_learning": "Q-Learning", "random": "Random Search"}
    caption = f"Same start {start} and target {target} | Actual evaluation trajectories"
    legend = "Numbers mark sampled steps. Circles mark revisits. JSON includes every step."
    q, random_path = (best[key] for key in names)
    best_caption = "Best episodes selected independently."
    if q is not None and random_path is not None:
        differing = [key for key in ("run", "episode") if q[key] != random_path[key]]
        if differing:
            best_caption += f" The selected paths differ by {' and '.join(differing)}."
        else:
            best_caption += " Both paths came from the same run and episode."

    figures = [
        ("q_learning_best", {"q_learning": q}, None, ""),
        ("random_search_best", {"random": random_path}, None, ""),
        ("best_path_comparison", best, "Q-Learning vs Random Search — Same Target", best_caption),
        ("paired_comparison", paired or dict.fromkeys(names),
         "Q-Learning vs Random Search — Paired Comparison",
         "Same run, episode, and RSSI noise seed for both paths." if paired else
         "No run and episode produced a success for both algorithms."),
    ]
    outputs = []
    for stem, selections, heading, selection_caption in figures:
        single = len(selections) == 1
        figure = Figure(figsize=(7.5 if single else 13.5, 7.9))
        FigureCanvasAgg(figure)
        panels = figure.add_gridspec(1, len(selections), left=0.09 if single else 0.06,
                                     right=0.91 if single else 0.95, top=0.74, bottom=0.17,
                                     wspace=0.30)
        for index, (algorithm, record) in enumerate(selections.items()):
            panel = panels[0, index].subgridspec(1, 2, width_ratios=(24, 1), wspace=0.10)
            is_paired = stem == "paired_comparison"
            title = f"{names[algorithm]} Paired Path" if is_paired else None
            missing = "No jointly successful episode" if is_paired else "No successful evaluation episode"
            _draw_best_path(figure, figure.add_subplot(panel[0, 0]), record, names[algorithm],
                            presentation_positions=positions, title=title, missing_label=missing,
                            colorbar_ax=figure.add_subplot(panel[0, 1]))
        if heading:
            figure.suptitle(heading, y=0.955, fontsize=18, fontweight="bold")
        footer = "\n".join(line for line in (caption, selection_caption, legend) if line)
        figure.text(0.5, 0.035, footer, ha="center", fontsize=9, linespacing=1.45)
        for extension in ("png", "pdf"):
            path = directory / f"{stem}.{extension}"
            figure.savefig(path, dpi=240)
            outputs.append(path)
        figure.clear()
    return tuple(outputs)


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
