"""Reward sweep 결과 CSV를 지표별 막대그래프로 시각화한다.

프로젝트 루트에서 다음과 같이 실행한다.

    python -m src.plot_sweep
    python -m src.plot_sweep --input results/reward_sweep_results.csv --show
"""

import argparse
import csv
from pathlib import Path

import matplotlib

import config


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / config.RESULTS_DIR
    / "reward_sweep_results.csv"
)

# (CSV 컬럼, 그래프 제목, 높을수록 좋은지)
METRICS = (
    ("train_success_rate", "Train success rate", True),
    ("train_average_steps", "Train avg steps", False),
    ("train_average_total_reward", "Train avg total reward", True),
    ("eval_success_rate", "Eval success rate", True),
    ("eval_average_steps", "Eval avg steps", False),
    ("eval_failure_count", "Eval failures", False),
)

BASELINE_NAME = "baseline"
BAR_COLOR = "#2a78d6"
BASELINE_COLOR = "#8a8984"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
SURFACE = "#fcfcfb"


def load_results(input_path):
    """CSV를 읽어 숫자 컬럼을 float로 변환한 dict 리스트를 반환한다."""
    with Path(input_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            if key != "name":
                row[key] = float(value)
    return rows


def _format_value(column, value):
    if column.endswith("_rate"):
        return f"{value:.3f}"
    if value == int(value):
        return f"{int(value)}"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"


def plot_results(rows, output_path=None, show=False):
    """지표마다 하나의 패널을 그리는 small multiples 그림을 만든다."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    names = [row["name"] for row in rows]
    colors = [
        BASELINE_COLOR if name == BASELINE_NAME else BAR_COLOR
        for name in names
    ]
    baseline = next((row for row in rows if row["name"] == BASELINE_NAME), None)

    ncols = 3
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 0.45 * len(rows) * nrows + 1.6 * nrows),
        facecolor=SURFACE,
    )
    axes = axes.flatten()

    for ax, (column, title, higher_is_better) in zip(axes, METRICS):
        values = [row[column] for row in rows]
        positions = range(len(rows))
        ax.set_facecolor(SURFACE)
        ax.barh(positions, values, height=0.6, color=colors)

        # 기준선(baseline) 값을 점선으로 표시해 차이를 비교하기 쉽게 한다.
        if baseline is not None:
            ax.axvline(
                baseline[column], color=BASELINE_COLOR, linewidth=1, linestyle="--"
            )

        if column.endswith("_rate"):
            span = 1
        else:
            span = max(max(values), 0) - min(min(values), 0) or 1
        for position, value in zip(positions, values):
            ax.text(
                value + span * 0.02 if value >= 0 else value - span * 0.02,
                position,
                _format_value(column, value),
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
                color=TEXT_PRIMARY,
            )

        arrow = "higher is better" if higher_is_better else "lower is better"
        ax.set_title(
            f"{title}\n", loc="left", fontsize=11, color=TEXT_PRIMARY,
            fontweight="bold",
        )
        ax.text(
            0, 1.02, arrow, transform=ax.transAxes, fontsize=8,
            color=TEXT_SECONDARY,
        )
        ax.set_yticks(list(positions))
        ax.set_yticklabels(names, fontsize=9, color=TEXT_SECONDARY)
        ax.invert_yaxis()
        if column.endswith("_rate"):
            ax.set_xlim(0, 1.15)
        else:
            low = min(min(values), 0)
            ax.set_xlim(low - span * 0.05 if low < 0 else 0, max(max(values), 0) + span * 0.2)
        if column.endswith("_count"):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8, colors=TEXT_SECONDARY, length=0)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(METRICS):]:
        ax.set_visible(False)

    episodes = rows[0]
    fig.suptitle(
        "Reward sweep comparison "
        f"(train {int(episodes['train_episodes'])} / "
        f"eval {int(episodes['eval_episodes'])} episodes)",
        x=0.01, ha="left", fontsize=13, color=TEXT_PRIMARY, fontweight="bold",
    )
    fig.text(
        0.01, 0.005,
        "Gray bar and dashed line = baseline. Total reward depends on each "
        "config's reward scale, so compare it with care.",
        fontsize=8, color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot Reward sweep results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Sweep result CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: <input>.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the figure in a window as well as saving it.",
    )
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".png")
    rows = load_results(args.input)
    saved = plot_results(rows, output_path=output_path, show=args.show)
    print(f"Sweep plot saved to: {saved}")


if __name__ == "__main__":
    main()
