"""Gaussian Noise 실험 집계 CSV를 평균±표준편차 그래프로 시각화한다."""

import argparse
import csv
from pathlib import Path

import matplotlib

import config


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / config.RESULTS_DIR
    / "noise_sweep_summary.csv"
)

METRICS = (
    ("eval_success_rate", "Evaluation success rate", "higher is better"),
    ("eval_average_steps", "Evaluation average steps", "lower is better"),
    ("eval_average_total_reward", "Evaluation average reward", "higher is better"),
    ("train_success_rate", "Training success rate", "higher is better"),
    ("train_average_steps", "Training average steps", "lower is better"),
    ("train_average_final_distance", "Training final distance", "lower is better"),
)


def load_summary(input_path):
    """Noise 집계 CSV를 읽어 숫자 값으로 변환한다."""
    with Path(input_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key, value in row.items():
            row[key] = float(value) if value != "" else None
    return rows


def plot_noise_summary(rows, output_path, show=False):
    """Noise 수준별 평균과 Seed 간 표준편차를 6개 지표로 그린다."""
    if not rows:
        raise ValueError("rows must not be empty.")
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda row: row["noise_std"])
    noise_stds = [row["noise_std"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (metric, title, direction) in zip(axes, METRICS):
        means = [row[f"{metric}_mean"] for row in rows]
        errors = [row[f"{metric}_std"] for row in rows]
        ax.errorbar(
            noise_stds,
            means,
            yerr=errors,
            color="#2369a1",
            marker="o",
            linewidth=2,
            markersize=6,
            capsize=4,
        )
        ax.set_title(
            f"{title}\n{direction}",
            loc="left",
            fontweight="bold",
            fontsize=11,
            color="#202020",
        )
        ax.set_xlabel("Noise std (dBm)")
        ax.set_xticks(noise_stds)
        ax.grid(color="#dedede", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if metric.endswith("success_rate"):
            ax.set_ylim(0, 1.05)

    seed_count = int(rows[0]["seed_count"])
    train_episodes = int(rows[0]["train_episodes"])
    eval_episodes = int(rows[0]["eval_episodes"])
    fig.suptitle(
        "Gaussian Noise sensitivity of RSSI Q-Learning\n"
        f"mean ± sample std across {seed_count} seeds "
        f"(train {train_episodes}, eval {eval_episodes} episodes per seed)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=3.0, w_pad=2.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot Gaussian Noise sweep results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    output_path = args.output or args.input.with_suffix(".png")
    rows = load_summary(args.input)
    saved = plot_noise_summary(rows, output_path, show=args.show)
    print(f"Noise sweep plot saved to: {saved}")


if __name__ == "__main__":
    main()
