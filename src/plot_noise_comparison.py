"""네 가지 비교 지표를 Run별 평균과 표본 표준편차 오차 막대로 시각화한다."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

import config


METRICS = (
    ("success_rate", "Success rate", "Success rate"),
    ("average_steps", "Average steps (all episodes)", "Steps"),
    ("average_steps_on_success", "Average steps on success", "Steps"),
    ("average_final_distance", "Average final distance", "Distance (grid units)"),
)
ALGORITHMS = (
    ("q_learning", "Q-Learning", "#2369a1", "o"),
    ("random", "Random", "#d97706", "s"),
)


def _number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def load_summary(input_path):
    """알고리즘 이름은 문자열로 유지하고, 빈 값이나 유한하지 않은 지표는 None으로 읽는다."""
    with Path(input_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return [
        {key: value if key == "algorithm" else _number(value) for key, value in row.items()}
        for row in rows
    ]


def plot_noise_comparison(rows, output_path=None):
    """GUI 없이 2×2 그래프를 저장하며, 평균의 결측값은 0 대신 빈 구간으로 표시한다.

    표준편차가 없으면 오차 막대를 표시하지 않는다. 성공이 없는 Run은
    성공 시 평균 Step 계산에서 제외한다. 데이터 값은 임의로 보정하지 않는다.
    """
    if not rows:
        raise ValueError("rows must not be empty.")
    rows = [row for row in rows if _number(row.get("noise_std")) is not None]
    if not rows:
        raise ValueError("rows must contain a finite noise_std.")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / config.RESULTS_DIR / "noise_comparison.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    try:
        for ax, (metric, title, ylabel) in zip(axes.flat, METRICS):
            any_values = False
            for algorithm, label, color, marker in ALGORITHMS:
                by_noise = {float(row["noise_std"]): row for row in rows
                            if row.get("algorithm") == algorithm}
                means, error_x, error_means, errors = [], [], [], []
                for noise in noise_stds:
                    row = by_noise.get(noise, {})
                    mean = _number(row.get(f"{metric}_mean"))
                    std = _number(row.get(f"{metric}_std"))
                    means.append(mean if mean is not None else math.nan)
                    if mean is not None:
                        any_values = True
                        if std is not None and std >= 0:
                            error_x.append(noise)
                            error_means.append(mean)
                            errors.append(std)
                ax.plot(noise_stds, means, color=color, marker=marker, label=label, linewidth=2)
                if error_x:
                    ax.errorbar(error_x, error_means, yerr=errors, fmt="none",
                                ecolor=color, capsize=4)
            if not any_values:
                ax.text(0.5, 0.5, "No valid runs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel("Noise standard deviation (dB)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(noise_stds)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Q-Learning vs Random: mean ± sample standard deviation across runs")
        fig.text(0.5, 0.015, "Successful-step metrics use only runs with successes; missing values remain gaps.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot Q-Learning/Random noise comparison.")
    parser.add_argument("--input", type=Path,
                        default=root / config.RESULTS_DIR / "noise_comparison_summary.csv")
    parser.add_argument("--output", type=Path,
                        default=root / config.RESULTS_DIR / "noise_comparison.png")
    args = parser.parse_args()
    saved = plot_noise_comparison(load_summary(args.input), args.output)
    print(f"Noise comparison plot saved to: {saved}")


if __name__ == "__main__":
    main()
