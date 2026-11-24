"""네 가지 비교 지표를 Run별 평균과 표본 표준편차 오차 막대로 시각화한다."""

import argparse
import math
from pathlib import Path

import matplotlib

import config


METRICS = (
    ("success_rate", "Success rate", "Success rate (fraction)"),
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
    """빈 성공 Step만 결측으로 읽고 손상된 숫자를 결측으로 숨기지 않는다."""
    from src.analyze_paper_results import read_result_csv
    from src.compare_noise_sweep import METRICS as metric_names, NOISE_SUMMARY_FIELDS

    rows = read_result_csv(input_path, NOISE_SUMMARY_FIELDS)
    keys = [(row["noise_std"], row["algorithm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("중복된 Noise/알고리즘 Summary 행")
    for row in rows:
        if row["run_count"] < 1 or row["eval_episodes"] < 1:
            raise ValueError("Run/평가 Episode 수는 양수여야 합니다.")
        for metric in metric_names:
            mean, sd, count = (row[f"{metric}_{suffix}"] for suffix in ("mean", "std", "run_count"))
            if count is None or not 0 <= count <= row["run_count"]:
                raise ValueError(f"유효 Run 수 오류: {metric}")
            if metric != "average_steps_on_success" and count != row["run_count"]:
                raise ValueError(f"전체 Run 지표의 표본 수 누락: {metric}")
            if (mean is None) != (count == 0) or (sd is None) != (count == 0):
                raise ValueError(f"결측값과 유효 Run 수 불일치: {metric}")
        if not 0 <= row["success_rate_mean"] <= 1:
            raise ValueError("성공률은 0~1 비율이어야 합니다.")
    return rows


def plot_noise_comparison(rows, output_path=None, *, success_percent=False, dpi=160):
    """GUI 없이 2×2 그래프를 저장하며, 평균의 결측값은 0 대신 빈 구간으로 표시한다.

    표준편차가 없으면 오차 막대를 표시하지 않는다. 성공이 없는 Run은
    성공 시 평균 Step 계산에서 제외한다. 데이터 값은 임의로 보정하지 않는다.
    """
    if not rows:
        raise ValueError("rows must not be empty.")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    keys = set()
    for row in rows:
        noise = _number(row.get("noise_std"))
        if noise is None or noise < 0 or row.get("algorithm") not in {a[0] for a in ALGORITHMS}:
            raise ValueError("rows must contain a nonnegative finite noise_std and known algorithm.")
        key = noise, row["algorithm"]
        if key in keys:
            raise ValueError("duplicate noise/algorithm row")
        keys.add(key)
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
            scale = 100 if success_percent and metric == "success_rate" else 1
            for algorithm, label, color, marker in ALGORITHMS:
                by_noise = {float(row["noise_std"]): row for row in rows
                            if row.get("algorithm") == algorithm}
                means, error_x, error_means, errors = [], [], [], []
                for noise in noise_stds:
                    row = by_noise.get(noise, {})
                    mean = _number(row.get(f"{metric}_mean"))
                    std = _number(row.get(f"{metric}_std"))
                    mean = mean * scale if mean is not None else None
                    std = std * scale if std is not None else None
                    means.append(mean if mean is not None else math.nan)
                    if mean is not None:
                        any_values = True
                        count = row.get(f"{metric}_run_count", row.get("run_count"))
                        if std is not None and std >= 0 and (count is None or count >= 2):
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
            ax.set_ylabel("Success rate (%)" if scale == 100 else ylabel)
            ax.set_xticks(noise_stds)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Q-Learning vs Random: mean ± sample standard deviation across runs")
        fig.text(0.5, 0.015, "Error bars: sample SD (not CI); omitted for n < 2.\n"
                 "Successful-step metrics use only runs with successes; missing values remain gaps.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
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
    parser.add_argument("--percent", action="store_true", help="Display success rate and SD in percent units.")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    saved = plot_noise_comparison(load_summary(args.input), args.output,
                                  success_percent=args.percent, dpi=args.dpi)
    print(f"Noise comparison plot saved to: {saved}")


if __name__ == "__main__":
    main()
