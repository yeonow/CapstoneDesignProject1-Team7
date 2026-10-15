"""검증된 논문용 표에서 Run 단위 95% 신뢰구간 그래프를 PNG·PDF로 생성한다."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

import config
from src.plot_noise_comparison import ALGORITHMS, METRICS


FIGURES = (
    ("success_rate_percent", "success_rate_comparison", "noise_success_rate", "Success rate", "Success rate (%)"),
    ("average_steps", "average_steps_comparison", "noise_average_steps", METRICS[1][1], "Steps"),
    ("average_steps_on_success", "average_success_steps_comparison", "noise_success_steps", METRICS[2][1], "Steps"),
    ("average_final_distance", "final_distance_comparison", "noise_final_distance", METRICS[3][1], "Distance (grid units)"),
)


def load_performance_table(path):
    """분석 출력을 읽고 결측값은 None으로 유지하며 원본을 보정하지 않는다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"분석 결과가 없습니다: {path}. 먼저 python -m src.analyze_paper_results 를 실행하세요.")
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"algorithm", "noise_std", "run_count", "success_count_total"}
        required.update(f"{metric}_{suffix}" for metric, *_ in FIGURES
                        for suffix in ("mean", "ci95_low", "ci95_high", "valid_run_count"))
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"논문용 표의 필수 열이 없습니다: {path}")
        rows = []
        for raw in reader:
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"논문용 표의 열 개수가 일치하지 않습니다: {path}")
            row = {}
            for key, value in raw.items():
                if key == "algorithm":
                    if value not in {entry[0] for entry in ALGORITHMS}:
                        raise ValueError(f"알 수 없는 알고리즘: {value}")
                    row[key] = value
                elif value == "":
                    row[key] = None
                else:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(f"유한하지 않은 그래프 입력값: {path}/{key}")
                    row[key] = number
            rows.append(row)
    keys = [(row["noise_std"], row["algorithm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"논문용 표에 중복된 알고리즘·Noise 행이 있습니다: {path}")
    return rows


def _add_ci(ax, x, row, metric, color):
    """유효 Run이 둘 이상이고 구간이 있을 때만 비대칭 CI 오차 막대를 그린다."""
    mean, low, high = (row.get(f"{metric}_{suffix}") for suffix in ("mean", "ci95_low", "ci95_high"))
    if mean is None or low is None or high is None or row.get(f"{metric}_valid_run_count", 0) < 2:
        return
    if not low <= mean <= high:
        raise ValueError(f"평균을 포함하지 않는 신뢰구간: {metric}")
    ax.errorbar([x], [mean], yerr=[[mean - low], [high - mean]],
                fmt="none", ecolor=color, capsize=4, linewidth=1.3)


def plot_metric(rows, metric, title, ylabel, output_base, noise=False):
    """한 지표의 평균과 CI를 그리며 결측 Noise 지점을 연결하거나 보간하지 않는다."""
    if not rows:
        raise ValueError("그래프를 생성할 데이터가 없습니다.")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "lines.linewidth": 1.7, "pdf.fonttype": 42, "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        try:
            any_values = False
            if noise:
                levels = sorted({row["noise_std"] for row in rows if row["noise_std"] is not None})
                if not levels:
                    raise ValueError("Noise 그래프에 noise_std 값이 없습니다.")
                for algorithm, label, color, marker in ALGORITHMS:
                    by_noise = {row["noise_std"]: row for row in rows if row["algorithm"] == algorithm}
                    means = [by_noise.get(level, {}).get(f"{metric}_mean") for level in levels]
                    ax.plot(levels, [value if value is not None else math.nan for value in means],
                            color=color, marker=marker, label=label)
                    for level in levels:
                        row = by_noise.get(level, {})
                        any_values |= row.get(f"{metric}_mean") is not None
                        _add_ci(ax, level, row, metric, color)
                ax.set_xlabel("RSSI noise standard deviation (dB)")
                ax.set_xticks(levels)
            else:
                by_algorithm = {row["algorithm"]: row for row in rows}
                labels = []
                for index, (algorithm, label, color, _) in enumerate(ALGORITHMS):
                    row = by_algorithm.get(algorithm, {})
                    mean = row.get(f"{metric}_mean")
                    ax.bar(index, mean if mean is not None else math.nan, color=color, label=label, width=0.55)
                    labels.append(f"{label}\nvalid runs={int(row.get(f'{metric}_valid_run_count') or 0)}")
                    if mean is None:
                        ax.text(index, 0.05, "NA", transform=ax.get_xaxis_transform(), ha="center")
                    else:
                        any_values = True
                    _add_ci(ax, index, row, metric, color)
                ax.set_xticks(range(len(ALGORITHMS)), labels)
            if not any_values:
                ax.text(0.5, 0.5, "No valid runs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend()
            caption = "Error bars: 95% Student's t CI across runs (not SD); unavailable for n < 2."
            if metric == "average_steps_on_success":
                counts = "; ".join(f"{row['algorithm']}" + (f" @ {row['noise_std']:g} dB" if noise else "")
                                   + f": {int(row['success_count_total'])}" for row in rows)
                caption += "\nSuccessful episodes: " + counts
            fig.text(0.5, 0.015, caption, ha="center", fontsize=7, wrap=True)
            fig.tight_layout(rect=(0, 0.12 if metric == "average_steps_on_success" else 0.07, 1, 1))
            paths = []
            for suffix in (".png", ".pdf"):
                path = output_base.with_suffix(suffix)
                fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
                paths.append(path)
        finally:
            plt.close(fig)
    return tuple(paths)


def plot_paper_results(input_dir=None, output_dir=None):
    """독립 CLI에서 기존 분석 표만 읽어 기본 4종·Noise 4종 그래프를 저장한다."""
    root = Path(__file__).resolve().parents[1]
    results = root / config.RESULTS_DIR
    input_dir = root / input_dir if input_dir is not None else results / "paper_tables"
    output_dir = root / output_dir if output_dir is not None else results / "paper_figures"
    basic = load_performance_table(input_dir / "algorithm_performance.csv")
    noise_rows = load_performance_table(input_dir / "noise_performance.csv")
    if not basic and not noise_rows:
        raise ValueError("분석 가능한 표가 없습니다. 분석 보고서의 검증 실패 내용을 확인하세요.")
    paths = []
    for noise, rows in ((False, basic), (True, noise_rows)):
        if not rows:
            print(f"{'Noise' if noise else '기본'} 비교는 검증된 분석 행이 없어 그래프를 생략합니다.")
            continue
        for metric, basic_name, noise_name, title, ylabel in FIGURES:
            paths.extend(plot_metric(rows, metric, title, ylabel,
                                     output_dir / (noise_name if noise else basic_name), noise=noise))
    print(f"논문용 그래프 {len(paths)}개 파일 저장: {output_dir}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Plot validated paper results with 95% confidence intervals.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        plot_paper_results(args.input_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
