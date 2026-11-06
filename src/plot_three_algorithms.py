"""compare_three_algorithms 결과로 성능 비교 그래프와 경로 그림을 만든다.

python -m src.plot_three_algorithms --input-dir <결과 폴더> 명령으로 실행한다.
입력: three_algorithm_summary.csv, three_algorithm_paths.json (원본은 수정하지 않는다)
출력:
- three_algorithm_comparison.png/.pdf: 지표 4개(성공률, 전체 평균 Step, 성공 시 Path Ratio,
  평균 최종 거리) 2×2 막대그래프. 막대는 Run 평균, 오차 막대는 Run 간 표본 표준편차.
- three_algorithm_paths.png/.pdf: 알고리즘별 최적 성공 경로(위)와 대표 경로(아래).
  실제 기록 좌표를 그대로 그린다.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

from src.plot_grid_comparison import METRICS, PLOTTED_METRICS, TARGET_MODE_TITLES


# (알고리즘, 이름, 색)
ALGORITHMS = (
    ("q_learning", "Q-Learning", "#2369a1"),
    ("q_learning_relative", "Q-Learning\n(relative)", "#7b4ea3"),
    ("greedy", "Greedy", "#2e8b57"),
    ("random", "Random", "#d97706"),
)
METRIC_NAMES = tuple(metric for metric, *_ in METRICS)
INTEGER_FIELDS = ("grid_size", "max_steps", "run_count", "train_episodes", "eval_episodes")


def _number(value):
    if value in ("", None):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"유한하지 않은 값: {value!r}")
    return number


def load_summary(path):
    """요약 CSV를 읽어 숫자 열을 float/int로 바꾼다."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"요약 CSV가 비어 있습니다: {path}")
    parsed = []
    for row in rows:
        item = {"target_mode": row["target_mode"], "algorithm": row["algorithm"]}
        for field in INTEGER_FIELDS:
            item[field] = int(row[field])
        for metric in METRIC_NAMES:
            item[f"{metric}_mean"] = _number(row[f"{metric}_mean"])
            item[f"{metric}_std"] = _number(row[f"{metric}_std"])
            item[f"{metric}_run_count"] = int(row[f"{metric}_run_count"])
        parsed.append(item)
    return parsed


def _save(fig, output_base):
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = output_base.with_suffix(suffix)
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        paths.append(path)
    return tuple(paths)


def plot_metrics(rows, output_base):
    """알고리즘을 x축 막대로 두고 지표마다 별도 패널로 그린다."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_algorithm = {row["algorithm"]: row for row in rows}
    present = [entry for entry in ALGORITHMS if entry[0] in by_algorithm]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
        try:
            for ax, (metric, title, ylabel, scale, _) in zip(axes.flat, PLOTTED_METRICS):
                top = 0.0
                for index, (algorithm, label, color) in enumerate(present):
                    row = by_algorithm[algorithm]
                    mean, std = row[f"{metric}_mean"], row[f"{metric}_std"]
                    if mean is None:
                        ax.text(index, 0.02, "NA", ha="center", va="bottom", fontsize=8,
                                color="#555555", transform=ax.get_xaxis_transform())
                        continue
                    value = mean * scale
                    show_error = std is not None and row["run_count"] > 1 and std > 0
                    error = std * scale if show_error else 0.0
                    ax.bar(index, value, width=0.6, color=color, zorder=2)
                    if show_error:
                        ax.errorbar(index, value, yerr=error, fmt="none", ecolor="#333333",
                                    capsize=3, linewidth=1.0, zorder=3)
                    top = max(top, value + error)
                    label_text = f"{value:.2f}" if metric == "path_ratio_on_success" else f"{value:.1f}"
                    valid_runs = row[f"{metric}_run_count"]
                    if valid_runs < row["run_count"]:
                        # 성공한 Run만 평균한 값이면 유효 Run 수를 함께 적는다.
                        label_text += f"\n(n={valid_runs})"
                    ax.annotate(label_text, (index, value + error), xytext=(0, 3),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=8, color="#222222")
                ax.set_xticks(range(len(present)), [label for _, label, _ in present])
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_ylim(0, (top or 1.0) * 1.18)
                if metric == "path_ratio_on_success":
                    # 1.0 = 8방향 최단 경로로 도착
                    ax.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
                ax.grid(axis="y", alpha=0.25, zorder=0)
                ax.spines[["top", "right"]].set_visible(False)
            first = rows[0]
            fig.suptitle(
                f"{TARGET_MODE_TITLES[first['target_mode']]} | "
                f"{first['grid_size']}×{first['grid_size']} grid, 8 directions, "
                f"max {first['max_steps']} steps",
                fontsize=12, y=1.0,
            )
            runs = sorted({row["run_count"] for row in rows})
            fig.text(0.5, 0.01,
                     f"Bars: mean across runs (runs={', '.join(map(str, runs))}). "
                     "Error bars: sample SD across runs; omitted for a single run. "
                     "NA: no successful episode. Dashed line: shortest path (ratio 1.0).",
                     ha="center", fontsize=7.5, color="#444444")
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
            return _save(fig, output_base)
        finally:
            plt.close(fig)


def plot_paths(paths, output_base):
    """알고리즘별 최적(위)·대표(아래) 성공 경로를 Grid 위에 그린다."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_size = paths["grid_size"]
    start = paths["start"]
    kinds = (("best", "Best"), ("representative", "Representative"))
    present = [entry for entry in ALGORITHMS if entry[0] in paths["algorithms"]]
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42,
                         "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, len(present), figsize=(3.4 * len(present), 7.6),
                                 squeeze=False)
        try:
            for column, (algorithm, label, color) in enumerate(present):
                label = label.replace("\n", " ")
                selected = paths["algorithms"][algorithm]
                for row_index, (kind, kind_label) in enumerate(kinds):
                    ax = axes[row_index][column]
                    entry = selected.get(kind)
                    ax.set_xlim(-0.5, grid_size - 0.5)
                    # 환경의 UP이 y-1이므로 y축을 뒤집어 화면 위쪽이 UP이 되게 한다.
                    ax.set_ylim(grid_size - 0.5, -0.5)
                    ax.set_aspect("equal")
                    ax.set_xticks(range(0, grid_size, max(1, grid_size // 5)))
                    ax.set_yticks(range(0, grid_size, max(1, grid_size // 5)))
                    ax.grid(alpha=0.2)
                    if entry is None:
                        ax.set_title(f"{label} — {kind_label}\n(no successful episode)")
                        continue
                    xs = [position[0] for position in entry["path"]]
                    ys = [position[1] for position in entry["path"]]
                    ax.plot(xs, ys, color=color, linewidth=1.3, alpha=0.85, zorder=2)
                    ax.scatter([start[0]], [start[1]], marker="o", s=40, color="#222222",
                               zorder=3, label="Start")
                    ax.scatter([entry["target"][0]], [entry["target"][1]], marker="*", s=110,
                               color="#c62828", zorder=3, label="Target")
                    ax.set_title(
                        f"{label} — {kind_label}\n"
                        f"run {entry['run']}, ep {entry['episode']}: "
                        f"{entry['steps']} steps / {entry['shortest_path_steps']} "
                        f"(ratio {entry['path_ratio']:.2f})"
                    )
            handles, labels = [], []
            for ax in axes.flat:
                for handle, text in zip(*ax.get_legend_handles_labels()):
                    if text not in labels:
                        handles.append(handle)
                        labels.append(text)
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False,
                           bbox_to_anchor=(0.5, 1.0))
            fig.text(0.5, 0.005,
                     "Best: lowest path ratio among successful episodes. Representative: successful "
                     "episode whose path ratio is closest to the mean path ratio across runs.",
                     ha="center", fontsize=7.5, color="#444444")
            fig.tight_layout(rect=(0, 0.02, 1, 0.96))
            return _save(fig, output_base)
        finally:
            plt.close(fig)


def make_three_algorithm_report(input_dir, output_dir=None):
    """요약 CSV와 경로 JSON을 읽어 두 그림을 저장하고 경로를 반환한다."""
    input_dir = Path(input_dir)
    output_dir = input_dir if output_dir is None else Path(output_dir)
    rows = load_summary(input_dir / "three_algorithm_summary.csv")
    with (input_dir / "three_algorithm_paths.json").open(encoding="utf-8") as file:
        paths = json.load(file)
    figure_paths = (
        *plot_metrics(rows, output_dir / "three_algorithm_comparison"),
        *plot_paths(paths, output_dir / "three_algorithm_paths"),
    )
    print("Figures saved to:\n" + "\n".join(str(path) for path in figure_paths))
    return figure_paths


def main():
    parser = argparse.ArgumentParser(description="Plot the three-algorithm comparison.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    make_three_algorithm_report(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
