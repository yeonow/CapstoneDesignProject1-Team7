"""Grid 크기별 Q-Learning/Random 비교 결과를 숫자 표(CSV)와 그래프(PNG·PDF)로 만든다.

python -m src.plot_grid_comparison 명령으로 실행한다. 입력은 compare_grid_sizes가 만든
grid_comparison_summary.csv이며, 원본 CSV는 수정하지 않는다.
- grid_comparison_table.csv: 논문 표에 바로 옮길 수 있게 반올림하고 성공률을 %로 바꾼 표
- grid_comparison.png / .pdf: 지표 4개를 각자의 y축으로 그린 2×2 막대그래프
  (Target이 Episode마다 바뀌면 거리가 달라지므로 성공 시 Step 대신 최단 경로 대비 Step 비율을 그린다)
오차 막대는 Run 간 표본 표준편차(SD)이며, Run이 하나면 그리지 않는다.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

import config
from src.plot_noise_comparison import ALGORITHMS


# (지표, 그래프 제목, y축 이름, 배율, 그래프에 그릴지 여부)
METRICS = (
    ("success_rate", "Success rate", "Success rate (%)", 100.0, True),
    ("average_steps", "Average steps (all episodes)", "Steps", 1.0, True),
    ("average_steps_on_success", "Average steps on success", "Steps", 1.0, False),
    ("path_ratio_on_success", "Path ratio on success (steps / shortest path)",
     "Steps / shortest path", 1.0, True),
    ("average_final_distance", "Average final distance", "Distance (grid units)", 1.0, True),
)
PLOTTED_METRICS = tuple(metric for metric in METRICS if metric[4])
TABLE_FIELDS = (
    "target_mode", "grid", "algorithm", "max_steps", "runs", "train_episodes", "eval_episodes",
    "success_rate_percent_mean", "success_rate_percent_std",
    "average_steps_mean", "average_steps_std",
    "average_steps_on_success_mean", "average_steps_on_success_std",
    "average_steps_on_success_valid_runs",
    "path_ratio_on_success_mean", "path_ratio_on_success_std",
    "path_ratio_on_success_valid_runs",
    "average_final_distance_mean", "average_final_distance_std",
)
TARGET_MODE_TITLES = {
    "fixed": "Fixed target",
    "random": "Random target per episode",
    "split": "Unseen targets (checkerboard train/eval split)",
}
ALGORITHM_LABELS = {algorithm: label for algorithm, label, _, _ in ALGORITHMS}


def _number(value):
    if value in ("", None):
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"유한하지 않은 값: {value!r}")
    return number


def _default_dir(path):
    root = Path(__file__).resolve().parents[1]
    return root / (config.RESULTS_DIR if path is None else path)


def load_grid_summary(path):
    """grid_comparison_summary.csv를 읽고 빈 값은 None으로 둔다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Grid 비교 결과가 없습니다: {path}. 먼저 python -m src.compare_grid_sizes 를 실행하세요."
        )
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"target_mode", "grid_size", "max_steps", "algorithm", "run_count",
                    "train_episodes", "eval_episodes"}
        required.update(f"{metric}_{suffix}" for metric, *_ in METRICS
                        for suffix in ("mean", "std", "run_count"))
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Grid 비교 결과에 필수 열이 없습니다: {sorted(missing)}")
        rows = []
        for raw in reader:
            if raw["algorithm"] not in ALGORITHM_LABELS:
                raise ValueError(f"알 수 없는 알고리즘: {raw['algorithm']}")
            if raw["target_mode"] not in TARGET_MODE_TITLES:
                raise ValueError(f"알 수 없는 Target 모드: {raw['target_mode']}")
            row = {"algorithm": raw["algorithm"], "target_mode": raw["target_mode"]}
            for key in required - {"algorithm", "target_mode"}:
                row[key] = _number(raw[key])
            for key in ("grid_size", "max_steps", "run_count", "train_episodes", "eval_episodes"):
                row[key] = int(row[key])
            rows.append(row)
    if not rows:
        raise ValueError(f"Grid 비교 결과에 데이터가 없습니다: {path}")
    keys = [(row["grid_size"], row["algorithm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Grid·알고리즘 조합이 중복되었습니다.")
    if len({row["target_mode"] for row in rows}) != 1:
        raise ValueError("한 요약 파일에 여러 Target 모드가 섞여 있습니다.")
    return sorted(rows, key=lambda row: (row["grid_size"], row["algorithm"]))


def _rounded(value, scale=1.0):
    return None if value is None else round(value * scale, 3)


def build_table(rows):
    """논문 표 형식(성공률 %, 소수 셋째 자리)으로 변환한다."""
    table = []
    for row in rows:
        size = row["grid_size"]
        entry = {
            "target_mode": row["target_mode"],
            "grid": f"{size}x{size}",
            "algorithm": ALGORITHM_LABELS[row["algorithm"]],
            "max_steps": row["max_steps"], "runs": row["run_count"],
            "train_episodes": row["train_episodes"], "eval_episodes": row["eval_episodes"],
            "average_steps_on_success_valid_runs": int(row["average_steps_on_success_run_count"]),
            "path_ratio_on_success_valid_runs": int(row["path_ratio_on_success_run_count"]),
        }
        for metric, _, _, scale, _ in METRICS:
            name = "success_rate_percent" if metric == "success_rate" else metric
            entry[f"{name}_mean"] = _rounded(row[f"{metric}_mean"], scale)
            entry[f"{name}_std"] = _rounded(row[f"{metric}_std"], scale)
        table.append(entry)
    return table


def save_table(table, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(table)
    return output_path


def format_table(table):
    """터미널에서 바로 읽을 수 있는 고정폭 표 문자열을 만든다."""
    # Windows 콘솔(cp949)에서도 깨지지 않도록 ± 대신 +/-를 쓴다.
    def cell(mean, std):
        if mean is None:
            return "NA"
        return f"{mean:.2f} +/- {std:.2f}" if std is not None else f"{mean:.2f}"

    def success_cell(row, metric):
        text = cell(row[f"{metric}_mean"], row[f"{metric}_std"])
        valid = row[f"{metric}_valid_runs"]
        # 성공한 Run만 평균하므로 일부 Run만 유효하면 그 수를 함께 보여준다.
        return text if valid in (0, row["runs"]) else f"{text} (n={valid})"

    header = ("Grid", "Algorithm", "Runs", "Success rate (%)", "Avg steps",
              "Avg steps on success", "Path ratio on success", "Final distance")
    lines = [header]
    for row in table:
        lines.append((
            row["grid"], row["algorithm"], str(row["runs"]),
            cell(row["success_rate_percent_mean"], row["success_rate_percent_std"]),
            cell(row["average_steps_mean"], row["average_steps_std"]),
            success_cell(row, "average_steps_on_success"),
            success_cell(row, "path_ratio_on_success"),
            cell(row["average_final_distance_mean"], row["average_final_distance_std"]),
        ))
    widths = [max(len(line[index]) for line in lines) for index in range(len(header))]
    text = [" | ".join(value.ljust(width) for value, width in zip(line, widths)) for line in lines]
    text.insert(1, "-+-".join("-" * width for width in widths))
    return "\n".join(text)


def plot_grid_comparison(rows, output_base):
    """Grid 크기를 x축, 알고리즘을 막대로 두고 지표마다 별도 패널(y축 하나)로 그린다."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    grid_sizes = sorted({row["grid_size"] for row in rows})
    by_key = {(row["grid_size"], row["algorithm"]): row for row in rows}
    width = 0.36
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
        try:
            for ax, (metric, title, ylabel, scale, _) in zip(axes.flat, PLOTTED_METRICS):
                top = 0.0
                for offset, (algorithm, label, color, _) in zip((-width / 2, width / 2), ALGORITHMS):
                    for index, size in enumerate(grid_sizes):
                        row = by_key.get((size, algorithm), {})
                        mean = row.get(f"{metric}_mean")
                        std = row.get(f"{metric}_std")
                        x = index + offset
                        if mean is None:
                            ax.text(x, 0.02, "NA", ha="center", va="bottom", fontsize=8,
                                    color="#555555", transform=ax.get_xaxis_transform())
                            continue
                        value = mean * scale
                        show_error = std is not None and row.get("run_count", 0) > 1 and std > 0
                        error = std * scale if show_error else 0.0
                        ax.bar(x, value, width=width * 0.94, color=color,
                               label=label if index == 0 else None, zorder=2)
                        if show_error:
                            ax.errorbar(x, value, yerr=error, fmt="none", ecolor="#333333",
                                        capsize=3, linewidth=1.0, zorder=3)
                        top = max(top, value + error)
                        label_text = f"{value:.2f}" if metric == "path_ratio_on_success" else f"{value:.1f}"
                        valid_runs = int(row.get(f"{metric}_run_count") or 0)
                        if valid_runs < row.get("run_count", 0):
                            # 성공한 Run만 평균한 값이면 유효 Run 수를 함께 적는다.
                            label_text += f"\n(n={valid_runs})"
                        ax.annotate(label_text, (x, value + error), xytext=(0, 3),
                                    textcoords="offset points", ha="center", va="bottom",
                                    fontsize=8, color="#222222")
                ax.set_xticks(range(len(grid_sizes)), [f"{size}×{size}" for size in grid_sizes])
                ax.set_xlabel("Grid size")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.set_ylim(0, (top or 1.0) * 1.18)
                if metric == "path_ratio_on_success":
                    # 1.0 = 최단 경로로 도착
                    ax.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--", zorder=1)
                ax.grid(axis="y", alpha=0.25, zorder=0)
                ax.spines[["top", "right"]].set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.suptitle(TARGET_MODE_TITLES[rows[0]["target_mode"]], fontsize=12, y=1.0)
            fig.legend(handles, labels, loc="upper center", ncol=len(ALGORITHMS), frameon=False,
                       bbox_to_anchor=(0.5, 0.965))
            runs = sorted({row["run_count"] for row in rows})
            fig.text(0.5, 0.01,
                     f"Bars: mean across runs (runs={', '.join(map(str, runs))}). "
                     "Error bars: sample SD across runs; omitted for a single run. "
                     "NA: no successful episode. Dashed line: shortest path (ratio 1.0).",
                     ha="center", fontsize=7.5, color="#444444")
            fig.tight_layout(rect=(0, 0.03, 1, 0.95))
            paths = []
            for suffix in (".png", ".pdf"):
                path = output_base.with_suffix(suffix)
                fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
                paths.append(path)
        finally:
            plt.close(fig)
    return tuple(paths)


def make_grid_report(input_dir=None, output_dir=None):
    """요약 CSV를 읽어 숫자 표 CSV와 그래프를 저장하고, 표를 출력한다."""
    input_dir = _default_dir(input_dir)
    output_dir = _default_dir(output_dir)
    rows = load_grid_summary(input_dir / "grid_comparison_summary.csv")
    table = build_table(rows)
    table_path = save_table(table, output_dir / "grid_comparison_table.csv")
    figure_paths = plot_grid_comparison(rows, output_dir / "grid_comparison")
    print(format_table(table))
    print("Table saved to: " + str(table_path))
    print("Figures saved to:\n" + "\n".join(str(path) for path in figure_paths))
    return {"table": table, "table_path": table_path, "figure_paths": figure_paths}


def main():
    parser = argparse.ArgumentParser(description="Make a table and figure for the grid-size comparison.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        make_grid_report(args.input_dir, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
