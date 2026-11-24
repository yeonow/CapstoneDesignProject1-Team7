"""200/300/400 Step의 원시 CSV를 검증하고 성공률 Run 평균 ± SD를 그린다.

Noise 2 dB 등 CSV에 없는 환경 조건은 사용자가 확인해야 한다. 학습하지 않는다.
"""

import argparse
from pathlib import Path

import matplotlib

from src.analyze_paper_results import _write_csv, run_statistics, validate_dataset
from src.graph_artifacts import PHASES, ROOT, capture_inputs, reserve_output, write_manifest
from src.plot_noise_comparison import ALGORITHMS


STEPS = (200, 300, 400)
FILES = ("algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv")


def build_step_summary(directories, expected_runs=100, train_episodes=3000, eval_episodes=100):
    """사전 집계 숫자를 믿지 않고 Episode→Run 정합성 확인 후 Run SD를 계산한다."""
    if set(directories) != set(STEPS):
        raise ValueError("200/300/400 Step 입력이 각각 필요합니다.")
    for value in (expected_runs, train_episodes, eval_episodes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Run/학습/평가 수는 양의 정수여야 합니다.")
    rows, warnings, seed_pairs, target_pairs = [], [], {}, {}
    for limit in STEPS:
        status = validate_dataset(directories[limit])
        if not status["valid"]:
            raise ValueError(f"{limit} Step 검증 실패: " + "; ".join(status["errors"]))
        warnings.extend(f"{limit}: {text}" for text in status["warnings"])
        for row in status["episodes"]:
            if row["steps"] > limit or (not row["success"] and row["steps"] != limit):
                raise ValueError(f"선언한 {limit} Step 상한과 CSV가 다릅니다.")
            key = row["run"], row["episode"]
            target = row.get("target_x"), row.get("target_y")
            if key in target_pairs and target_pairs[key] != target:
                raise ValueError("Step 조건 간 Target 대응 불일치")
            target_pairs[key] = target
        for row in status["runs"]:
            run = row["run"]
            pair = row["train_seed"], row["eval_seed"]
            if run in seed_pairs and seed_pairs[run] != pair:
                raise ValueError("Step 조건 간 Seed 대응 불일치")
            seed_pairs[run] = pair
            if row["eval_episodes"] != eval_episodes or row["train_episodes"] != (
                train_episodes if row["algorithm"] == "q_learning" else 0
            ):
                raise ValueError("요청한 학습/평가 Episode 수와 CSV가 다릅니다.")
        for algorithm, *_ in ALGORITHMS:
            subset = [r for r in status["runs"] if r["algorithm"] == algorithm]
            if len(subset) != expected_runs:
                raise ValueError("요청한 Run 수와 CSV가 다릅니다.")
            stats = run_statistics([r["success_rate"] * 100 for r in subset])
            rows.append({"max_steps": limit, "algorithm": algorithm, "run_count": len(subset),
                         "success_rate_mean_percent": stats["mean"],
                         "success_rate_sd_percentage_points": stats["std"]})
    warnings.append("Noise 2 dB/20×20/8방향/split/거리 필터는 이 CSV 형식만으로 인증할 수 없습니다.")
    return rows, warnings


def plot_step_summary(rows, directory):
    """백분율 평균과 %p 단위 SD를 같은 배율로 표시하고 SD를 CI라고 부르지 않는다."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            for algorithm, label, color, marker in ALGORITHMS:
                selected = sorted((r for r in rows if r["algorithm"] == algorithm), key=lambda r: r["max_steps"])
                means = [r["success_rate_mean_percent"] for r in selected]
                sd = [r["success_rate_sd_percentage_points"] for r in selected]
                ax.plot([r["max_steps"] for r in selected], means, color=color, marker=marker, label=label)
                for row, mean, error in zip(selected, means, sd):
                    if error is not None:
                        ax.errorbar(row["max_steps"], mean, yerr=error, fmt="none", ecolor=color, capsize=4)
            ax.set(xlabel="Maximum steps", ylabel="Success rate (%)", xticks=STEPS,
                   title="Success Rate under Different Step Limits")
            lower, upper = ax.get_ylim()
            ax.set_ylim(min(0, lower), max(100, upper))  # 범위 밖 SD도 잘라 숨기지 않는다.
            ax.grid(alpha=.25)
            ax.legend()
            fig.text(.5, .015, "Mean ± sample SD across runs (not CI); n < 2: no error bar.\n"
                     "Noise 2 dB is a declared source condition; verify experiment metadata.", ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .10, 1, 1))
            paths = tuple(directory / f"step_success_rate.{suffix}" for suffix in ("png", "pdf"))
            for path in paths:
                fig.savefig(path, dpi=300, bbox_inches="tight")
        finally:
            plt.close(fig)
    return paths


def generate_step_graph(directories, output_root, name, expected_runs=100,
                        train_episodes=3000, eval_episodes=100, phase="pre_calibration"):
    inputs = capture_inputs(Path(directories[step]) / filename for step in STEPS for filename in FILES)
    rows, warnings = build_step_summary(directories, expected_runs, train_episodes, eval_episodes)
    directory = reserve_output(output_root, phase, name)
    table = directory / "step_success_summary.csv"
    _write_csv(table, rows, list(rows[0]))
    figures = plot_step_summary(rows, directory)
    manifest = write_manifest(directory, inputs, {
        "phase": phase, "declared_noise_std_db": 2, "environment_verified_from_csv": False,
        "expected_runs": expected_runs, "train_episodes": train_episodes, "eval_episodes": eval_episodes,
        "error_bars": "sample SD across runs", "dpi": 300, "warnings": warnings,
    }, [table, *figures], [Path(__file__), ROOT / "src/analyze_paper_results.py",
                           ROOT / "src/compare_algorithms.py", ROOT / "src/graph_artifacts.py"])
    return {"rows": rows, "output_dir": directory, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for step in STEPS:
        parser.add_argument(f"--step{step}", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--expected-runs", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--phase", choices=PHASES, default="pre_calibration")
    args = parser.parse_args()
    try:
        result = generate_step_graph({s: getattr(args, f"step{s}") for s in STEPS}, args.output_root,
                                     args.name, args.expected_runs, args.train_episodes, args.eval_episodes, args.phase)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(result["output_dir"])


if __name__ == "__main__":
    main()
