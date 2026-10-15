"""Grid 크기(기본 10×10, 20×20)별로 Q-Learning과 Random 탐색을 비교한다.

python -m src.compare_grid_sizes 명령으로 실행한다.
Grid마다 config의 GRID_SIZE, TARGET_POSITION, MAX_STEPS를 GRID_ENV_SETTINGS 값으로
잠시 바꾼 뒤 기존 비교 파이프라인(compare_algorithms, 선택적으로 compare_noise_sweep,
analyze_paper_results)을 그대로 실행하고, 끝나면 원래 config 값으로 되돌린다.
결과는 <output_dir>/grid_<N>x<N>/ 아래에 기존과 같은 파일 이름으로 저장되며,
Grid 간 요약은 <output_dir>/grid_comparison_runs.csv, grid_comparison_summary.csv에 저장하고,
숫자 표(grid_comparison_table.csv)와 그래프(grid_comparison.png/.pdf)도 함께 만든다.
Seed 규칙은 compare_algorithms와 같으므로 Grid 크기만 다르고 Seed는 같은 조건에서 비교된다.
"""

import argparse
import csv
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean, stdev

import config
from src.analyze_paper_results import analyze_paper_results
from src.compare_algorithms import SUMMARY_FIELDS, compare_algorithms
from src.compare_noise_sweep import run_noise_comparison
from src.experiment_noise_sweep import (
    DEFAULT_NOISE_STDS,
    DEFAULT_SEEDS,
    DEFAULT_TRAIN_EPISODES,
)
from src.plot_grid_comparison import make_grid_report


GRID_METRICS = (
    "success_rate", "average_steps", "average_steps_on_success", "average_final_distance",
)
GRID_RUN_FIELDS = ("grid_size", "max_steps", *SUMMARY_FIELDS)
GRID_SUMMARY_FIELDS = (
    "grid_size", "max_steps", "algorithm", "run_count", "train_episodes", "eval_episodes",
    *(f"{metric}_{suffix}" for metric in GRID_METRICS for suffix in ("mean", "std", "run_count")),
)


def grid_settings(grid_size):
    """config.GRID_ENV_SETTINGS에 정의된 Grid 크기의 Target·최대 Step을 반환한다."""
    if grid_size not in config.GRID_ENV_SETTINGS:
        raise ValueError(
            f"grid_size {grid_size} is not in config.GRID_ENV_SETTINGS "
            f"({sorted(config.GRID_ENV_SETTINGS)})."
        )
    settings = config.GRID_ENV_SETTINGS[grid_size]
    target_x, target_y = settings["target_position"]
    start_x, start_y = config.AGENT_START
    if not all(0 <= value < grid_size for value in (target_x, target_y, start_x, start_y)):
        raise ValueError(f"Start/Target must be inside the {grid_size}x{grid_size} grid.")
    return settings


@contextmanager
def grid_config(grid_size):
    """블록 안에서만 config의 Grid 설정을 바꾸고, 예외가 나도 원래 값으로 되돌린다."""
    settings = grid_settings(grid_size)
    previous = (config.GRID_SIZE, config.TARGET_POSITION, config.MAX_STEPS)
    config.GRID_SIZE = grid_size
    config.TARGET_POSITION = settings["target_position"]
    config.MAX_STEPS = settings["max_steps"]
    try:
        yield settings
    finally:
        config.GRID_SIZE, config.TARGET_POSITION, config.MAX_STEPS = previous


def grid_output_dir(output_dir, grid_size):
    base = Path(config.RESULTS_DIR if output_dir is None else output_dir)
    return base / f"grid_{grid_size}x{grid_size}"


def aggregate_grid_comparison(run_rows):
    """Grid·알고리즘별로 Run에 같은 가중치를 주어 평균과 표본 표준편차를 계산한다."""
    grouped = {}
    for row in run_rows:
        grouped.setdefault((row["grid_size"], row["algorithm"]), []).append(row)
    summaries = []
    for (grid_size, algorithm), rows in sorted(grouped.items()):
        summary = {
            "grid_size": grid_size, "max_steps": rows[0]["max_steps"],
            "algorithm": algorithm, "run_count": len(rows),
            "train_episodes": rows[0]["train_episodes"],
            "eval_episodes": rows[0]["eval_episodes"],
        }
        for metric in GRID_METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[f"{metric}_mean"] = fmean(values) if values else None
            summary[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0 if values else None
            )
            summary[f"{metric}_run_count"] = len(values)
        summaries.append(summary)
    return summaries


def save_grid_comparison_results(run_rows, summary_rows, output_dir=None):
    """프로젝트 루트 기준으로 Grid 간 비교 CSV 두 개를 저장한다."""
    root = Path(__file__).resolve().parents[1]
    directory = root / (config.RESULTS_DIR if output_dir is None else output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, rows, fields in (
        ("grid_comparison_runs.csv", run_rows, GRID_RUN_FIELDS),
        ("grid_comparison_summary.csv", summary_rows, GRID_SUMMARY_FIELDS),
    ):
        path = directory / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return tuple(paths)


def run_grid_comparison(
    grid_sizes=None, train_episodes=None, eval_episodes=None, seed=None, runs=1,
    noise=False, noise_stds=DEFAULT_NOISE_STDS, noise_seeds=DEFAULT_SEEDS,
    noise_train_episodes=DEFAULT_TRAIN_EPISODES, analyze=False, plot=True, output_dir=None,
):
    """각 Grid 크기에서 두 알고리즘을 비교하고 Grid 간 요약을 저장·반환한다.

    noise=True면 Grid마다 Noise 수준별 비교도 실행하고, analyze=True면
    Grid별 결과 폴더에 논문용 표와 보고서를 생성한다(Noise 결과가 없으면 해당 부분만 제외).
    plot=True면 Grid 간 숫자 표 CSV와 그래프를 만든다.
    """
    grid_sizes = tuple(config.GRID_SIZES if grid_sizes is None else grid_sizes)
    if not grid_sizes or len(set(grid_sizes)) != len(grid_sizes):
        raise ValueError("grid_sizes must be non-empty and must not contain duplicates.")
    # 오래 걸리는 학습을 시작하기 전에 모든 Grid 설정을 검증한다.
    for grid_size in grid_sizes:
        grid_settings(grid_size)

    run_rows = []
    for grid_size in grid_sizes:
        directory = grid_output_dir(output_dir, grid_size)
        with grid_config(grid_size) as settings:
            print(
                f"=== Grid {grid_size}x{grid_size} | start={config.AGENT_START} | "
                f"target={settings['target_position']} | max_steps={settings['max_steps']} ==="
            )
            result = compare_algorithms(
                train_episodes=train_episodes, eval_episodes=eval_episodes,
                seed=seed, runs=runs, output_dir=directory,
            )
            run_rows.extend(
                {"grid_size": grid_size, "max_steps": settings["max_steps"], **row}
                for row in result["summary"]
            )
            if noise:
                run_noise_comparison(
                    noise_stds=noise_stds, seeds=noise_seeds,
                    train_episodes=noise_train_episodes, eval_episodes=eval_episodes,
                    output_dir=directory,
                )
            if analyze:
                analyze_paper_results(input_dir=directory, output_dir=directory)

    summaries = aggregate_grid_comparison(run_rows)
    paths = save_grid_comparison_results(run_rows, summaries, output_dir)
    for row in summaries:
        print(
            f"Grid {row['grid_size']}x{row['grid_size']} | {row['algorithm']} | "
            f"runs={row['run_count']} | success_rate={row['success_rate_mean']:.2%} | "
            f"average_steps={row['average_steps_mean']:.2f} | "
            f"average_steps_on_success={row['average_steps_on_success_mean']} | "
            f"average_final_distance={row['average_final_distance_mean']:.3f}"
        )
    print("Grid comparison saved to:\n" + "\n".join(str(path) for path in paths))
    if plot:
        make_grid_report(output_dir, output_dir)
    return {"runs": run_rows, "summary": summaries}


def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-Learning and random search on multiple grid sizes.",
    )
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=list(config.GRID_SIZES))
    parser.add_argument("--train-episodes", type=int, default=config.NUM_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=config.NUM_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--noise", action="store_true",
                        help="Also run the Gaussian noise comparison for each grid size.")
    parser.add_argument("--noise-stds", type=float, nargs="+", default=list(DEFAULT_NOISE_STDS))
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--noise-train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--analyze", action="store_true",
                        help="Write paper tables and report into each grid folder.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip the grid comparison table and figure.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for name in ("train_episodes", "eval_episodes", "runs", "noise_train_episodes"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    try:
        run_grid_comparison(
            grid_sizes=args.grid_sizes, train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes, seed=args.seed, runs=args.runs,
            noise=args.noise, noise_stds=args.noise_stds, noise_seeds=args.noise_seeds,
            noise_train_episodes=args.noise_train_episodes, analyze=args.analyze,
            plot=not args.no_plot, output_dir=args.output_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
