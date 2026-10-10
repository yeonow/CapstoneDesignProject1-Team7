"""Gaussian Noise 수준과 Random Seed별 학습·평가 실험을 실행한다.

기본 실행:

    python3 -m src.experiment_noise_sweep

빠른 확인:

    python3 -m src.experiment_noise_sweep \
        --noise-stds 0 5 10 --seeds 42 123 --train-episodes 10 --eval-episodes 5
"""

import argparse
import csv
import random
from pathlib import Path
from statistics import fmean, stdev

import config
from src.experiment import build_agent, build_environment, run_episode


DEFAULT_NOISE_STDS = (0.0, 5.0, 10.0)
DEFAULT_SEEDS = (42, 123, 2026, 7777, 10000)
DEFAULT_TRAIN_EPISODES = 1000

RUN_METRICS = (
    "train_success_rate",
    "train_average_steps",
    "train_average_total_reward",
    "train_average_final_distance",
    "eval_success_rate",
    "eval_average_steps",
    "eval_average_steps_on_success",
    "eval_average_total_reward",
    "eval_average_final_distance",
    "eval_failure_count",
)


def summarize_episodes(results):
    """Episode 결과 목록에서 성공·효율·보상·잔여거리 지표를 계산한다."""
    if not results:
        raise ValueError("results must not be empty.")

    successful_steps = [row["steps"] for row in results if row["success"]]
    return {
        "success_rate": len(successful_steps) / len(results),
        "average_steps": fmean(row["steps"] for row in results),
        "average_steps_on_success": (
            fmean(successful_steps) if successful_steps else None
        ),
        "average_total_reward": fmean(row["total_reward"] for row in results),
        "average_final_distance": fmean(row["final_distance"] for row in results),
        "failure_count": len(results) - len(successful_steps),
    }


def run_noise_seed(noise_std, seed, train_episodes, eval_episodes):
    """하나의 Noise·Seed 조합을 학습하고 같은 Agent를 별도로 평가한다."""
    if noise_std < 0:
        raise ValueError("noise_std must be greater than or equal to 0.")
    if train_episodes <= 0:
        raise ValueError("train_episodes must be greater than 0.")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than 0.")

    random.seed(seed)
    train_env = build_environment(noise_std=noise_std)
    agent = build_agent(seed=seed)
    train_results = []

    for _ in range(train_episodes):
        train_results.append(run_episode(train_env, agent))
        agent.decay_epsilon()

    random.seed(seed + 1)
    eval_env = build_environment(noise_std=noise_std)
    eval_results = [
        run_episode(eval_env, agent, training=False)
        for _ in range(eval_episodes)
    ]

    train_metrics = summarize_episodes(train_results)
    eval_metrics = summarize_episodes(eval_results)
    return {
        "noise_std": float(noise_std),
        "seed": int(seed),
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "train_success_rate": train_metrics["success_rate"],
        "train_average_steps": train_metrics["average_steps"],
        "train_average_total_reward": train_metrics["average_total_reward"],
        "train_average_final_distance": train_metrics["average_final_distance"],
        "eval_success_rate": eval_metrics["success_rate"],
        "eval_average_steps": eval_metrics["average_steps"],
        "eval_average_steps_on_success": eval_metrics["average_steps_on_success"],
        "eval_average_total_reward": eval_metrics["average_total_reward"],
        "eval_average_final_distance": eval_metrics["average_final_distance"],
        "eval_failure_count": eval_metrics["failure_count"],
    }


def aggregate_noise_results(run_results):
    """Seed별 결과를 Noise 수준별 평균과 표본 표준편차로 집계한다."""
    if not run_results:
        raise ValueError("run_results must not be empty.")

    grouped = {}
    for row in run_results:
        grouped.setdefault(row["noise_std"], []).append(row)

    summaries = []
    for noise_std in sorted(grouped):
        rows = grouped[noise_std]
        summary = {
            "noise_std": noise_std,
            "seed_count": len(rows),
            "train_episodes": rows[0]["train_episodes"],
            "eval_episodes": rows[0]["eval_episodes"],
        }
        for metric in RUN_METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[f"{metric}_mean"] = fmean(values) if values else None
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def _write_csv(rows, output_path, fieldnames):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_noise_results(run_results, summary_results, output_dir=None):
    """Seed별 원본 결과와 Noise별 집계 결과를 각각 CSV로 저장한다."""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / config.RESULTS_DIR
    output_dir = Path(output_dir)

    runs_path = _write_csv(
        run_results,
        output_dir / "noise_sweep_runs.csv",
        [
            "noise_std",
            "seed",
            "train_episodes",
            "eval_episodes",
            *RUN_METRICS,
        ],
    )
    summary_fields = ["noise_std", "seed_count", "train_episodes", "eval_episodes"]
    for metric in RUN_METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    summary_path = _write_csv(
        summary_results,
        output_dir / "noise_sweep_summary.csv",
        summary_fields,
    )
    return runs_path, summary_path


def run_noise_sweep(
    noise_stds=DEFAULT_NOISE_STDS,
    seeds=DEFAULT_SEEDS,
    train_episodes=DEFAULT_TRAIN_EPISODES,
    eval_episodes=None,
    output_dir=None,
):
    """모든 Noise·Seed 조합을 실행하고 원본·집계 CSV를 저장한다."""
    if eval_episodes is None:
        eval_episodes = config.NUM_EVAL_EPISODES
    if not noise_stds:
        raise ValueError("noise_stds must not be empty.")
    if not seeds:
        raise ValueError("seeds must not be empty.")

    run_results = []
    for noise_std in noise_stds:
        for seed in seeds:
            row = run_noise_seed(noise_std, seed, train_episodes, eval_episodes)
            run_results.append(row)
            print(
                f"noise_std={noise_std:g} seed={seed} | "
                f"train_success={row['train_success_rate']:.3f} | "
                f"eval_success={row['eval_success_rate']:.3f} | "
                f"eval_steps={row['eval_average_steps']:.2f}"
            )

    summary_results = aggregate_noise_results(run_results)
    runs_path, summary_path = save_noise_results(
        run_results,
        summary_results,
        output_dir=output_dir,
    )
    print(f"Noise runs saved to: {runs_path}")
    print(f"Noise summary saved to: {summary_path}")
    return run_results, summary_results


def main():
    parser = argparse.ArgumentParser(description="Compare Gaussian Noise levels.")
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=DEFAULT_NOISE_STDS,
        help="Gaussian Noise standard deviations.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Random seeds used for independent repetitions.",
    )
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=DEFAULT_TRAIN_EPISODES,
        help="Training episodes per Noise·Seed pair.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=config.NUM_EVAL_EPISODES,
        help="Evaluation episodes per Noise·Seed pair.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: config.RESULTS_DIR).",
    )
    args = parser.parse_args()
    run_noise_sweep(
        noise_stds=args.noise_stds,
        seeds=args.seeds,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
