"""Gaussian Noise 수준별로 Q-Learning과 Random 탐색의 성능을 비교한다.

Run 번호 r은 1부터 시작하며, 기준 Seed는 seeds[r-1]이다.
학습 Seed는 train_seed = base_seed + 2*(r-1),
평가 Seed는 eval_seed = train_seed + 1로 설정한다.
같은 Run에서는 모든 Noise 수준에 동일한 Seed를 사용한다.
기존 비교 평가 함수는 f"{eval_seed}:{episode}"로 환경 Noise를 대응시키고,
Action 선택용 난수 생성기는 독립적으로 유지한다.

기존 RSSI 환경은 Python의 전역 random을 사용하므로 실험은 순차 실행한다.
기본 설정은 학습 Episode 1,000회를 포함해 기존 Q-Learning 전용 Noise Sweep을 따른다.
요약 통계에서는 각 Run에 같은 가중치를 부여한다. 성공 시 평균 Step은
성공이 없는 Run을 제외하고 계산하며, 유효한 Run 수도 함께 기록한다.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean, stdev

import config
from src.compare_algorithms import (
    EPISODE_FIELDS,
    SUMMARY_FIELDS,
    calculate_metrics,
    evaluate_q_learning,
    evaluate_random_agent,
    train_q_learning_agent,
)
from src.experiment_noise_sweep import (
    DEFAULT_NOISE_STDS,
    DEFAULT_SEEDS,
    DEFAULT_TRAIN_EPISODES,
)


METRICS = (
    "success_rate", "average_steps", "average_steps_on_success",
    "average_final_distance", "success_count", "failure_count",
)
NOISE_EPISODE_FIELDS = ("noise_std", *EPISODE_FIELDS)
NOISE_RUN_FIELDS = ("noise_std", *SUMMARY_FIELDS)
NOISE_SUMMARY_FIELDS = (
    "noise_std", "algorithm", "run_count", "train_episodes", "eval_episodes",
    *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std", "run_count")),
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _noise_value(value):
    try:
        noise_std = float(value)
    except (TypeError, ValueError):
        raise ValueError("noise_std must be finite and greater than or equal to 0.") from None
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and greater than or equal to 0.")
    return noise_std


def _run_seeds(seed, run):
    _positive_integer(run, "run")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    train_seed = seed + 2 * (run - 1)
    if not 0 <= seed <= train_seed <= 2**32 - 1:
        raise ValueError("seed and derived train_seed must be in [0, 2**32 - 1].")
    return train_seed, train_seed + 1


def run_noise_seed_comparison(
    noise_std, seed, train_episodes=DEFAULT_TRAIN_EPISODES, eval_episodes=None, run=1,
):
    """하나의 Noise·Run 조합에서 새 Q-Table을 학습하고 두 알고리즘을 평가한다.

    seed는 해당 Run의 기준 Seed이며, 반환하는 결과에는 계산된 학습·평가 Seed를 기록한다.
    호출 측에서 여러 실험 결과를 합칠 수 있도록 이 함수에서는 CSV를 저장하지 않는다.
    """
    noise_std = _noise_value(noise_std)
    train_episodes = _positive_integer(train_episodes, "train_episodes")
    eval_episodes = _positive_integer(
        config.NUM_EVAL_EPISODES if eval_episodes is None else eval_episodes,
        "eval_episodes",
    )
    train_seed, eval_seed = _run_seeds(seed, run)
    print(
        f"Noise={noise_std:g} dB | run={run} | base_seed={seed} | "
        f"train_seed={train_seed} | eval_seed={eval_seed}"
    )
    agent = train_q_learning_agent(train_episodes, seed=train_seed, noise_std=noise_std)
    q_results = evaluate_q_learning(
        agent, eval_episodes, seed=eval_seed, noise_std=noise_std,
    )
    random_results = evaluate_random_agent(
        eval_episodes, seed=eval_seed, noise_std=noise_std,
    )
    episodes, runs = [], []
    for algorithm, results in (("q_learning", q_results), ("random", random_results)):
        metadata = {
            "noise_std": noise_std, "run": run, "algorithm": algorithm,
            "train_seed": train_seed, "eval_seed": eval_seed,
        }
        episodes.extend({**metadata, **row} for row in results)
        runs.append({
            **metadata,
            "train_episodes": train_episodes if algorithm == "q_learning" else 0,
            "eval_episodes": eval_episodes,
            **calculate_metrics(results),
        })
    return {"episodes": episodes, "runs": runs}


def aggregate_noise_comparison(run_results):
    """각 Run에 같은 가중치를 적용해 평균과 표본 표준편차(ddof=1)를 계산한다.

    유효한 Run이 없으면 평균과 표준편차는 None, 하나면 표준편차는 0으로 처리한다.
    전체 run_count와 별도로 각 지표의 유효한 Run 수를 *_run_count에 기록한다.
    """
    if not run_results:
        raise ValueError("run_results must not be empty.")
    grouped = {}
    for row in run_results:
        grouped.setdefault((row["noise_std"], row["algorithm"]), []).append(row)
    summaries = []
    for (noise_std, algorithm), rows in sorted(grouped.items()):
        if len({(row["train_episodes"], row["eval_episodes"]) for row in rows}) != 1:
            raise ValueError("Episode counts must match within each noise/algorithm group.")
        summary = {
            "noise_std": noise_std, "algorithm": algorithm, "run_count": len(rows),
            "train_episodes": rows[0]["train_episodes"],
            "eval_episodes": rows[0]["eval_episodes"],
        }
        for metric in METRICS:
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[f"{metric}_mean"] = fmean(values) if values else None
            summary[f"{metric}_std"] = (
                stdev(values) if len(values) > 1 else 0.0 if values else None
            )
            summary[f"{metric}_run_count"] = len(values)
        summaries.append(summary)
    return summaries


def save_noise_comparison_results(episode_results, run_results, summary_results, output_dir=None):
    """프로젝트 루트를 기준으로 Noise 비교 결과 CSV 세 개만 저장한다."""
    root = Path(__file__).resolve().parents[1]
    directory = root / (config.RESULTS_DIR if output_dir is None else output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, rows, fields in (
        ("noise_comparison_episodes.csv", episode_results, NOISE_EPISODE_FIELDS),
        ("noise_comparison_runs.csv", run_results, NOISE_RUN_FIELDS),
        ("noise_comparison_summary.csv", summary_results, NOISE_SUMMARY_FIELDS),
    ):
        path = directory / filename
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return tuple(paths)


def run_noise_comparison(
    noise_stds=DEFAULT_NOISE_STDS, seeds=DEFAULT_SEEDS,
    train_episodes=DEFAULT_TRAIN_EPISODES, eval_episodes=None, output_dir=None,
):
    """모든 Noise·기준 Seed 조합을 실행하고 Episode별·Run별·요약 결과를 반환한다."""
    noise_stds = tuple(_noise_value(value) for value in noise_stds)
    seeds = tuple(seeds)
    if not noise_stds or not seeds:
        raise ValueError("noise_stds and seeds must not be empty.")
    _positive_integer(train_episodes, "train_episodes")
    eval_episodes = _positive_integer(
        config.NUM_EVAL_EPISODES if eval_episodes is None else eval_episodes,
        "eval_episodes",
    )
    # 시간이 오래 걸리는 실험이나 파일 저장을 시작하기 전에 모든 조합을 검증한다.
    for run, seed in enumerate(seeds, 1):
        _run_seeds(seed, run)
    if len(set(noise_stds)) != len(noise_stds):
        raise ValueError("noise_stds must not contain duplicates.")
    if len({_run_seeds(seed, run)[0] for run, seed in enumerate(seeds, 1)}) != len(seeds):
        raise ValueError("Derived train seeds must be distinct between runs.")

    episodes, runs = [], []
    for noise_std in noise_stds:
        for run, seed in enumerate(seeds, 1):
            result = run_noise_seed_comparison(
                noise_std, seed, train_episodes, eval_episodes, run=run,
            )
            episodes.extend(result["episodes"])
            runs.extend(result["runs"])
    summaries = aggregate_noise_comparison(runs)
    paths = save_noise_comparison_results(episodes, runs, summaries, output_dir)
    for row in summaries:
        print(
            f"Noise={row['noise_std']:g} dB | {row['algorithm']} | "
            f"runs={row['run_count']} | success_rate="
            f"{row['success_rate_mean']:.2%} ± {row['success_rate_std']:.2%} | "
            f"average_steps={row['average_steps_mean']:.2f} ± {row['average_steps_std']:.2f} | "
            f"average_steps_on_success={row['average_steps_on_success_mean']} | "
            f"valid_success_runs={row['average_steps_on_success_run_count']} | "
            f"average_final_distance={row['average_final_distance_mean']:.3f}"
        )
    print("Results saved to:\n" + "\n".join(str(path) for path in paths))
    return {"episodes": episodes, "runs": runs, "summary": summaries}


def main():
    parser = argparse.ArgumentParser(description="Compare algorithms across Gaussian noise levels.")
    parser.add_argument("--noise-stds", type=float, nargs="+", default=DEFAULT_NOISE_STDS)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
        help="Base seed per run; run r trains with seeds[r-1] + 2*(r-1).",
    )
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=config.NUM_EVAL_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        run_noise_comparison(
            noise_stds=args.noise_stds, seeds=args.seeds,
            train_episodes=args.train_episodes, eval_episodes=args.eval_episodes,
            output_dir=args.output_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
