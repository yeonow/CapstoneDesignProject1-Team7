"""동일한 RSSI Grid 환경에서 Q-Learning을 학습시키고, Random 탐색 알고리즘과 성능을 비교한다.

python -m src.compare_algorithms 명령어로 실행한다.
각 실행(Run)은 1부터 시작하며, 실행 번호 r에 따라 다음과 같이 Seed를 설정한다.
- 학습 Seed: seed + 2*(r-1)
- 평가 Seed: train_seed + 1
각 평가 Episode e에서는 두 알고리즘 모두 환경의 Noise 생성에 f"{eval_seed}:{e}" 형식의 문자열을 Seed로 사용한다.
이렇게 하면 서로 다른 Run 사이에서 정수형 Episode Seed가 중복되는 것을 방지할 수 있다.
각 Episode마다 Noise 난수 생성기를 초기화하면 두 알고리즘이 사용하는 Noise 시퀀스의 앞부분이 동일해진다.
다만 Episode 길이가 서로 다를 수 있으며, Noise 시퀀스가 동일하다고 해서 서로 다른 위치에서 측정한 RSSI 값까지 동일하다는 의미는 아니다.
Action 선택에 사용되는 난수 생성기(RNG)는 독립적으로 관리되며, 평가 시작 시 eval_seed로 초기화한다.
이 난수 생성기는 환경의 Noise 생성과 독립적이다.
config.TARGET_MODE가 "random"/"split"이면 Target 위치도 Episode마다 바뀐다(src/targets.py).
학습 Target은 train_seed로 만든 별도 RNG에서, 평가 Target은 f"target:{eval_seed}:{e}"에서 고르므로
같은 평가 Episode에서는 두 알고리즘이 같은 Target을 받는다.
또한 실행 과정에서 사용한 Python 및 NumPy의 전역 난수 생성기 상태는 사용 후 원래 상태로 복원한다.
기존 Environment는 전역 random을 사용하며 수정하지 않았으므로, 실험은 병렬이 아닌 순차적으로 실행해야 한다.
"""

import argparse
import csv
import random
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean

import numpy as np

import config
from src import targets
from src.experiment import build_agent, build_environment, run_episode
from src.random_agent import RandomAgent
from src.random_experiment import run_random_episode


# 분석 코드(analyze_paper_results)가 검증하는 기본 열
BASE_EPISODE_FIELDS = (
    "run", "algorithm", "episode", "train_seed", "eval_seed",
    "success", "steps", "final_distance",
)
# Episode별 Target 위치와 시작점→Target 최단 경로 Step 수
TARGET_FIELDS = ("target_x", "target_y", "shortest_path_steps")
EPISODE_FIELDS = (*BASE_EPISODE_FIELDS, *TARGET_FIELDS)
SUMMARY_FIELDS = (
    "run", "algorithm", "train_seed", "eval_seed", "train_episodes",
    "eval_episodes", "success_count", "failure_count", "success_rate",
    "average_steps", "average_steps_on_success", "average_final_distance",
)


def _positive_count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


@contextmanager
def _noise_seed(seed):
    previous_state = random.getstate()
    try:
        random.seed(seed)
        yield
    finally:
        random.setstate(previous_state)


def train_q_learning_agent(train_episodes=None, seed=None, noise_std=None):
    """Return a newly trained agent, using existing updates and epsilon decay.

    No Q-table file is needed: the returned object is passed to evaluation.
    noise_std is an optional override for callers; None uses current config.
    """
    count = _positive_count(
        config.NUM_EPISODES if train_episodes is None else train_episodes,
        "train_episodes",
    )
    seed = config.RANDOM_SEED if seed is None else seed
    targets.validate_target_config()
    candidates = targets.target_candidates("train")
    target_rng = targets.training_target_rng(seed)
    numpy_state = np.random.get_state()
    try:
        with _noise_seed(seed):
            env = build_environment(noise_std=noise_std)
            agent = build_agent(seed=seed)
            for episode in range(1, count + 1):
                if candidates is None:
                    run_episode(env, agent, training=True)
                else:
                    run_episode(env, agent, training=True,
                                target_position=target_rng.choice(candidates))
                agent.decay_epsilon()
                if (
                    config.LOG_INTERVAL > 0 and episode % config.LOG_INTERVAL == 0
                ) or episode == count:
                    print(f"Training episode {episode}/{count} | seed={seed}")
    finally:
        # Existing QLearningAgent construction seeds global NumPy random.
        np.random.set_state(numpy_state)
    return agent


def _evaluate_episodes(env, agent, count, seed, *, q_learning):
    targets.validate_target_config()
    candidates = targets.target_candidates("eval")
    results = []
    for episode in range(1, count + 1):
        target_kwargs = {} if candidates is None else {
            "target_position": targets.evaluation_target(candidates, seed, episode),
        }
        with _noise_seed(f"{seed}:{episode}"):
            if q_learning:
                result = run_episode(env, agent, training=False, **target_kwargs)
            else:
                result = run_random_episode(env, agent, **target_kwargs)
        target = env.target_position
        # Reward is internal to Q-Learning and is not a comparison metric.
        results.append({
            "episode": episode,
            "success": bool(result["success"]),
            "steps": int(result["steps"]),
            "final_distance": float(result["final_distance"]),
            "target_x": target[0],
            "target_y": target[1],
            "shortest_path_steps": targets.shortest_path_steps(env.agent_start, target),
        })
    return results


def evaluate_q_learning(agent, eval_episodes=None, seed=None, noise_std=None):
    """Evaluate the trained agent greedily in a separate, identically set grid.

    Only its tie-breaking RNG is reseeded; Q-table and epsilon remain intact.
    Uses the same run_episode(training=False) path as evaluate_agent().
    """
    count = _positive_count(
        config.NUM_EVAL_EPISODES if eval_episodes is None else eval_episodes,
        "eval_episodes",
    )
    seed = config.RANDOM_SEED + 1 if seed is None else seed
    env = build_environment(noise_std=noise_std)
    agent.random.seed(seed)
    return _evaluate_episodes(env, agent, count, seed, q_learning=True)


def evaluate_random_agent(eval_episodes=None, seed=None, noise_std=None):
    """Evaluate the existing RandomAgent/episode runner without any learning."""
    count = _positive_count(
        config.NUM_EVAL_EPISODES if eval_episodes is None else eval_episodes,
        "eval_episodes",
    )
    seed = config.RANDOM_SEED + 1 if seed is None else seed
    env = build_environment(noise_std=noise_std)
    agent = RandomAgent(seed=seed)
    return _evaluate_episodes(env, agent, count, seed, q_learning=False)


def calculate_metrics(results):
    """Aggregate actual episode outcomes, including failures in overall means."""
    if not results:
        raise ValueError("results must contain at least one episode.")
    successful_steps = [row["steps"] for row in results if row["success"]]
    success_count = len(successful_steps)
    return {
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "success_rate": success_count / len(results),
        "average_steps": fmean(row["steps"] for row in results),
        "average_steps_on_success": (
            fmean(successful_steps) if successful_steps else None
        ),
        "average_final_distance": fmean(row["final_distance"] for row in results),
    }


def save_comparison_results(episode_results, summary_results, output_dir=None):
    """Overwrite only the two comparison CSVs; relative paths use project root."""
    root = Path(__file__).resolve().parents[1]
    directory = root / (config.RESULTS_DIR if output_dir is None else output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    episode_path = directory / "algorithm_comparison_episodes.csv"
    summary_path = directory / "algorithm_comparison_summary.csv"
    for path, fields, rows in (
        (episode_path, EPISODE_FIELDS, episode_results),
        (summary_path, SUMMARY_FIELDS, summary_results),
    ):
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return episode_path, summary_path


def compare_algorithms(
    train_episodes=None, eval_episodes=None, seed=None, runs=1, output_dir=None,
):
    """Retrain for every run, evaluate both algorithms, save and return rows.

    Returns {"episodes": [...], "summary": [...]} with per-run metrics.
    Random's train_seed records the paired run, not a training operation.
    """
    train_episodes = _positive_count(
        config.NUM_EPISODES if train_episodes is None else train_episodes,
        "train_episodes",
    )
    eval_episodes = _positive_count(
        config.NUM_EVAL_EPISODES if eval_episodes is None else eval_episodes,
        "eval_episodes",
    )
    _positive_count(runs, "runs")
    seed = config.RANDOM_SEED if seed is None else seed
    episodes, summaries = [], []
    print("Episode noise seed: '<eval_seed>:<episode>' | action seed: eval_seed")
    for run in range(1, runs + 1):
        train_seed = seed + 2 * (run - 1)
        eval_seed = train_seed + 1
        print(f"Run {run}/{runs} | train_seed={train_seed} | eval_seed={eval_seed}")
        agent = train_q_learning_agent(train_episodes, seed=train_seed)
        q_results = evaluate_q_learning(agent, eval_episodes, seed=eval_seed)
        random_results = evaluate_random_agent(eval_episodes, seed=eval_seed)
        for algorithm, results in (("q_learning", q_results), ("random", random_results)):
            metadata = {
                "run": run, "algorithm": algorithm,
                "train_seed": train_seed, "eval_seed": eval_seed,
            }
            episodes.extend({**metadata, **row} for row in results)
            summary = {
                **metadata,
                "train_episodes": train_episodes if algorithm == "q_learning" else 0,
                "eval_episodes": eval_episodes,
                **calculate_metrics(results),
            }
            summaries.append(summary)
            print(
                f"{algorithm} | success_rate={summary['success_rate']:.2%} | "
                f"successes={summary['success_count']} | failures={summary['failure_count']} | "
                f"average_steps={summary['average_steps']:.2f} | "
                f"average_steps_on_success={summary['average_steps_on_success']} | "
                f"average_final_distance={summary['average_final_distance']:.3f}"
            )
    paths = save_comparison_results(episodes, summaries, output_dir)
    print("Results saved to:\n" + "\n".join(str(path) for path in paths))
    return {"episodes": episodes, "summary": summaries}


def main():
    parser = argparse.ArgumentParser(description="Compare Q-Learning and random search.")
    parser.add_argument("--train-episodes", type=int, default=config.NUM_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=config.NUM_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for name in ("train_episodes", "eval_episodes", "runs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    compare_algorithms(
        train_episodes=args.train_episodes, eval_episodes=args.eval_episodes,
        seed=args.seed, runs=args.runs, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
