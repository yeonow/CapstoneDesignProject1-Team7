"""여러 Reward 설정을 같은 조건에서 학습하고 결과를 CSV로 비교한다.

프로젝트 루트에서 다음과 같이 실행한다.

    python3 -m src.experiment_sweep
    python3 -m src.experiment_sweep --train-episodes 500 --eval-episodes 100
"""

import argparse
import csv
import random
from pathlib import Path
from statistics import fmean

import config
from src.experiment import (
    build_agent,
    build_environment,
    evaluate_agent,
    run_episode,
)


REWARD_FIELDS = (
    "reward_up",
    "reward_keep",
    "reward_down",
    "move_cost",
    "terminal_reward",
)

# 한 번에 하나의 Reward 성격을 주로 바꿔 결과를 해석하기 쉽게 구성한다.
DEFAULT_REWARD_COMBINATIONS = (
    {
        "name": "baseline",
        "reward_up": config.REWARD_UP,
        "reward_keep": config.REWARD_KEEP,
        "reward_down": config.REWARD_DOWN,
        "move_cost": config.MOVE_COST,
        "terminal_reward": config.TERMINAL_REWARD,
    },
    {
        "name": "strong_up",
        "reward_up": 2.0,
        "reward_keep": config.REWARD_KEEP,
        "reward_down": config.REWARD_DOWN,
        "move_cost": config.MOVE_COST,
        "terminal_reward": config.TERMINAL_REWARD,
    },
    {
        "name": "strong_down_penalty",
        "reward_up": config.REWARD_UP,
        "reward_keep": config.REWARD_KEEP,
        "reward_down": -2.0,
        "move_cost": config.MOVE_COST,
        "terminal_reward": config.TERMINAL_REWARD,
    },
    {
        "name": "high_move_cost",
        "reward_up": config.REWARD_UP,
        "reward_keep": config.REWARD_KEEP,
        "reward_down": config.REWARD_DOWN,
        "move_cost": 0.3,
        "terminal_reward": config.TERMINAL_REWARD,
    },
    {
        "name": "high_terminal_reward",
        "reward_up": config.REWARD_UP,
        "reward_keep": config.REWARD_KEEP,
        "reward_down": config.REWARD_DOWN,
        "move_cost": config.MOVE_COST,
        "terminal_reward": 20.0,
    },
)


def evaluate_reward_combination(
    reward_combination,
    train_episodes,
    eval_episodes=None,
):
    """하나의 Reward 조합을 학습한 뒤 같은 Agent를 평가한다."""
    if train_episodes <= 0:
        raise ValueError("train_episodes must be greater than 0.")
    if eval_episodes is None:
        eval_episodes = config.NUM_EVAL_EPISODES
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be greater than 0.")

    missing_fields = {"name", *REWARD_FIELDS} - reward_combination.keys()
    if missing_fields:
        raise ValueError(f"Missing reward fields: {sorted(missing_fields)}")

    # 각 조합을 같은 난수 조건에서 시작시켜 Reward 차이를 비교하기 쉽게 한다.
    random.seed(config.RANDOM_SEED)
    train_env = build_environment()
    agent = build_agent()
    reward_config = {
        field: reward_combination[field]
        for field in REWARD_FIELDS
    }
    train_results = []

    for _ in range(train_episodes):
        train_results.append(
            run_episode(train_env, agent, reward_config=reward_config)
        )
        agent.decay_epsilon()

    # Evaluation은 별도 환경에서 수행하되 학습된 Agent와 Q-table은 그대로 사용한다.
    random.seed(config.RANDOM_SEED + 1)
    eval_env = build_environment()
    evaluation = evaluate_agent(
        eval_env,
        agent,
        eval_episodes,
        reward_config=reward_config,
    )

    return {
        "name": reward_combination["name"],
        **reward_config,
        "train_episodes": train_episodes,
        "train_success_rate": sum(result["success"] for result in train_results)
        / train_episodes,
        "train_average_steps": fmean(
            result["steps"] for result in train_results
        ),
        "train_average_total_reward": fmean(
            result["total_reward"] for result in train_results
        ),
        "eval_episodes": eval_episodes,
        "eval_success_rate": evaluation["success_rate"],
        "eval_average_steps": evaluation["average_steps"],
        "eval_average_steps_on_success": evaluation[
            "average_steps_on_success"
        ],
        "eval_failure_count": evaluation["failure_count"],
    }


def save_sweep_results(results, output_path=None):
    """집계 결과를 CSV로 저장하고 저장 경로를 반환한다."""
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[1]
            / config.RESULTS_DIR
            / "reward_sweep_results.csv"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                *REWARD_FIELDS,
                "train_episodes",
                "train_success_rate",
                "train_average_steps",
                "train_average_total_reward",
                "eval_episodes",
                "eval_success_rate",
                "eval_average_steps",
                "eval_average_steps_on_success",
                "eval_failure_count",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    return output_path


def run_sweep(
    reward_combinations=None,
    train_episodes=None,
    eval_episodes=None,
    output_path=None,
    num_episodes=None,
):
    """모든 Reward 조합을 실행하고 CSV 저장 후 결과를 반환한다."""
    if reward_combinations is None:
        reward_combinations = DEFAULT_REWARD_COMBINATIONS
    if num_episodes is not None:
        if train_episodes is not None:
            raise ValueError("Use either train_episodes or num_episodes, not both.")
        train_episodes = num_episodes
    if train_episodes is None:
        train_episodes = config.NUM_EPISODES
    if eval_episodes is None:
        eval_episodes = config.NUM_EVAL_EPISODES

    results = []
    for reward_combination in reward_combinations:
        result = evaluate_reward_combination(
            reward_combination,
            train_episodes,
            eval_episodes,
        )
        results.append(result)
        print(
            f"{result['name']} | "
            f"train_success_rate={result['train_success_rate']:.3f} | "
            f"eval_success_rate={result['eval_success_rate']:.3f} | "
            f"eval_average_steps={result['eval_average_steps']:.2f} | "
            f"eval_failures={result['eval_failure_count']}"
        )

    saved_path = save_sweep_results(results, output_path)
    print(f"Sweep results saved to: {saved_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Reward configurations.")
    parser.add_argument(
        "--train-episodes",
        "--episodes",
        dest="train_episodes",
        type=int,
        default=config.NUM_EPISODES,
        help="Number of training episodes per Reward configuration.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=config.NUM_EVAL_EPISODES,
        help="Number of evaluation episodes per Reward configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args()
    run_sweep(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
