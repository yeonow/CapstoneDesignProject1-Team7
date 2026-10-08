"""현재 config의 고정 설정으로 학습한 Agent를 평가하고 단일 summary CSV를 저장한다.

프로젝트 루트에서 python -m src.final_evaluation 으로 실행한다.
"""

import argparse
import csv
from pathlib import Path

import config
from src.experiment import get_reward_config
from src.experiment_sweep import evaluate_reward_combination


def save_final_evaluation(summary, output_path=None):
    """기존 결과 저장 정책대로 동일 파일을 덮어쓰며 None은 빈 CSV 값으로 저장한다."""
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[1]
            / config.RESULTS_DIR
            / "final_evaluation_results.csv"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    return output_path


def run_final_experiment(train_episodes=None, eval_episodes=None, output_path=None):
    """기존 단일 설정 학습/평가를 재사용하며 실행마다 새 Agent로 시작한다.

    학습 환경과 Agent는 RANDOM_SEED, 평가 환경은 RANDOM_SEED + 1을 사용한다.
    평가 Agent의 난수 상태는 기존 실행 흐름대로 학습 후 상태를 이어받는다.
    """
    if train_episodes is None:
        train_episodes = config.NUM_EPISODES
    if eval_episodes is None:
        eval_episodes = config.NUM_EVAL_EPISODES
    reward_config = get_reward_config()

    # Sweep 목록을 선택하거나 순회하지 않고 현재 config의 한 설정만 실행한다.
    result = evaluate_reward_combination(
        {"name": "config", **reward_config}, train_episodes, eval_episodes,
    )
    summary = {
        "eval_episodes": result["eval_episodes"],
        "success_count": result["eval_episodes"] - result["eval_failure_count"],
        "failure_count": result["eval_failure_count"],
        "success_rate": result["eval_success_rate"],
        "average_steps": result["eval_average_steps"],
        "average_steps_on_success": result["eval_average_steps_on_success"],
        "seed": config.RANDOM_SEED,
        "eval_seed": config.RANDOM_SEED + 1,
        "train_episodes": result["train_episodes"],
        "max_steps": config.MAX_STEPS,
        **reward_config,
    }
    saved_path = save_final_evaluation(summary, output_path)
    print(f"Training complete: {train_episodes} episodes")
    print(f"Evaluation summary: {summary}")
    print(f"Final evaluation results saved to: {saved_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the current config.")
    parser.add_argument("--train-episodes", type=int, default=config.NUM_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=config.NUM_EVAL_EPISODES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_final_experiment(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
