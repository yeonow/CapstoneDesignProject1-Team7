"""python -m src.random_experiment 명령어로 Random 베이스라인을 실행한다"""

import argparse
import csv
import random
from pathlib import Path
from statistics import fmean

import config
from src import env_factory
from src.random_agent import RandomAgent


def build_environment():
    """Q-Learning과 같은 환경 생성 함수를 쓰되 학습 모듈은 import하지 않는다."""
    return env_factory.build_environment()


def run_random_episode(env, agent):
    """Reset and run until the environment ends, without state or reward.

    reset() samples initial RSSI. Each move() samples again; get_rssi()
    only reads that cached measurement and never feeds it to the agent.
    RNG seeding belongs to the caller, so episodes continue the RNG streams.
    """
    env.reset()
    done = env.is_done()
    while not done:
        action = agent.choose_action()
        env.move(action)
        env.get_rssi()
        done = env.is_done()

    info = env.get_info()
    return {
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "final_distance": float(info["distance"]),
    }


def save_random_results(results, output_path=None):
    """Write episode rows with a header, creating missing parent directories."""
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[1]
            / config.RESULTS_DIR
            / "random_results.csv"
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["episode", "success", "steps", "final_distance"],
        )
        writer.writeheader()
        writer.writerows(results)
    return output_path


def run_random_experiment(num_episodes=None, seed=None, output_path=None):
    """Run fresh seeded streams, save episode results, and return the rows.

    None uses config.NUM_EVAL_EPISODES and config.RANDOM_SEED respectively.
    Actions use RandomAgent's private RNG. The unchanged environment uses
    global random.gauss; seed it once and restore its prior state even on
    failure. This supports sequential experiments, not concurrent isolation
    from other code using the same global RNG.
    """
    num_episodes = config.NUM_EVAL_EPISODES if num_episodes is None else num_episodes
    seed = config.RANDOM_SEED if seed is None else seed
    if isinstance(num_episodes, bool) or not isinstance(num_episodes, int):
        raise ValueError("num_episodes must be a positive integer.")
    if num_episodes <= 0:
        raise ValueError("num_episodes must be a positive integer.")

    previous_rng_state = random.getstate()
    results = []
    try:
        random.seed(seed)
        env = build_environment()
        agent = RandomAgent(seed=seed)
        for episode in range(1, num_episodes + 1):
            result = {"episode": episode, **run_random_episode(env, agent)}
            results.append(result)
            if (
                config.LOG_INTERVAL > 0 and episode % config.LOG_INTERVAL == 0
            ) or episode == num_episodes:
                print(
                    f"Episode {episode}/{num_episodes} | "
                    f"success={result['success']} | steps={result['steps']} | "
                    f"final_distance={result['final_distance']:.2f}"
                )
    finally:
        random.setstate(previous_rng_state)

    saved_path = save_random_results(results, output_path)
    success_count = sum(result["success"] for result in results)
    print(
        f"Random experiment complete: episodes={num_episodes} | seed={seed} | "
        f"successes={success_count} | success_rate={success_count / num_episodes:.2%} | "
        f"average_steps={fmean(result['steps'] for result in results):.2f}"
    )
    print(f"Random results saved to: {saved_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the random RSSI search baseline.")
    parser.add_argument("--episodes", type=int, default=config.NUM_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")
    run_random_experiment(
        num_episodes=args.episodes, seed=args.seed, output_path=args.output,
    )


if __name__ == "__main__":
    main()
