"""python -m src.greedy_experiment 명령어로 Greedy Search 베이스라인을 실행한다"""

import argparse
import csv
import random
from pathlib import Path
from statistics import fmean

import config
from src import env_factory
from src.greedy_agent import GreedySearchAgent


def build_environment():
    """Q-Learning·Random과 같은 환경 생성 함수를 쓴다."""
    return env_factory.build_environment()


def run_greedy_episode(env, agent, target_position=None):
    """Episode를 처음부터 끝까지 실행한다. 학습·Reward는 없다.

    RSSI는 reset()에서 1회, move()마다 1회 생성된 값을 get_rssi()로 읽기만 한다.
    Agent는 그 ΔRSSI와 경계 충돌 여부만 받고 Target 좌표는 받지 않는다.
    RNG Seed 설정은 호출하는 쪽이 담당한다.
    target_position을 주면 이번 Episode의 Target을 그 위치로 바꾼다.
    """
    if target_position is None:
        previous_rssi = env.reset()
    else:
        previous_rssi = env.reset(target_position)
    agent.reset()
    done = env.is_done()
    while not done:
        action = agent.choose_action()
        blocked = env.move(action)
        current_rssi = env.get_rssi()
        agent.observe(current_rssi, previous_rssi, blocked)
        previous_rssi = current_rssi
        done = env.is_done()

    info = env.get_info()
    return {
        "success": bool(info["success"]),
        "steps": int(info["step_count"]),
        "final_distance": float(info["distance"]),
    }


def save_greedy_results(results, output_path=None):
    """Episode별 결과를 CSV로 저장하고, 상위 폴더가 없으면 만든다."""
    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[1]
            / config.RESULTS_DIR
            / "greedy_results.csv"
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


def run_greedy_experiment(num_episodes=None, seed=None, output_path=None):
    """Seed를 고정해 Greedy Episode를 실행하고 결과를 저장·반환한다.

    None이면 config.NUM_EVAL_EPISODES와 config.RANDOM_SEED를 쓴다.
    방향 선택은 GreedySearchAgent의 자체 RNG를 쓰고, 환경 Noise가 쓰는 전역 random은
    한 번 Seed를 고정한 뒤 실패해도 원래 상태로 되돌린다.
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
        agent = GreedySearchAgent(seed=seed)
        for episode in range(1, num_episodes + 1):
            result = {"episode": episode, **run_greedy_episode(env, agent)}
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

    saved_path = save_greedy_results(results, output_path)
    success_count = sum(result["success"] for result in results)
    print(
        f"Greedy experiment complete: episodes={num_episodes} | seed={seed} | "
        f"successes={success_count} | success_rate={success_count / num_episodes:.2%} | "
        f"average_steps={fmean(result['steps'] for result in results):.2f}"
    )
    print(f"Greedy results saved to: {saved_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run the greedy RSSI search baseline.")
    parser.add_argument("--episodes", type=int, default=config.NUM_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be a positive integer")
    run_greedy_experiment(
        num_episodes=args.episodes, seed=args.seed, output_path=args.output,
    )


if __name__ == "__main__":
    main()
