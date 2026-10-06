"""기존 환경, State/Reward, 임시 Agent를 연결하는 Q-Learning 학습 실행기.

프로젝트 루트에서 python -m src.experiment 명령으로 실행한다.
"""

import csv
import random
from pathlib import Path

import config
from src.environment import RSSIGridEnv
from src.q_learning_stub import QLearningAgent
from src.state_reward import calculate_reward, make_state


def build_environment():
    """config의 환경 및 Simulation 설정으로 환경을 생성한다."""
    return RSSIGridEnv(
        grid_size=config.GRID_SIZE,
        max_steps=config.MAX_STEPS,
        agent_start=config.AGENT_START,
        target_position=config.TARGET_POSITION,
        reference_rssi=config.REFERENCE_RSSI,
        reference_distance=config.REFERENCE_DISTANCE,
        path_loss_exponent=config.PATH_LOSS_EXPONENT,
        noise_std=config.NOISE_STD,
    )


def build_agent():
    """임시 Agent 생성부. 실제 구현 도입 시 import와 이 어댑터를 교체한다."""
    return QLearningAgent(
        actions=(config.UP, config.DOWN, config.LEFT, config.RIGHT),
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
    )


def select_action(agent, state):
    """Agent의 Action 선택 인터페이스를 연결한다."""
    return agent.choose_action(state)


def update_agent(agent, state, action, reward, next_state, done):
    """Agent의 Q-table 업데이트 인터페이스를 연결한다."""
    agent.update_q(state, action, reward, next_state, done)


def run_episode(env, agent):
    """Episode 한 번을 학습하고 평가용 결과를 반환한다."""
    current_rssi = env.reset()
    state = make_state(current_rssi, None, config.NONE_ACTION)
    total_reward = 0.0
    # 시작부터 Target에 있거나 Step 한도에 도달했다면 이동하지 않는다.
    done = env.is_done()

    while not done:
        action = select_action(agent, state)
        env.move(action)
        # 이동당 한 번 조회한 RSSI를 State와 Reward가 함께 사용한다.
        new_rssi = env.get_rssi()
        next_state = make_state(new_rssi, current_rssi, action)
        reward = calculate_reward(new_rssi, current_rssi)
        done = env.is_done()
        update_agent(agent, state, action, reward, next_state, done)

        total_reward += reward
        state = next_state
        current_rssi = new_rssi

    # Ground Truth는 학습 State에 넣지 않고 종료 후 평가에만 사용한다.
    info = env.get_info()
    return {
        "success": info["success"],
        "steps": info["step_count"],
        "total_reward": total_reward,
        "final_distance": info["distance"],
    }


def train():
    """동일 Agent로 여러 Episode를 학습하고 결과를 저장 및 반환한다."""
    random.seed(config.RANDOM_SEED)
    env = build_environment()
    agent = build_agent()
    results = []

    for episode in range(1, config.NUM_EPISODES + 1):
        result = {"episode": episode, **run_episode(env, agent)}
        results.append(result)
        if episode % config.LOG_INTERVAL == 0 or episode == config.NUM_EPISODES:
            print(
                f"Episode {episode}/{config.NUM_EPISODES} | "
                f"success={result['success']} | steps={result['steps']} | "
                f"total_reward={result['total_reward']:.2f} | "
                f"final_distance={result['final_distance']:.2f}"
            )

    output_path = save_results(results)
    print(f"Results saved to: {output_path}")
    return results


def save_results(results):
    """결과를 프로젝트 루트 기준 RESULTS_DIR에 CSV로 저장한다.

    동일 파일이 있으면 이번 학습 결과로 덮어쓴다.
    """
    results_dir = Path(__file__).resolve().parents[1] / config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "training_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["episode", "success", "steps", "total_reward", "final_distance"],
        )
        writer.writeheader()
        writer.writerows(results)
    return output_path


if __name__ == "__main__":
    train()
