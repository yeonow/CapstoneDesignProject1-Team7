"""환경, State/Reward, Q-Learning Agent를 연결하는 학습 실행기.

프로젝트 루트에서 python -m src.experiment 명령으로 실행한다.
"""

import csv
import random
from pathlib import Path
from statistics import fmean

import config
from src.environment import RSSIGridEnv
from src.q_learning import QLearningAgent
from src.state_reward import calculate_reward, make_state


def build_environment(noise_std=None):
    """config 설정과 선택적 Noise override로 환경을 생성한다."""
    return RSSIGridEnv(
        grid_size=config.GRID_SIZE,
        max_steps=config.MAX_STEPS,
        agent_start=config.AGENT_START,
        target_position=config.TARGET_POSITION,
        reference_rssi=config.REFERENCE_RSSI,
        reference_distance=config.REFERENCE_DISTANCE,
        path_loss_exponent=config.PATH_LOSS_EXPONENT,
        noise_std=config.NOISE_STD if noise_std is None else noise_std,
    )


def build_agent(seed=None):
    """config 학습 설정과 선택적 Seed override로 Agent를 생성한다."""
    return QLearningAgent(
        num_actions=4,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        seed=config.RANDOM_SEED if seed is None else seed,
    )


def select_action(agent, state, training=True):
    """학습 시 epsilon-greedy, 평가 시 greedy 방식으로 Action을 선택한다."""
    if training:
        return agent.choose_action(state)
    return agent.choose_action(state, training=False)


def update_agent(agent, state, action, reward, next_state, done):
    """Agent의 Q-table 업데이트 인터페이스를 연결한다."""
    agent.update_q(state, action, reward, next_state, done)


def get_reward_config():
    """config.py의 Reward 설정을 calculate_reward 인자 형식으로 반환한다."""
    return {
        "reward_up": config.REWARD_UP,
        "reward_keep": config.REWARD_KEEP,
        "reward_down": config.REWARD_DOWN,
        "move_cost": config.MOVE_COST,
        "terminal_reward": config.TERMINAL_REWARD,
    }


def run_episode(env, agent, reward_config=None, training=True):
    """학습 또는 평가 Episode 한 번을 실행하고 결과를 반환한다."""
    reward_config = get_reward_config() if reward_config is None else reward_config
    current_rssi = env.reset()
    state = make_state(current_rssi, None, config.NONE_ACTION)
    total_reward = 0.0
    terminal_info = None
    # 시작부터 Target에 있거나 Step 한도에 도달했다면 이동하지 않는다.
    done = env.is_done()

    while not done:
        action = select_action(agent, state, training=training)
        # 경계에 막힌 Action인지 Reward 계산에 함께 전달한다.
        blocked = env.move(action)
        # 이동당 한 번 조회한 RSSI를 State와 Reward가 함께 사용한다.
        new_rssi = env.get_rssi()
        next_state = make_state(new_rssi, current_rssi, action)
        done = env.is_done()
        if done:
            terminal_info = env.get_info()
        reward = calculate_reward(
            new_rssi,
            current_rssi,
            success=bool(terminal_info and terminal_info["success"]),
            blocked=blocked,
            **reward_config,
        )
        if training:
            update_agent(agent, state, action, reward, next_state, done)

        total_reward += reward
        state = next_state
        current_rssi = new_rssi

    # Ground Truth는 학습 State에 넣지 않고 종료 후 평가에만 사용한다.
    info = terminal_info if terminal_info is not None else env.get_info()
    return {
        "success": info["success"],
        "steps": info["step_count"],
        "total_reward": total_reward,
        "final_distance": info["distance"],
    }


def evaluate_agent(env, agent, num_episodes, reward_config=None):
    """학습된 Agent를 greedy 정책으로 평가하고 집계 결과를 반환한다."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be greater than 0.")

    results = [
        run_episode(
            env,
            agent,
            reward_config=reward_config,
            training=False,
        )
        for _ in range(num_episodes)
    ]
    successful_steps = [
        result["steps"]
        for result in results
        if result["success"]
    ]
    failure_count = num_episodes - len(successful_steps)

    return {
        "success_rate": len(successful_steps) / num_episodes,
        "average_steps": fmean(result["steps"] for result in results),
        "average_steps_on_success": (
            fmean(successful_steps) if successful_steps else None
        ),
        "failure_count": failure_count,
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
        agent.decay_epsilon()
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
