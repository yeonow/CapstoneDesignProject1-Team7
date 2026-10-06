# test_q_learning.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.q_learning import QLearningAgent


def test_q_table_initialization():
    agent = QLearningAgent()

    state = (1, 0, -1)

    q_values = agent.get_q_values(state)

    assert len(q_values) == 4
    assert all(q == 0.0 for q in q_values)

    print("[PASS] Q-table initialization")


def test_q_update():
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.0,
        epsilon_min=0.0,
    )

    state = (1, 0, -1)
    next_state = (2, 1, 3)

    action = 3
    reward = 10

    updated_q = agent.update_q(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=False,
    )

    # 처음 Q값은 모두 0이므로
    #
    # Q = 0 + 0.1 * (10 + 0.9 * 0 - 0)
    #   = 1.0

    assert abs(updated_q - 1.0) < 1e-9

    print("[PASS] Q-value update")


def test_terminal_q_update():
    agent = QLearningAgent(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.0,
        epsilon_min=0.0,
    )

    state = (1, 0, -1)
    next_state = (2, 1, 3)

    action = 0
    reward = 100

    updated_q = agent.update_q(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=True,
    )

    # done=True이면 미래 Q-value를 사용하지 않음
    #
    # Q = 0 + 0.1 * (100 - 0)
    #   = 10

    assert abs(updated_q - 10.0) < 1e-9

    print("[PASS] Terminal Q-value update")


def test_best_action():
    agent = QLearningAgent(
        epsilon=0.0,
        epsilon_min=0.0,
    )

    state = (2, 1, 0)

    # State 생성
    agent.get_q_values(state)

    # 임의로 Q값 설정
    agent.q_table[state][0] = 1.0
    agent.q_table[state][1] = 2.0
    agent.q_table[state][2] = 3.0
    agent.q_table[state][3] = 10.0

    action = agent.choose_action(
        state,
        training=False,
    )

    assert action == 3

    print("[PASS] Greedy action selection")


def test_epsilon_decay():
    agent = QLearningAgent(
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.5,
    )

    agent.decay_epsilon()

    assert abs(agent.epsilon - 0.5) < 1e-9

    agent.decay_epsilon()

    assert abs(agent.epsilon - 0.25) < 1e-9

    print("[PASS] Epsilon decay")


def test_invalid_action():
    agent = QLearningAgent()

    state = (1, 0, -1)
    next_state = (1, 0, 0)

    try:
        agent.update_q(
            state,
            10,
            1,
            next_state,
            False,
        )

    except ValueError:
        print("[PASS] Invalid action handling")
        return

    raise AssertionError(
        "Invalid action did not raise ValueError."
    )


if __name__ == "__main__":

    test_q_table_initialization()
    test_q_update()
    test_terminal_q_update()
    test_best_action()
    test_epsilon_decay()
    test_invalid_action()

    print()
    print("==============================")
    print("All Q-Learning tests passed.")
    print("==============================")
