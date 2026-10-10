"""통합 학습의 호출 순서, 종료 조건, Episode 반복과 CSV 저장을 검증한다."""

import csv
import math
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import experiment


def test_build_agent():
    """완성형 Agent 생성자에 config의 학습 설정과 seed를 전달하는지 확인한다."""
    with patch.object(experiment, "QLearningAgent") as agent_class:
        agent = experiment.build_agent()

    agent_class.assert_called_once_with(
        num_actions=4,
        alpha=config.ALPHA,
        gamma=config.GAMMA,
        epsilon=config.EPSILON,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        seed=config.RANDOM_SEED,
    )
    assert agent is agent_class.return_value


def test_episode_transition():
    """State/Reward가 동일 RSSI를 쓰고 종료 Step까지 Q를 갱신하는지 확인한다."""
    real_env = experiment.build_environment()
    real_env.agent_start = (0, 0)
    real_env.target_position = (2, 0)
    real_env.max_steps = 5
    env = Mock(wraps=real_env)
    agent = Mock(wraps=experiment.build_agent())
    agent.choose_action.return_value = config.RIGHT
    events = Mock()
    events.attach_mock(env, "env")
    events.attach_mock(agent, "agent")

    with (
        patch.object(real_env, "_generate_rssi", side_effect=[-65.0, -61.0, -64.0]),
        patch.object(experiment, "make_state", wraps=experiment.make_state) as state,
        patch.object(experiment, "calculate_reward", wraps=experiment.calculate_reward) as reward,
    ):
        events.attach_mock(state, "state")
        events.attach_mock(reward, "reward")
        result = experiment.run_episode(env, agent)

    initial = experiment.make_state(-65.0, None, config.NONE_ACTION)
    middle = experiment.make_state(-61.0, -65.0, config.RIGHT)
    final = experiment.make_state(-64.0, -61.0, config.RIGHT)
    reward_config = experiment.get_reward_config()
    first_reward = experiment.calculate_reward(
        -61.0, -65.0, success=False, blocked=False, **reward_config
    )
    last_reward = experiment.calculate_reward(
        -64.0, -61.0, success=True, blocked=False, **reward_config
    )
    assert events.mock_calls == [
        call.env.reset(),
        call.state(-65.0, None, config.NONE_ACTION),
        call.env.is_done(),
        call.agent.choose_action(initial),
        call.env.move(config.RIGHT),
        call.env.get_rssi(),
        call.state(-61.0, -65.0, config.RIGHT),
        call.env.is_done(),
        call.reward(-61.0, -65.0, success=False, blocked=False, **reward_config),
        call.agent.update_q(initial, config.RIGHT, first_reward, middle, False),
        call.agent.choose_action(middle),
        call.env.move(config.RIGHT),
        call.env.get_rssi(),
        call.state(-64.0, -61.0, config.RIGHT),
        call.env.is_done(),
        call.env.get_info(),
        call.reward(-64.0, -61.0, success=True, blocked=False, **reward_config),
        call.agent.update_q(middle, config.RIGHT, last_reward, final, True),
    ]
    assert result["success"] is True
    assert result["steps"] == 2
    assert result["final_distance"] == 0.0
    assert math.isclose(result["total_reward"], first_reward + last_reward)


def test_initially_done():
    """시작 위치가 Target이면 추가 Action이나 Q 업데이트가 없어야 한다."""
    env = experiment.build_environment()
    env.target_position = env.agent_start
    agent = Mock()
    result = experiment.run_episode(env, agent)
    assert result == {
        "success": True, "steps": 0, "total_reward": 0.0, "final_distance": 0.0,
    }
    assert agent.mock_calls == []


def test_evaluation_episode_uses_greedy_policy_without_learning():
    """평가에서는 exploration, Q 업데이트, epsilon decay를 수행하지 않는다."""
    env = experiment.build_environment()
    env.agent_start = (0, 0)
    env.target_position = (2, 0)
    env.max_steps = 5
    agent = Mock()
    agent.choose_action.return_value = config.RIGHT

    with patch.object(env, "_generate_rssi", side_effect=[-65.0, -61.0, -64.0]):
        result = experiment.run_episode(env, agent, training=False)

    initial = experiment.make_state(-65.0, None, config.NONE_ACTION)
    middle = experiment.make_state(-61.0, -65.0, config.RIGHT)
    assert agent.choose_action.call_args_list == [
        call(initial, training=False),
        call(middle, training=False),
    ]
    agent.update_q.assert_not_called()
    agent.decay_epsilon.assert_not_called()
    assert result["success"] is True
    assert result["steps"] == 2


def test_evaluate_agent_metrics():
    """평가 결과에서 성공률, 평균 Step, 성공 Step, 실패 수를 계산한다."""
    env = Mock()
    agent = Mock()
    episode_results = [
        {"success": True, "steps": 4},
        {"success": False, "steps": 10},
        {"success": True, "steps": 6},
    ]

    with patch.object(
        experiment,
        "run_episode",
        side_effect=episode_results,
    ) as run_episode:
        result = experiment.evaluate_agent(env, agent, 3)

    assert run_episode.call_args_list == [
        call(env, agent, reward_config=None, training=False),
        call(env, agent, reward_config=None, training=False),
        call(env, agent, reward_config=None, training=False),
    ]
    agent.decay_epsilon.assert_not_called()
    assert result == {
        "success_rate": 2 / 3,
        "average_steps": 20 / 3,
        "average_steps_on_success": 5.0,
        "failure_count": 1,
    }


def _build_evaluation_integrity_case():
    env = experiment.build_environment()
    env.agent_start = (0, 0)
    env.target_position = (2, 2)
    env.max_steps = 2
    agent = experiment.build_agent()
    initial = experiment.make_state(-65.0, None, config.NONE_ACTION)
    learned_next = experiment.make_state(-61.0, -65.0, config.RIGHT)
    agent.update_q(initial, config.RIGHT, 10.0, learned_next, False)
    agent.decay_epsilon()
    unseen = experiment.make_state(-45.0, -65.0, config.RIGHT)
    assert unseen not in agent.q_table
    return env, agent


def test_evaluate_agent_preserves_epsilon():
    """실제 학습된 Agent의 epsilon은 여러 평가 Episode 후에도 정확히 같다."""
    env, agent = _build_evaluation_integrity_case()
    epsilon_before = agent.epsilon

    with patch.object(env, "_generate_rssi", side_effect=[-65.0, -45.0, -45.0] * 3):
        result = experiment.evaluate_agent(env, agent, 3)

    assert result["average_steps"] == 2.0
    assert epsilon_before == agent.epsilon


def test_evaluate_agent_preserves_q_table():
    """기존 Q-value뿐 아니라 처음 보는 State의 key도 평가 중 추가하지 않는다."""
    env, agent = _build_evaluation_integrity_case()
    q_table_before = deepcopy(agent.q_table)

    with patch.object(env, "_generate_rssi", side_effect=[-65.0, -45.0, -45.0] * 3):
        result = experiment.evaluate_agent(env, agent, 3)

    assert result["average_steps"] == 2.0
    assert set(q_table_before) == set(agent.q_table)
    for state, q_values in q_table_before.items():
        np.testing.assert_array_equal(q_values, agent.q_table[state])


def test_training_and_csv():
    """Episode별 reset, 최대 Step 종료, Agent 재사용 및 CSV 내용을 확인한다."""
    real_env = experiment.build_environment()
    real_env.agent_start = (0, 0)
    real_env.target_position = (2, 2)
    real_env.max_steps = 2
    env = Mock(wraps=real_env)
    agent = Mock(wraps=experiment.build_agent())
    agent.choose_action.return_value = config.UP

    with (
        TemporaryDirectory() as directory,
        patch.object(config, "NUM_EPISODES", 3),
        patch.object(config, "LOG_INTERVAL", 2),
        patch.object(config, "RESULTS_DIR", directory),
        patch.object(experiment, "build_environment", return_value=env) as build_env,
        patch.object(experiment, "build_agent", return_value=agent) as build_agent,
    ):
        results = experiment.train()
        with (Path(directory) / "training_results.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == [
                "episode", "success", "steps", "total_reward", "final_distance",
            ]
            rows = list(reader)
        assert rows == [{key: str(value) for key, value in row.items()} for row in results]

    build_env.assert_called_once_with()
    build_agent.assert_called_once_with()
    assert env.reset.call_count == 3
    assert env.get_rssi.call_count == agent.update_q.call_count == 6
    assert agent.decay_epsilon.call_count == 3
    assert [row["episode"] for row in results] == [1, 2, 3]
    assert all(row["steps"] == 2 and row["success"] is False for row in results)
    assert [args.args[-1] for args in agent.update_q.call_args_list] == [False, True] * 3


if __name__ == "__main__":
    test_build_agent()
    test_episode_transition()
    test_initially_done()
    test_evaluation_episode_uses_greedy_policy_without_learning()
    test_evaluate_agent_metrics()
    test_evaluate_agent_preserves_epsilon()
    test_evaluate_agent_preserves_q_table()
    test_training_and_csv()
    print("All experiment tests passed.")
