"""통합 학습의 호출 순서, 종료 조건, Episode 반복과 CSV 저장을 검증한다."""

import csv
import math
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import experiment


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
    first_reward = experiment.calculate_reward(-61.0, -65.0)
    last_reward = experiment.calculate_reward(-64.0, -61.0)
    assert events.mock_calls == [
        call.env.reset(),
        call.state(-65.0, None, config.NONE_ACTION),
        call.env.is_done(),
        call.agent.choose_action(initial),
        call.env.move(config.RIGHT),
        call.env.get_rssi(),
        call.state(-61.0, -65.0, config.RIGHT),
        call.reward(-61.0, -65.0),
        call.env.is_done(),
        call.agent.update_q(initial, config.RIGHT, first_reward, middle, False),
        call.agent.choose_action(middle),
        call.env.move(config.RIGHT),
        call.env.get_rssi(),
        call.state(-64.0, -61.0, config.RIGHT),
        call.reward(-64.0, -61.0),
        call.env.is_done(),
        call.agent.update_q(middle, config.RIGHT, last_reward, final, True),
        call.env.get_info(),
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
    assert [row["episode"] for row in results] == [1, 2, 3]
    assert all(row["steps"] == 2 and row["success"] is False for row in results)
    assert [args.args[-1] for args in agent.update_q.call_args_list] == [False, True] * 3


if __name__ == "__main__":
    test_episode_transition()
    test_initially_done()
    test_training_and_csv()
    print("All experiment tests passed.")
