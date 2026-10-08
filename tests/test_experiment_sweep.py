"""Reward sweep의 반복 실행, 집계, CSV 저장을 검증한다."""

import csv
import random
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

import pytest

from src import experiment_sweep


def make_reward_combination():
    return {
        "name": "test_reward",
        "reward_up": 2.0,
        "reward_keep": 0.0,
        "reward_down": -2.0,
        "move_cost": 0.2,
        "terminal_reward": 10.0,
    }


def test_evaluate_reward_combination():
    train_env = Mock()
    eval_env = Mock()
    agent = Mock()
    train_results = [
        {"success": True, "steps": 4, "total_reward": 8.0},
        {"success": False, "steps": 10, "total_reward": -2.0},
    ]
    evaluation = {
        "success_rate": 0.75,
        "average_steps": 6.0,
        "average_steps_on_success": 4.0,
        "failure_count": 1,
    }
    combination = make_reward_combination()
    expected_reward_config = {
        key: combination[key]
        for key in experiment_sweep.REWARD_FIELDS
    }

    with (
        patch.object(
            experiment_sweep,
            "build_environment",
            side_effect=[train_env, eval_env],
        ),
        patch.object(experiment_sweep, "build_agent", return_value=agent),
        patch.object(
            experiment_sweep,
            "run_episode",
            side_effect=train_results,
        ) as run_episode,
        patch.object(
            experiment_sweep,
            "evaluate_agent",
            return_value=evaluation,
        ) as evaluate_agent,
    ):
        result = experiment_sweep.evaluate_reward_combination(combination, 2, 4)

    assert run_episode.call_args_list == [
        call(train_env, agent, reward_config=expected_reward_config),
        call(train_env, agent, reward_config=expected_reward_config),
    ]
    evaluate_agent.assert_called_once_with(
        eval_env,
        agent,
        4,
        reward_config=expected_reward_config,
    )
    assert agent.decay_epsilon.call_count == 2
    assert result["train_success_rate"] == 0.5
    assert result["train_average_steps"] == 7.0
    assert result["train_average_total_reward"] == 3.0
    assert result["eval_success_rate"] == 0.75
    assert result["eval_average_steps_on_success"] == 4.0
    assert result["eval_failure_count"] == 1


def test_run_sweep_saves_csv():
    combination = make_reward_combination()
    with TemporaryDirectory() as directory:
        output_path = Path(directory) / "sweep.csv"
        with patch.object(experiment_sweep.config, "MAX_STEPS", 2):
            results = experiment_sweep.run_sweep(
                reward_combinations=[combination],
                train_episodes=2,
                eval_episodes=2,
                output_path=output_path,
            )

        with output_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    assert len(results) == len(rows) == 1
    assert results[0]["name"] == rows[0]["name"] == "test_reward"
    assert results[0]["train_episodes"] == 2
    assert results[0]["eval_episodes"] == 2
    assert results[0]["train_average_steps"] == 2.0
    assert results[0]["eval_average_steps"] == 2.0
    assert results[0]["eval_average_steps_on_success"] is None
    assert rows[0]["eval_average_steps_on_success"] == ""
    assert rows[0]["eval_failure_count"] == "2"


def _config_snapshot():
    return deepcopy({
        name: value
        for name, value in vars(experiment_sweep.config).items()
        if name.isupper()
    })


def test_reward_combinations_start_with_independent_agents_and_equal_conditions(tmp_path):
    """실제 학습 후에도 다음 조합은 빈 Q-table과 초기 epsilon/seed로 시작한다."""
    agents = []
    environments = []
    build_agent = experiment_sweep.build_agent
    build_environment = experiment_sweep.build_environment

    def record_agent():
        if agents:
            # 이전 조합이 실제로 학습되어 초기 상태와 달라졌음을 확인한다.
            assert agents[-1].q_table
            assert any(values.any() for values in agents[-1].q_table.values())
            assert agents[-1].epsilon < experiment_sweep.config.EPSILON
        agent = build_agent()
        assert all(agent is not previous for previous in agents)
        assert all(agent.q_table is not previous.q_table for previous in agents)
        assert all(agent.random is not previous.random for previous in agents)
        assert agent.q_table == {}
        assert agent.epsilon == experiment_sweep.config.EPSILON
        for name in ("alpha", "gamma", "epsilon_min", "epsilon_decay"):
            assert getattr(agent, name) == getattr(experiment_sweep.config, name.upper())
        assert agent.num_actions == 4
        assert agent.random.getstate() == random.Random(
            experiment_sweep.config.RANDOM_SEED
        ).getstate()
        assert _config_snapshot() == config_before
        agents.append(agent)
        return agent

    def record_environment():
        # 환경 생성 시점은 각 조합의 학습 seed, 평가 seed로 각각 초기화되어야 한다.
        seed = experiment_sweep.config.RANDOM_SEED + len(environments) % 2
        assert random.getstate() == random.Random(seed).getstate()
        env = build_environment()
        assert all(env is not previous for previous in environments)
        for name in (
            "grid_size", "agent_start", "target_position", "max_steps",
            "reference_rssi", "reference_distance", "path_loss_exponent", "noise_std",
        ):
            assert getattr(env, name) == getattr(experiment_sweep.config, name.upper())
        assert _config_snapshot() == config_before
        environments.append(env)
        return env

    with (
        patch.object(experiment_sweep.config, "MAX_STEPS", 4),
        patch.object(experiment_sweep, "build_agent", side_effect=record_agent) as factory,
        patch.object(experiment_sweep, "build_environment", side_effect=record_environment),
        patch.object(experiment_sweep, "run_episode", wraps=experiment_sweep.run_episode) as train,
        patch.object(experiment_sweep, "evaluate_agent", wraps=experiment_sweep.evaluate_agent) as evaluate,
    ):
        config_before = _config_snapshot()
        results = experiment_sweep.run_sweep(
            train_episodes=2, eval_episodes=3, output_path=tmp_path / "independence.csv",
        )
        assert _config_snapshot() == config_before

    combinations = experiment_sweep.DEFAULT_REWARD_COMBINATIONS
    assert factory.call_count == len(agents) == len(results) == len(combinations)
    assert len(environments) == 2 * len(combinations)
    assert train.call_count == 2 * len(combinations)
    assert evaluate.call_count == len(combinations)
    for index, combination in enumerate(combinations):
        reward_config = {field: combination[field] for field in experiment_sweep.REWARD_FIELDS}
        assert train.call_args_list[index * 2:(index + 1) * 2] == [
            call(environments[index * 2], agents[index], reward_config=reward_config),
        ] * 2
        assert evaluate.call_args_list[index] == call(
            environments[index * 2 + 1], agents[index], 3, reward_config=reward_config,
        )
        assert results[index]["train_episodes"] == 2
        assert results[index]["eval_episodes"] == 3


@pytest.mark.parametrize("failure_stage", ["run_episode", "evaluate_agent"])
def test_sweep_does_not_change_config_on_failure(tmp_path, failure_stage):
    """학습 또는 평가가 실패해도 글로벌 설정에 Reward 변경이 남지 않는다."""
    config_before = _config_snapshot()
    combinations_before = deepcopy(experiment_sweep.DEFAULT_REWARD_COMBINATIONS)
    with patch.object(experiment_sweep, failure_stage, side_effect=RuntimeError("test failure")):
        with pytest.raises(RuntimeError, match="test failure"):
            experiment_sweep.run_sweep(
                train_episodes=1, eval_episodes=1, output_path=tmp_path / "failed.csv",
            )
    assert _config_snapshot() == config_before
    assert experiment_sweep.DEFAULT_REWARD_COMBINATIONS == combinations_before


def test_sweep_is_reproducible_with_same_seed_and_independent_of_order(tmp_path):
    """동일 seed의 전체 지표/CSV가 같고 조합 순서와 외부 난수 소비에 영향받지 않는다."""
    combinations_before = deepcopy(experiment_sweep.DEFAULT_REWARD_COMBINATIONS)
    config_before = _config_snapshot()
    with (
        patch.object(experiment_sweep.config, "MAX_STEPS", 8),
        patch.object(experiment_sweep.config, "TARGET_POSITION", (1, 0)),
    ):
        first = experiment_sweep.run_sweep(
            train_episodes=6, eval_episodes=4, output_path=tmp_path / "first.csv",
        )
        # 앞선 실험에서 남은 전역 난수 상태에 의존하는 구현을 검출한다.
        for _ in range(17):
            random.random()
            random.gauss(0.0, 1.0)
        second = experiment_sweep.run_sweep(
            train_episodes=6, eval_episodes=4, output_path=tmp_path / "second.csv",
        )
        reordered = experiment_sweep.run_sweep(
            reward_combinations=list(reversed(experiment_sweep.DEFAULT_REWARD_COMBINATIONS)),
            train_episodes=6, eval_episodes=4, output_path=tmp_path / "reordered.csv",
        )

    assert len(first) == len(experiment_sweep.DEFAULT_REWARD_COMBINATIONS)
    assert first == second
    assert (tmp_path / "first.csv").read_bytes() == (tmp_path / "second.csv").read_bytes()
    assert {row["name"]: row for row in first} == {row["name"]: row for row in reordered}
    assert _config_snapshot() == config_before
    assert experiment_sweep.DEFAULT_REWARD_COMBINATIONS == combinations_before
