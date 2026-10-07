"""Reward sweep의 반복 실행, 집계, CSV 저장을 검증한다."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, call, patch

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
    assert rows[0]["eval_average_steps_on_success"] == ""
    assert rows[0]["eval_failure_count"] == "2"
