"""고정 설정의 최종 학습/평가 경로와 summary CSV를 검증한다."""

import csv
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

import config
from src import experiment, experiment_sweep, final_evaluation


def _assert_summary_csv(summary, output_path):
    with output_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        required = {
            "eval_episodes", "success_count", "failure_count", "success_rate",
            "average_steps", "average_steps_on_success",
        }
        assert required <= set(reader.fieldnames)
    assert rows == [{
        key: "" if value is None else str(value)
        for key, value in summary.items()
    }]


def test_final_experiment_reuses_trained_agent_without_learning_in_evaluation(tmp_path):
    """최종 경로는 학습 Agent/별도 환경을 전달하며 실제 평가와 CSV가 일치한다."""
    train_calls = []
    evaluation_results = []
    run_episode = experiment.run_episode

    def record_training(env, agent, reward_config):
        train_calls.append((env, agent, reward_config))
        return run_episode(env, agent, reward_config=reward_config)

    def check_evaluation(env, agent, num_episodes, reward_config):
        assert train_calls
        assert all(agent is trained_agent for _, trained_agent, _ in train_calls)
        assert all(env is not train_env for train_env, _, _ in train_calls)
        assert agent.q_table
        assert agent.epsilon < config.EPSILON
        q_before = deepcopy(agent.q_table)
        epsilon_before = agent.epsilon
        with (
            patch.object(agent, "update_q", wraps=agent.update_q) as update,
            patch.object(agent, "decay_epsilon", wraps=agent.decay_epsilon) as decay,
        ):
            result = experiment.evaluate_agent(env, agent, num_episodes, reward_config)
        update.assert_not_called()
        decay.assert_not_called()
        assert agent.epsilon == epsilon_before
        assert set(agent.q_table) == set(q_before)
        for state, values in q_before.items():
            np.testing.assert_array_equal(agent.q_table[state], values)
        evaluation_results.append(result)
        return result

    with (
        patch.object(config, "MAX_STEPS", 4),
        patch.object(experiment_sweep, "build_agent", wraps=experiment.build_agent) as build_agent,
        patch.object(experiment_sweep, "run_episode", side_effect=record_training),
        patch.object(experiment_sweep, "evaluate_agent", side_effect=check_evaluation) as evaluate,
    ):
        output_path = tmp_path / "nested" / "final.csv"
        summary = final_evaluation.run_final_experiment(2, 3, output_path)

    build_agent.assert_called_once_with()
    assert len(train_calls) == 2
    assert evaluate.call_count == len(evaluation_results) == 1
    evaluation = evaluation_results[0]
    assert summary["eval_episodes"] == 3
    assert summary["success_count"] == 3 - evaluation["failure_count"]
    for field in ("failure_count", "success_rate", "average_steps", "average_steps_on_success"):
        assert summary[field] == evaluation[field]
    assert summary["seed"] == config.RANDOM_SEED
    assert summary["eval_seed"] == config.RANDOM_SEED + 1
    assert summary["train_episodes"] == 2
    assert summary["max_steps"] == 4
    expected_reward = experiment.get_reward_config()
    assert all(reward == expected_reward for _, _, reward in train_calls)
    assert {field: summary[field] for field in expected_reward} == expected_reward
    _assert_summary_csv(summary, output_path)


def test_final_experiment_restarts_reproducibly_and_overwrites_csv(tmp_path):
    """새 실행은 Agent를 재사용하지 않으며 성공 0건은 None/빈 CSV 값으로 남긴다."""
    agents = []

    def record_agent():
        agent = experiment.build_agent()
        assert agent.q_table == {}
        assert agent.epsilon == config.EPSILON
        agents.append(agent)
        return agent

    with (
        patch.object(config, "MAX_STEPS", 2),
        patch.object(config, "NUM_EPISODES", 2),
        patch.object(config, "NUM_EVAL_EPISODES", 3),
        patch.object(config, "RESULTS_DIR", str(tmp_path)),
        patch.object(config, "REWARD_UP", config.REWARD_UP + 1.0),
        patch.object(experiment_sweep, "build_agent", side_effect=record_agent),
    ):
        first = final_evaluation.run_final_experiment()
        output_path = tmp_path / "final_evaluation_results.csv"
        first_bytes = output_path.read_bytes()
        output_path.write_text("old result\n", encoding="utf-8")
        second = final_evaluation.run_final_experiment()
        assert second["reward_up"] == config.REWARD_UP

    assert len(agents) == 2
    assert agents[0] is not agents[1]
    assert first == second
    assert output_path.read_bytes() == first_bytes
    assert second["eval_episodes"] == second["failure_count"] == 3
    assert second["success_count"] == 0
    assert second["success_rate"] == 0.0
    assert second["average_steps_on_success"] is None
    _assert_summary_csv(second, output_path)


@pytest.mark.parametrize("train_episodes,eval_episodes", [(0, 1), (1, 0)])
def test_final_experiment_rejects_nonpositive_episode_counts(tmp_path, train_episodes, eval_episodes):
    output_path = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="episodes must be greater than 0"):
        final_evaluation.run_final_experiment(train_episodes, eval_episodes, output_path)
    assert not output_path.exists()


def test_final_evaluation_cli_passes_episode_counts_and_output(tmp_path):
    output_path = tmp_path / "cli.csv"
    with (
        patch("sys.argv", [
            "final_evaluation", "--train-episodes", "2", "--eval-episodes", "3",
            "--output", str(output_path),
        ]),
        patch.object(final_evaluation, "run_final_experiment") as run,
    ):
        final_evaluation.main()
    run.assert_called_once_with(train_episodes=2, eval_episodes=3, output_path=output_path)
