"""학습된 Agent의 재사용, 동일한 Noise 조건 적용, 성능 지표 계산, 실험 재현성 및 CSV 파일 저장을 검증한다."""

import csv
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

import config
from src import compare_algorithms as comparison
from src import experiment
from src.random_agent import RandomAgent


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    for name, value in {
        "GRID_SIZE": 4, "AGENT_START": (0, 0), "TARGET_POSITION": (3, 3),
        "MAX_STEPS": 6, "NOISE_STD": 2.0, "NUM_EPISODES": 3,
        "NUM_EVAL_EPISODES": 3, "RANDOM_SEED": 42, "LOG_INTERVAL": 0,
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def assert_numpy_state_equal(before):
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_training_reuses_existing_updates_and_decay():
    agent = experiment.build_agent(seed=0)
    with (
        patch.object(comparison, "build_agent", return_value=agent) as build,
        patch.object(comparison, "run_episode", wraps=experiment.run_episode) as run,
        patch.object(agent, "update_q", wraps=agent.update_q) as update,
        patch.object(agent, "decay_epsilon", wraps=agent.decay_epsilon) as decay,
    ):
        trained = comparison.train_q_learning_agent(3, seed=0)
    assert trained is agent
    build.assert_called_once_with(seed=0)
    assert run.call_count == decay.call_count == 3
    assert all(args.kwargs == {"training": True} for args in run.call_args_list)
    assert update.call_count == 18
    assert agent.q_table
    assert agent.epsilon == pytest.approx(config.EPSILON * config.EPSILON_DECAY ** 3)


def test_q_evaluation_uses_greedy_and_preserves_values_keys_and_epsilon():
    agent = comparison.train_q_learning_agent(2, seed=42)
    initial = experiment.make_state(-65.0, None, config.NONE_ACTION)
    agent.get_q_values(initial)
    agent.q_table[initial][:] = [0.0, 0.0, 0.0, 10.0]
    unseen = experiment.make_state(-45.0, -65.0, config.RIGHT)
    agent.q_table.pop(unseen, None)
    before = deepcopy(agent.q_table)
    epsilon = agent.epsilon
    env = comparison.build_environment()
    with (
        patch.object(comparison, "build_environment", return_value=env),
        patch.object(env, "_generate_rssi", side_effect=[-65.0] + [-45.0] * 6),
        patch.object(agent, "choose_action", wraps=agent.choose_action) as choose,
        patch.object(agent, "update_q", wraps=agent.update_q) as update,
        patch.object(agent, "decay_epsilon", wraps=agent.decay_epsilon) as decay,
        patch.object(comparison, "build_agent", side_effect=AssertionError("new agent")),
    ):
        results = comparison.evaluate_q_learning(agent, 1, seed=43)
    assert len(results) == 1
    assert choose.call_args_list[0] == call(initial, training=False)
    assert all(args.kwargs == {"training": False} for args in choose.call_args_list)
    update.assert_not_called()
    decay.assert_not_called()
    assert epsilon == agent.epsilon
    assert before.keys() == agent.q_table.keys()
    for state, values in before.items():
        np.testing.assert_array_equal(agent.q_table[state], values)
    assert "total_reward" not in results[0]


def test_random_reuses_agent_and_episode_without_state_or_reward():
    with (
        patch.object(comparison, "RandomAgent", wraps=RandomAgent) as agent_class,
        patch.object(comparison, "run_random_episode", wraps=comparison.run_random_episode) as run,
        patch.object(experiment, "make_state", side_effect=AssertionError("state")),
        patch.object(experiment, "calculate_reward", side_effect=AssertionError("reward")),
        patch.object(comparison, "build_agent", side_effect=AssertionError("Q-Learning")),
    ):
        results = comparison.evaluate_random_agent(3, seed=43)
    agent_class.assert_called_once_with(seed=43)
    assert run.call_count == 3
    agents = [args.args[1] for args in run.call_args_list]
    assert all(agent is agents[0] for agent in agents)
    assert isinstance(agents[0], RandomAgent)
    assert not hasattr(agents[0], "q_table")
    assert [row["episode"] for row in results] == [1, 2, 3]
    assert all(set(row) == {"episode", "success", "steps", "final_distance",
                            "target_x", "target_y", "shortest_path_steps"}
               for row in results)


def test_comparison_retrains_each_run_and_passes_same_trained_agent(tmp_path):
    agents, environments, evaluated = [], [], []
    real_train = comparison.train_q_learning_agent
    real_evaluate = comparison.evaluate_q_learning

    def record_build(seed):
        agent = experiment.build_agent(seed=seed)
        assert agent.q_table == {}
        agents.append(agent)
        return agent

    def record_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        environments.append(env)
        return env

    def check_evaluation(agent, eval_episodes, seed):
        assert agent is agents[-1]
        assert agent.q_table
        assert agent.epsilon < config.EPSILON
        evaluated.append(agent)
        return real_evaluate(agent, eval_episodes, seed=seed)

    with (
        patch.object(comparison, "build_agent", side_effect=record_build) as build,
        patch.object(comparison, "build_environment", side_effect=record_environment),
        patch.object(comparison, "train_q_learning_agent", wraps=real_train) as train,
        patch.object(comparison, "evaluate_q_learning", side_effect=check_evaluation) as evaluate,
        patch.object(comparison, "evaluate_random_agent", wraps=comparison.evaluate_random_agent) as rand,
    ):
        result = comparison.compare_algorithms(2, 3, seed=42, runs=2, output_dir=tmp_path)

    assert build.call_args_list == [call(seed=42), call(seed=44)]
    assert train.call_args_list == [call(2, seed=42), call(2, seed=44)]
    assert agents[0] is not agents[1]
    assert evaluated == agents
    assert [args.args[1] for args in evaluate.call_args_list] == [3, 3]
    assert rand.call_args_list == [call(3, seed=43), call(3, seed=45)]
    assert len(environments) == 6
    assert len({id(env) for env in environments}) == 6
    for env in environments:
        for name in (
            "GRID_SIZE", "MAX_STEPS", "AGENT_START", "TARGET_POSITION",
            "CARRIER_FREQUENCY_HZ", "TX_POWER_DBM", "PATH_LOSS_EXPONENT", "NOISE_STD",
        ):
            assert getattr(env, name.lower()) == getattr(config, name)
    assert len(result["episodes"]) == 12
    for run in (1, 2):
        for algorithm in ("q_learning", "random"):
            rows = [row for row in result["episodes"]
                    if row["run"] == run and row["algorithm"] == algorithm]
            assert [row["episode"] for row in rows] == [1, 2, 3]
            assert all(row["train_seed"] == 42 + 2 * (run - 1) for row in rows)
            assert all(row["eval_seed"] == 43 + 2 * (run - 1) for row in rows)
    for row in result["summary"]:
        assert row["train_episodes"] == (2 if row["algorithm"] == "q_learning" else 0)


def test_paired_noise_restarts_per_episode_despite_different_path_lengths():
    # Q goes right to target in two steps; random is forced against a boundary
    # for six steps. Compare noise, not RSSI: measurement positions differ.
    agent = experiment.build_agent(seed=42)
    traces = []
    current = []
    real_gauss = random.gauss

    def record_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        env.target_position = (2, 0)
        original_reset = env.reset

        def reset():
            nonlocal current
            current = []
            traces.append(current)
            return original_reset()

        env.reset = reset
        return env

    def record_noise(mu, sigma):
        value = real_gauss(mu, sigma)
        current.append(value)
        return value

    random_agent = Mock(spec=RandomAgent)
    random_agent.choose_action.return_value = config.UP
    with (
        patch.object(comparison, "build_environment", side_effect=record_environment),
        patch.object(agent, "choose_action", return_value=config.RIGHT),
        patch.object(comparison, "RandomAgent", return_value=random_agent),
        patch("src.environment.random.gauss", side_effect=record_noise),
    ):
        q_results = comparison.evaluate_q_learning(agent, 3, seed=43)
        random_results = comparison.evaluate_random_agent(3, seed=43)
    assert all(row["steps"] == 2 and row["success"] for row in q_results)
    assert all(row["steps"] == 6 and not row["success"] for row in random_results)
    assert [len(trace) for trace in traces] == [3, 3, 3, 7, 7, 7]
    for episode in range(3):
        expected_rng = random.Random(f"43:{episode + 1}")
        expected = [expected_rng.gauss(0.0, config.NOISE_STD) for _ in range(7)]
        assert traces[episode] == traces[episode + 3][:3] == expected[:3]
        assert traces[episode + 3] == expected


@pytest.mark.parametrize("results,expected", [
    ([
        {"success": True, "steps": 2, "final_distance": 0.0},
        {"success": False, "steps": 10, "final_distance": 6.0},
        {"success": True, "steps": 4, "final_distance": 0.0},
    ], {"success_count": 2, "failure_count": 1, "success_rate": 2 / 3,
        "average_steps": 16 / 3, "average_steps_on_success": 3.0,
        "average_final_distance": 2.0}),
    ([{"success": False, "steps": 5, "final_distance": 3.0}],
     {"success_count": 0, "failure_count": 1, "success_rate": 0.0,
      "average_steps": 5.0, "average_steps_on_success": None,
      "average_final_distance": 3.0}),
    ([{"success": True, "steps": 0, "final_distance": 0.0}],
     {"success_count": 1, "failure_count": 0, "success_rate": 1.0,
      "average_steps": 0.0, "average_steps_on_success": 0.0,
      "average_final_distance": 0.0}),
])
def test_metrics(results, expected):
    assert comparison.calculate_metrics(results) == expected


def test_empty_metrics_rejected():
    with pytest.raises(ValueError, match="at least one"):
        comparison.calculate_metrics([])


def test_reproducibility_csv_fields_and_existing_file_preservation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    originals = {}
    for filename in ("training_results.csv", "random_results.csv", "final_evaluation_results.csv"):
        path = tmp_path / filename
        path.write_bytes(b"existing result\r\n")
        originals[path] = path.read_bytes()
    first = comparison.compare_algorithms(seed=0, runs=2)
    paths = [tmp_path / name for name in (
        "algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv",
    )]
    first_bytes = [path.read_bytes() for path in paths]
    for path in paths:
        path.write_text("stale comparison", encoding="utf-8")
    random.seed(999)
    random.gauss(0.0, 2.0)
    np.random.seed(999)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    second = comparison.compare_algorithms(seed=0, runs=2)

    assert first == second
    assert random.getstate() == python_before
    assert_numpy_state_equal(numpy_before)
    assert first_bytes == [path.read_bytes() for path in paths]
    for path, original in originals.items():
        assert path.read_bytes() == original
    for path, key, fields in zip(paths, ("episodes", "summary"),
                                 (comparison.EPISODE_FIELDS, comparison.SUMMARY_FIELDS)):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == list(fields)
            rows = list(reader)
        assert rows == [{key: "" if value is None else str(value)
                         for key, value in row.items()} for row in second[key]]
        assert not any("reward" in field for field in fields)
    assert all((row["average_steps_on_success"] is None) == (row["success_count"] == 0)
               for row in second["summary"])
    assert all(row["train_episodes"] == config.NUM_EPISODES
               for row in second["summary"] if row["algorithm"] == "q_learning")


def test_csv_relative_directory_uses_project_root_from_other_cwd(tmp_path, monkeypatch):
    root = Path(comparison.__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    # Simulate a module located in a temporary project to avoid test artifacts.
    monkeypatch.setattr(comparison, "__file__", str(tmp_path / "project" / "src" / "compare_algorithms.py"))
    episodes, summary = comparison.save_comparison_results([], [], "relative_results")
    assert episodes.parent == summary.parent == tmp_path / "project" / "relative_results"
    assert root != episodes.parent
    assert episodes.exists() and summary.exists()


@pytest.mark.parametrize("argument,value", [
    ("train_episodes", 0), ("train_episodes", -1), ("eval_episodes", 0),
    ("eval_episodes", -1), ("runs", 0), ("runs", -1), ("runs", 1.5),
])
def test_invalid_counts_fail_before_training_or_saving(argument, value, tmp_path):
    with patch.object(comparison, "train_q_learning_agent") as train:
        with pytest.raises(ValueError, match=argument):
            comparison.compare_algorithms(**{argument: value}, output_dir=tmp_path / "absent")
    train.assert_not_called()
    assert not (tmp_path / "absent").exists()


def test_training_failure_restores_rngs():
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    with patch.object(comparison, "run_episode", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            comparison.train_q_learning_agent(seed=0)
    assert random.getstate() == python_before
    assert_numpy_state_equal(numpy_before)


def test_evaluation_failure_restores_noise_rng():
    python_before = random.getstate()
    with patch.object(comparison, "run_random_episode", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            comparison.evaluate_random_agent(seed=0)
    assert random.getstate() == python_before


def test_noise_override_is_available_without_modifying_config():
    original = config.NOISE_STD
    environments = []

    def record_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        environments.append(env)
        return env

    with patch.object(comparison, "build_environment", side_effect=record_environment):
        agent = comparison.train_q_learning_agent(1, noise_std=0.0)
        comparison.evaluate_q_learning(agent, 1, noise_std=0.0)
        comparison.evaluate_random_agent(1, noise_std=0.0)
    assert all(env.noise_std == 0.0 for env in environments)
    assert config.NOISE_STD == original


def test_cli_options(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "compare_algorithms", "--train-episodes", "2", "--eval-episodes", "3",
        "--seed", "0", "--runs", "2", "--output-dir", str(tmp_path),
    ])
    with patch.object(comparison, "compare_algorithms") as compare:
        comparison.main()
    compare.assert_called_once_with(
        train_episodes=2, eval_episodes=3, seed=0, runs=2, output_dir=tmp_path,
    )


@pytest.mark.parametrize("option", ["--train-episodes", "--eval-episodes", "--runs"])
def test_cli_rejects_nonpositive_counts(option, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["compare_algorithms", option, "0"])
    with pytest.raises(SystemExit) as error:
        comparison.main()
    assert error.value.code == 2
    assert f"{option} must be greater than 0" in capsys.readouterr().err


def test_module_entrypoint(tmp_path):
    process = subprocess.run(
        [sys.executable, "-m", "src.compare_algorithms", "--train-episodes", "1",
         "--eval-episodes", "2", "--seed", "0", "--runs", "2",
         "--output-dir", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    assert "Run 1/2" in process.stdout and "Run 2/2" in process.stdout
    with (tmp_path / "algorithm_comparison_episodes.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 8
    with (tmp_path / "algorithm_comparison_summary.csv").open(newline="", encoding="utf-8") as file:
        assert len(list(csv.DictReader(file))) == 4
