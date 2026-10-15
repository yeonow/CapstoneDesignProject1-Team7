"""Noise별 비교, 새 Agent 학습, 대응되는 난수 시퀀스, 통계와 그래프를 검증한다."""

import csv
import math
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
from src import compare_noise_sweep as sweep
from src import experiment
from src import experiment_noise_sweep as old_sweep
from src import plot_noise_comparison as plotting
from src.random_agent import RandomAgent


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    for name, value in {
        "GRID_SIZE": 4, "MAX_STEPS": 4, "AGENT_START": (0, 0),
        "TARGET_POSITION": (3, 3), "NOISE_STD": 2.0,
        "NUM_EVAL_EPISODES": 3, "LOG_INTERVAL": 0,
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def assert_rngs_unchanged(python_state, numpy_state):
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]


def test_defaults_reuse_existing_noise_sweep():
    assert sweep.DEFAULT_NOISE_STDS == old_sweep.DEFAULT_NOISE_STDS == (0.0, 5.0, 10.0)
    assert sweep.DEFAULT_SEEDS == old_sweep.DEFAULT_SEEDS
    assert sweep.DEFAULT_TRAIN_EPISODES == old_sweep.DEFAULT_TRAIN_EPISODES == 1000


def test_each_noise_run_trains_fresh_agent_then_evaluates_without_learning(tmp_path):
    agents, environments, evaluations = [], [], []
    real_evaluate = comparison.evaluate_q_learning

    def build_agent(seed):
        agent = experiment.build_agent(seed=seed)
        assert agent.q_table == {}
        agents.append(agent)
        return agent

    def build_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        environments.append(env)
        return env

    def evaluate(agent, count, seed, noise_std):
        assert agent is agents[-1]
        assert agent.q_table
        before, epsilon = deepcopy(agent.q_table), agent.epsilon
        with (
            patch.object(agent, "choose_action", wraps=agent.choose_action) as choose,
            patch.object(agent, "update_q", wraps=agent.update_q) as update,
            patch.object(agent, "decay_epsilon", wraps=agent.decay_epsilon) as decay,
        ):
            rows = real_evaluate(agent, count, seed=seed, noise_std=noise_std)
        assert all(args.kwargs == {"training": False} for args in choose.call_args_list)
        update.assert_not_called()
        decay.assert_not_called()
        assert before.keys() == agent.q_table.keys()
        for state in before:
            np.testing.assert_array_equal(before[state], agent.q_table[state])
        assert agent.epsilon == epsilon
        evaluations.append((agent, seed, noise_std, count))
        return rows

    with (
        patch.object(comparison, "build_agent", side_effect=build_agent) as builder,
        patch.object(comparison, "build_environment", side_effect=build_environment),
        patch.object(sweep, "evaluate_q_learning", side_effect=evaluate),
    ):
        result = sweep.run_noise_comparison([0, 5], [42, 123], 2, 3, tmp_path)
    assert builder.call_args_list == [call(seed=42), call(seed=125)] * 2
    assert len(agents) == len({id(agent.q_table) for agent in agents}) == 4
    assert [entry[0] for entry in evaluations] == agents
    assert [(seed, noise, count) for _, seed, noise, count in evaluations] == [
        (43, 0.0, 3), (126, 0.0, 3), (43, 5.0, 3), (126, 5.0, 3),
    ]
    assert len(environments) == len({id(env) for env in environments}) == 12
    for index, env in enumerate(environments):
        assert env.noise_std == (0.0 if index < 6 else 5.0)
        for setting in (
            "GRID_SIZE", "MAX_STEPS", "AGENT_START", "TARGET_POSITION",
            "REFERENCE_RSSI", "REFERENCE_DISTANCE", "PATH_LOSS_EXPONENT",
        ):
            assert getattr(env, setting.lower()) == getattr(config, setting)
    assert config.NOISE_STD == 2.0
    assert len(result["episodes"]) == 24
    assert len(result["runs"]) == 8
    assert len(result["summary"]) == 4
    for row in result["runs"]:
        assert row["train_episodes"] == (2 if row["algorithm"] == "q_learning" else 0)
        assert row["eval_episodes"] == 3


def test_random_paths_and_outcomes_ignore_noise_but_measure_rssi():
    environments, paths, measurements = [], [], []

    def build_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        environments.append(env)
        moves, rssi = [], []
        paths.append(moves)
        measurements.append(rssi)
        reset, move = env.reset, env.move

        def record_reset():
            value = reset()
            moves.append(("reset", env.agent_position))
            rssi.append(value)
            return value

        def record_move(action):
            blocked = move(action)
            moves.append((action, env.agent_position))
            return blocked

        def get_rssi():
            rssi.append(env.current_rssi)
            return env.current_rssi

        env.reset, env.move, env.get_rssi = record_reset, record_move, get_rssi
        return env

    outcomes = []
    with (
        patch.object(comparison, "build_environment", side_effect=build_environment),
        patch.object(comparison, "RandomAgent", wraps=RandomAgent) as agent_class,
        patch.object(experiment, "make_state", side_effect=AssertionError("state")),
        patch.object(experiment, "calculate_reward", side_effect=AssertionError("reward")),
        patch.object(comparison, "run_episode", side_effect=AssertionError("Q-Learning")),
        patch("src.environment.random.gauss", wraps=random.gauss) as noise,
    ):
        for std in (0.0, 5.0, 10.0):
            outcomes.append(sweep.evaluate_random_agent(3, seed=43, noise_std=std))
    assert agent_class.call_args_list == [call(seed=43)] * 3
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert paths[0] == paths[1] == paths[2]
    assert all(len(values) == 15 for values in measurements)
    assert measurements[0] != measurements[1] != measurements[2]
    assert noise.call_count == 45
    assert [args.args[1] for args in noise.call_args_list] == [0.0] * 15 + [5.0] * 15 + [10.0] * 15


@pytest.mark.parametrize("std", [0.0, 5.0, 10.0])
def test_existing_episode_noise_seed_rule_pairs_noise(std):
    agent = experiment.build_agent(seed=42)
    traces, current = [], []
    gaussian = random.gauss

    def build_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        reset = env.reset

        def record_reset():
            nonlocal current
            current = []
            traces.append(current)
            return reset()

        env.reset = record_reset
        return env

    def record_noise(mu, sigma):
        value = gaussian(mu, sigma)
        current.append(value)
        return value

    with (
        patch.object(comparison, "build_environment", side_effect=build_environment),
        patch("src.environment.random.gauss", side_effect=record_noise),
    ):
        sweep.evaluate_q_learning(agent, 2, seed=43, noise_std=std)
        sweep.evaluate_random_agent(2, seed=43, noise_std=std)
    for index in range(2):
        rng = random.Random(f"43:{index + 1}")
        expected = [rng.gauss(0.0, std) for _ in range(5)]
        assert traces[index] == traces[index + 2] == expected


def make_run(success_count, success_steps, final_distance=2.0):
    return {
        "noise_std": 5.0, "algorithm": "q_learning",
        "train_episodes": 2, "eval_episodes": 4,
        "success_count": success_count, "failure_count": 4 - success_count,
        "success_rate": success_count / 4, "average_steps": 4.0 + success_count,
        "average_steps_on_success": success_steps, "average_final_distance": final_distance,
    }


def test_aggregation_sample_std_and_metric_specific_valid_counts():
    rows = [make_run(1, 2.0, 4.0), make_run(3, 4.0, 2.0), make_run(0, None, 6.0)]
    summary = sweep.aggregate_noise_comparison(rows)[0]
    assert summary["run_count"] == 3
    assert summary["success_rate_mean"] == pytest.approx(1 / 3)
    assert summary["success_rate_std"] == pytest.approx(math.sqrt(7 / 48))
    assert summary["average_steps_mean"] == pytest.approx(16 / 3)
    assert summary["average_steps_std"] == pytest.approx(math.sqrt(7 / 3))
    assert summary["average_steps_on_success_mean"] == 3.0
    assert summary["average_steps_on_success_std"] == pytest.approx(math.sqrt(2))
    assert summary["average_steps_on_success_run_count"] == 2
    assert summary["average_final_distance_mean"] == 4.0
    assert summary["average_final_distance_std"] == 2.0
    assert summary["success_count_mean"] == pytest.approx(4 / 3)
    assert summary["failure_count_mean"] == pytest.approx(8 / 3)
    assert summary["success_rate_run_count"] == 3


def test_zero_or_one_valid_successful_run_is_not_treated_as_zero():
    summary = sweep.aggregate_noise_comparison([make_run(0, None)])[0]
    assert summary["run_count"] == 1
    assert summary["success_rate_std"] == 0.0
    assert summary["average_steps_on_success_mean"] is None
    assert summary["average_steps_on_success_std"] is None
    assert summary["average_steps_on_success_run_count"] == 0
    summary = sweep.aggregate_noise_comparison([make_run(0, None), make_run(1, 8.0)])[0]
    assert summary["average_steps_on_success_mean"] == 8.0
    assert summary["average_steps_on_success_std"] == 0.0
    assert summary["average_steps_on_success_run_count"] == 1


def test_aggregation_rejects_empty_or_mixed_episode_counts():
    with pytest.raises(ValueError, match="empty"):
        sweep.aggregate_noise_comparison([])
    rows = [make_run(0, None), make_run(1, 2.0)]
    rows[1]["eval_episodes"] = 5
    with pytest.raises(ValueError, match="counts must match"):
        sweep.aggregate_noise_comparison(rows)


def test_reproducibility_all_csv_fields_overwrite_and_existing_files_preserved(tmp_path):
    old_files = {}
    for name in (
        "training_results.csv", "random_results.csv", "final_evaluation_results.csv",
        "algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv",
        "noise_sweep_runs.csv", "noise_sweep_summary.csv",
    ):
        path = tmp_path / name
        path.write_bytes(b"preserved result\r\n")
        old_files[path] = path.read_bytes()
    first = sweep.run_noise_comparison([0, 5], [0, 42], 2, 3, tmp_path)
    paths = [tmp_path / f"noise_comparison_{kind}.csv" for kind in ("episodes", "runs", "summary")]
    contents = [path.read_bytes() for path in paths]
    for path in paths:
        path.write_text("stale data", encoding="utf-8")
    random.seed(999)
    random.gauss(0.0, 1.0)
    np.random.seed(999)
    python_before, numpy_before = random.getstate(), np.random.get_state()
    second = sweep.run_noise_comparison([0, 5], [0, 42], 2, 3, tmp_path)
    assert first == second
    assert_rngs_unchanged(python_before, numpy_before)
    assert contents == [path.read_bytes() for path in paths]
    for path, content in old_files.items():
        assert path.read_bytes() == content
    for path, key, fields in zip(paths, ("episodes", "runs", "summary"), (
        sweep.NOISE_EPISODE_FIELDS, sweep.NOISE_RUN_FIELDS, sweep.NOISE_SUMMARY_FIELDS,
    )):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            assert reader.fieldnames == list(fields)
            assert list(reader) == [{name: "" if value is None else str(value)
                                    for name, value in row.items()} for row in second[key]]
        assert not any("reward" in name for name in fields)
    assert all(row["average_steps_on_success_mean"] is None for row in second["summary"])


def test_default_output_uses_project_root_even_from_other_cwd(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "__file__", str(tmp_path / "project" / "src" / "compare_noise_sweep.py"))
    monkeypatch.setattr(config, "RESULTS_DIR", "custom_results")
    monkeypatch.chdir(tmp_path)
    paths = sweep.save_noise_comparison_results([], [], [])
    assert all(path.parent == tmp_path / "project" / "custom_results" for path in paths)
    assert all(path.exists() for path in paths)


@pytest.mark.parametrize("kwargs", [
    {"noise_stds": [-1]}, {"noise_stds": [math.nan]}, {"noise_stds": [math.inf]},
    {"noise_stds": []}, {"seeds": []}, {"seeds": [-1]}, {"seeds": [1.5]},
    {"seeds": [2**32]}, {"train_episodes": 0}, {"eval_episodes": 0},
    {"train_episodes": -1}, {"eval_episodes": -1}, {"noise_stds": [0, -1]},
])
def test_invalid_inputs_fail_before_any_experiment(kwargs, tmp_path):
    args = {"noise_stds": [0], "seeds": [42], "train_episodes": 2, "eval_episodes": 3}
    args.update(kwargs)
    with patch.object(sweep, "run_noise_seed_comparison") as run:
        with pytest.raises(ValueError):
            sweep.run_noise_comparison(**args, output_dir=tmp_path / "absent")
    run.assert_not_called()
    assert not (tmp_path / "absent").exists()


def test_failure_restores_rngs_without_creating_results(tmp_path):
    python_before, numpy_before = random.getstate(), np.random.get_state()
    with patch.object(comparison, "run_episode", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            sweep.run_noise_comparison([5], [0], 2, 3, tmp_path)
    assert_rngs_unchanged(python_before, numpy_before)
    assert not (tmp_path / "noise_comparison_runs.csv").exists()


def test_plot_with_all_successful_step_metrics_missing(tmp_path):
    result = sweep.run_noise_comparison([0, 5], [42], 1, 2, tmp_path)
    rows = plotting.load_summary(tmp_path / "noise_comparison_summary.csv")
    assert {row["algorithm"] for row in rows} == {"random", "q_learning"}
    assert all(row["average_steps_on_success_mean"] is None for row in rows)
    path = plotting.plot_noise_comparison(rows, tmp_path / "figures" / "comparison.png")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(result["summary"]) == 4


def test_plot_means_errors_and_missing_values_are_preserved(tmp_path):
    # 값이 정해진 오차 막대와 결측 지표를 제공하고, 그래프 요소에 저장된 데이터를 확인한다.
    # 함수가 그래프를 닫더라도 확인할 수 있도록 생성된 Figure 객체를 보관한다.
    import matplotlib.pyplot as plt

    rows = [
        {"noise_std": 0, "algorithm": "q_learning", "success_rate_mean": 0.8,
         "success_rate_std": 0.1},
        {"noise_std": 5, "algorithm": "q_learning", "success_rate_mean": None},
        {"noise_std": 0, "algorithm": "random", "success_rate_mean": 0.2,
         "success_rate_std": 0.0},
    ]
    subplots = plt.subplots
    figures = []

    def capture(*args, **kwargs):
        fig, axes = subplots(*args, **kwargs)
        figures.append((fig, axes))
        return fig, axes

    with patch.object(plt, "subplots", side_effect=capture):
        plotting.plot_noise_comparison(rows, tmp_path / "missing.png")
    fig, axes = figures[0]
    q_line = next(line for line in axes.flat[0].lines if line.get_label() == "Q-Learning")
    random_line = next(line for line in axes.flat[0].lines if line.get_label() == "Random")
    assert list(q_line.get_xdata()) == [0, 5]
    assert q_line.get_ydata()[0] == 0.8
    assert math.isnan(q_line.get_ydata()[1])
    assert random_line.get_ydata()[0] == 0.2
    assert len(axes.flat[0].containers) == 2
    segments = axes.flat[0].containers[0].lines[2][0].get_segments()[0]
    np.testing.assert_allclose(segments[:, 1], [0.7, 0.9])
    assert not plt.fignum_exists(fig.number)


def test_plot_empty_input_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        plotting.plot_noise_comparison([], tmp_path / "none.png")


def test_cli_options_and_default_episode_count(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "compare_noise_sweep", "--noise-stds", "0", "5", "--seeds", "42", "123",
        "--train-episodes", "2", "--eval-episodes", "3", "--output-dir", str(tmp_path),
    ])
    with patch.object(sweep, "run_noise_comparison") as run:
        sweep.main()
    run.assert_called_once_with(
        noise_stds=[0.0, 5.0], seeds=[42, 123], train_episodes=2,
        eval_episodes=3, output_dir=tmp_path,
    )
    monkeypatch.setattr(sys, "argv", ["compare_noise_sweep"])
    with patch.object(sweep, "run_noise_comparison") as run:
        sweep.main()
    assert run.call_args.kwargs["train_episodes"] == old_sweep.DEFAULT_TRAIN_EPISODES
    assert run.call_args.kwargs["eval_episodes"] == config.NUM_EVAL_EPISODES


@pytest.mark.parametrize("option,value", [
    ("--noise-stds", "-1"), ("--noise-stds", "nan"),
    ("--train-episodes", "0"), ("--eval-episodes", "0"),
])
def test_cli_rejects_invalid_values(option, value, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["compare_noise_sweep", option, value])
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_both_module_entrypoints_run_headlessly(tmp_path):
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, "-m", "src.compare_noise_sweep", "--noise-stds", "0", "5",
         "--seeds", "0", "42", "--train-episodes", "1", "--eval-episodes", "2",
         "--output-dir", str(tmp_path)],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "train_seed=44" in process.stdout and "eval_seed=45" in process.stdout
    path = tmp_path / "plot.png"
    subprocess.run(
        [sys.executable, "-m", "src.plot_noise_comparison", "--input",
         str(tmp_path / "noise_comparison_summary.csv"), "--output", str(path)],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert path.read_bytes().startswith(b"\x89PNG")
