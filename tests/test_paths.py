"""Actual trajectories, deterministic selection, and experiment integrity."""

import json
import random
from unittest.mock import Mock, patch

import pytest

import config
from src import compare_algorithms as comparison
from src import compare_grid_sizes as grids
from src.experiment import build_environment, run_episode
from src.random_experiment import run_random_episode
from src.paths import select_paths


@pytest.mark.parametrize("runner", [run_episode, run_random_episode])
def test_diagonal_actual_path(runner):
    env = build_environment()
    env.agent_start = (0, 0)
    agent = Mock()
    agent.choose_action.side_effect = [7, 7]
    kwargs = {"training": False} if runner is run_episode else {}
    result = runner(env, agent, target_position=(2, 2), record_path=True, **kwargs)
    assert result["success"] and result["steps"] == 2
    assert result["trajectory"]["path"] == [(0, 0), (1, 1), (2, 2)]
    assert result["trajectory"]["shortest_path_steps"] == 2
    assert result["trajectory"]["path_ratio"] == 1.0


@pytest.mark.parametrize("runner", [run_episode, run_random_episode])
def test_positions_blocked_moves_and_measurement_count(runner):
    env = build_environment()
    env.agent_start = (0, 0)
    env.target_position = (2, 0)
    agent = Mock()
    agent.choose_action.side_effect = [0, 3, 3]
    kwargs = {"training": False} if runner is run_episode else {}
    with patch.object(env, "_generate_rssi", wraps=env._generate_rssi) as measure:
        result = runner(env, agent, record_path=True, **kwargs)
    path = result["trajectory"]
    assert path["path"] == [(0, 0), (0, 0), (1, 0), (2, 0)]
    assert len(path["path"]) == result["steps"] + 1 == 4
    assert path["blocked_move_count"] == 1
    assert path["path_ratio"] == 1.5
    assert measure.call_count == result["steps"] + 1
    agent.update_q.assert_not_called()


@pytest.mark.parametrize("runner", [run_episode, run_random_episode])
def test_zero_step_episode(runner):
    env = build_environment()
    env.target_position = env.agent_start
    kwargs = {"training": False} if runner is run_episode else {}
    agent = Mock()
    result = runner(env, agent, record_path=True, **kwargs)
    assert result["steps"] == 0
    assert result["trajectory"]["path"] == [env.agent_start]
    assert result["trajectory"]["path_ratio"] is None
    agent.choose_action.assert_not_called()


def test_selection_uses_ratios_real_successes_and_deterministic_ties():
    def record(ratio, steps, run=1, episode=1, success=True):
        return dict(path_ratio=ratio, steps=steps, run=run, episode=episode, success=success)
    rows = [record(1, 10, 2), record(1, 4, 2), record(1, 4, 1, 2),
            record(1, 4), record(2, 8), record(4, 16), record(0, 0, success=False)]
    selections = select_paths(rows)
    assert selections["best"] is rows[3]
    assert selections["representative"] is rows[4]  # successful mean ratio = 10/6
    assert select_paths(list(reversed(rows))) == selections
    assert select_paths([]) == {"best": None, "representative": None}
    assert select_paths(rows[-1:]) == {"best": None, "representative": None}
    tied = [record(1, 4, 2), record(3, 6), record(1, 4)]
    assert select_paths(tied)["representative"] is tied[2]
    zero = record(None, 0)
    assert select_paths([zero])["best"] is zero


@pytest.fixture
def small_config(monkeypatch):
    for key, value in {
        "GRID_SIZE": 4, "AGENT_START": (0, 0), "TARGET_POSITION": (3, 3),
        "MAX_STEPS": 30, "TARGET_MIN_DISTANCE": 2, "LOG_INTERVAL": 0,
        "GRID_ENV_SETTINGS": {4: {"target_position": (3, 3), "max_steps": 30}},
    }.items():
        monkeypatch.setattr(config, key, value)


@pytest.mark.parametrize("mode", ["fixed", "random", "split"])
def test_modes_preserve_metrics_csv_rng_and_reproduce_paths(mode, small_config, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TARGET_MODE", mode)
    kwargs = dict(train_episodes=3, eval_episodes=12, seed=7, runs=2)
    state = random.getstate()
    baseline = comparison.compare_algorithms(**kwargs, output_dir=tmp_path / "baseline")
    recorded = comparison.compare_algorithms(**kwargs, output_dir=tmp_path / "recorded", record_paths=True)
    repeated = comparison.compare_algorithms(**kwargs, output_dir=tmp_path / "repeat", record_paths=True)
    assert random.getstate() == state
    assert baseline["episodes"] == recorded["episodes"]
    assert baseline["summary"] == recorded["summary"]
    assert recorded == repeated
    for filename in ("algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv"):
        assert (tmp_path / "baseline" / filename).read_bytes() == (tmp_path / "recorded" / filename).read_bytes()
    # Inspect every evaluation trajectory, including failures, and repeated RNG streams.
    for evaluate in (comparison.evaluate_random_agent,
                     lambda **kw: comparison.evaluate_q_learning(comparison.train_q_learning_agent(3, seed=7), **kw)):
        rows = evaluate(eval_episodes=12, seed=8, record_paths=True)
        assert rows == evaluate(eval_episodes=12, seed=8, record_paths=True)
        for row in rows:
            path = row["trajectory"]
            assert len(path["path"]) == row["steps"] + 1
            assert path["path"][0] == (0, 0)
            assert (path["path"][-1] == path["target_position"]) == row["success"]
    assert not (tmp_path / "baseline" / "paths").exists()


@pytest.mark.parametrize("mode", ["fixed", "random", "split"])
def test_grid_writes_json_and_headless_figures(mode, small_config, tmp_path):
    result = grids.run_grid_comparison(
        grid_sizes=(4,), train_episodes=2, eval_episodes=8, seed=3, runs=1,
        target_mode=mode, plot_paths=True, plot=False, output_dir=tmp_path)
    directory = result["output_dir"] / "grid_4x4" / "paths"
    assert len(list(directory.glob("*.json"))) == 4
    for filename in directory.glob("*.json"):
        payload = json.loads(filename.read_text(encoding="utf-8"))
        assert payload["target_mode"] == mode
        record = payload["trajectory"]
        if record is not None:
            assert record["success"]
            assert len(record["path"]) == record["steps"] + 1
            assert record["path"][-1] == record["target_position"]
            assert record["train_seed"] == 3 and record["eval_seed"] == 4
    assert (directory / "path_comparison.png").read_bytes().startswith(b"\x89PNG")
    assert (directory / "path_comparison.pdf").read_bytes().startswith(b"%PDF")


def test_no_success_overwrites_previous_selections(small_config, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TARGET_MODE", "fixed")
    kwargs = dict(train_episodes=1, eval_episodes=20, seed=0, output_dir=tmp_path, record_paths=True)
    comparison.compare_algorithms(**kwargs)
    monkeypatch.setattr(config, "MAX_STEPS", 1)
    result = comparison.compare_algorithms(**kwargs)
    assert all(record is None for selected in result["paths"].values() for record in selected.values())
    for path in (tmp_path / "paths").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "no_successful_episode"
        assert payload["trajectory"] is None


def test_cli_enables_path_pipeline(monkeypatch):
    monkeypatch.setattr("sys.argv", ["compare_grid_sizes", "--plot-paths"])
    with patch.object(grids, "run_grid_comparison") as run:
        grids.main()
    assert run.call_args.kwargs["plot_paths"] is True
