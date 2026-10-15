"""Random 탐색의 Episode 실행, RSSI 샘플링, Seed 기반 반복 실행, CSV 출력 및 CLI 기능을 검증한다."""

import csv
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

import config
from src import random_experiment as experiment
from src.random_agent import RandomAgent


@pytest.fixture(autouse=True)
def preserve_global_rng():
    before = random.getstate()
    try:
        yield
    finally:
        random.setstate(before)


def make_env():
    env = experiment.build_environment()
    env.agent_start = (0, 0)
    env.target_position = (2, 0)
    env.max_steps = 3
    return env


def test_environment_uses_current_config(monkeypatch):
    settings = {
        "GRID_SIZE": 7, "MAX_STEPS": 15, "AGENT_START": (1, 2),
        "TARGET_POSITION": (5, 6), "CARRIER_FREQUENCY_HZ": 5e9,
        "TX_POWER_DBM": 3.0, "PATH_LOSS_EXPONENT": 2.5, "NOISE_STD": 3.0,
    }
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    env = experiment.build_environment()
    for name, value in settings.items():
        assert getattr(env, name.lower()) == value


def test_successful_episode_call_order_and_single_rssi_sample_per_step():
    real_env = make_env()
    env = Mock(wraps=real_env)
    agent = Mock(spec=RandomAgent)
    agent.choose_action.return_value = config.RIGHT
    events = Mock()
    events.attach_mock(env, "env")
    events.attach_mock(agent, "agent")
    with patch("src.environment.random.gauss", side_effect=[0.0, 1.0, -1.0]) as noise:
        result = experiment.run_random_episode(env, agent)

    assert result == {"success": True, "steps": 2, "final_distance": 0.0}
    assert type(result["success"]) is bool
    assert type(result["steps"]) is int
    assert type(result["final_distance"]) is float
    assert noise.call_count == 3  # Initial measurement plus two moves.
    assert events.mock_calls == [
        call.env.reset(), call.env.is_done(),
        call.agent.choose_action(), call.env.move(config.RIGHT),
        call.env.get_rssi(), call.env.is_done(),
        call.agent.choose_action(), call.env.move(config.RIGHT),
        call.env.get_rssi(), call.env.is_done(), call.env.get_info(),
    ]


def test_blocked_moves_consume_steps_and_stop_at_limit():
    real_env = make_env()
    env = Mock(wraps=real_env)
    agent = Mock(spec=RandomAgent)
    agent.choose_action.return_value = config.UP
    with patch("src.environment.random.gauss", return_value=0.0) as noise:
        result = experiment.run_random_episode(env, agent)

    assert result == {"success": False, "steps": 3, "final_distance": 2.0}
    assert real_env.agent_position == (0, 0)
    assert real_env.last_move_blocked is True
    assert env.move.call_count == env.get_rssi.call_count == 3
    assert noise.call_count == 4


@pytest.mark.parametrize("at_target", [True, False])
def test_initially_done_does_not_move(at_target):
    real_env = make_env()
    if at_target:
        real_env.target_position = real_env.agent_start
    else:
        real_env.max_steps = 0
    env = Mock(wraps=real_env)
    agent = Mock(spec=RandomAgent)
    with patch("src.environment.random.gauss", return_value=0.0) as noise:
        result = experiment.run_random_episode(env, agent)

    assert result == {
        "success": at_target, "steps": 0,
        "final_distance": 0.0 if at_target else 2.0,
    }
    env.reset.assert_called_once_with()
    agent.choose_action.assert_not_called()
    env.move.assert_not_called()
    assert noise.call_count == 1


def test_real_random_agent_produces_valid_moves():
    env = make_env()
    with patch.object(env, "move", wraps=env.move) as move:
        result = experiment.run_random_episode(env, RandomAgent(seed=42))
    assert 0 < result["steps"] <= env.max_steps
    assert move.call_count == result["steps"]
    assert all(args.args[0] in {config.UP, config.DOWN, config.LEFT, config.RIGHT}
               for args in move.call_args_list)
    assert set(result) == {"success", "steps", "final_distance"}


def test_multiple_episodes_defaults_resets_progress_and_csv(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "NUM_EVAL_EPISODES", 3)
    monkeypatch.setattr(config, "RANDOM_SEED", 17)
    monkeypatch.setattr(config, "LOG_INTERVAL", 2)
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "new_results"))
    env = Mock(wraps=make_env())
    agent = Mock(spec=RandomAgent)
    agent.choose_action.return_value = config.UP
    with (
        patch.object(experiment, "build_environment", return_value=env) as build,
        patch.object(experiment, "RandomAgent", return_value=agent) as agent_class,
    ):
        results = experiment.run_random_experiment()

    build.assert_called_once_with()
    agent_class.assert_called_once_with(seed=17)
    assert env.reset.call_count == 3
    assert agent.choose_action.call_args_list == [call()] * 9
    assert [row["episode"] for row in results] == [1, 2, 3]
    assert all(row["steps"] == 3 and row["success"] is False for row in results)
    path = tmp_path / "new_results" / "random_results.csv"
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["episode", "success", "steps", "final_distance"]
        assert list(reader) == [{key: str(value) for key, value in row.items()}
                                for row in results]
    output = capsys.readouterr().out
    assert "Episode 1/3" not in output
    assert "Episode 2/3" in output and "Episode 3/3" in output
    assert "success_rate=0.00%" in output


def test_seeded_results_and_actual_noise_are_reproducible(tmp_path):
    traces = []
    environments = iter([make_env(), make_env()])

    def build_recorded_environment():
        env = next(environments)
        trace = []
        traces.append(trace)
        sample = env._generate_rssi

        def record_sample():
            value = sample()
            trace.append(value)
            return value

        env._generate_rssi = record_sample
        return env

    before = random.getstate()
    with patch.object(experiment, "build_environment", side_effect=build_recorded_environment):
        first = experiment.run_random_experiment(4, seed=0, output_path=tmp_path / "a.csv")
        assert random.getstate() == before
        random.seed(999)
        random.gauss(0.0, 2.0)  # Also populate the global Gaussian cache.
        second_before = random.getstate()
        second = experiment.run_random_experiment(4, seed=0, output_path=tmp_path / "b.csv")
        assert random.getstate() == second_before

    assert first == second
    assert traces[0] and traces[0] == traces[1]
    assert len(traces[0]) == sum(row["steps"] + 1 for row in first)
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_rng_restored_on_failure(tmp_path):
    before = random.getstate()
    with patch.object(experiment, "run_random_episode", side_effect=RuntimeError("failed")):
        with pytest.raises(RuntimeError, match="failed"):
            experiment.run_random_experiment(1, seed=0, output_path=tmp_path / "failed.csv")
    assert random.getstate() == before
    assert not (tmp_path / "failed.csv").exists()


@pytest.mark.parametrize("episodes", [0, -1, 1.5, True])
def test_invalid_episode_count_has_no_side_effects(episodes, tmp_path):
    with patch.object(experiment, "build_environment") as build:
        with pytest.raises(ValueError, match="positive integer"):
            experiment.run_random_experiment(episodes, output_path=tmp_path / "invalid.csv")
    build.assert_not_called()
    assert not (tmp_path / "invalid.csv").exists()


def test_default_csv_does_not_change_training_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    training_path = tmp_path / "training_results.csv"
    original = b"episode,success,steps,total_reward,final_distance\n1,True,2,10.0,0.0\n"
    training_path.write_bytes(original)
    results = [{"episode": 1, "success": True, "steps": 2, "final_distance": 0.0}]

    path = experiment.save_random_results(results)
    assert path == tmp_path / "random_results.csv"
    assert training_path.read_bytes() == original
    assert "total_reward" not in path.read_text(encoding="utf-8")


def test_explicit_csv_path_creates_parents(tmp_path):
    path = tmp_path / "nested" / "custom.csv"
    assert experiment.save_random_results([], str(path)) == path
    assert path.read_text(encoding="utf-8").strip() == "episode,success,steps,final_distance"


def test_cli_runs_without_q_learning_or_state_reward(tmp_path):
    # A fresh interpreter prevents already-imported learning modules hiding dependencies.
    code = (
        "import runpy, sys; "
        "sys.modules['src.experiment'] = None; "
        "sys.modules['src.q_learning'] = None; "
        "sys.modules['src.state_reward'] = None; "
        "sys.argv = ['src.random_experiment'] + sys.argv[1:]; "
        "runpy.run_module('src.random_experiment', run_name='__main__')"
    )
    path = tmp_path / "cli.csv"
    process = subprocess.run(
        [sys.executable, "-c", code, "--episodes", "2", "--seed", "0",
         "--output", str(path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, check=True,
    )
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["episode"] for row in rows] == ["1", "2"]
    assert "episodes=2 | seed=0" in process.stdout


def test_cli_rejects_nonpositive_episodes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["random_experiment", "--episodes", "0"])
    with pytest.raises(SystemExit) as error:
        experiment.main()
    assert error.value.code == 2
    assert "--episodes must be a positive integer" in capsys.readouterr().err
