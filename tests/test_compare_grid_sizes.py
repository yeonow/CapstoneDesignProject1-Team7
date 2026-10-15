"""Grid 크기별 비교 실행의 config 적용·복원, 결과 저장 위치, Grid 간 요약을 검증한다."""

import csv
import random
from unittest.mock import patch

import numpy as np
import pytest

import config
from src import compare_algorithms as comparison
from src import compare_grid_sizes as grids
from src import experiment


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    for name, value in {
        "GRID_SIZE": 10, "AGENT_START": (0, 0), "TARGET_POSITION": (9, 9),
        "MAX_STEPS": 100, "NUM_EPISODES": 2, "NUM_EVAL_EPISODES": 2,
        "RANDOM_SEED": 0, "LOG_INTERVAL": 0,
        "GRID_SIZES": (3, 4),
        "GRID_ENV_SETTINGS": {
            3: {"target_position": (2, 2), "max_steps": 5},
            4: {"target_position": (3, 3), "max_steps": 7},
        },
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_grid_config_applies_and_restores_settings():
    with grids.grid_config(4) as settings:
        assert settings == {"target_position": (3, 3), "max_steps": 7}
        env = experiment.build_environment()
        assert (env.grid_size, env.target_position, env.max_steps) == (4, (3, 3), 7)
    assert (config.GRID_SIZE, config.TARGET_POSITION, config.MAX_STEPS) == (10, (9, 9), 100)


def test_grid_config_restores_settings_on_failure():
    with pytest.raises(RuntimeError, match="failed"):
        with grids.grid_config(3):
            raise RuntimeError("failed")
    assert (config.GRID_SIZE, config.TARGET_POSITION, config.MAX_STEPS) == (10, (9, 9), 100)


@pytest.mark.parametrize("grid_sizes", [(3, 5), (3, 3), ()])
def test_invalid_grid_sizes_fail_before_training(grid_sizes, tmp_path):
    with patch.object(grids, "compare_algorithms") as compare:
        with pytest.raises(ValueError):
            grids.run_grid_comparison(grid_sizes=grid_sizes, output_dir=tmp_path)
    compare.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_target_outside_grid_is_rejected(monkeypatch):
    monkeypatch.setitem(config.GRID_ENV_SETTINGS, 3, {"target_position": (3, 3), "max_steps": 5})
    with pytest.raises(ValueError, match="inside"):
        grids.grid_settings(3)


def test_both_algorithms_run_on_every_grid_size(tmp_path):
    environments = []

    def record_environment(noise_std=None):
        env = experiment.build_environment(noise_std=noise_std)
        environments.append(env)
        return env

    with patch.object(comparison, "build_environment", side_effect=record_environment):
        result = grids.run_grid_comparison(
            train_episodes=2, eval_episodes=2, seed=0, runs=2, output_dir=tmp_path,
        )

    # Grid마다 Run별로 학습·Q 평가·Random 평가 환경을 하나씩 만든다.
    assert [env.grid_size for env in environments] == [3] * 6 + [4] * 6
    assert {(env.grid_size, env.target_position, env.max_steps) for env in environments} == {
        (3, (2, 2), 5), (4, (3, 3), 7),
    }
    assert (config.GRID_SIZE, config.TARGET_POSITION, config.MAX_STEPS) == (10, (9, 9), 100)

    for grid_size in (3, 4):
        directory = tmp_path / f"grid_{grid_size}x{grid_size}"
        with (directory / "algorithm_comparison_summary.csv").open(newline="", encoding="utf-8") as file:
            assert len(list(csv.DictReader(file))) == 4

    assert [(row["grid_size"], row["algorithm"], row["run"]) for row in result["runs"]] == [
        (grid_size, algorithm, run)
        for grid_size in (3, 4) for run in (1, 2) for algorithm in ("q_learning", "random")
    ]
    assert all(row["steps"] <= {3: 5, 4: 7}[grid_size]
               for grid_size in (3, 4)
               for row in _read_csv(tmp_path / f"grid_{grid_size}x{grid_size}"
                                    / "algorithm_comparison_episodes.csv", int_fields=("steps",)))
    assert [(row["grid_size"], row["algorithm"], row["run_count"]) for row in result["summary"]] == [
        (3, "q_learning", 2), (3, "random", 2), (4, "q_learning", 2), (4, "random", 2),
    ]
    for row in result["summary"]:
        runs = [run for run in result["runs"]
                if (run["grid_size"], run["algorithm"]) == (row["grid_size"], row["algorithm"])]
        assert row["success_rate_mean"] == pytest.approx(
            sum(run["success_rate"] for run in runs) / len(runs))

    summary_rows = _read_csv(tmp_path / "grid_comparison_summary.csv")
    assert list(summary_rows[0]) == list(grids.GRID_SUMMARY_FIELDS)
    assert len(summary_rows) == 4
    assert len(_read_csv(tmp_path / "grid_comparison_runs.csv")) == 8
    assert len(_read_csv(tmp_path / "grid_comparison_table.csv")) == 4
    assert (tmp_path / "grid_comparison.png").exists()
    assert (tmp_path / "grid_comparison.pdf").exists()


def test_same_seeds_are_used_on_each_grid(tmp_path):
    with patch.object(grids, "compare_algorithms", wraps=grids.compare_algorithms) as compare:
        grids.run_grid_comparison(train_episodes=1, eval_episodes=1, seed=5, runs=1,
                                  output_dir=tmp_path)
    assert [args.kwargs["seed"] for args in compare.call_args_list] == [5, 5]
    assert [args.kwargs["output_dir"].name for args in compare.call_args_list] == [
        "grid_3x3", "grid_4x4",
    ]


def test_cli_options(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "compare_grid_sizes", "--grid-sizes", "3", "4", "--train-episodes", "2",
        "--eval-episodes", "3", "--seed", "1", "--runs", "2", "--noise", "--analyze",
        "--output-dir", str(tmp_path),
    ])
    with patch.object(grids, "run_grid_comparison") as run:
        grids.main()
    kwargs = run.call_args.kwargs
    assert kwargs["grid_sizes"] == [3, 4]
    assert (kwargs["train_episodes"], kwargs["eval_episodes"], kwargs["seed"], kwargs["runs"]) == (2, 3, 1, 2)
    assert kwargs["noise"] is True and kwargs["analyze"] is True
    assert kwargs["output_dir"] == tmp_path


def _read_csv(path, int_fields=()):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in int_fields:
            row[field] = int(row[field])
    return rows
