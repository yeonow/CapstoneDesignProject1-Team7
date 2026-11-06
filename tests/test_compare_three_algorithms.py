"""세 알고리즘 비교가 기존 Q-Learning·Random 평가와 같은 결과를 내고, 같은 Target·Noise 조건,
경로 기록, config 복원, 경로 선택, 파일 저장·그림 생성을 올바르게 하는지 검증한다."""

import json
import random

import numpy as np
import pytest

import config
from src import compare_algorithms as two
from src import compare_three_algorithms as three
from src import targets
from src.plot_three_algorithms import load_summary, make_three_algorithm_report


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    for name, value in {
        "GRID_SIZE": 6, "AGENT_START": (0, 0), "TARGET_POSITION": (5, 5),
        "MAX_STEPS": 30, "NOISE_STD": 2.0, "NUM_EPISODES": 4,
        "NUM_EVAL_EPISODES": 4, "RANDOM_SEED": 42, "LOG_INTERVAL": 0,
        "TARGET_MODE": "random", "TARGET_MIN_DISTANCE": 2,
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _without_path(rows):
    return [
        {key: value for key, value in row.items() if key not in ("path", "path_ratio")}
        for row in rows
    ]


def test_q_learning_and_random_match_existing_two_algorithm_evaluation():
    agent = two.train_q_learning_agent(4, seed=42)
    expected_q = two.evaluate_q_learning(agent, 5, seed=43)
    expected_random = two.evaluate_random_agent(5, seed=43)
    assert _without_path(three.evaluate_algorithm("q_learning", 5, 43, agent)) == expected_q
    assert _without_path(three.evaluate_algorithm("random", 5, 43)) == expected_random


def _trained_agent(algorithm, episodes=2):
    mode = three.Q_ACTION_MODES.get(algorithm)
    return None if mode is None else two.train_q_learning_agent(episodes, seed=42, action_mode=mode)


def test_all_algorithms_get_same_targets_per_episode():
    results = {
        algorithm: three.evaluate_algorithm(algorithm, 6, 43, _trained_agent(algorithm))
        for algorithm in three.ALGORITHMS
    }
    target_lists = {
        algorithm: [(row["target_x"], row["target_y"]) for row in rows]
        for algorithm, rows in results.items()
    }
    assert len({tuple(values) for values in target_lists.values()}) == 1
    candidates = targets.target_candidates("eval")
    assert target_lists["greedy"] == [
        targets.evaluation_target(candidates, 43, episode) for episode in range(1, 7)
    ]


@pytest.mark.parametrize("algorithm", three.ALGORITHMS)
def test_recorded_path_is_consistent_with_episode_result(algorithm):
    for row in three.evaluate_algorithm(algorithm, 6, 43, _trained_agent(algorithm, 3)):
        path = row["path"]
        assert len(path) == row["steps"] + 1
        assert path[0] == config.AGENT_START
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            assert max(abs(x1 - x0), abs(y1 - y0)) <= 1
            assert 0 <= x1 < config.GRID_SIZE and 0 <= y1 < config.GRID_SIZE
        target = (row["target_x"], row["target_y"])
        assert (path[-1] == target) == row["success"]
        if row["success"]:
            assert row["path_ratio"] == row["steps"] / row["shortest_path_steps"]
        else:
            assert row["path_ratio"] is None


def test_q_learning_requires_trained_agent():
    with pytest.raises(ValueError):
        three.evaluate_algorithm("q_learning", 2, 43)
    with pytest.raises(ValueError):
        three.evaluate_algorithm("unknown", 2, 43)


def test_experiment_config_is_restored_even_on_error():
    before = (config.GRID_SIZE, config.MAX_STEPS, config.TARGET_POSITION,
              config.TARGET_MODE, config.TARGET_MIN_DISTANCE)
    with pytest.raises(RuntimeError):
        with three.experiment_config(20, 400, "split", 5):
            assert (config.GRID_SIZE, config.MAX_STEPS) == (20, 400)
            assert config.TARGET_POSITION == (19, 19)
            assert (config.TARGET_MODE, config.TARGET_MIN_DISTANCE) == ("split", 5)
            raise RuntimeError
    assert (config.GRID_SIZE, config.MAX_STEPS, config.TARGET_POSITION,
            config.TARGET_MODE, config.TARGET_MIN_DISTANCE) == before


def test_experiment_config_uses_corner_target_for_undefined_grid():
    with three.experiment_config(7, 50):
        assert config.TARGET_POSITION == (6, 6)


def _episode(algorithm, run, episode, steps, shortest):
    return {
        "algorithm": algorithm, "run": run, "episode": episode, "steps": steps,
        "shortest_path_steps": shortest, "target_x": shortest, "target_y": 0,
        "path_ratio": steps / shortest, "path": [(0, 0)] * (steps + 1),
    }


def test_select_paths_picks_best_and_closest_to_mean():
    episodes = [
        _episode("greedy", 1, 1, 10, 5),   # 2.0
        _episode("greedy", 1, 2, 6, 5),    # 1.2 → best
        _episode("greedy", 2, 1, 15, 5),   # 3.0
        {**_episode("greedy", 2, 2, 50, 5), "path_ratio": None},  # 실패
    ]
    summaries = [
        {"algorithm": "greedy", "path_ratio_on_success_mean": 2.1},
        {"algorithm": "random", "path_ratio_on_success_mean": None},
    ]
    selected = three.select_paths(episodes, summaries)
    assert (selected["greedy"]["best"]["run"], selected["greedy"]["best"]["episode"]) == (1, 2)
    representative = selected["greedy"]["representative"]
    assert (representative["run"], representative["episode"]) == (1, 1)
    assert representative["path"] == [[0, 0]] * 11
    assert selected["random"] == {"mean_path_ratio": None, "best": None, "representative": None}
    assert selected["q_learning"]["best"] is None


def test_compare_three_algorithms_saves_results_and_figures(tmp_path):
    before = (config.GRID_SIZE, config.MAX_STEPS, config.TARGET_MODE)
    result = three.compare_three_algorithms(
        grid_size=6, max_steps=30, train_episodes=3, eval_episodes=4, runs=2,
        target_mode="random", target_min_distance=2, output_dir=tmp_path, plot=True,
    )
    assert (config.GRID_SIZE, config.MAX_STEPS, config.TARGET_MODE) == before
    assert len(result["episodes"]) == 2 * 3 * 4
    assert len(result["runs"]) == 2 * 3
    assert {row["algorithm"] for row in result["summary"]} == set(three.DEFAULT_ALGORITHMS)
    assert all(row["run_count"] == 2 for row in result["summary"])
    for name in (
        "three_algorithm_episodes.csv", "three_algorithm_runs.csv",
        "three_algorithm_summary.csv", "three_algorithm_paths.json",
        "three_algorithm_comparison.png", "three_algorithm_comparison.pdf",
        "three_algorithm_paths.png", "three_algorithm_paths.pdf",
    ):
        assert (tmp_path / name).exists(), name
    episodes_csv = (tmp_path / "three_algorithm_episodes.csv").read_text(encoding="utf-8")
    assert "path_ratio" in episodes_csv.splitlines()[0]
    paths = json.loads((tmp_path / "three_algorithm_paths.json").read_text(encoding="utf-8"))
    assert paths["grid_size"] == 6 and set(paths["algorithms"]) == set(three.DEFAULT_ALGORITHMS)
    assert len(load_summary(tmp_path / "three_algorithm_summary.csv")) == 3


def test_same_arguments_reproduce_same_results(tmp_path):
    kwargs = dict(grid_size=6, max_steps=30, train_episodes=3, eval_episodes=3, runs=1,
                  target_mode="split", target_min_distance=2, plot=False)
    first = three.compare_three_algorithms(output_dir=tmp_path / "a", **kwargs)
    second = three.compare_three_algorithms(output_dir=tmp_path / "b", **kwargs)
    assert first["episodes"] == second["episodes"]


def test_report_handles_algorithm_without_success(tmp_path):
    three.compare_three_algorithms(
        grid_size=6, max_steps=30, train_episodes=2, eval_episodes=2, runs=1,
        target_mode="random", target_min_distance=2, output_dir=tmp_path, plot=False,
    )
    paths_file = tmp_path / "three_algorithm_paths.json"
    paths = json.loads(paths_file.read_text(encoding="utf-8"))
    paths["algorithms"]["random"] = {"mean_path_ratio": None, "best": None, "representative": None}
    paths_file.write_text(json.dumps(paths), encoding="utf-8")
    figures = make_three_algorithm_report(tmp_path)
    assert all(path.exists() for path in figures)
