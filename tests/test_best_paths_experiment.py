"""Paper conditions, actual episode provenance, RNG integrity and artifacts."""

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

import config
from src import compare_algorithms as comparison
from src import best_paths_experiment as paper
from src.paths import select_best_paths
from src.plot_paths import _draw_best_path, plot_best_paths


def test_paper_settings_restore_on_exception():
    previous = vars(config).copy()
    with pytest.raises(RuntimeError):
        with paper.paper_config():
            assert config.MAX_STEPS == 400 and config.NOISE_STD == 2
            assert config.TARGET_DISTANCE_METRIC == "chebyshev"
            assert len(paper.targets.target_candidates("train")) == 187
            assert len(paper.targets.target_candidates("eval")) == 188
            raise RuntimeError("interrupted")
    assert vars(config) == previous


def test_default_output_is_separate_from_past_manhattan_results():
    assert paper.DEFAULT_OUTPUT == Path("results/best_paths_8dir/chebyshev_target/grid_20x20")
    assert paper.DEFAULT_OUTPUT != Path("results/best_paths_8dir/random_target/grid_20x20")


@pytest.mark.parametrize("cli", [False, True])
def test_existing_output_is_refused_before_training(monkeypatch, tmp_path, cli):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "past.json").write_text("past result", encoding="utf-8")

    def must_not_train(**kwargs):
        raise AssertionError("training must not start for an existing output")

    monkeypatch.setattr(paper, "compare_algorithms", must_not_train)
    previous = vars(config).copy()
    with pytest.raises(FileExistsError, match="already exists"):
        if cli:
            monkeypatch.setattr(sys, "argv", ["best_paths_experiment", "--train-episodes", "1",
                                              "--eval-episodes", "1", "--runs", "1",
                                              "--output-dir", str(output)])
            paper.main()
        else:
            paper.run_best_path_experiment(train_episodes=1, eval_episodes=1, runs=1,
                                           output_dir=output)
    assert [path.name for path in output.iterdir()] == ["past.json"]
    assert (output / "past.json").read_text(encoding="utf-8") == "past result"
    assert vars(config) == previous


def test_new_output_is_saved_and_cannot_be_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOG_INTERVAL", 0)
    output = tmp_path / "chebyshev_target" / "grid_20x20"
    kwargs = dict(train_episodes=1, eval_episodes=2, runs=1, seed=42, output_dir=output)
    paper.run_best_path_experiment(**kwargs)
    saved = {path.name: path.read_bytes() for path in output.iterdir()}
    assert {"q_learning_best.json", "random_search_best.json", "experiment_summary.csv",
            "algorithm_comparison_episodes.csv", "best_path_comparison.pdf"} <= set(saved)
    with pytest.raises(FileExistsError):
        paper.run_best_path_experiment(**kwargs)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == saved


def test_actual_paired_evaluations_artifacts_and_rng_integrity(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOG_INTERVAL", 0)
    kwargs = dict(train_episodes=2, eval_episodes=8, runs=2, seed=42)
    before = random.getstate()
    numpy_before = np.random.get_state()
    with paper.paper_config():
        baseline = comparison.compare_algorithms(**kwargs, output_dir=tmp_path / "baseline")
    evaluations = []
    original = comparison._evaluate_episodes

    def observe(*args, **kwargs):
        rows = original(*args, **kwargs)
        evaluations.append([dict(row) for row in rows])
        return rows

    monkeypatch.setattr(comparison, "_evaluate_episodes", observe)
    output = tmp_path / "recorded"
    result = paper.run_best_path_experiment(**kwargs, output_dir=output)
    assert result["episodes"] == baseline["episodes"]
    assert result["summary"] == baseline["summary"]
    assert random.getstate() == before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    records = []
    for index, rows in enumerate(evaluations):
        run = index // 2 + 1
        algorithm = "q_learning" if index % 2 == 0 else "random"
        paired = evaluations[index - 1] if index % 2 else None
        for row in rows:
            path = row["trajectory"]
            coordinates = path["path"]
            assert len(coordinates) == row["steps"] + 1
            assert (coordinates[-1] == path["target_position"]) == row["success"]
            assert all(max(abs(a - b) for a, b in zip(p, q)) <= 1
                       for p, q in zip(coordinates, coordinates[1:]))
            if paired:
                other = paired[row["episode"] - 1]
                assert (row["target_x"], row["target_y"]) == (other["target_x"], other["target_y"])
            records.append({**row, **path, "run": run, "algorithm": algorithm})
    expected = select_best_paths(records)
    for algorithm, stem in (("q_learning", "q_learning"), ("random", "random_search")):
        selected = result["paths"][algorithm]
        assert selected is not None
        assert selected["path"] == expected[algorithm]["path"]
        paper.validate_best_path(selected)
        saved = json.loads((output / f"{stem}_best.json").read_text(encoding="utf-8"))
        assert saved["trajectory"]["path"] == [list(p) for p in selected["path"]]
        assert saved["train_target_count"] == 187 and saved["eval_target_count"] == 188
        assert (saved["target_distance_metric"], saved["target_min_distance"],
                saved["target_candidate_count"], saved["shortest_distance_metric"]) == (
                    "chebyshev", 5, 375, "chebyshev")
        target = tuple(selected["target_position"])
        assert paper.targets.shortest_path_steps((0, 0), target) >= 5
        assert paper.targets.checkerboard_split(target) == "eval"
        assert saved["trajectory"]["train_seed"] == 42 + 2 * (selected["run"] - 1)
    assert not (output / "paths").exists()  # no representative artifacts
    for filename in ("algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv"):
        assert (output / filename).read_bytes() == (tmp_path / "baseline" / filename).read_bytes()
    for filename in ("q_learning_best.png", "random_search_best.png", "best_path_comparison.png"):
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "best_path_comparison.pdf").read_bytes().startswith(b"%PDF")
    with (output / "experiment_summary.csv").open(encoding="utf-8", newline="") as file:
        summaries = list(csv.DictReader(file))
    assert len(summaries) == 2 and all(int(row["total_eval_episodes"]) == 16 for row in summaries)
    assert all((row["target_distance_metric"], row["train_target_count"], row["eval_target_count"])
               == ("chebyshev", "187", "188") for row in summaries)
    # The plot keeps every segment and the original diagonal geometry.
    figure = Figure()
    ax = figure.subplots()
    _draw_best_path(figure, ax, result["paths"]["q_learning"], "Q-Learning")
    segments = ax.collections[0].get_segments()
    path = result["paths"]["q_learning"]["path"]
    np.testing.assert_array_equal(segments, np.stack((path[:-1], path[1:]), axis=1))
    figure.clear()


@pytest.mark.parametrize("mode,grid", [("fixed", 20), ("random", 20), ("split", 10)])
def test_paper_plot_rejects_other_modes_and_grids(mode, grid, tmp_path):
    with pytest.raises(ValueError, match="require"):
        plot_best_paths({"q_learning": {"target_mode": mode, "grid_size": grid}}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_no_success_is_saved_explicitly(tmp_path):
    row = dict(episode=1, success=False, steps=400, final_distance=10, shortest_path_steps=10)
    result = {"paths": {"q_learning": None, "random": None},
              "episodes": [{**row, "algorithm": algorithm} for algorithm in ("q_learning", "random")]}
    paper.save_best_path_results(result, tmp_path, train_episodes=1, eval_episodes=1, runs=1, seed=42)
    payload = json.loads((tmp_path / "q_learning_best.json").read_text(encoding="utf-8"))
    assert payload["status"] == "no_successful_episode" and payload["trajectory"] is None
