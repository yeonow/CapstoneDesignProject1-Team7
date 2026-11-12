"""Paper conditions, actual episode provenance, RNG integrity and artifacts."""

import csv
import json
import random

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
            assert config.TARGET_DISTANCE_METRIC == "manhattan"
            assert len(paper.targets.target_candidates("train")) == 193
            assert len(paper.targets.target_candidates("eval")) == 192
            raise RuntimeError("interrupted")
    assert vars(config) == previous


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
        assert saved["train_target_count"] == 193 and saved["eval_target_count"] == 192
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
