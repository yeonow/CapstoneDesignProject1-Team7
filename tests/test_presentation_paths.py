"""Verify held-out target trajectories, provenance, isolation, and exports."""

import csv
import json
import random
from unittest.mock import Mock

import numpy as np
import pytest
from matplotlib.figure import Figure

import config
from src import compare_algorithms as comparison
from src import presentation_paths as presentation
from src import targets
from src.plot_paths import _draw_best_path
from src.random_experiment import run_random_episode


def test_presentation_split_validation_restores_config(monkeypatch):
    previous = vars(config).copy()
    with presentation.presentation_config() as counts:
        train, evaluation = (targets.target_candidates(phase) for phase in ("train", "eval"))
        assert not set(train) & set(evaluation)
        assert presentation.TARGET in evaluation
        assert all(targets.shortest_path_steps((0, 0), target) >= 5
                   for target in (*train, *evaluation))
        assert counts == {"train_target_count": len(train), "eval_target_count": len(evaluation)}
        assert sum(counts.values()) == 375
        assert presentation.SHORTEST_STEPS == 18
    assert vars(config) == previous

    monkeypatch.setattr(presentation, "TARGET", (19, 19))
    with pytest.raises(ValueError, match="held out"):
        with presentation.presentation_config():
            pytest.fail("A training target must never reach evaluation")
    assert vars(config) == previous


def test_actual_paths_reproduce_and_leave_standard_comparison_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_INTERVAL", 0)
    monkeypatch.setattr(config, "TARGET_MODE", "split")
    kwargs = dict(train_episodes=2, eval_episodes=8, runs=2, seed=42)
    baseline_directory = tmp_path / "baseline"
    baseline = comparison.compare_algorithms(**kwargs, output_dir=baseline_directory)
    baseline_files = {path.name: path.read_bytes() for path in baseline_directory.iterdir()}
    previous_config = vars(config).copy()
    previous_rng = random.getstate()
    previous_numpy = np.random.get_state()

    training_targets = []
    original = comparison.run_episode

    def observe_training(*args, **kwargs):
        if kwargs["training"]:
            training_targets.append(kwargs["target_position"])
        return original(*args, **kwargs)

    with monkeypatch.context() as observe:
        observe.setattr(comparison, "run_episode", observe_training)
        output = tmp_path / "presentation"
        result = presentation.run_presentation_paths(**kwargs, output_dir=output)
        with presentation.presentation_config():
            train_candidates = targets.target_candidates("train")
            replayed = [
                row for run in (1, 2)
                for row in presentation._record_run(run, 2, 2, 8, 42)
            ]
    expected_training_targets = []
    for seed in (42, 44):
        rng = targets.training_target_rng(seed)
        expected_training_targets.extend(rng.choice(train_candidates) for _ in range(2))
    assert training_targets == expected_training_targets * 2
    assert result["records"] == replayed
    assert len(replayed) == 32
    assert any(row["success"] for row in replayed)
    assert any(not row["success"] for row in replayed)
    assert any(row["blocked_move_count"] for row in replayed)
    for row in replayed:
        path = row["path"]
        assert path[0] == (0, 0)
        assert row["target_position"] == (17, 18)
        assert row["shortest_path_steps"] == 18
        assert len(path) == row["steps"] + 1
        assert row["success"] == (path[-1] == (17, 18))
        assert row["blocked_move_count"] == sum(a == b for a, b in zip(path, path[1:]))
        assert all(targets.shortest_path_steps(a, b) <= 1 for a, b in zip(path, path[1:]))
        assert row["train_seed"] == 42 + 2 * (row["run"] - 1)
        assert row["eval_seed"] == row["train_seed"] + 1
        assert row["noise_seed"] == f"{row['eval_seed']}:{row['episode']}"

    assert vars(config) == previous_config
    assert random.getstate() == previous_rng
    np.testing.assert_equal(np.random.get_state(), previous_numpy)
    after_directory = tmp_path / "after"
    assert comparison.compare_algorithms(**kwargs, output_dir=after_directory) == baseline
    for filename, data in baseline_files.items():
        assert (baseline_directory / filename).read_bytes() == data
        assert (after_directory / filename).read_bytes() == data
        assert not (output / filename).exists()
    with pytest.raises(ValueError, match="dedicated"):
        presentation.run_presentation_paths(**kwargs, output_dir=baseline_directory)

    for stem in ("q_learning_best", "random_search_best", "best_path_comparison",
                 "paired_comparison"):
        assert (output / f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (output / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
    with (output / "presentation_path_summary.csv").open(encoding="utf-8", newline="") as file:
        saved_rows = list(csv.DictReader(file))
    assert len(saved_rows) == 2
    assert [row["Algorithm"] for row in saved_rows] == ["Q-Learning", "Random Search"]
    for (algorithm, stem), row in zip(presentation.STEMS.items(), saved_rows):
        payload = json.loads((output / f"{stem}.json").read_text(encoding="utf-8"))
        selected = result["best"][algorithm]
        assert payload["trajectory"] == (json.loads(json.dumps(selected)) if selected else None)
        assert payload["status"] == ("selected" if selected else "no_successful_episode")
        assert row["Target"] == "(17, 18)"
        assert row["Shortest Steps"] == "18"
        assert row["Success"] == str(selected is not None)
        assert row["Actual Steps"] == (str(selected["steps"]) if selected else "")
    paired = json.loads((output / "paired_comparison.json").read_text(encoding="utf-8"))
    assert paired["trajectories"] is None and paired["status"] == "no_joint_success"

    selected = result["best"]["random"]
    figure = Figure()
    axis = figure.subplots()
    _draw_best_path(figure, axis, selected, "Random Search",
                    presentation_positions=((0, 0), (17, 18)))
    path = selected["path"]
    np.testing.assert_array_equal(axis.collections[2].get_segments(),
                                  np.stack((path[:-1], path[1:]), axis=1))


def test_boundary_actions_record_duplicate_coordinates():
    with presentation.presentation_config():
        env = comparison.build_environment()
        agent = Mock()
        agent.choose_action.side_effect = [
            config.UP_LEFT, config.DOWN_RIGHT, config.UP_LEFT, config.LEFT,
            *([config.DOWN_RIGHT] * 17), config.DOWN,
        ]
        with comparison._noise_seed("43:1"):
            outcome = run_random_episode(env, agent, target_position=(17, 18), record_path=True)
    trajectory = outcome.pop("trajectory")
    record = {**outcome, **trajectory}
    assert outcome["success"] and outcome["steps"] == 22
    assert trajectory["path"][:5] == [(0, 0), (0, 0), (1, 1), (0, 0), (0, 0)]
    assert trajectory["blocked_move_count"] == 2
    presentation.validate_trajectory(record)


def test_independent_and_paired_rankings():
    def row(algorithm, run, episode, steps, *, success=True, target=(17, 18)):
        return dict(algorithm=algorithm, run=run, episode=episode, steps=steps, success=success,
                    target_position=target, path_ratio=steps / 18)

    rows = [
        row("q_learning", 1, 1, 18, target=(18, 17)),
        row("random", 1, 1, 18, success=False),
        row("q_learning", 2, 1, 19), row("q_learning", 1, 7, 19),
        row("q_learning", 1, 4, 19), row("random", 2, 8, 22),
        row("random", 1, 4, 50),
        row("q_learning", 3, 3, 30), row("random", 3, 3, 24),
        row("q_learning", 3, 2, 28), row("random", 3, 2, 26),
        row("q_learning", 4, 1, 28), row("random", 4, 1, 26),
    ]
    best, paired = presentation.select_presentation_paths(reversed(rows))
    assert best["q_learning"] == rows[4]
    assert best["random"] == rows[5]
    assert paired == {"q_learning": rows[9], "random": rows[10]}
