"""Gaussian Noise sweep 실행, 집계, 저장을 검증한다."""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from src import experiment_noise_sweep, plot_noise_sweep


def test_run_noise_seed_is_reproducible_and_uses_requested_noise():
    with patch.object(experiment_noise_sweep.config, "MAX_STEPS", 3):
        result1 = experiment_noise_sweep.run_noise_seed(5.0, 42, 2, 2)
        result2 = experiment_noise_sweep.run_noise_seed(5.0, 42, 2, 2)

    assert result1 == result2
    assert result1["noise_std"] == 5.0
    assert result1["seed"] == 42
    assert result1["train_episodes"] == result1["eval_episodes"] == 2


def test_aggregate_noise_results_calculates_mean_and_sample_std():
    rows = []
    for noise_std, values in ((0.0, (0.2, 0.4)), (5.0, (0.6, 1.0))):
        for seed, success_rate in enumerate(values):
            row = {
                "noise_std": noise_std,
                "seed": seed,
                "train_episodes": 2,
                "eval_episodes": 2,
            }
            for metric in experiment_noise_sweep.RUN_METRICS:
                row[metric] = success_rate
            rows.append(row)

    summaries = experiment_noise_sweep.aggregate_noise_results(rows)

    assert [row["noise_std"] for row in summaries] == [0.0, 5.0]
    assert summaries[0]["eval_success_rate_mean"] == pytest.approx(0.3)
    assert summaries[0]["eval_success_rate_std"] > 0
    assert summaries[1]["eval_success_rate_mean"] == pytest.approx(0.8)


def test_noise_sweep_saves_csv_and_plot():
    with TemporaryDirectory() as directory:
        output_dir = Path(directory)
        with patch.object(experiment_noise_sweep.config, "MAX_STEPS", 2):
            runs, summaries = experiment_noise_sweep.run_noise_sweep(
                noise_stds=[0.0, 5.0],
                seeds=[42, 123],
                train_episodes=2,
                eval_episodes=2,
                output_dir=output_dir,
            )

        runs_path = output_dir / "noise_sweep_runs.csv"
        summary_path = output_dir / "noise_sweep_summary.csv"
        plot_path = output_dir / "noise_sweep_summary.png"
        loaded = plot_noise_sweep.load_summary(summary_path)
        plot_noise_sweep.plot_noise_summary(loaded, plot_path)

        with runs_path.open(newline="", encoding="utf-8") as file:
            saved_runs = list(csv.DictReader(file))
        with summary_path.open(newline="", encoding="utf-8") as file:
            saved_summaries = list(csv.DictReader(file))

        assert len(runs) == len(saved_runs) == 4
        assert len(summaries) == len(saved_summaries) == 2
        assert plot_path.exists() and plot_path.stat().st_size > 0
