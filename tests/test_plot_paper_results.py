"""가상 통계 표로 논문용 그림의 지표·CI·결측값·PNG/PDF 출력을 검증한다."""

import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest
from PIL import Image

from src import analyze_paper_results as analysis
from src import plot_paper_results as plotting


@pytest.fixture
def fixed_tables(tmp_path):
    """테스트 전용 가상 Run 통계이며 실제 논문용 결과에는 사용하지 않는다."""
    runs = []
    for run in (1, 2):
        for algorithm in ("q_learning", "random"):
            successes = 2 if algorithm == "q_learning" else 1 if run == 1 else 0
            runs.append({
                "run": run, "algorithm": algorithm, "train_seed": 42 + 2 * (run - 1),
                "eval_seed": 43 + 2 * (run - 1), "train_episodes": 5 if algorithm == "q_learning" else 0,
                "eval_episodes": 3, "success_count": successes, "failure_count": 3 - successes,
                "success_rate": successes / 3, "average_steps": 4.0 + run,
                "average_steps_on_success": 2.0 + run if successes else None,
                "average_final_distance": 1.0 + run,
            })
    basic = analysis.summarize_runs(runs)
    noise = analysis.summarize_runs([{**row, "noise_std": level} for level in (0.0, 5.0, 10.0) for row in runs])
    analysis._write_csv(tmp_path / "algorithm_performance.csv", basic, analysis.PERFORMANCE_FIELDS)
    analysis._write_csv(tmp_path / "noise_performance.csv", noise, analysis.PERFORMANCE_FIELDS)
    return tmp_path, basic, noise


def test_all_eight_figures_saved_in_png_and_pdf_at_300dpi(fixed_tables):
    directory, _, _ = fixed_tables
    paths = plotting.plot_paper_results(directory, directory / "figures")
    assert len(paths) == 16
    assert len({path.stem for path in paths}) == 8
    for path in paths:
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG")
            with Image.open(path) as image:
                assert image.info["dpi"][0] >= 299.9
                assert min(image.size) > 1000
        else:
            assert path.read_bytes().startswith(b"%PDF")


def test_noise_plot_preserves_algorithms_metric_ci_and_missing_gap(tmp_path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {"noise_std": 0.0, "algorithm": "q_learning", "average_steps_mean": 4.0,
         "average_steps_ci95_low": 2.0, "average_steps_ci95_high": 6.0,
         "average_steps_valid_run_count": 3},
        {"noise_std": 5.0, "algorithm": "q_learning", "average_steps_mean": None},
        {"noise_std": 10.0, "algorithm": "q_learning", "average_steps_mean": 9.0},
        {"noise_std": 0.0, "algorithm": "random", "average_steps_mean": 10.0},
    ]
    real_subplots, figures = plt.subplots, []

    def capture(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        figures.append((fig, ax))
        return fig, ax

    with patch.object(plt, "subplots", side_effect=capture):
        plotting.plot_metric(rows, "average_steps", "Average steps", "Steps", tmp_path / "noise", noise=True)
    fig, ax = figures[0]
    q_line = next(line for line in ax.lines if line.get_label() == "Q-Learning")
    random_line = next(line for line in ax.lines if line.get_label() == "Random")
    assert list(q_line.get_xdata()) == [0.0, 5.0, 10.0]
    assert q_line.get_ydata()[0] == 4
    assert math.isnan(q_line.get_ydata()[1])
    assert q_line.get_ydata()[2] == 9
    assert random_line.get_ydata()[0] == 10
    segments = ax.containers[0].lines[2][0].get_segments()[0]
    np.testing.assert_allclose(segments[:, 1], [2.0, 6.0])
    assert ax.get_ylabel() == "Steps"
    assert not plt.fignum_exists(fig.number)


def test_single_run_does_not_show_fake_confidence_interval(tmp_path):
    import matplotlib.pyplot as plt

    rows = [{"noise_std": None, "algorithm": "q_learning", "success_rate_percent_mean": 100.0,
             "success_rate_percent_valid_run_count": 1, "success_count_total": 3}]
    axes = []
    real_subplots = plt.subplots

    def capture(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        axes.append(ax)
        return fig, ax

    with patch.object(plt, "subplots", side_effect=capture):
        plotting.plot_metric(rows, "success_rate_percent", "Success rate", "Success rate (%)", tmp_path / "single")
    assert len(axes[0].containers) == 2  # 두 알고리즘 막대 컨테이너만 있고 CI 컨테이너는 없다.
    assert all(type(container).__name__ == "BarContainer" for container in axes[0].containers)


def test_all_missing_success_steps_are_not_zero(tmp_path):
    rows = [{"noise_std": 0.0, "algorithm": "random", "average_steps_on_success_mean": None,
             "average_steps_on_success_valid_run_count": 0, "success_count_total": 0}]
    paths = plotting.plot_metric(rows, "average_steps_on_success", "Average steps on success", "Steps",
                                 tmp_path / "missing", noise=True)
    assert all(path.exists() for path in paths)


def test_missing_table_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="analyze_paper_results"):
        plotting.plot_paper_results(tmp_path, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_empty_tables_do_not_generate_fake_plots(tmp_path):
    for name in ("algorithm_performance.csv", "noise_performance.csv"):
        analysis._write_csv(tmp_path / name, [], analysis.PERFORMANCE_FIELDS)
    with pytest.raises(ValueError, match="분석 가능한 표"):
        plotting.plot_paper_results(tmp_path, tmp_path / "figures")


def test_loader_preserves_labels_percent_units_and_none(fixed_tables):
    directory, _, _ = fixed_tables
    rows = plotting.load_performance_table(directory / "algorithm_performance.csv")
    assert {row["algorithm"] for row in rows} == {"q_learning", "random"}
    q = next(row for row in rows if row["algorithm"] == "q_learning")
    assert q["success_rate_percent_mean"] == pytest.approx(200 / 3)
    random_row = next(row for row in rows if row["algorithm"] == "random")
    assert random_row["average_steps_on_success_ci95_low"] is None


def test_invalid_ci_is_rejected(tmp_path):
    rows = [{"noise_std": 0.0, "algorithm": "q_learning", "average_steps_mean": 4.0,
             "average_steps_ci95_low": 5.0, "average_steps_ci95_high": 6.0,
             "average_steps_valid_run_count": 2}]
    with pytest.raises(ValueError, match="신뢰구간"):
        plotting.plot_metric(rows, "average_steps", "Steps", "Steps", tmp_path / "invalid", noise=True)


def test_independent_plot_module_cli(fixed_tables):
    directory, _, _ = fixed_tables
    process = subprocess.run(
        [sys.executable, "-m", "src.plot_paper_results", "--input-dir", str(directory),
         "--output-dir", str(directory / "cli")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True,
    )
    assert process.returncode == 0
    assert len(list((directory / "cli").glob("*.pdf"))) == 8
