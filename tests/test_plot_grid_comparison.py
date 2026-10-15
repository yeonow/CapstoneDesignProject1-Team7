"""Grid 비교 숫자 표 CSV·그래프 생성과 입력 검증을 확인한다."""

import csv

import pytest

from src import compare_grid_sizes as grids
from src import plot_grid_comparison as plot


def summary_row(grid_size, algorithm, success_rate, steps, steps_on_success, distance, runs=2):
    row = {
        "grid_size": grid_size, "max_steps": grid_size * 10, "algorithm": algorithm,
        "run_count": runs, "train_episodes": 500 if algorithm == "q_learning" else 0,
        "eval_episodes": 100,
    }
    for metric, (mean, std) in {
        "success_rate": success_rate, "average_steps": steps,
        "average_steps_on_success": steps_on_success, "average_final_distance": distance,
    }.items():
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
        row[f"{metric}_run_count"] = 0 if mean is None else runs
    return row


@pytest.fixture
def summary_dir(tmp_path):
    rows = [
        summary_row(10, "q_learning", (1.0, 0.0), (19.2, 0.8), (19.2, 0.8), (0.0, 0.0)),
        summary_row(10, "random", (0.0833, 0.04), (97.9, 1.0), (74.8, 0.2), (7.2, 0.2)),
        summary_row(20, "q_learning", (1.0, 0.0), (47.9, 9.8), (47.9, 9.8), (0.0, 0.0)),
        summary_row(20, "random", (0.0, 0.0), (200.0, 0.0), (None, None), (17.8, 0.3)),
    ]
    grids.save_grid_comparison_results([], rows, tmp_path)
    return tmp_path


def test_report_writes_rounded_table_and_figures(summary_dir, capsys):
    result = plot.make_grid_report(summary_dir, summary_dir)

    with (summary_dir / "grid_comparison_table.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == list(plot.TABLE_FIELDS)
        table = list(reader)
    assert [(row["grid"], row["algorithm"]) for row in table] == [
        ("10x10", "Q-Learning"), ("10x10", "Random"), ("20x20", "Q-Learning"), ("20x20", "Random"),
    ]
    # 성공률은 % 단위, 소수 셋째 자리로 반올림한다.
    assert table[1]["success_rate_percent_mean"] == "8.33"
    assert table[1]["success_rate_percent_std"] == "4.0"
    # 성공이 없는 지표는 빈 칸이며, 유효 Run 수를 함께 남긴다.
    assert table[3]["average_steps_on_success_mean"] == ""
    assert table[3]["average_steps_on_success_valid_runs"] == "0"

    for path in result["figure_paths"]:
        assert path.exists() and path.stat().st_size > 0
    assert {path.suffix for path in result["figure_paths"]} == {".png", ".pdf"}

    output = capsys.readouterr().out
    assert "Success rate (%)" in output
    assert "8.33 +/- 4.00" in output
    assert "NA" in output


def test_missing_summary_explains_how_to_create_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="compare_grid_sizes"):
        plot.make_grid_report(tmp_path, tmp_path)
    assert not list(tmp_path.iterdir())


def test_duplicate_rows_are_rejected(summary_dir):
    path = summary_dir / "grid_comparison_summary.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        plot.load_grid_summary(path)


def test_single_run_has_no_error_bars(tmp_path):
    rows = [summary_row(10, algorithm, (0.5, 0.0), (10.0, 0.0), (10.0, 0.0), (1.0, 0.0), runs=1)
            for algorithm in ("q_learning", "random")]
    grids.save_grid_comparison_results([], rows, tmp_path)
    loaded = plot.load_grid_summary(tmp_path / "grid_comparison_summary.csv")
    paths = plot.plot_grid_comparison(loaded, tmp_path / "single")
    assert all(path.exists() for path in paths)
