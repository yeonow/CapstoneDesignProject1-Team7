"""실제 논문 결과와 분리된 고정 가상 CSV로 검증·통계·보고서 출력을 확인한다."""

import csv
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src import analyze_paper_results as analysis
from src.compare_algorithms import EPISODE_FIELDS, SUMMARY_FIELDS, calculate_metrics
from src.compare_noise_sweep import (
    NOISE_EPISODE_FIELDS, NOISE_RUN_FIELDS, NOISE_SUMMARY_FIELDS, aggregate_noise_comparison,
)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixed_csvs(tmp_path):
    """학습 없이 고정 Episode로 만든 테스트 자료이며 실제 결과 경로에는 저장하지 않는다."""
    basic_episodes, basic_runs, noise_episodes, noise_runs = [], [], [], []
    for level in (None, 0.0, 5.0):
        episodes, runs = [], []
        for run in (1, 2):
            for algorithm in ("q_learning", "random"):
                metadata = {"run": run, "algorithm": algorithm,
                            "train_seed": 42 + 2 * (run - 1), "eval_seed": 43 + 2 * (run - 1)}
                if level is not None:
                    metadata["noise_std"] = level
                values = [(True, 2, 0.0), (True, 4, 0.0), (False, 10, 3.0)] if algorithm == "q_learning" else [
                    (True, 6, 0.0), (False, 10, 4.0), (False, 10, 5.0),
                ]
                rows = [{**metadata, "episode": episode, "success": success,
                         "steps": steps + ((run - 1) + (1 if level == 5 and algorithm == "q_learning" else 0) if success else 0),
                         "final_distance": distance}
                        for episode, (success, steps, distance) in enumerate(values, 1)]
                episodes.extend(rows)
                runs.append({**metadata, "train_episodes": 5 if algorithm == "q_learning" else 0,
                             "eval_episodes": 3, **calculate_metrics(rows)})
        if level is None:
            basic_episodes, basic_runs = episodes, runs
        else:
            noise_episodes.extend(episodes)
            noise_runs.extend(runs)
    data = {
        "algorithm_comparison_episodes.csv": (basic_episodes, EPISODE_FIELDS),
        "algorithm_comparison_summary.csv": (basic_runs, SUMMARY_FIELDS),
        "noise_comparison_episodes.csv": (noise_episodes, NOISE_EPISODE_FIELDS),
        "noise_comparison_runs.csv": (noise_runs, NOISE_RUN_FIELDS),
        "noise_comparison_summary.csv": (aggregate_noise_comparison(noise_runs), NOISE_SUMMARY_FIELDS),
    }
    for name, (rows, fields) in data.items():
        write_csv(tmp_path / name, rows, fields)
    return tmp_path, data


@pytest.mark.parametrize("df,expected", [
    (1, 12.7062047364), (2, 4.30265272975), (4, 2.77644510520),
    (9, 2.26215716280), (29, 2.04522964213), (99, 1.98421695151),
])
def test_student_t_critical_against_known_quantiles(df, expected):
    """NIST 표와 일치하는 97.5% t 분위수의 수치 정확도를 확인한다."""
    assert analysis.student_t_critical(df) == pytest.approx(expected, rel=1e-9)


def test_mean_sample_sd_and_t_confidence_interval():
    stats = analysis.run_statistics([1.0, 2.0, None, 3.0])
    assert stats["mean"] == 2.0
    assert stats["std"] == 1.0
    assert stats["valid_run_count"] == 3
    margin = 4.30265272975 / math.sqrt(3)
    assert stats["ci95_low"] == pytest.approx(2 - margin)
    assert stats["ci95_high"] == pytest.approx(2 + margin)


@pytest.mark.parametrize("values,mean,count", [([], None, 0), ([None], None, 0), ([4.0, None], 4.0, 1)])
def test_insufficient_runs_do_not_create_sd_or_ci(values, mean, count):
    assert analysis.run_statistics(values) == {
        "mean": mean, "std": None, "ci95_low": None, "ci95_high": None, "valid_run_count": count,
    }


def test_nan_not_silently_excluded():
    with pytest.raises(ValueError, match="NaN"):
        analysis.run_statistics([1.0, math.nan])


def test_valid_data_and_summary_recalculation(fixed_csvs):
    directory, _ = fixed_csvs
    basic, noise = analysis.validate_dataset(directory), analysis.validate_dataset(directory, noise=True)
    assert basic["valid"] and noise["valid"]
    assert len(basic["episodes"]) == 12
    assert len(noise["episodes"]) == 24
    assert len(noise["runs"]) == 8
    assert noise["warnings"]


def test_failure_step_limit_is_consistent_across_algorithms(fixed_csvs):
    directory, data = fixed_csvs
    episodes, fields = data["algorithm_comparison_episodes.csv"]
    episodes[2]["steps"] = 9
    runs, run_fields = data["algorithm_comparison_summary.csv"]
    runs[0].update(calculate_metrics(episodes[:3]))
    write_csv(directory / "algorithm_comparison_episodes.csv", episodes, fields)
    write_csv(directory / "algorithm_comparison_summary.csv", runs, run_fields)
    result = analysis.validate_dataset(directory)
    assert not result["valid"]
    assert any("실패 종료 Step 불일치" in error for error in result["errors"])


@pytest.mark.parametrize("mutation,expected", [
    ("missing_column", "필수 열"), ("duplicate_episode", "중복 Episode"),
    ("missing_episode", "Episode 수"), ("bad_summary", "재계산 결과"),
    ("negative_distance", "final_distance"), ("nan_steps", "steps"),
    ("bad_success", "success"), ("bad_algorithm", "algorithm"),
    ("inconsistent_distance", "거리 불일치"), ("bad_seed", "Seed"),
    ("bad_rate", "성공률"), ("bad_counts", "성공·실패"),
])
def test_invalid_basic_data_is_rejected_without_correction(fixed_csvs, mutation, expected):
    directory, data = fixed_csvs
    episodes, fields = data["algorithm_comparison_episodes.csv"]
    runs, run_fields = data["algorithm_comparison_summary.csv"]
    if mutation == "missing_column":
        fields = tuple(field for field in fields if field != "steps")
        episodes = [{key: value for key, value in row.items() if key != "steps"} for row in episodes]
    elif mutation == "duplicate_episode":
        episodes.append(episodes[0].copy())
    elif mutation == "missing_episode":
        episodes.pop()
    elif mutation == "bad_summary":
        runs[0]["average_steps"] = 999
    elif mutation == "negative_distance":
        episodes[0]["final_distance"] = -1
    elif mutation == "nan_steps":
        episodes[0]["steps"] = "NaN"
    elif mutation == "bad_success":
        episodes[0]["success"] = "yes"
    elif mutation == "bad_algorithm":
        episodes[0]["algorithm"] = "unknown"
    elif mutation == "inconsistent_distance":
        episodes[0]["final_distance"] = 1
    elif mutation == "bad_seed":
        runs[0]["eval_seed"] = 999
    elif mutation == "bad_rate":
        runs[0]["success_rate"] = 2
    elif mutation == "bad_counts":
        runs[0]["failure_count"] = 99
    write_csv(directory / "algorithm_comparison_episodes.csv", episodes, fields)
    write_csv(directory / "algorithm_comparison_summary.csv", runs, run_fields)
    status = analysis.validate_dataset(directory)
    assert not status["valid"]
    assert expected in "\n".join(status["errors"])
    assert status["runs"] == status["episodes"] == []


@pytest.mark.parametrize("mutation,expected", [
    ("bad_noise", "noise_std"), ("summary", "Summary 재계산"),
    ("random_changes", "Noise 독립성"), ("missing_run", "알고리즘 Run 누락"),
])
def test_noise_validation_failures(fixed_csvs, mutation, expected):
    directory, data = fixed_csvs
    if mutation == "bad_noise":
        rows, fields = data["noise_comparison_episodes.csv"]
        rows[0]["noise_std"] = -1
        write_csv(directory / "noise_comparison_episodes.csv", rows, fields)
    elif mutation == "summary":
        rows, fields = data["noise_comparison_summary.csv"]
        rows[0]["average_steps_mean"] = 99
        write_csv(directory / "noise_comparison_summary.csv", rows, fields)
    elif mutation == "random_changes":
        rows, fields = data["noise_comparison_episodes.csv"]
        target = next(row for row in rows if row["algorithm"] == "random" and row["noise_std"] == 5)
        target["steps"] += 1
        write_csv(directory / "noise_comparison_episodes.csv", rows, fields)
    else:
        rows, fields = data["noise_comparison_runs.csv"]
        rows.pop()
        write_csv(directory / "noise_comparison_runs.csv", rows, fields)
    status = analysis.validate_dataset(directory, noise=True)
    assert not status["valid"]
    assert expected in "\n".join(status["errors"])


def test_paired_differences_sign_units_and_run_count(fixed_csvs):
    _, data = fixed_csvs
    runs, _ = data["algorithm_comparison_summary.csv"]
    summaries, pairs = analysis.paired_comparisons(runs, "algorithm")
    success = next(row for row in summaries if row["metric"] == "success_rate")
    assert success["mean"] == pytest.approx(100 / 3)
    assert success["unit"] == "percentage points"
    assert success["valid_run_count"] == 2
    steps = next(row for row in summaries if row["metric"] == "average_steps")
    assert steps["mean"] == pytest.approx(-19 / 6)
    assert steps["ci95_low"] < steps["mean"] < steps["ci95_high"]
    assert len(pairs) == 12
    assert all(row["difference"] == pytest.approx(row["q_value"] - row["random_value"]) for row in pairs)


def test_unverified_pairs_are_not_matched(fixed_csvs):
    _, data = fixed_csvs
    runs, _ = data["algorithm_comparison_summary.csv"]
    for row in runs:
        if row["algorithm"] == "random":
            row["eval_seed"] = 999
    stats, pairs = analysis.paired_comparisons(runs, "algorithm")
    assert pairs == []
    assert all(row["valid_run_count"] == 0 and row["mean"] is None for row in stats)


def test_no_success_is_missing_with_valid_run_counts(fixed_csvs):
    _, data = fixed_csvs
    runs, _ = data["algorithm_comparison_summary.csv"]
    for row in runs:
        if row["algorithm"] == "random":
            row["average_steps_on_success"] = None
    tables = analysis.summarize_runs(runs)
    row = next(row for row in tables if row["algorithm"] == "random")
    assert row["average_steps_on_success_mean"] is None
    assert row["average_steps_on_success_valid_run_count"] == 0
    stats, pairs = analysis.paired_comparisons(runs, "algorithm")
    assert next(row for row in stats if row["metric"] == "average_steps_on_success")["valid_run_count"] == 0
    assert all(row["metric"] != "average_steps_on_success" for row in pairs)


def test_outputs_reports_and_source_files_unchanged(fixed_csvs):
    directory, _ = fixed_csvs
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.glob("*.csv")}
    output = directory / "analysis_output"
    result = analysis.analyze_paper_results(directory, output)
    assert all(status["valid"] for status in result["validation"])
    for name in ("algorithm_performance", "noise_performance", "algorithm_differences"):
        assert (output / "paper_tables" / f"{name}.csv").exists()
        assert (output / "paper_tables" / f"{name}.md").exists()
    assert (output / "paper_tables" / "paired_run_differences.csv").exists()
    report = (output / "paper_analysis_summary.md").read_text(encoding="utf-8")
    for phrase in ("검증 통과", "Student", "현재 config.py", "입증할 수 없다", "SHA-256", "일반화", "독립"):
        assert phrase in report
    for path, digest in hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_invalid_dataset_is_excluded_while_valid_dataset_is_analyzed(fixed_csvs):
    directory, _ = fixed_csvs
    (directory / "algorithm_comparison_summary.csv").write_text("wrong\n", encoding="utf-8")
    output = directory / "partial"
    result = analysis.analyze_paper_results(directory, output)
    assert result["algorithm"] == []
    assert result["noise"]
    assert "데이터셋 전체 분석 제외" in (output / "paper_analysis_summary.md").read_text(encoding="utf-8")
    with (output / "paper_tables" / "algorithm_performance.csv").open(encoding="utf-8", newline="") as file:
        assert list(csv.DictReader(file)) == []


def test_missing_sources_do_not_start_training(tmp_path, monkeypatch):
    from src import compare_algorithms

    def forbidden(*args, **kwargs):
        raise AssertionError("분석에서 학습을 실행하면 안 됩니다.")

    monkeypatch.setattr(compare_algorithms, "train_q_learning_agent", forbidden)
    result = analysis.analyze_paper_results(tmp_path / "absent", tmp_path / "out")
    assert not any(status["valid"] for status in result["validation"])
    assert result["algorithm"] == result["noise"] == []


def test_analysis_module_cli(fixed_csvs):
    directory, _ = fixed_csvs
    process = subprocess.run(
        [sys.executable, "-m", "src.analyze_paper_results", "--input-dir", str(directory),
         "--output-dir", str(directory / "cli")],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True,
    )
    assert process.returncode == 0
    assert (directory / "cli" / "paper_analysis_summary.md").exists()
