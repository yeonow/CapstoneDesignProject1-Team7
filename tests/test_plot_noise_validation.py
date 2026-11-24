"""실험을 실행하지 않는 가상 Summary CSV 검증 테스트."""

import csv

import pytest

from src.compare_noise_sweep import NOISE_SUMMARY_FIELDS, aggregate_noise_comparison
from src.plot_noise_comparison import load_summary, plot_noise_comparison


@pytest.fixture
def summary_csv(tmp_path):
    row = {"noise_std": 2.0, "algorithm": "q_learning", "train_episodes": 3,
           "eval_episodes": 2, "success_count": 0, "failure_count": 2,
           "success_rate": 0.0, "average_steps": 10.0,
           "average_steps_on_success": None, "average_final_distance": 5.0}
    rows = aggregate_noise_comparison([row])
    path = tmp_path / "synthetic.csv"

    def write():
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=NOISE_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    return rows, write


@pytest.mark.parametrize("value", ["NaN", "inf", "oops", "", -1])
def test_noise_loader_rejects_corrupt_mean(summary_csv, value):
    rows, write = summary_csv
    rows[0]["average_steps_mean"] = value
    with pytest.raises(ValueError):
        load_summary(write())


def test_noise_loader_preserves_no_success_and_rejects_duplicates(summary_csv):
    rows, write = summary_csv
    assert load_summary(write())[0]["average_steps_on_success_mean"] is None
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="중복"):
        load_summary(write())


@pytest.mark.parametrize("content", ["", "algorithm\nrandom\n", "noise_std,noise_std\n2,2\n"])
def test_noise_loader_requires_nonempty_complete_schema(tmp_path, content):
    path = tmp_path / "invalid.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_summary(path)


def test_direct_plot_rejects_bad_identity_before_creating_output(tmp_path):
    output = tmp_path / "never.png"
    row = {"noise_std": 2, "algorithm": "q_learning"}
    with pytest.raises(ValueError, match="duplicate"):
        plot_noise_comparison([row, row], output)
    with pytest.raises(ValueError, match="finite"):
        plot_noise_comparison([{**row, "noise_std": float("nan")}], output)
    assert not output.exists()
