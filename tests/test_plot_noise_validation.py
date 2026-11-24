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


def test_percent_sd_and_five_noise_success_gaps(tmp_path, monkeypatch):
    import math
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # 모든 수치는 선의 단절/단위 변환만 확인하기 위한 가상 테스트 데이터이다.
    rows = [{"noise_std": level, "algorithm": "q_learning", "run_count": 3,
             "success_rate_mean": .4, "success_rate_std": .1,
             "average_steps_on_success_mean": None if level == 6 else 20.0,
             "average_steps_on_success_std": None if level == 6 else 2.0,
             "average_steps_on_success_run_count": 0 if level == 6 else 1}
            for level in (2, 4, 6, 8, 10)]
    captured = []
    original = plt.subplots

    def capture(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured.append((fig, axes))
        return fig, axes

    monkeypatch.setattr(plt, "subplots", capture)
    output = tmp_path / "synthetic.png"
    plot_noise_comparison(rows, output, success_percent=True, dpi=300)
    fig, axes = captured[0]
    rate, successes = axes.flat[0], axes.flat[2]
    np.testing.assert_allclose(rate.lines[0].get_ydata(), [40] * 5)
    np.testing.assert_allclose(rate.containers[0].lines[2][0].get_segments()[0][:, 1], [30, 50])
    assert rate.get_ylabel() == "Success rate (%)"
    assert list(successes.lines[0].get_xdata()) == [2, 4, 6, 8, 10]
    assert math.isnan(successes.lines[0].get_ydata()[2])
    assert not successes.containers  # 유효 Run 한 개의 SD=0 관례를 오차막대로 오해하지 않음
    assert any("not CI" in text.get_text() for text in fig.texts)
    with Image.open(output) as picture:
        assert picture.info["dpi"][0] >= 299.9
