import random

import pytest

from src.calibration_analysis import run_analysis
from src.calibration_profile import load_profile


def test_full_measured_analysis_writes_profile_and_preserves_rng(tmp_path):
    output = tmp_path / "analysis"
    state = random.getstate()
    report = run_analysis(output)
    assert random.getstate() == state
    assert report["quality"]["primary_count"] == 90
    profile = load_profile(output / "profile.json")
    assert profile["path_loss_exponent"] == report["primary_fit"]["path_loss_exponent"]
    for name in ("distance_model_comparison.png", "distance_model_comparison.pdf", "movement_trends.png"):
        assert (output / name).stat().st_size > 100
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_analysis(output)
    assert (output / "report.json").read_bytes() == before
