import numpy as np
import pytest

from src.calibration_data import DATA, read_measurements
from src.calibration_fit import fit_measurements, rssi_from_rx


def test_fit_recovers_known_equation_one_curve():
    rows = [{"distance_m": d, "rssi_dbm": float(rssi_from_rx(-55 - 24 * np.log10(d), -90))}
            for d in (1, 2, 3, 5, 10, 15) for _ in range(3)]
    result = fit_measurements(rows)
    assert result["reference_rx_power_dbm"] == pytest.approx(-55, abs=1e-6)
    assert result["path_loss_exponent"] == pytest.approx(2.4, abs=1e-6)
    assert result["rmse_db"] < 1e-6


def test_real_primary_and_strict_remain_separate():
    primary = fit_measurements(read_measurements(DATA / "scenario1_provisional.csv"))
    strict = fit_measurements(read_measurements(DATA / "scenario1_strict_provisional.csv"))
    assert (primary["sample_count"], strict["sample_count"]) == (90, 89)
    assert [r["count"] for r in primary["distance_statistics"]] == [15] * 6
    assert primary["path_loss_exponent"] > 0
    assert primary["within_distance_repeat_sd_db"] > 0
    assert primary["rmse_db"] > 0


def test_invalid_noise_floor_is_not_silently_fitted():
    rows = read_measurements(DATA / "scenario1_provisional.csv")
    with pytest.raises(ValueError):
        fit_measurements(rows, -40)
