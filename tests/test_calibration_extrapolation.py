"""실측 범위(1~15m) 밖인 20m에서 기존 보정 회귀모델의 외삽 예측을 검증한다.

배포된 프로필을 그대로 읽으며 다시 피팅하거나 파라미터/RSSI 수식을 바꾸지 않는다.
"""
import math
import random

import pytest

import config
from src.calibration_analysis import compare_curves
from src.calibration_data import DATA, read_measurements
from src.calibration_fit import rssi_from_rx
from src.calibration_profile import load_profile
from src.env_factory import build_environment


PROFILE = DATA / "calibration_profile.json"
# da5f284에서 배포한 최종 90개 피팅값(문서 기록값). 이번 검증에서 다시 피팅하지 않는다.
FITTED_REFERENCE_RX_DBM = -60.27895489260387
FITTED_EXPONENT = 1.583968492442464
# (0,0)에서 Euclidean 20m인 20x20 Grid 칸. Chebyshev 16, Manhattan 28과 구분된다.
TWENTY_METERS = (12, 16)


@pytest.fixture
def grid20(monkeypatch):
    for name, value in {"GRID_SIZE": 20, "AGENT_START": (0, 0), "CELL_SIZE_M": 1.0,
                        "MIN_DISTANCE_M": 1.0, "CHANNEL_STATE": "los", "NOISE_FLOOR_DBM": -90.0,
                        "RSSI_CALIBRATION_FILE": PROFILE}.items():
        monkeypatch.setattr(config, name, value)


def test_profile_keeps_fitted_parameters_and_measured_range():
    profile = load_profile(PROFILE)
    assert profile["reference_rx_power_dbm"] == FITTED_REFERENCE_RX_DBM
    assert profile["path_loss_exponent"] == FITTED_EXPONENT
    assert profile["noise_floor_dbm"] == -90.0
    assert profile["measurement_range_m"] == [1, 15]
    assert "20m disconnected and unobserved" in profile["limitations"]


def test_calibrated_model_predicts_20m_from_euclidean_distance(grid20, monkeypatch):
    monkeypatch.setattr(config, "RSSI_MODEL", "calibrated")
    env = build_environment(noise_std=0)
    assert env.calibrated_reference_rx_power_dbm == FITTED_REFERENCE_RX_DBM
    assert env.path_loss_exponent == FITTED_EXPONENT
    env.reset(target_position=TWENTY_METERS)
    info = env.get_info()
    assert info["distance"] == 20.0
    expected_rx = FITTED_REFERENCE_RX_DBM - 10 * FITTED_EXPONENT * math.log10(20)
    assert info["current_rx_power"] == pytest.approx(expected_rx, abs=1e-9)
    assert info["current_rssi"] == pytest.approx(float(rssi_from_rx(expected_rx, -90.0)), abs=1e-9)
    assert info["current_rssi"] == pytest.approx(-80.384, abs=1e-3)


@pytest.mark.parametrize("mode", ["baseline", "calibrated"])
def test_far_targets_give_finite_rssi_with_runtime_noise(grid20, monkeypatch, mode):
    monkeypatch.setattr(config, "RSSI_MODEL", mode)
    state = random.getstate()
    try:
        random.seed(f"extrapolation:{mode}")
        env = build_environment(noise_std=2.0)
        for target in (TWENTY_METERS, (19, 19)):  # 20m와 Grid 최대 약 26.9m
            for _ in range(200):
                env.reset(target_position=target)
                info = env.get_info()
                assert all(math.isfinite(info[key])
                           for key in ("current_rx_power", "current_snr", "current_rssi"))
    finally:
        random.setstate(state)


def test_20m_is_reported_as_extrapolation_not_measurement():
    primary = read_measurements(DATA / "scenario1_provisional.csv")
    assert {row["distance_m"] for row in primary} == {1, 2, 3, 5, 10, 15}
    rows = compare_curves(primary, load_profile(PROFILE), draws=2)
    twenty = {row["mode"]: row for row in rows if row["distance_m"] == 20}
    assert set(twenty) == {"baseline", "calibrated"}
    for row in twenty.values():
        assert row["observed"] is False
        assert row["measured_mean_dbm"] is None and row["measured_sd_db"] is None
        assert math.isfinite(row["noise_free_rssi_dbm"])
    assert twenty["calibrated"]["noise_free_rssi_dbm"] == pytest.approx(-80.384, abs=1e-3)
    assert all(row["observed"] for row in rows if row["distance_m"] <= 15)


def test_baseline_and_calibrated_remain_selectable_at_20m(grid20, monkeypatch):
    environments, rssi = {}, {}
    for mode in ("baseline", "calibrated"):
        monkeypatch.setattr(config, "RSSI_MODEL", mode)
        environments[mode] = build_environment(noise_std=0)
        rssi[mode] = environments[mode].reset(target_position=TWENTY_METERS)
    baseline = environments["baseline"]
    assert baseline.calibrated_reference_rx_power_dbm is None
    assert baseline.path_loss_exponent == config.PATH_LOSS_EXPONENT
    assert baseline.carrier_frequency_hz == config.CARRIER_FREQUENCY_HZ
    assert environments["calibrated"].carrier_frequency_hz == 2452e6
    assert rssi["baseline"] == pytest.approx(-69.048, abs=1e-3)
    assert rssi["calibrated"] == pytest.approx(-80.384, abs=1e-3)
