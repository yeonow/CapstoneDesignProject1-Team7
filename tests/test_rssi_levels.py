"""RSSI Level 경계 선택: 기존 경계 보존, 선택 방식, Calibrated 조정 경계 검증."""
import random

import pytest

import config
from src import rssi_levels
from src import state_reward as sr
from src.calibration_data import DATA


ORIGINAL = [-80.0, -70.0, -60.0, -50.0]
# 진단 실험(results/rssi_level_validation_seed10042_10runs/validation_manifest.json)에 기록된 값
DIAGNOSIS_ADJUSTED = [-89.04901890526463, -81.25897503062048,
                      -73.46893115597632, -65.67888728133218]


@pytest.fixture
def calibrated(monkeypatch):
    monkeypatch.setattr(config, "RSSI_MODEL", "calibrated")
    monkeypatch.setattr(config, "RSSI_CALIBRATION_FILE", DATA / "calibration_profile.json")


def test_default_settings_keep_original_thresholds():
    assert config.RSSI_THRESHOLDS == ORIGINAL
    assert config.RSSI_LEVEL_MODE == "original"
    assert sr.rssi_thresholds() == ORIGINAL
    assert sr.num_rssi_levels() == 5
    assert [sr.get_rssi_level(r) for r in (-90, -80, -75, -65, -55, -50, -30)] == [0, 1, 1, 2, 3, 4, 4]


def test_unset_mode_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", None)
    assert sr.rssi_thresholds() == ORIGINAL


@pytest.mark.parametrize("mode", ["adjusted", "ORIGINAL", ""])
def test_unknown_level_mode_is_rejected(monkeypatch, mode):
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", mode)
    with pytest.raises(ValueError, match="RSSI_LEVEL_MODE"):
        sr.get_rssi_level(-65)


def test_adjusted_thresholds_equal_the_diagnosis_values(calibrated, monkeypatch):
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", "calibrated_adjusted")
    assert sr.rssi_thresholds() == DIAGNOSIS_ADJUSTED
    assert sr.num_rssi_levels() == 5
    mapping = rssi_levels.calibrated_level_mapping()
    assert mapping["reference_distances_m"] == [1, 15]
    assert mapping["baseline_noise_free_rssi_dbm"] == pytest.approx([-43.0622201, -66.5643994])
    assert mapping["calibrated_noise_free_rssi_dbm"] == pytest.approx([-60.2743263, -78.5826271])
    assert mapping["slope"] == pytest.approx(0.7790043874644151, abs=1e-15)
    assert mapping["intercept"] == pytest.approx(-26.72866790811142, abs=1e-12)
    assert mapping["original_thresholds"] == ORIGINAL
    # 대응식이 두 기준점을 정확히 잇는다.
    for base, cal in zip(mapping["baseline_noise_free_rssi_dbm"],
                         mapping["calibrated_noise_free_rssi_dbm"]):
        assert mapping["intercept"] + mapping["slope"] * base == pytest.approx(cal, abs=1e-12)


def test_calibrated_model_can_keep_original_levels(calibrated):
    assert config.RSSI_LEVEL_MODE == "original"
    assert sr.rssi_thresholds() == ORIGINAL


def test_adjusted_levels_are_rejected_for_baseline_model(monkeypatch):
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", "calibrated_adjusted")
    assert config.RSSI_MODEL == "baseline"
    with pytest.raises(ValueError, match="RSSI_MODEL='calibrated'"):
        sr.get_rssi_level(-65)


def test_mapping_restores_config_and_random_state(calibrated):
    before_config = {k: v for k, v in vars(config).items() if k.isupper()}
    before_random = random.getstate()
    rssi_levels.calibrated_level_mapping()
    assert random.getstate() == before_random
    assert {k: v for k, v in vars(config).items() if k.isupper()} == before_config


def test_adjusted_thresholds_follow_changed_settings(calibrated, monkeypatch):
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", "calibrated_adjusted")
    assert sr.rssi_thresholds() == DIAGNOSIS_ADJUSTED
    monkeypatch.setattr(config, "RSSI_THRESHOLDS", [-85.0, -75.0, -65.0, -55.0])
    mapping = rssi_levels.calibrated_level_mapping()
    assert sr.rssi_thresholds() == mapping["adjusted_thresholds"] != DIAGNOSIS_ADJUSTED
