"""RSSI Level 경계 선택: 기존 경계 보존과 선택 방식 검증."""
import pytest

import config
from src import state_reward as sr


ORIGINAL = [-80.0, -70.0, -60.0, -50.0]


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
