import math
from unittest.mock import patch

import pytest

from src.env_factory import build_environment


def test_factory_selects_profile_without_changing_baseline(tmp_path, monkeypatch):
    import config
    import json
    from src.calibration_data import EXPECTED_HASHES
    profile = dict(schema_version=1, source_sha256=EXPECTED_HASHES, reference_distance_m=1,
                   channel_state="los", reference_rx_power_dbm=-55, path_loss_exponent=2.4,
                   carrier_frequency_hz=2452e6, noise_floor_dbm=-90)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(config, "RSSI_CALIBRATION_FILE", path)
    monkeypatch.setattr(config, "RSSI_MODEL", "baseline")  # 기본값은 calibrated이므로 명시 선택
    baseline = build_environment(noise_std=0)
    monkeypatch.setattr(config, "RSSI_MODEL", "calibrated")
    calibrated = build_environment(noise_std=0)
    assert calibrated.path_loss_exponent == 2.4
    assert calibrated.carrier_frequency_hz == 2452e6
    assert baseline.calibrated_reference_rx_power_dbm is None
    assert baseline.path_loss_exponent == config.PATH_LOSS_EXPONENT
    calibrated.reset(target_position=(1, 0))
    assert calibrated.current_rx_power == pytest.approx(-55)
    monkeypatch.setattr(config, "NOISE_FLOOR_DBM", -80)
    with pytest.raises(ValueError, match="noise floor"):
        build_environment()


@pytest.mark.parametrize("mode", ["typo", "calibrated"])
def test_bad_or_missing_mode_settings_fail(monkeypatch, mode):
    import config
    monkeypatch.setattr(config, "RSSI_MODEL", mode)
    monkeypatch.setattr(config, "RSSI_CALIBRATION_FILE", None)
    with pytest.raises(ValueError):
        build_environment()


def test_calibrated_reference_and_cached_single_noise_sample():
    env = build_environment(noise_std=2)
    env.calibrated_reference_rx_power_dbm = -55.0
    env.path_loss_exponent = 2.4
    with patch("src.environment.random.gauss", return_value=1.5) as noise:
        env.reset(target_position=(3, 4))
        expected_rx = -55 - 24 * math.log10(5) + 1.5
        assert env.current_rx_power == pytest.approx(expected_rx)
        expected_rssi = 10 * math.log10(10 ** (expected_rx / 10) + 10 ** (env.noise_floor_dbm / 10))
        assert env.get_rssi() == pytest.approx(expected_rssi)
        before = env.get_info()
        assert env.get_info() == before
        assert noise.call_count == 1
        env.move(0)  # 경계 충돌도 동일하게 측정 1회
        assert noise.call_count == 2
        assert env.step_count == 1


def test_calibration_does_not_invent_nlos_transfer():
    env = build_environment()
    env.calibrated_reference_rx_power_dbm = -55
    env.channel_state = "nlos"
    with pytest.raises(ValueError, match="NLoS"):
        env.reset()
