import math
from unittest.mock import patch

import pytest

from src.env_factory import build_environment


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
