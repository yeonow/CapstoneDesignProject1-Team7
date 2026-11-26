"""기본 실행 환경이 최종 실측 보정 환경(20×20, 400 Step, split, 보정 RSSI + 조정 Level)인지 검증한다."""
import pytest

import config
from src import compare_algorithms as comparison
from src import rssi_level_experiment as level_experiment
from src import state_reward as sr
from src import targets
from src.env_factory import build_environment


ORIGINAL = [-80.0, -70.0, -60.0, -50.0]
ADJUSTED = [-89.04901890526463, -81.25897503062048, -73.46893115597632, -65.67888728133218]


def test_config_defaults_select_the_calibrated_environment():
    expected = {
        "GRID_SIZE": 20, "MAX_STEPS": 400, "AGENT_START": (0, 0), "TARGET_POSITION": (19, 19),
        "TARGET_MODE": "split", "TARGET_MIN_DISTANCE": 5, "TARGET_DISTANCE_METRIC": "chebyshev",
        "RSSI_MODEL": "calibrated", "RSSI_LEVEL_MODE": "calibrated_adjusted",
        "RSSI_CALIBRATION_FILE": "data/rssi_calibration_20261125/calibration_profile.json",
        "NUM_EPISODES": 3000, "NUM_EVAL_EPISODES": 100, "NOISE_STD": 2.0,
        # 보정 모델과 Level 변환의 기준값이므로 바꾸지 않는다.
        "PATH_LOSS_EXPONENT": 2.0, "RSSI_THRESHOLDS": ORIGINAL,
    }
    assert {name: getattr(config, name) for name in expected} == expected


def test_default_environment_applies_profile_and_adjusted_levels():
    env = build_environment()
    assert env.path_loss_exponent == 1.583968492442464
    assert env.calibrated_reference_rx_power_dbm == -60.27895489260387
    assert (round(env.path_loss_exponent, 3), round(env.calibrated_reference_rx_power_dbm, 3)) == (
        1.584, -60.279)
    assert env.carrier_frequency_hz == 2452e6
    assert (env.grid_size, env.max_steps, env.noise_std) == (20, 400, 2.0)
    env.noise_std = 0
    env.reset(target_position=(1, 0))  # 1m 기준 수신 전력
    assert env.current_rx_power == pytest.approx(-60.27895489260387, abs=1e-9)
    assert sr.rssi_thresholds() == ADJUSTED
    assert [round(value, 2) for value in sr.rssi_thresholds()] == [-89.05, -81.26, -73.47, -65.68]


def test_default_split_has_187_train_and_188_eval_targets():
    assert targets.validate_target_config() == "split"
    train, evaluation = targets.target_candidates("train"), targets.target_candidates("eval")
    assert (len(train), len(evaluation)) == (187, 188)
    assert not set(train) & set(evaluation)


def test_baseline_and_original_levels_remain_selectable(monkeypatch):
    monkeypatch.setattr(config, "RSSI_MODEL", "baseline")
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", "original")
    env = build_environment()
    assert env.calibrated_reference_rx_power_dbm is None
    assert (env.path_loss_exponent, env.carrier_frequency_hz) == (2.0, 2.4e9)
    assert sr.rssi_thresholds() == ORIGINAL


def test_defaults_reproduce_the_previous_explicit_calibrated_settings(tmp_path):
    kwargs = dict(train_episodes=20, eval_episodes=10, runs=2, seed=42)
    # 이전 방식: 공통 설정과 calibrated_adjusted를 실행기에서 명시 적용
    level_experiment.run_level_experiment(tmp_path / "explicit", settings=["calibrated_adjusted"],
                                          **kwargs)
    # 새 방식: 아무 설정 변경 없이 기본값으로 실행
    comparison.compare_algorithms(output_dir=tmp_path / "default", **kwargs)
    for name in ("algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv"):
        assert ((tmp_path / "default" / name).read_bytes()
                == (tmp_path / "explicit" / "calibrated_adjusted" / name).read_bytes())
