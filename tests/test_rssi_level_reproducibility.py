"""조정 RSSI Level의 State 반영, 기존 모드 재현성, 세 설정의 동일 평가 조건 검증."""
import math
import random
from bisect import bisect_right

import numpy as np
import pytest

import config
from src import compare_algorithms as comparison
from src import experiment
from src import state_reward as sr
from src.calibration_data import DATA


ORIGINAL = [-80.0, -70.0, -60.0, -50.0]
DIAGNOSIS_ADJUSTED = [-89.04901890526463, -81.25897503062048,
                      -73.46893115597632, -65.67888728133218]
SETTINGS = {
    "baseline_original": {"RSSI_MODEL": "baseline", "RSSI_LEVEL_MODE": "original"},
    "calibrated_original": {"RSSI_MODEL": "calibrated", "RSSI_LEVEL_MODE": "original"},
    "calibrated_adjusted": {"RSSI_MODEL": "calibrated", "RSSI_LEVEL_MODE": "calibrated_adjusted"},
}


@pytest.fixture(autouse=True)
def paper_condition(monkeypatch):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # 논문 Target/Noise 조건. MAX_STEPS만 테스트 시간을 위해 줄인다.
    for name, value in {"GRID_SIZE": 20, "AGENT_START": (0, 0), "TARGET_POSITION": (19, 19),
                        "MAX_STEPS": 60, "TARGET_MODE": "split", "TARGET_MIN_DISTANCE": 5,
                        "TARGET_DISTANCE_METRIC": "chebyshev", "NOISE_STD": 2.0,
                        "LOG_INTERVAL": 0,
                        "RSSI_CALIBRATION_FILE": DATA / "calibration_profile.json"}.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def apply(monkeypatch, setting):
    for name, value in SETTINGS[setting].items():
        monkeypatch.setattr(config, name, value)


def compare(output, **kwargs):
    return comparison.compare_algorithms(train_episodes=4, eval_episodes=6, seed=42, runs=2,
                                         output_dir=output, **kwargs)


def test_adjusted_level_intervals_and_state(monkeypatch):
    apply(monkeypatch, "calibrated_adjusted")
    assert sr.rssi_thresholds() == DIAGNOSIS_ADJUSTED
    for level, threshold in enumerate(DIAGNOSIS_ADJUSTED, start=1):
        assert sr.get_rssi_level(threshold) == level
        assert sr.get_rssi_level(math.nextafter(threshold, -math.inf)) == level - 1
    assert sr.get_rssi_level(-95.0) == 0 and sr.get_rssi_level(-40.0) == 4
    # Trend(±2 dB)와 Previous Action은 Level 모드와 무관하다.
    for current, previous in ((-70.0, -73.5), (-80.0, -79.0), (-66.0, None)):
        trend = sr.get_rssi_trend(current, previous)
        assert sr.make_state(current, previous, 3) == (bisect_right(DIAGNOSIS_ADJUSTED, current), trend, 3)
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", "original")
    assert sr.get_rssi_trend(-70.0, -73.5) == 1 and sr.get_rssi_trend(-80.0, -79.0) == 0


@pytest.mark.parametrize("setting", ["baseline_original", "calibrated_original"])
def test_original_mode_reproduces_previous_level_function(monkeypatch, tmp_path, setting):
    apply(monkeypatch, setting)
    current = compare(tmp_path / "current")
    # 변경 전 구현: 기본 경계 [-80, -70, -60, -50]에 bisect_right를 직접 적용
    monkeypatch.setattr(sr, "get_rssi_level", lambda rssi: bisect_right(ORIGINAL, rssi))
    previous = compare(tmp_path / "previous")
    assert current == previous


def test_adjusted_mode_equals_diagnosis_threshold_override(monkeypatch, tmp_path):
    apply(monkeypatch, "calibrated_adjusted")
    adjusted = compare(tmp_path / "mode")
    # 진단 실험처럼 original 모드에서 경계 값만 덮어쓴 경우와 결과가 같아야 한다.
    monkeypatch.setattr(config, "RSSI_LEVEL_MODE", "original")
    monkeypatch.setattr(config, "RSSI_THRESHOLDS", list(DIAGNOSIS_ADJUSTED))
    override = compare(tmp_path / "override")
    assert adjusted == override


def test_adjusted_levels_reach_training_and_evaluation_states(monkeypatch):
    apply(monkeypatch, "calibrated_adjusted")
    phase = {"name": None}
    seen = {"train": [], "eval": []}
    real_run, real_state = comparison.run_episode, experiment.make_state

    def run(env, agent, training=True, **kwargs):
        phase["name"] = "train" if training else "eval"
        return real_run(env, agent, training=training, **kwargs)

    def state(current_rssi, previous_rssi, previous_action):
        result = real_state(current_rssi, previous_rssi, previous_action)
        seen[phase["name"]].append((current_rssi, result))
        return result

    monkeypatch.setattr(comparison, "run_episode", run)
    monkeypatch.setattr(experiment, "make_state", state)
    agent = comparison.train_q_learning_agent(4, seed=42)
    comparison.evaluate_q_learning(agent, 6, seed=43)
    for name in ("train", "eval"):
        assert seen[name]
        assert all(s[0] == bisect_right(DIAGNOSIS_ADJUSTED, rssi) for rssi, s in seen[name])
        assert any(bisect_right(DIAGNOSIS_ADJUSTED, rssi) != bisect_right(ORIGINAL, rssi)
                   for rssi, _ in seen[name])
    # Q-table의 State 키 (level, trend, previous_action)에 조정 Level이 그대로 쓰인다.
    q_levels = {key[0] for key in agent.q_table}
    assert {s[0] for _, s in seen["train"]} <= q_levels <= set(range(5))


def test_three_settings_share_targets_seeds_and_random_results(monkeypatch, tmp_path):
    results = {}
    for setting in SETTINGS:
        apply(monkeypatch, setting)
        results[setting] = compare(tmp_path / setting)
    identity = {}
    for setting, result in results.items():
        for row in result["episodes"]:
            key = (row["run"], row["episode"])
            value = (row["train_seed"], row["eval_seed"], row["target_x"], row["target_y"])
            identity.setdefault(key, set()).add(value)
    assert len(identity) == 2 * 6 and all(len(values) == 1 for values in identity.values())
    random_rows = [[row for row in result["episodes"] if row["algorithm"] == "random"]
                   for result in results.values()]
    assert random_rows[0] == random_rows[1] == random_rows[2]
