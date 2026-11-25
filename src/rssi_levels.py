"""Calibrated 전용 조정 RSSI Level 경계.

종합설계 최종 발표를 위한 추가 개선이며, 이미 제출한 논문 결과의 조건이 아니다.
실측 RSSI로 경계를 직접 최적화한 것이 아니라, 기존 경계가 거리별로 나누던 구분을
보정 모델에서 비슷하게 유지하도록 옮긴 것이다.

1. Baseline 모델의 1m, 15m noise-free RSSI(식 (1) 포함)를 계산한다.
2. Calibrated 모델의 같은 거리 noise-free RSSI를 계산한다.
3. 두 점을 잇는 선형 대응 calibrated = intercept + slope * baseline을 구한다.
4. 기존 경계(config.RSSI_THRESHOLDS) 4개에 같은 변환을 적용한다.
"""
import random
from contextlib import contextmanager

import config
from src.env_factory import build_environment


REFERENCE_DISTANCES_M = (1, 15)  # 실측 범위의 양 끝
# 대응 결과에 영향을 주는 설정. 이 값이 바뀌면 다시 계산한다.
_MAPPING_SETTINGS = (
    "RSSI_MODEL", "RSSI_CALIBRATION_FILE", "RSSI_THRESHOLDS", "CARRIER_FREQUENCY_HZ",
    "TX_POWER_DBM", "ETA_LOS", "ETA_NLOS", "PATH_LOSS_EXPONENT", "MULTIPATH_FADING",
    "MIN_DISTANCE_M", "NOISE_FLOOR_DBM", "CHANNEL_STATE",
)
_cache = {}


@contextmanager
def _settings(**values):
    previous = {name: getattr(config, name) for name in values}
    try:
        for name, value in values.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def noise_free_rssi(model, distances_m=REFERENCE_DISTANCES_M):
    """현재 물리 설정에서 model("baseline"/"calibrated")의 거리(m)별 noise-free RSSI.

    실제 실험과 같은 환경 생성 함수를 쓰며, 1m 격자에서 (0,0)→(d,0) 거리로 계산한다.
    전역 config와 random 상태는 호출 전으로 복원한다.
    """
    state = random.getstate()
    try:
        with _settings(RSSI_MODEL=model, GRID_SIZE=max(distances_m) + 1, AGENT_START=(0, 0),
                       CELL_SIZE_M=1.0):
            env = build_environment(noise_std=0)
            return {d: env.reset(target_position=(d, 0)) for d in distances_m}
    finally:
        random.setstate(state)


def calibrated_level_mapping():
    """조정 경계와 계산 근거(기준 RSSI, slope, intercept)를 반환한다."""
    if config.RSSI_MODEL != "calibrated":
        raise ValueError("calibrated_adjusted RSSI levels require RSSI_MODEL='calibrated'")
    near, far = REFERENCE_DISTANCES_M
    baseline = noise_free_rssi("baseline")
    calibrated = noise_free_rssi("calibrated")
    slope = (calibrated[far] - calibrated[near]) / (baseline[far] - baseline[near])
    intercept = calibrated[near] - slope * baseline[near]
    original = [float(t) for t in config.RSSI_THRESHOLDS]
    adjusted = [intercept + slope * t for t in original]
    if not slope > 0 or any(a >= b for a, b in zip(adjusted, adjusted[1:])):
        raise ValueError("Adjusted RSSI thresholds must keep the original increasing order")
    return {
        "method": "linear map of baseline->calibrated noise-free RSSI at 1 m and 15 m",
        "reference_distances_m": list(REFERENCE_DISTANCES_M),
        "baseline_noise_free_rssi_dbm": [baseline[d] for d in REFERENCE_DISTANCES_M],
        "calibrated_noise_free_rssi_dbm": [calibrated[d] for d in REFERENCE_DISTANCES_M],
        "slope": slope, "intercept": intercept,
        "original_thresholds": original, "adjusted_thresholds": adjusted,
    }


def calibrated_adjusted_thresholds():
    """현재 설정의 조정 경계. State 생성마다 호출되므로 설정값별로 한 번만 계산한다."""
    key = tuple(tuple(value) if isinstance(value, list) else value
                for value in (getattr(config, name) for name in _MAPPING_SETTINGS))
    if key not in _cache:
        _cache[key] = calibrated_level_mapping()["adjusted_thresholds"]
    return _cache[key]
