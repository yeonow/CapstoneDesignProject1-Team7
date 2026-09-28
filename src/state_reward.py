"""
state_reward.py  
Raw RSSI -> RSSI 대표값 / Level / Trend / State / Reward 변환
"""

from bisect import bisect_right
from statistics import median

try:
    import config
except ImportError: 
    config = None

# ---------------------------------------------------------------
# 공통 상수 
# ---------------------------------------------------------------
NONE_ACTION = getattr(config, "NONE_ACTION", -1)   
TREND_DOWN, TREND_KEEP, TREND_UP = -1, 0, 1

# ---------------------------------------------------------------
# 기본값 
# ---------------------------------------------------------------
_DEFAULTS = {
    "RSSI_THRESHOLDS": [-80.0, -70.0, -60.0, -50.0],
    "TREND_THRESHOLD": 2.0,   # dBm. 이보다 크게 변해야 증가/감소로 인정 (노이즈보다 커야 함)
    "REWARD_UP": 1.0,
    "REWARD_KEEP": 0.0,
    "REWARD_DOWN": -1.0,
    "MOVE_COST": 0.1,
}


def _cfg(name):
    """config.py 값 우선, 없거나 None이면 기본값."""
    value = getattr(config, name, None) if config is not None else None
    return _DEFAULTS[name] if value is None else value


# ---------------------------------------------------------------
# 1) RSSI 대표값
# ---------------------------------------------------------------
def get_representative_rssi(samples):
    """
    여러 번 측정한 RSSI 샘플 -> 대표값 1개.
    - 시뮬레이션: 샘플이 1개면 그대로 반환
    - 실측(Pi): 최근 3~5개 샘플의 중앙값 (튀는 값에 강함)
    """
    if isinstance(samples, (int, float)):
        return float(samples)
    return float(median(samples))


# ---------------------------------------------------------------
# 2) RSSI Level
# ---------------------------------------------------------------
def get_rssi_level(rssi):
    """
    RSSI 세기를 구간 번호로 변환. 클수록 신호가 강함.
    thresholds = [-80, -70, -60, -50]
      rssi < -80        -> 0
      -80 <= rssi < -70 -> 1
      ...
      rssi >= -50       -> 4
    """
    return bisect_right(_cfg("RSSI_THRESHOLDS"), rssi)


def num_rssi_levels():
    """Level 종류 수 (Q-table 크기 계산, 로그용)."""
    return len(_cfg("RSSI_THRESHOLDS")) + 1


# ---------------------------------------------------------------
# 3) RSSI Trend
# ---------------------------------------------------------------
def get_rssi_trend(current_rssi, previous_rssi):
    """
    이전 대비 증가 1 / 유지 0 / 감소 -1.
    previous_rssi가 None(첫 State)이면 0(유지)으로 통일.
    """
    if previous_rssi is None:
        return TREND_KEEP
    diff = current_rssi - previous_rssi
    threshold = _cfg("TREND_THRESHOLD")
    if diff > threshold:
        return TREND_UP
    if diff < -threshold:
        return TREND_DOWN
    return TREND_KEEP


# ---------------------------------------------------------------
# 4) State
# ---------------------------------------------------------------
def make_state(current_rssi, previous_rssi, previous_action):
    """State = (rssi_level, rssi_trend, previous_action). 튜플이라 dict 키로 바로 사용 가능."""
    level = get_rssi_level(current_rssi)
    trend = get_rssi_trend(current_rssi, previous_rssi)
    return (level, trend, previous_action)


# ---------------------------------------------------------------
# 5) Reward
# ---------------------------------------------------------------
def calculate_reward(current_rssi, previous_rssi):
    """
    RSSI 변화(Trend) 기준 보상 - 이동 비용.
    첫 스텝(previous_rssi가 None)은 trend=유지로 처리되어 보상 = REWARD_KEEP - MOVE_COST.
    """
    trend = get_rssi_trend(current_rssi, previous_rssi)
    base = {
        TREND_UP: _cfg("REWARD_UP"),
        TREND_KEEP: _cfg("REWARD_KEEP"),
        TREND_DOWN: _cfg("REWARD_DOWN"),
    }[trend]
    return base - _cfg("MOVE_COST")


# ---------------------------------------------------------------
# 보조 도구: 2번 환경의 RSSI 분포를 보고 Level 임계값 정하기
# ---------------------------------------------------------------
def suggest_thresholds(rssi_samples, n_levels=5):
    """
    환경에서 랜덤 위치/랜덤 이동으로 모은 RSSI 샘플의 분위수로 임계값 후보를 제안.
    예: n_levels=5 -> 20/40/60/80% 지점 4개 경계.
    """
    data = sorted(rssi_samples)
    cuts = []
    for i in range(1, n_levels):
        idx = int(len(data) * i / n_levels)
        cuts.append(round(data[min(idx, len(data) - 1)], 1))
    return cuts
