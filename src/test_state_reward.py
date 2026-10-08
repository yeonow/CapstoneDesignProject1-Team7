"""

state_reward.py 단독 검증. `python test_state_reward.py` 로 실행.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import state_reward as sr


def test_level_boundaries():
    assert sr.get_rssi_level(-90) == 0
    assert sr.get_rssi_level(-80) == 1     
    assert sr.get_rssi_level(-65) == 2
    assert sr.get_rssi_level(-50) == 4
    assert sr.get_rssi_level(-30) == 4


def test_level_monotonic():
    """RSSI가 커질수록 Level이 줄어들면 안 된다."""
    levels = [sr.get_rssi_level(x) for x in range(-100, -20)]
    assert levels == sorted(levels)


def test_trend():
    assert sr.get_rssi_trend(-60, -65) == 1     # +5 -> 증가
    assert sr.get_rssi_trend(-70, -65) == -1    # -5 -> 감소
    assert sr.get_rssi_trend(-64, -65) == 0     # +1 -> 임계값 이내, 유지
    assert sr.get_rssi_trend(-65, None) == 0    # 첫 State


def test_state_format():
    state = sr.make_state(-65, None, sr.NONE_ACTION)
    assert state == (2, 0, -1)
    assert isinstance(state, tuple)
    hash(state)  

def test_reward_order():
    up = sr.calculate_reward(-58, -65)
    keep = sr.calculate_reward(-65, -65)
    down = sr.calculate_reward(-72, -65)
    assert up > keep > down


def test_explicit_reward_config_and_terminal_reward():
    reward = sr.calculate_reward(
        -58,
        -65,
        reward_up=2.0,
        reward_keep=0.0,
        reward_down=-2.0,
        move_cost=0.25,
        terminal_reward=5.0,
        success=True,
    )
    assert reward == 6.75


def test_blocked_default_matches_normal_action():
    """기본 설정에서는 막힌 Action도 일반 Action과 같은 Reward를 받는다."""
    for current in (-58, -65, -72):
        assert sr.calculate_reward(current, -65, blocked=True) == sr.calculate_reward(current, -65)


def test_blocked_options():
    kwargs = dict(reward_up=1.0, reward_keep=0.0, reward_down=-1.0, move_cost=0.1)
    # Noise로 RSSI가 올라가도 Trend를 KEEP으로 보면 REWARD_KEEP만 받는다.
    assert sr.calculate_reward(
        -58, -65, blocked=True, blocked_trend_keep=True, **kwargs
    ) == -0.1
    # MOVE_COST를 실제 이동에만 적용하면 막힌 Action에는 빼지 않는다.
    assert sr.calculate_reward(
        -65, -65, blocked=True, blocked_apply_move_cost=False, **kwargs
    ) == 0.0
    # blocked penalty는 막힌 Action에만 추가된다.
    assert sr.calculate_reward(
        -65, -65, blocked=True, blocked_penalty=0.5, **kwargs
    ) == -0.6
    assert sr.calculate_reward(
        -65, -65, blocked=False, blocked_penalty=0.5, **kwargs
    ) == -0.1


def test_representative():
    assert sr.get_representative_rssi(-60) == -60.0
    assert sr.get_representative_rssi([-60, -61, -90]) == -61.0   # 튀는 값에 강함


def test_suggest_thresholds():
    cuts = sr.suggest_thresholds(list(range(-100, -50)), n_levels=5)
    assert len(cuts) == 4
    assert cuts == sorted(cuts)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
