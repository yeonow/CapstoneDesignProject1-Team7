"""현재 진행 방향(heading) 기준의 상대 행동을 8방향 절대 Action으로 바꿔 환경에 전달한다.

상대 행동(Q-Learning이 고르는 Action):
    0 = STRAIGHT  직진
    1 = TURN_45   45° 회전
    2 = TURN_90   90° 회전
    3 = TURN_135  135° 회전
    4 = REVERSE   되돌아가기(180°)
회전(45°·90°·135°)의 좌/우는 Wrapper의 RNG로 1/2 확률로 정한다.

환경(environment.py)은 바꾸지 않는다. RelativeActionEnv가 환경을 감싸
move(상대 행동)를 move(절대 방향)으로 바꿔 호출하고, 나머지는 원래 환경에 맡긴다.
- reset 때 초기 heading을 8방향 중 무작위로 정한다.
- 경계에 막혀도 heading은 시도한 방향으로 바뀐다(Previous Action = 시도한 Action 규칙과 같음).
- RNG는 환경 Noise가 쓰는 전역 random과 분리되어 Noise 시퀀스를 바꾸지 않는다.
"""

import random

from config import DOWN, DOWN_LEFT, DOWN_RIGHT, LEFT, RIGHT, UP, UP_LEFT, UP_RIGHT


# 시계 방향 순서. 인덱스 1칸 차이가 45°다.
CLOCKWISE = (UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT)

STRAIGHT, TURN_45, TURN_90, TURN_135, REVERSE = range(5)
RELATIVE_ACTIONS = (STRAIGHT, TURN_45, TURN_90, TURN_135, REVERSE)
# 상대 행동별 회전량(45° 단위)
_TURN_STEPS = {STRAIGHT: 0, TURN_45: 1, TURN_90: 2, TURN_135: 3, REVERSE: 4}


def to_absolute(heading, relative_action, side=1):
    """heading에서 relative_action만큼 회전한 절대 방향. side는 1(시계)·-1(반시계)."""
    if relative_action not in _TURN_STEPS:
        raise ValueError(f"relative_action must be one of {RELATIVE_ACTIONS}: {relative_action!r}")
    if side not in (1, -1):
        raise ValueError("side must be 1 or -1.")
    index = CLOCKWISE.index(heading)
    return CLOCKWISE[(index + side * _TURN_STEPS[relative_action]) % len(CLOCKWISE)]


class RelativeActionEnv:
    """상대 행동을 받아 절대 방향으로 이동시키는 환경 Wrapper."""

    def __init__(self, env, seed=None):
        self._env = env
        self.random = random.Random(seed)
        self.heading = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        rssi = self._env.reset(*args, **kwargs)
        self.heading = self.random.choice(CLOCKWISE)
        return rssi

    def move(self, relative_action):
        if self.heading is None:
            raise RuntimeError("reset() must be called before move().")
        steps = _TURN_STEPS.get(relative_action)
        if steps is None:
            raise ValueError(f"relative_action must be one of {RELATIVE_ACTIONS}: {relative_action!r}")
        # 직진·되돌아가기는 좌우 구분이 없으므로 RNG를 쓰지 않는다.
        side = self.random.choice((1, -1)) if steps in (1, 2, 3) else 1
        self.heading = to_absolute(self.heading, relative_action, side)
        return self._env.move(self.heading)
