"""RSSI 변화량만 보고 이동 방향을 정하는 Greedy Search Agent.

학습 없이 직전 이동의 결과(ΔRSSI)만으로 다음 방향을 결정한다.
- Episode 시작 시 8방향 중 하나를 무작위로 골라 초기 방향으로 삼는다.
- 한 칸 이동한 뒤 ΔRSSI = 현재 RSSI - 직전 RSSI를 계산한다.
- ΔRSSI가 Trend 임계값 τ보다 크면(Q-Learning State의 '증가'와 같은 기준) 현재 방향을 유지한다.
- 유지·감소이거나 Grid 경계에 막혔으면 현재 방향을 제외한 7방향 중 하나를 무작위로 고른다.

공정성: Target 좌표·거리, 이동하지 않은 주변 칸의 RSSI는 사용하지 않는다.
RSSI는 환경과 동일하게 시작 시 1회, 이동 후 1회만 관측한다.
경계에 막혔는지(blocked)는 Agent 자신의 이동 결과이므로 사용한다.

노이즈 대응은 최소한으로 둔다.
- 임계값: state_reward.get_rssi_trend의 τ(TREND_THRESHOLD, 기본 2 dB)를 그대로 쓴다.
- 평활화(이동평균 등): 적용하지 않는다.
- 반복 방문 방지: 별도 방문 기록 없이, 방향을 바꿀 때 현재 방향만 후보에서 제외한다.
"""

import random
from typing import Optional

from config import ACTIONS
from src.state_reward import TREND_UP, get_rssi_trend


class GreedySearchAgent:
    """직전 이동에서 RSSI가 충분히 증가했으면 방향을 유지하고, 아니면 다른 방향을 고른다."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """환경 Noise와 독립적인 RNG를 만든다. 같은 Seed면 같은 방향 선택 순서가 재현된다."""
        self.random = random.Random(seed)
        self.direction: Optional[int] = None

    def reset(self) -> int:
        """새 Episode의 초기 방향을 8방향 중 무작위로 고른다."""
        self.direction = self.random.choice(ACTIONS)
        return self.direction

    def choose_action(self) -> int:
        """현재 유지 중인 방향을 반환한다. reset 전이면 초기 방향을 먼저 고른다."""
        if self.direction is None:
            self.reset()
        return self.direction

    def observe(self, current_rssi: float, previous_rssi: float, blocked: bool = False) -> bool:
        """이동 결과를 반영해 다음 방향을 정하고, 방향을 유지했는지 반환한다."""
        improved = not blocked and get_rssi_trend(current_rssi, previous_rssi) == TREND_UP
        if not improved:
            self.direction = self.random.choice(
                [action for action in ACTIONS if action != self.direction]
            )
        return improved
