"""Independent random-action baseline for the RSSI grid environment."""

import random
from typing import Optional

from config import DOWN, LEFT, RIGHT, UP


class RandomAgent:
    """State(상태 정보)나 학습 과정 없이, 설정된 4개의 Action(행동) 중 하나를 선택한다."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """독립적인 난수 생성기(RNG)를 생성하고, 고정된 Seed를 사용하면 동일한 Action 선택 순서가 재현되도록 한다."""
        self.random = random.Random(seed)

    def choose_action(self) -> int:
        """각 방향을 1/4의 확률로 선택한다. 경계선에서도 마찬가지이다.

        이동, RSSI 샘플링, 그리고 종료는 여전히 환경의 책임이다.
        상태, RSSI, 보상, 또는 학습 플래그는 필요하지 않다.
        """
        return self.random.choice((UP, DOWN, LEFT, RIGHT))
