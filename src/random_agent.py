"""Independent random-action baseline for the RSSI grid environment."""

import random
from typing import Optional

from config import DOWN, LEFT, RIGHT, UP


class RandomAgent:
    """Select among the four configured actions without state or learning."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Create a private RNG; a fixed seed reproduces the action sequence."""
        self.random = random.Random(seed)

    def choose_action(self) -> int:
        """Choose each direction with probability 1/4, including at boundaries.

        Movement, RSSI sampling, and termination remain the environment's
        responsibility. No state, RSSI, reward, or training flag is required.
        """
        return self.random.choice((UP, DOWN, LEFT, RIGHT))
