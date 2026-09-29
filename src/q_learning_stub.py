"""q_learning.py가 완성되기 전 통합 테스트용 임시 Q-Learning Agent."""

import random
from collections import defaultdict


class QLearningAgent:
    """experiment.py 통합 테스트를 위한 임시 Q-Learning Agent."""

    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.9,
    ):
        self.actions = tuple(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # State별 각 Action의 Q-value 저장
        self.q_table = defaultdict(
            lambda: [0.0 for _ in self.actions]
        )

    def choose_action(self, state):
        """epsilon-greedy 방식으로 Action을 선택한다."""

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Exploitation
        q_values = self.q_table[state]
        max_q = max(q_values)

        best_actions = [
            action
            for action, q_value in zip(self.actions, q_values)
            if q_value == max_q
        ]

        return random.choice(best_actions)

    def update_q(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """Q-Learning 업데이트를 수행한다."""

        action_index = self.actions.index(action)

        current_q = self.q_table[state][action_index]

        if done:
            next_max_q = 0.0
        else:
            next_max_q = max(self.q_table[next_state])

        target_q = reward + self.gamma * next_max_q

        new_q = current_q + self.alpha * (
            target_q - current_q
        )

        self.q_table[state][action_index] = new_q