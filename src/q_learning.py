# q_learning.py

import random
import pickle
from pathlib import Path
from typing import Hashable, Optional

import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning Agent

    State 예시:
        (rssi_level, rssi_trend, previous_action)

    Action:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
    """

    def __init__(
        self,
        num_actions: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        num_actions : int
            가능한 Action 개수.
            현재 프로젝트에서는 UP/DOWN/LEFT/RIGHT 4개.

        alpha : float
            Learning Rate.
            새로운 경험을 Q-value에 얼마나 반영할지 결정.

        gamma : float
            Discount Factor.
            미래 Reward를 얼마나 중요하게 고려할지 결정.

        epsilon : float
            초기 Exploration 확률.

        epsilon_min : float
            epsilon이 감소할 수 있는 최소값.

        epsilon_decay : float
            Episode 종료 후 epsilon 감소 비율.

        seed : int | None
            실험 재현성을 위한 Random Seed.
        """

        if num_actions <= 0:
            raise ValueError("num_actions must be greater than 0.")

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the range (0, 1].")

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the range [0, 1].")

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in the range [0, 1].")

        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError("epsilon_min must be in the range [0, 1].")

        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in the range (0, 1].")

        if epsilon_min > epsilon:
            raise ValueError(
                "epsilon_min cannot be greater than initial epsilon."
            )

        self.num_actions = num_actions

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # {state: np.array([Q_up, Q_down, Q_left, Q_right])}
        self.q_table = {}

        self.random = random.Random(seed)

        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------
    # Q-Table
    # ------------------------------------------------------------

    def _ensure_state(self, state: Hashable) -> None:
        """
        Q-table에 State가 존재하지 않는 경우
        모든 Action의 Q-value를 0으로 초기화한다.
        """

        try:
            hash(state)
        except TypeError as exc:
            raise TypeError(
                "state must be hashable. "
                "Use a tuple such as "
                "(rssi_level, rssi_trend, previous_action)."
            ) from exc

        if state not in self.q_table:
            self.q_table[state] = np.zeros(
                self.num_actions,
                dtype=np.float64,
            )

    def get_q_values(self, state: Hashable) -> np.ndarray:
        """
        특정 State의 모든 Action에 대한 Q-value 반환.

        반환값은 복사본이므로 외부에서 수정해도
        실제 Q-table은 변경되지 않는다.
        """

        self._ensure_state(state)

        return self.q_table[state].copy()

    def get_q_value(self, state: Hashable, action: int) -> float:
        """
        특정 State-Action의 Q-value 반환.
        """

        self._validate_action(action)
        self._ensure_state(state)

        return float(self.q_table[state][action])

    # ------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------

    def choose_action(
        self,
        state: Hashable,
        training: bool = True,
    ) -> int:
        """
        epsilon-greedy 방식으로 Action을 선택한다.

        training=True:
            epsilon 확률로 Random Action 선택
            1-epsilon 확률로 가장 높은 Q-value의 Action 선택

        training=False:
            Exploration 없이 가장 높은 Q-value만 선택
        """

        self._ensure_state(state)

        if training and self.random.random() < self.epsilon:
            return self.random.randrange(self.num_actions)

        return self._greedy_action(state)

    def _greedy_action(self, state: Hashable) -> int:
        """
        가장 높은 Q-value를 가진 Action을 선택한다.

        동일한 최대 Q-value가 여러 개이면
        한쪽 Action에 편향되지 않도록 Random하게 선택한다.
        """

        self._ensure_state(state)

        q_values = self.q_table[state]

        max_q = np.max(q_values)

        best_actions = np.flatnonzero(
            np.isclose(q_values, max_q)
        )

        return int(self.random.choice(best_actions.tolist()))

    # ------------------------------------------------------------
    # Q-Learning Update
    # ------------------------------------------------------------

    def update_q(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
    ) -> float:
        """
        Q-Learning Update 수행.

        Q(s, a) =
            Q(s, a)
            + alpha *
              (
                reward
                + gamma * max_a' Q(s', a')
                - Q(s, a)
              )

        Episode이 종료된 경우(done=True),
        다음 State의 미래 Reward는 사용하지 않는다.

        Returns
        -------
        float
            업데이트된 Q-value
        """

        self._validate_action(action)

        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self.q_table[state][action]

        if done:
            target = float(reward)
        else:
            max_next_q = float(
                np.max(self.q_table[next_state])
            )

            target = (
                float(reward)
                + self.gamma * max_next_q
            )

        updated_q = (
            current_q
            + self.alpha * (target - current_q)
        )

        self.q_table[state][action] = updated_q

        return float(updated_q)

    # ------------------------------------------------------------
    # Epsilon
    # ------------------------------------------------------------

    def decay_epsilon(self) -> float:
        """
        Episode 종료 후 Exploration 확률 감소.

        epsilon =
            max(
                epsilon_min,
                epsilon * epsilon_decay
            )
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        return self.epsilon

    def reset_epsilon(self, epsilon: float) -> None:
        """
        epsilon 값을 직접 재설정한다.
        """

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be in the range [0, 1]."
            )

        if epsilon < self.epsilon_min:
            raise ValueError(
                "epsilon cannot be smaller than epsilon_min."
            )

        self.epsilon = epsilon

    # ------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------

    def save_q_table(self, file_path: str) -> None:
        """
        학습한 Q-table과 Agent 설정값을 파일에 저장한다.

        예:
            agent.save_q_table("models/q_table.pkl")
        """

        path = Path(file_path)

        if path.parent != Path("."):
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        data = {
            "q_table": self.q_table,
            "num_actions": self.num_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }

        with path.open("wb") as file:
            pickle.dump(data, file)

    def load_q_table(self, file_path: str) -> None:
        """
        저장된 Q-table과 Agent 설정값을 불러온다.

        예:
            agent.load_q_table("models/q_table.pkl")
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Q-table file not found: {file_path}"
            )

        with path.open("rb") as file:
            data = pickle.load(file)

        required_keys = {
            "q_table",
            "num_actions",
            "alpha",
            "gamma",
            "epsilon",
            "epsilon_min",
            "epsilon_decay",
        }

        missing_keys = required_keys - data.keys()

        if missing_keys:
            raise ValueError(
                f"Invalid Q-table file. "
                f"Missing keys: {missing_keys}"
            )

        self.q_table = data["q_table"]

        self.num_actions = data["num_actions"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]

        self.epsilon = data["epsilon"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    def get_best_action(self, state: Hashable) -> int:
        """
        Exploration 없이 현재 Q-table 기준으로
        가장 좋은 Action을 반환한다.
        """

        return self._greedy_action(state)

    def get_q_table_size(self) -> int:
        """
        현재까지 Q-table에 등록된 State 개수 반환.
        """

        return len(self.q_table)

    def clear_q_table(self) -> None:
        """
        Q-table 초기화.
        """

        self.q_table.clear()

    def _validate_action(self, action: int) -> None:
        """
        Action 값이 유효한지 확인한다.
        """

        if not isinstance(action, (int, np.integer)):
            raise TypeError(
                "action must be an integer."
            )

        if not 0 <= int(action) < self.num_actions:
            raise ValueError(
                f"action must be between "
                f"0 and {self.num_actions - 1}. "
                f"Received: {action}"
            )

    def __len__(self) -> int:
        """
        len(agent) 호출 시
        현재 Q-table의 State 개수 반환.
        """

        return len(self.q_table)