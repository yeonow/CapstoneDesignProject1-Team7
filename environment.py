import math


class RSSIGridEnv:
    def __init__(
        self,
        grid_size: int,
        max_steps: int,
        agent_start: tuple[int, int],
        target_position: tuple[int, int],
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_start = agent_start
        self.agent_position = self.agent_start
        self.target_position = target_position
        self.step_count = 0

    def reset(self) -> None:
        """Restore the agent's starting position and clear the step count."""
        self.agent_position = self.agent_start
        self.step_count = 0

    def move(self, action: int) -> None:
        """Apply one action, counting blocked moves as steps."""
        x, y = self.agent_position

        if action == 0:  # UP
            y -= 1
        elif action == 1:  # DOWN
            y += 1
        elif action == 2:  # LEFT
            x -= 1
        elif action == 3:  # RIGHT
            x += 1
        else:
            raise ValueError("action must be 0 (UP), 1 (DOWN), 2 (LEFT), or 3 (RIGHT)")

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.agent_position = (x, y)

        self.step_count += 1

    def _calculate_distance(self) -> float:
        """Return the Euclidean distance between the agent and target."""
        agent_x, agent_y = self.agent_position
        target_x, target_y = self.target_position
        return math.hypot(target_x - agent_x, target_y - agent_y)

    def is_done(self) -> bool:
        """Check whether the target is reached or the step limit is reached."""
        return (
            self.agent_position == self.target_position
            or self.step_count >= self.max_steps
        )
