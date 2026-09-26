import math
import random


class RSSIGridEnv:
    """Grid simulation with provisional RSSI parameters for later calibration.

    Distances use grid units; reference_distance must be positive and
    noise_std must be nonnegative.
    """

    def __init__(
        self,
        grid_size: int,
        max_steps: int,
        agent_start: tuple[int, int],
        target_position: tuple[int, int],
        reference_rssi: float,
        reference_distance: float,
        path_loss_exponent: float,
        noise_std: float,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_start = agent_start
        self.agent_position = self.agent_start
        self.target_position = target_position
        self.step_count = 0
        self.reference_rssi = reference_rssi
        self.reference_distance = reference_distance
        self.path_loss_exponent = path_loss_exponent
        self.noise_std = noise_std
        self.current_rssi: float | None = None

    def reset(self) -> float:
        """Start a new episode and return one RSSI sample at the start."""
        self.agent_position = self.agent_start
        self.step_count = 0
        self.current_rssi = self._generate_rssi()
        return self.current_rssi

    def move(self, action: int) -> None:
        """Apply one action and sample RSSI, counting blocked moves as steps."""
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
        self.current_rssi = self._generate_rssi()

    def _calculate_distance(self) -> float:
        """Return the Euclidean distance between the agent and target."""
        agent_x, agent_y = self.agent_position
        target_x, target_y = self.target_position
        return math.hypot(target_x - agent_x, target_y - agent_y)

    def _generate_rssi(self) -> float:
        """Generate one raw RSSI sample using a provisional simulation model."""
        distance = self._calculate_distance()
        effective_distance = max(distance, self.reference_distance)
        base_rssi = self.reference_rssi - 10 * self.path_loss_exponent * math.log10(
            effective_distance / self.reference_distance
        )
        # Provisional Gaussian noise; replace or calibrate using measurements.
        noise = random.gauss(0.0, self.noise_std)
        return base_rssi + noise

    def get_rssi(self) -> float | None:
        """Return the stored sample without generating new noise."""
        return self.current_rssi

    def is_done(self) -> bool:
        """Check whether the target is reached or the step limit is reached."""
        return (
            self.agent_position == self.target_position
            or self.step_count >= self.max_steps
        )
