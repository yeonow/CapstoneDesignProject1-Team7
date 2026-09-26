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
