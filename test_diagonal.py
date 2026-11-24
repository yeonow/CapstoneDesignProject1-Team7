from src.experiment import build_environment
from src.random_experiment import run_random_episode

class DiagonalAgent:
    def choose_action(self):
        return 7  # 오른쪽 아래 대각선

env = build_environment()

result = run_random_episode(
    env,
    DiagonalAgent(),
    target_position=(2, 2),
    record_path=True
)

path = result["trajectory"]

print("Path:", path["path"])
print("Shortest:", path["shortest_path_steps"])
print("Ratio:", path["path_ratio"])

assert path["path"] == [(0, 0), (1, 1), (2, 2)]
assert path["shortest_path_steps"] == 2
assert path["path_ratio"] == 1.0