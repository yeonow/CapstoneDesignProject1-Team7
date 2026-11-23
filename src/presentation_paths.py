"""Generate presentation paths with a fixed, held-out evaluation target."""

from contextlib import contextmanager
from pathlib import Path

import config
from src import targets


START = (0, 0)
TARGET = (17, 18)
GRID_SIZE = 20
MAX_STEPS = 400
NOISE_STD = 2.0
TARGET_MODE = "split"
TARGET_MIN_DISTANCE = 5
TARGET_DISTANCE_METRIC = "chebyshev"
DEFAULT_TRAIN_EPISODES = 3000
DEFAULT_EVAL_EPISODES = 100
DEFAULT_RUNS = 30
DEFAULT_OUTPUT = Path("results/presentation_paths/target_17_18")


@contextmanager
def presentation_config():
    """Apply experiment-local settings for the existing, sequential runners."""
    settings = {
        "GRID_SIZE": GRID_SIZE, "AGENT_START": START, "TARGET_POSITION": (19, 19),
        "MAX_STEPS": MAX_STEPS, "NOISE_STD": NOISE_STD, "TARGET_MODE": TARGET_MODE,
        "TARGET_MIN_DISTANCE": TARGET_MIN_DISTANCE,
        "TARGET_DISTANCE_METRIC": TARGET_DISTANCE_METRIC,
    }
    previous = {name: getattr(config, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(config, name, value)
        targets.validate_target_config()
        train_targets = targets.target_candidates("train")
        eval_targets = targets.target_candidates("eval")
        if TARGET in train_targets or TARGET not in eval_targets:
            raise ValueError(f"Presentation target {TARGET} must be held out for evaluation")
        if config.ACTIONS != tuple(range(8)):
            raise ValueError("Presentation paths require all eight environment actions")
        yield {"train_target_count": len(train_targets), "eval_target_count": len(eval_targets)}
    finally:
        for name, value in previous.items():
            setattr(config, name, value)
