"""Reproduce the paper's 20x20, 8-direction Random Target (internal split) paths.

Run from the repository root: python -m src.best_paths_experiment
"""

import argparse
from contextlib import contextmanager
from pathlib import Path

import config
from src import targets
from src.compare_algorithms import compare_algorithms


DEFAULT_OUTPUT = Path("results/best_paths_8dir/random_target/grid_20x20")


@contextmanager
def paper_config():
    """Apply paper conditions locally and restore config even after a failure."""
    settings = {
        "GRID_SIZE": 20, "AGENT_START": (0, 0), "TARGET_POSITION": (19, 19),
        "MAX_STEPS": 400, "NOISE_STD": 2.0, "TARGET_MODE": "split",
        "TARGET_MIN_DISTANCE": 5, "TARGET_DISTANCE_METRIC": "manhattan",
    }
    previous = {name: getattr(config, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(config, name, value)
        targets.validate_target_config()
        if len(config.ACTIONS) != 8:
            raise ValueError("Paper experiment requires eight actions")
        counts = tuple(len(targets.target_candidates(phase)) for phase in ("train", "eval"))
        if counts != (193, 192):
            raise ValueError(f"Unexpected train/eval candidate counts: {counts}")
        yield
    finally:
        for name, value in previous.items():
            setattr(config, name, value)


def run_best_path_experiment(*, train_episodes=1000, eval_episodes=100, runs=30,
                             seed=None, output_dir=DEFAULT_OUTPUT):
    """Retrain each run and record only actual greedy/random evaluation paths."""
    directory = Path(__file__).resolve().parents[1] / output_dir
    with paper_config():
        result = compare_algorithms(
            train_episodes=train_episodes, eval_episodes=eval_episodes, runs=runs,
            seed=seed, output_dir=directory, record_paths=True, best_paths_only=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for name in ("train_episodes", "eval_episodes", "runs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    run_best_path_experiment(**vars(args))


if __name__ == "__main__":
    main()
