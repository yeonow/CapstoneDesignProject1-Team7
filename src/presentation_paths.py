"""Generate presentation paths with a fixed, held-out evaluation target."""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import config
from src import targets
from src.compare_algorithms import (
    _positive_count, evaluate_q_learning, evaluate_random_agent, train_q_learning_agent,
)
from src.paths import select_best_paths
from src.plot_paths import plot_presentation_paths


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

ALGORITHMS = {"q_learning": "Q-Learning", "random": "Random Search"}
STEMS = {"q_learning": "q_learning_best", "random": "random_search_best"}
SHORTEST_STEPS = targets.shortest_path_steps(START, TARGET)


def validate_trajectory(record):
    """Check the environment's observations before ranking or writing them."""
    path = record["path"]
    steps = record["steps"]
    if (record["grid_size"] != GRID_SIZE or record["noise_std"] != NOISE_STD
            or record["target_position"] != TARGET or record["start_position"] != START
            or len(path) != steps + 1 or path[0] != START or not 0 <= steps <= MAX_STEPS):
        raise ValueError("Recorded path does not match the presentation conditions")
    if record["success"] != (path[-1] == TARGET) or TARGET in path[:-1]:
        raise ValueError("Recorded endpoint does not match the episode outcome")
    if not record["success"] and steps != MAX_STEPS:
        raise ValueError("An unsuccessful episode must reach the step limit")
    if any(not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) for x, y in path):
        raise ValueError("Recorded path leaves the grid")
    blocked = 0
    for before, after in zip(path, path[1:]):
        if targets.shortest_path_steps(before, after) > 1:
            raise ValueError("Recorded move violates the eight-direction rule")
        if before == after:
            if not any(value in (0, GRID_SIZE - 1) for value in before):
                raise ValueError("Recorded stationary move is away from a boundary")
            blocked += 1
    if (record["shortest_path_steps"] != SHORTEST_STEPS
            or record["path_ratio"] != steps / SHORTEST_STEPS
            or record["blocked_move_count"] != blocked):
        raise ValueError("Recorded path metrics do not match the coordinates")


def select_presentation_paths(records):
    eligible = [row for row in records if row["success"] and row["target_position"] == TARGET]
    best = select_best_paths(eligible)
    by_episode = {}
    for row in eligible:
        key = (row["run"], row["episode"])
        by_episode.setdefault(key, {})[row["algorithm"]] = row
    pairs = [(key, pair) for key, pair in by_episode.items() if pair.keys() == ALGORITHMS.keys()]
    if not pairs:
        return best, None
    _, paired = min(pairs, key=lambda item: (
        sum(row["steps"] for row in item[1].values()), *item[0],
    ))
    return best, paired


def _record_run(run, runs, train_episodes, eval_episodes, seed):
    train_seed = seed + 2 * (run - 1)
    eval_seed = train_seed + 1
    print(f"Presentation run {run}/{runs} | train_seed={train_seed} | eval_seed={eval_seed}",
          flush=True)
    agent = train_q_learning_agent(train_episodes, seed=train_seed)
    evaluation = {
        "eval_episodes": eval_episodes, "seed": eval_seed,
        "fixed_target": TARGET, "record_paths": True,
    }
    for algorithm, rows in (
        ("q_learning", evaluate_q_learning(agent, **evaluation)),
        ("random", evaluate_random_agent(**evaluation)),
    ):
        for row in rows:
            trajectory = row.pop("trajectory")
            record = {
                **row, **trajectory, "algorithm": algorithm, "run": run,
                "train_seed": train_seed, "eval_seed": eval_seed, "target_mode": TARGET_MODE,
                "noise_seed": f"{eval_seed}:{row['episode']}",
            }
            validate_trajectory(record)
            yield record


def _write_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")


def _save_results(directory, settings, best, paired):
    directory.mkdir(parents=True, exist_ok=True)
    for algorithm, record in best.items():
        _write_json(directory / f"{STEMS[algorithm]}.json", {
            **settings, "algorithm": algorithm, "selection": "independent_best",
            "selection_rule": "minimum (path_ratio, steps, run, episode) among successes",
            "status": "selected" if record is not None else "no_successful_episode",
            "trajectory": record,
        })
    _write_json(directory / "paired_comparison.json", {
        **settings, "selection": "paired",
        "selection_rule": "minimum (combined steps, run, episode) among joint successes",
        "status": "selected" if paired is not None else "no_joint_success",
        "combined_steps": sum(row["steps"] for row in paired.values()) if paired else None,
        "trajectories": paired,
    })
    columns = {
        "Run": "run", "Episode": "episode", "Train Seed": "train_seed",
        "Eval Seed": "eval_seed", "Actual Steps": "steps", "Path Ratio": "path_ratio",
        "Blocked Moves": "blocked_move_count",
    }
    with (directory / "presentation_path_summary.csv").open(
        "w", encoding="utf-8", newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=[
            "Algorithm", "Target", "Run", "Episode", "Train Seed", "Eval Seed",
            "Success", "Actual Steps", "Shortest Steps", "Path Ratio", "Blocked Moves",
        ])
        writer.writeheader()
        for algorithm, record in best.items():
            writer.writerow({
                "Algorithm": ALGORITHMS[algorithm], "Target": str(TARGET),
                "Success": record is not None, "Shortest Steps": SHORTEST_STEPS,
                **{label: (record or {}).get(key) for label, key in columns.items()},
            })


def run_presentation_paths(*, train_episodes=DEFAULT_TRAIN_EPISODES,
                           eval_episodes=DEFAULT_EVAL_EPISODES, runs=DEFAULT_RUNS,
                           seed=None, output_dir=DEFAULT_OUTPUT):
    """Train each run, evaluate the held-out target, and save presentation selections."""
    for name, count in (("train_episodes", train_episodes), ("eval_episodes", eval_episodes),
                        ("runs", runs)):
        _positive_count(count, name)
    seed = config.RANDOM_SEED if seed is None else seed
    if not 0 <= seed <= 2**32 - 1 - 2 * (runs - 1):
        raise ValueError("All training seeds must fit in NumPy's unsigned 32-bit seed range")
    directory = (Path(__file__).resolve().parents[1] / output_dir).resolve()
    if any((directory / name).exists() for name in (
        "algorithm_comparison_episodes.csv", "algorithm_comparison_summary.csv",
        "experiment_summary.csv",
    )):
        raise ValueError("Choose an output directory dedicated to presentation paths")

    with presentation_config() as counts:
        settings = {
            **counts, "purpose": "presentation", "grid_size": GRID_SIZE,
            "start_position": START, "target_position": TARGET, "target_mode": TARGET_MODE,
            "target_min_distance": TARGET_MIN_DISTANCE,
            "target_distance_metric": TARGET_DISTANCE_METRIC,
            "actions": list(config.ACTIONS), "noise_std": NOISE_STD, "max_steps": MAX_STEPS,
            "shortest_path_steps": SHORTEST_STEPS, "seed": seed, "runs": runs,
            "train_episodes": train_episodes, "eval_episodes_per_run": eval_episodes,
        }
        records = [
            row for run in range(1, runs + 1)
            for row in _record_run(run, runs, train_episodes, eval_episodes, seed)
        ]
    best, paired = select_presentation_paths(records)
    _save_results(directory, settings, best, paired)
    plot_presentation_paths(best, paired, directory, start=START, target=TARGET)
    for algorithm, record in best.items():
        details = ("No successful evaluation episode" if record is None else
                   f"run={record['run']}, episode={record['episode']}, "
                   f"steps={record['steps']}, ratio={record['path_ratio']:.3f}")
        print(f"{ALGORITHMS[algorithm]} | {details}")
    print(f"Presentation artifacts saved to {directory}")
    return {"records": records, "best": best, "paired": paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        run_presentation_paths(**vars(args))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
