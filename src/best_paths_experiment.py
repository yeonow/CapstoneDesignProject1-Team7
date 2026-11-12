"""Reproduce the paper's 20x20, 8-direction Random Target (internal split) paths.

Run from the repository root: python -m src.best_paths_experiment
"""

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean

import config
from src import targets
from src.compare_algorithms import calculate_metrics, compare_algorithms
from src.plot_paths import plot_best_paths


DEFAULT_OUTPUT = Path("results/best_paths_8dir/random_target/grid_20x20")


def validate_best_path(record):
    """Reject inconsistent trajectories instead of repairing or synthesizing them."""
    if record is None:
        return
    path = record["path"]
    if not record["success"] or tuple(path[-1]) != tuple(record["target_position"]):
        raise ValueError("Best path must be a successful actual episode")
    if len(path) != record["steps"] + 1 or tuple(path[0]) != (0, 0):
        raise ValueError("Path length/start does not match evaluation")
    if record["grid_size"] != 20 or record["target_mode"] != "split":
        raise ValueError("Best path must use paper grid and target mode")
    if not all(0 <= v < 20 for point in path for v in point):
        raise ValueError("Path leaves grid")
    for before, after in zip(path, path[1:]):
        if max(abs(a - b) for a, b in zip(before, after)) > 1:
            raise ValueError("Path contains a non-eight-direction movement")
        if before == after and not any(v in (0, 19) for v in before):
            raise ValueError("Stationary step must be a boundary collision")
    shortest = targets.shortest_path_steps(path[0], record["target_position"])
    blocked = sum(a == b for a, b in zip(path, path[1:]))
    if (shortest != record["shortest_path_steps"] or shortest <= 0
            or record["steps"] / shortest != record["path_ratio"]
            or blocked != record["blocked_move_count"]):
        raise ValueError("Path metrics do not match observed coordinates")


def save_best_path_results(result, directory, *, train_episodes, eval_episodes, runs, seed):
    """Save only best selections; retain the existing raw comparison CSV schemas."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {
        "grid_size": 20, "target_mode": "split", "target_label": "Random Target",
        "target_distance_metric": "manhattan", "target_min_distance": 5,
        "target_candidate_count": 385, "train_target_count": 193, "eval_target_count": 192,
        "shortest_distance_metric": "chebyshev", "max_steps": 400,
        "noise_std": 2.0, "noise_distribution": "Gaussian", "seed": seed,
        "runs": runs, "train_episodes": train_episodes, "eval_episodes": eval_episodes,
        "actions": list(config.ACTIONS),
        "train_seed_rule": "seed + 2*(run-1)", "eval_seed_rule": "train_seed + 1",
        "noise_seed_rule": "<eval_seed>:<episode>",
        "evaluation_policy": "greedy Q-Learning; uniform random eight-action baseline",
        "model_config": {key: getattr(config, key) for key in (
            "ALPHA", "GAMMA", "EPSILON", "EPSILON_MIN", "EPSILON_DECAY",
            "REWARD_UP", "REWARD_KEEP", "REWARD_DOWN", "MOVE_COST", "TERMINAL_REWARD",
            "BLOCKED_APPLY_MOVE_COST", "BLOCKED_TREND_KEEP", "BLOCKED_PENALTY",
            "CELL_SIZE_M", "MIN_DISTANCE_M", "CARRIER_FREQUENCY_HZ", "TX_POWER_DBM",
            "NOISE_FLOOR_DBM", "ETA_LOS", "ETA_NLOS", "PATH_LOSS_EXPONENT",
            "MULTIPATH_FADING", "CHANNEL_STATE")},
    }
    summaries = []
    for algorithm, filename in (("q_learning", "q_learning_best"), ("random", "random_search_best")):
        record = result["paths"][algorithm]
        validate_best_path(record)
        payload = {
            **settings, "algorithm": algorithm, "selection": "best",
            "status": "selected" if record is not None else "no_successful_episode",
            "selection_rule": "success only; minimum (path_ratio, steps, run, episode)",
            "trajectory": record,
        }
        with (directory / f"{filename}.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        episodes = [row for row in result["episodes"] if row["algorithm"] == algorithm]
        ratios = [row["steps"] / row["shortest_path_steps"] for row in episodes if row["success"]]
        summaries.append({
            "algorithm": algorithm, "target_mode": "split", "target_label": "Random Target",
            "grid_size": 20, "runs": runs, "train_episodes": train_episodes if algorithm == "q_learning" else 0,
            "eval_episodes_per_run": eval_episodes, "total_eval_episodes": len(episodes),
            "seed": seed, "max_steps": 400, "noise_std": 2.0,
            "target_distance_metric": "manhattan", "train_target_count": 193, "eval_target_count": 192,
            **calculate_metrics(episodes), "path_ratio_on_success": fmean(ratios) if ratios else None,
            **{f"best_{key}": record[key] if record is not None else None for key in (
                "run", "episode", "train_seed", "eval_seed", "start_position", "target_position",
                "steps", "shortest_path_steps", "path_ratio", "blocked_move_count")},
        })
    with (directory / "experiment_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot_best_paths(result["paths"], directory)
    return summaries


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
        result["experiment_summary"] = save_best_path_results(
            result, directory, train_episodes=train_episodes, eval_episodes=eval_episodes,
            runs=runs, seed=config.RANDOM_SEED if seed is None else seed,
        )
        for algorithm, record in result["paths"].items():
            print(f"{algorithm} best: " + ("No successful episode" if record is None else
                  f"run={record['run']} episode={record['episode']} steps={record['steps']} "
                  f"shortest={record['shortest_path_steps']} ratio={record['path_ratio']}"))
        print(f"Best-path artifacts saved to: {directory}")
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
