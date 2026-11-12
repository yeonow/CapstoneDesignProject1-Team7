"""Observe actual positions without moving, measuring RSSI, or consuming RNG."""

import json
from pathlib import Path
from statistics import fmean

from src.targets import shortest_path_steps


class EpisodePath:
    def __init__(self, env):
        self.coordinates = [tuple(env.agent_position)]

    def append(self, env):
        self.coordinates.append(tuple(env.agent_position))

    def finish(self, env, steps):
        if len(self.coordinates) != steps + 1:
            raise ValueError("Path length must equal steps + 1")
        shortest = shortest_path_steps(self.coordinates[0], env.target_position)
        return {
            "start_position": self.coordinates[0],
            "target_position": tuple(env.target_position),
            "grid_size": env.grid_size,
            "path": self.coordinates,
            "shortest_path_steps": shortest,
            # A start already at the target has no defined steps/distance ratio.
            "path_ratio": steps / shortest if shortest else None,
            "noise_std": env.noise_std,
            "blocked_move_count": sum(a == b for a, b in zip(
                self.coordinates, self.coordinates[1:])),
        }


def select_paths(records):
    """Select real successes; break ties by steps, run, then episode."""
    successful = [row for row in records if row["success"]]
    if not successful:
        return {"best": None, "representative": None}
    ranked = [row for row in successful if row["path_ratio"] is not None]
    tie = lambda row: (row["steps"], row["run"], row["episode"])
    if not ranked:
        # All successes started on the target: ratio is undefined, steps are zero.
        selected = min(successful, key=tie)
        return {"best": selected, "representative": selected}
    mean_ratio = fmean(row["path_ratio"] for row in ranked)
    return {
        "best": min(ranked, key=lambda row: (row["path_ratio"], *tie(row))),
        "representative": min(ranked, key=lambda row: (
            abs(row["path_ratio"] - mean_ratio), *tie(row))),
    }


def select_best_paths(records):
    """Independently select each algorithm's best actual successful evaluation."""
    records = list(records)
    return {
        algorithm: select_paths(row for row in records if row["algorithm"] == algorithm)["best"]
        for algorithm in ("q_learning", "random")
    }


def save_selected_paths(records, output_dir, *, target_mode, grid_size):
    """Write only four selections, including explicit absence records."""
    directory = Path(output_dir) / "paths"
    directory.mkdir(parents=True, exist_ok=True)
    selections = {}
    for algorithm in ("q_learning", "random"):
        selections[algorithm] = select_paths([
            row for row in records if row["algorithm"] == algorithm])
        for kind, record in selections[algorithm].items():
            payload = {
                "target_mode": target_mode, "grid_size": grid_size,
                "algorithm": algorithm, "selection": kind,
                "status": "selected" if record is not None else "no_successful_episode",
                "trajectory": record,
            }
            with (directory / f"{algorithm}_{kind}.json").open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, allow_nan=False)
                file.write("\n")
        for kind, record in selections[algorithm].items():
            details = ("No successful episode" if record is None else
                       f"run={record['run']} episode={record['episode']} "
                       f"steps={record['steps']} ratio={record['path_ratio']}")
            print(f"{algorithm} {kind} path | {details}")
    print(f"Selected path JSONs saved to: {directory}")
    return selections
