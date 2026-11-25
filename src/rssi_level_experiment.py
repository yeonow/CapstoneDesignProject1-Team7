"""세 RSSI Level 설정을 같은 Seed/Target 조건으로 비교하는 실행기.

종합설계 최종 발표를 위한 추가 개선 실험이며, 이미 제출한 논문 결과를 바꾸지 않는다.

python -m src.rssi_level_experiment --output results/<새 폴더> [--settings ...] [--runs N]

- baseline_original:   Baseline RSSI 모델 + 기존 Level [-80, -70, -60, -50]
- calibrated_original: 실측 보정 모델 + 기존 Level
- calibrated_adjusted: 실측 보정 모델 + 조정 Level (src/rssi_levels.py)
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
from statistics import fmean

import config
from src import rssi_levels, state_reward
from src.calibration_data import ROOT
from src.calibration_profile import load_profile
from src.calibration_smoke import paired_targets, temporary_settings
from src.compare_algorithms import compare_algorithms


SETTINGS = {
    "baseline_original": {"RSSI_MODEL": "baseline", "RSSI_LEVEL_MODE": "original"},
    "calibrated_original": {"RSSI_MODEL": "calibrated", "RSSI_LEVEL_MODE": "original"},
    "calibrated_adjusted": {"RSSI_MODEL": "calibrated", "RSSI_LEVEL_MODE": "calibrated_adjusted"},
}
DEFAULT_PROFILE = Path("data/rssi_calibration_20261125/calibration_profile.json")
DEFAULT_TRAIN_EPISODES = 3000
DEFAULT_EVAL_EPISODES = 100
DEFAULT_RUNS = 100
DEFAULT_SEED = 42
RECORDED_CONFIG = (
    "GRID_SIZE", "AGENT_START", "MAX_STEPS", "TARGET_MODE", "TARGET_MIN_DISTANCE",
    "TARGET_DISTANCE_METRIC", "NOISE_STD", "NOISE_FLOOR_DBM", "CHANNEL_STATE", "CELL_SIZE_M",
    "MIN_DISTANCE_M", "ALPHA", "GAMMA", "EPSILON", "EPSILON_MIN", "EPSILON_DECAY",
    "REWARD_UP", "REWARD_KEEP", "REWARD_DOWN", "MOVE_COST", "TERMINAL_REWARD",
    "BLOCKED_APPLY_MOVE_COST", "BLOCKED_TREND_KEEP", "BLOCKED_PENALTY",
)


def _write(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def run_level_experiment(output_dir, *, settings=tuple(SETTINGS),
                         train_episodes=DEFAULT_TRAIN_EPISODES, eval_episodes=DEFAULT_EVAL_EPISODES,
                         runs=DEFAULT_RUNS, seed=DEFAULT_SEED, profile_path=DEFAULT_PROFILE):
    """선택한 설정마다 기존 compare_algorithms를 같은 Seed로 실행하고 manifest를 남긴다.

    출력 폴더가 이미 있으면 거부한다. 각 설정의 Level 경계와 근거는 학습 전에 기록한다.
    """
    settings = tuple(settings)
    if not settings or len(set(settings)) != len(settings) or not set(settings) <= set(SETTINGS):
        raise ValueError(f"settings must be distinct names from {tuple(SETTINGS)}: {settings}")
    profile_path = (ROOT / Path(profile_path)).resolve()
    profile = load_profile(profile_path)
    output = ROOT / Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    # 기존 보정 smoke와 같은 논문 조건(20x20, split, Chebyshev >= 5, 400 Step, Noise 2 dB)
    common = dict(GRID_SIZE=20, AGENT_START=(0, 0), TARGET_POSITION=(19, 19), MAX_STEPS=400,
                  TARGET_MODE="split", TARGET_MIN_DISTANCE=5, TARGET_DISTANCE_METRIC="chebyshev",
                  CHANNEL_STATE="los", CELL_SIZE_M=1.0, MIN_DISTANCE_M=1.0,
                  NOISE_FLOOR_DBM=profile["noise_floor_dbm"], NOISE_STD=2.0,
                  RSSI_CALIBRATION_FILE=profile_path, LOG_INTERVAL=1000)
    manifest = {
        "purpose": "Final capstone presentation add-on: RSSI Level comparison. "
                   "Not a change to the submitted paper results.",
        "status": "running", "profile": str(profile_path),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "python": platform.python_version(), "train_episodes": train_episodes,
        "eval_episodes": eval_episodes, "runs": runs, "seed": seed,
        "train_seed_rule": "seed + 2*(run-1)", "eval_seed_rule": "train_seed + 1",
        "noise_seed_rule": "<eval_seed>:<episode>", "settings": {},
    }
    for name in settings:
        with temporary_settings(**common, **SETTINGS[name]):
            adjusted = config.RSSI_LEVEL_MODE == "calibrated_adjusted"
            manifest["settings"][name] = {
                **SETTINGS[name],
                "rssi_thresholds": list(state_reward.rssi_thresholds()),
                "level_mapping": rssi_levels.calibrated_level_mapping() if adjusted else None,
                "trend_threshold_db": state_reward._cfg("TREND_THRESHOLD"),
                "config": {key: getattr(config, key) for key in RECORDED_CONFIG},
            }
    _write(output / "level_manifest.json", manifest)
    results = {}
    for name in settings:
        with temporary_settings(**common, **SETTINGS[name]):
            print(f"=== {name}: RSSI_MODEL={config.RSSI_MODEL}, "
                  f"thresholds={manifest['settings'][name]['rssi_thresholds']}")
            results[name] = compare_algorithms(train_episodes=train_episodes,
                                               eval_episodes=eval_episodes, seed=seed, runs=runs,
                                               output_dir=output / name)
    pairs = [paired_targets(result) for result in results.values()]
    if any(pair != pairs[0] for pair in pairs):
        raise ValueError("RSSI Level settings do not share evaluation targets/seeds")
    manifest["paired_targets_and_seeds_verified"] = True
    manifest["mean_success_rate"] = {
        name: {algorithm: fmean(row["success_rate"] for row in result["summary"]
                                if row["algorithm"] == algorithm)
               for algorithm in ("q_learning", "random")}
        for name, result in results.items()
    }
    manifest["status"] = "completed"
    _write(output / "level_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="new folder (must not exist)")
    parser.add_argument("--settings", nargs="+", choices=tuple(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    for name in ("train_episodes", "eval_episodes", "runs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    manifest = run_level_experiment(
        args.output, settings=args.settings, train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes, runs=args.runs, seed=args.seed, profile_path=args.profile)
    print(json.dumps(manifest["mean_success_rate"], indent=2))


if __name__ == "__main__":
    main()
