"""실측 보정의 통합 동작 확인용 소규모 실행. 연구 성능 평가/전체 실험이 아니다."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform

import config
from src.calibration_data import ROOT
from src.calibration_profile import load_profile
from src.compare_algorithms import compare_algorithms, _noise_seed
from src.env_factory import build_environment


@contextmanager
def temporary_settings(**settings):
    before = {name: getattr(config, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(config, name, value)


def paired_targets(result):
    groups = {}
    for row in result["episodes"]:
        key = row["run"], row["episode"]
        identity = (row["train_seed"], row["eval_seed"], row["target_x"], row["target_y"])
        groups.setdefault(key, {})[row["algorithm"]] = identity
    if any(set(g) != {"q_learning", "random"} or g["q_learning"] != g["random"] for g in groups.values()):
        raise ValueError("Paired algorithm target/seed mismatch")
    return {f"{run}:{episode}": g["q_learning"] for (run, episode), g in groups.items()}


def run_smoke(profile_path, output_dir, train_episodes=10, eval_episodes=10):
    """동일 1 Run(42/43), 양 모드 각각 새 학습. 출력 경로가 존재하면 거부한다."""
    for count in (train_episodes, eval_episodes):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError("Smoke episodes must be integers in 1..20; not a full research run")
    profile_path = (ROOT / Path(profile_path)).resolve()
    profile = load_profile(profile_path)
    output = ROOT / Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": "Smoke test only; not final research performance",
                "profile": profile, "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                "python": platform.python_version(), "train_episodes": train_episodes,
                "eval_episodes": eval_episodes, "runs": 1, "train_seed": 42, "eval_seed": 43,
                "noise_seed_rule": "<eval_seed>:<episode>", "settings": {}, "probes": {}}
    results = {}
    # 논문용 8방향 Split 선택 함수를 그대로 호출하며 파일/함수/Seed 규칙을 바꾸지 않는다.
    # Target 후보 필터만 최종 논문 기준인 Chebyshev이며, RSSI 감쇠는 환경의 Euclidean 거리를 쓴다.
    common = dict(GRID_SIZE=20, AGENT_START=(0, 0), TARGET_POSITION=(19, 19), MAX_STEPS=400,
                  TARGET_MODE="split", TARGET_MIN_DISTANCE=5, TARGET_DISTANCE_METRIC="chebyshev",
                  CHANNEL_STATE="los", CELL_SIZE_M=1.0, MIN_DISTANCE_M=1.0,
                  NOISE_FLOOR_DBM=profile["noise_floor_dbm"], NOISE_STD=2.0,
                  RSSI_CALIBRATION_FILE=profile_path)
    for mode in ("baseline", "calibrated"):
        with temporary_settings(**common, RSSI_MODEL=mode):
            manifest["settings"][mode] = {name: str(value) if isinstance(value, Path) else value
                                           for name, value in vars(config).items() if name.isupper()}
            probes = []
            for _ in range(2):
                with _noise_seed("calibration-probe:42"):
                    env = build_environment()
                    env.reset(target_position=(5, 0))
                    probes.append(env.get_info())
            if probes[0] != probes[1]:
                raise ValueError("Same model/distance/seed is not reproducible")
            manifest["probes"][mode] = probes[0]
            results[mode] = compare_algorithms(train_episodes=train_episodes,
                                               eval_episodes=eval_episodes, seed=42, runs=1,
                                               output_dir=output / mode)
    if paired_targets(results["baseline"]) != paired_targets(results["calibrated"]):
        raise ValueError("Baseline and calibrated evaluation target/seed mismatch")
    manifest["paired_targets_and_seeds_verified"] = True
    manifest["summary"] = {mode: result["summary"] for mode, result in results.items()}
    (output / "smoke_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                                          allow_nan=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()
    run_smoke(args.profile, args.output, args.train_episodes, args.eval_episodes)
