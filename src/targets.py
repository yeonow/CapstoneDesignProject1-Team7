"""Episode마다 사용할 Target 위치 후보를 만들고 Seed로 재현 가능하게 고른다.

config.TARGET_MODE
- "fixed": 기존처럼 config.TARGET_POSITION 한 곳만 사용한다.
- "random": 학습·평가 모두 같은 후보 칸 전체에서 Episode마다 무작위로 고른다(본 실험).
- "split": 칸을 체커보드로 나눠 학습용 칸과 평가용 칸을 분리한다(일반화 보조 실험).
  평가 Target은 학습 중 한 번도 Target으로 나오지 않은 칸이다.

체커보드는 1칸 단위가 아니라 2×2 블록 단위로 나눈다. 4방향 이동에서는 1칸 체커보드의
색이 시작점으로부터의 거리 홀짝과 같아서, 평가 Target이 모두 홀수 거리에만 놓이는
편향이 생기기 때문이다.

후보 칸은 시작 위치에서 맨해튼 거리 TARGET_MIN_DISTANCE 이상인 칸이다.
Target 선택용 RNG는 환경 Noise가 쓰는 전역 random과 분리되어 있어 Noise 시퀀스를 바꾸지 않는다.
"""

import random

import config


TARGET_MODES = ("fixed", "random", "split")
PHASES = ("train", "eval")


def shortest_path_steps(start, target):
    """4방향 Grid에서 시작점→Target 최단 경로의 Step 수(맨해튼 거리)."""
    return abs(target[0] - start[0]) + abs(target[1] - start[1])


def checkerboard_split(position):
    """2×2 블록 체커보드에서 칸이 속한 집합("train" 또는 "eval")을 반환한다."""
    x, y = position
    return "train" if (x // 2 + y // 2) % 2 == 0 else "eval"


def candidate_targets(grid_size, start, min_distance, split=None):
    """시작점에서 min_distance 이상 떨어진 칸 목록. split을 주면 해당 집합의 칸만 남긴다."""
    if split not in (None, *PHASES):
        raise ValueError(f"split must be None, 'train', or 'eval': {split!r}")
    return tuple(
        (x, y)
        for y in range(grid_size) for x in range(grid_size)
        if shortest_path_steps(start, (x, y)) >= min_distance
        and (split is None or checkerboard_split((x, y)) == split)
    )


def validate_target_config():
    """현재 config의 Target 설정을 확인하고 모드를 반환한다."""
    mode = config.TARGET_MODE
    if mode not in TARGET_MODES:
        raise ValueError(f"TARGET_MODE must be one of {TARGET_MODES}: {mode!r}")
    min_distance = config.TARGET_MIN_DISTANCE
    if isinstance(min_distance, bool) or not isinstance(min_distance, int) or min_distance < 1:
        raise ValueError("TARGET_MIN_DISTANCE must be a positive integer.")
    if mode != "fixed":
        for phase in PHASES:
            if not target_candidates(phase):
                raise ValueError(
                    f"No {phase} target candidates for mode={mode}, grid={config.GRID_SIZE}, "
                    f"min_distance={min_distance}."
                )
    return mode


def target_candidates(phase):
    """학습/평가 단계에 쓸 Target 후보. fixed 모드면 None(환경의 고정 Target 사용)."""
    if phase not in PHASES:
        raise ValueError(f"phase must be 'train' or 'eval': {phase!r}")
    mode = config.TARGET_MODE
    if mode == "fixed":
        return None
    if mode not in TARGET_MODES:
        raise ValueError(f"TARGET_MODE must be one of {TARGET_MODES}: {mode!r}")
    return candidate_targets(
        config.GRID_SIZE, config.AGENT_START, config.TARGET_MIN_DISTANCE,
        split=phase if mode == "split" else None,
    )


def training_target_rng(train_seed):
    """학습 Episode 전체에 이어서 쓰는 Target 선택 RNG."""
    return random.Random(f"target:{train_seed}")


def evaluation_target(candidates, eval_seed, episode):
    """평가 Episode별 Target. 같은 Seed·Episode면 두 알고리즘이 같은 Target을 받는다."""
    return random.Random(f"target:{eval_seed}:{episode}").choice(candidates)
