"""실측 보정 smoke의 최종 논문 Target 조건(20x20, (0,0), split, Chebyshev >= 5)을 검증한다.

src/targets.py의 후보/2x2 체커보드 분할/RNG 함수는 수정하지 않고 그대로 호출한다.
"""
import random

import numpy as np
import pytest

import config
from src import compare_algorithms as comparison
from src import targets
from src.calibration_data import DATA


@pytest.fixture
def paper_targets(monkeypatch):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Target 선택은 MAX_STEPS와 무관하므로 테스트 시간만 줄인다.
    for name, value in {"GRID_SIZE": 20, "AGENT_START": (0, 0), "TARGET_POSITION": (19, 19),
                        "TARGET_MODE": "split", "TARGET_MIN_DISTANCE": 5,
                        "TARGET_DISTANCE_METRIC": "chebyshev", "MAX_STEPS": 8,
                        "LOG_INTERVAL": 0, "RSSI_LEVEL_MODE": "original",
                        "RSSI_CALIBRATION_FILE": DATA / "calibration_profile.json"}.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_chebyshev_split_counts_disjoint_and_min_distance(paper_targets):
    train = targets.target_candidates("train")
    evaluation = targets.target_candidates("eval")
    assert (len(train), len(evaluation)) == (187, 188)
    assert len(set(train)) == len(train) and len(set(evaluation)) == len(evaluation)
    assert not set(train) & set(evaluation)
    everything = set(train) | set(evaluation)
    assert len(everything) == 375
    assert everything == set(targets.candidate_targets(20, (0, 0), 5))
    assert all(targets.shortest_path_steps((0, 0), cell) >= 5 for cell in everything)
    # 시작점 주변 Chebyshev < 5인 5x5 블록만 빠진다. Manhattan이면 포함되던 (2, 3)도 제외된다.
    excluded = {(x, y) for x in range(20) for y in range(20)} - everything
    assert excluded == {(x, y) for x in range(5) for y in range(5)}
    assert (2, 3) in excluded
    assert all(targets.checkerboard_split(cell) == "train" for cell in train)
    assert all(targets.checkerboard_split(cell) == "eval" for cell in evaluation)


def test_same_seed_reproduces_target_selection(paper_targets):
    train = targets.target_candidates("train")
    evaluation = targets.target_candidates("eval")

    def draw(train_seed):
        rng = targets.training_target_rng(train_seed)
        return ([rng.choice(train) for _ in range(30)],
                [targets.evaluation_target(evaluation, train_seed + 1, e) for e in range(1, 31)])

    first = draw(42)
    assert draw(42) == first
    assert draw(44) != first
    assert set(first[0]) <= set(train) and set(first[1]) <= set(evaluation)


def test_algorithms_and_rssi_models_share_reproducible_targets(paper_targets, monkeypatch, tmp_path):
    train_seen = []
    real_run = comparison.run_episode

    def record(env, agent, training=True, **kwargs):
        if training:
            train_seen.append(kwargs["target_position"])
        return real_run(env, agent, training=training, **kwargs)

    monkeypatch.setattr(comparison, "run_episode", record)
    observed = {}
    for label, mode in (("first", "baseline"), ("repeat", "baseline"), ("calibrated", "calibrated")):
        monkeypatch.setattr(config, "RSSI_MODEL", mode)
        train_seen.clear()
        result = comparison.compare_algorithms(10, 10, seed=42, runs=1, output_dir=tmp_path / label)
        evaluation = {}
        for row in result["episodes"]:
            evaluation.setdefault(row["algorithm"], []).append(
                (row["episode"], row["eval_seed"], row["target_x"], row["target_y"]))
        observed[label] = list(train_seen), evaluation
    train_first, eval_first = observed["first"]
    assert len(train_first) == 10 and len(eval_first["q_learning"]) == 10
    assert observed["repeat"] == observed["first"]
    assert observed["calibrated"] == observed["first"]
    assert eval_first["q_learning"] == eval_first["random"]
    assert set(train_first) <= set(targets.target_candidates("train"))
    assert {(x, y) for _, _, x, y in eval_first["random"]} <= set(targets.target_candidates("eval"))
