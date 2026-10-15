"""Target 후보 생성, 체커보드 분리, Seed 재현성, 학습/평가 Target 적용을 검증한다."""

import random

import numpy as np
import pytest

import config
from src import compare_algorithms as comparison
from src import experiment
from src import targets


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    for name, value in {
        "GRID_SIZE": 6, "AGENT_START": (0, 0), "TARGET_POSITION": (5, 5),
        "MAX_STEPS": 8, "NUM_EPISODES": 3, "NUM_EVAL_EPISODES": 3,
        "RANDOM_SEED": 0, "LOG_INTERVAL": 0,
        "TARGET_MODE": "fixed", "TARGET_MIN_DISTANCE": 3,
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_candidates_respect_min_distance_and_grid():
    candidates = targets.candidate_targets(6, (0, 0), 3)
    assert candidates
    assert all(0 <= x < 6 and 0 <= y < 6 for x, y in candidates)
    assert all(x + y >= 3 for x, y in candidates)
    assert len(candidates) == 36 - 6  # x+y<3 인 칸 6개 제외


def test_checkerboard_split_is_disjoint_complete_and_not_parity_biased():
    train = set(targets.candidate_targets(10, (0, 0), 5, split="train"))
    evaluation = set(targets.candidate_targets(10, (0, 0), 5, split="eval"))
    assert train and evaluation
    assert not train & evaluation
    assert train | evaluation == set(targets.candidate_targets(10, (0, 0), 5))
    # 2×2 블록이라 두 집합 모두 짝수·홀수 거리를 포함한다.
    for cells in (train, evaluation):
        assert {(x + y) % 2 for x, y in cells} == {0, 1}


def test_phase_candidates_follow_mode(monkeypatch):
    assert targets.target_candidates("train") is None
    monkeypatch.setattr(config, "TARGET_MODE", "random")
    assert targets.target_candidates("train") == targets.target_candidates("eval")
    monkeypatch.setattr(config, "TARGET_MODE", "split")
    assert all(targets.checkerboard_split(cell) == "train"
               for cell in targets.target_candidates("train"))
    assert all(targets.checkerboard_split(cell) == "eval"
               for cell in targets.target_candidates("eval"))


@pytest.mark.parametrize("name,value,message", [
    ("TARGET_MODE", "corner", "TARGET_MODE"),
    ("TARGET_MIN_DISTANCE", 0, "TARGET_MIN_DISTANCE"),
    ("TARGET_MIN_DISTANCE", 99, "candidates"),
])
def test_invalid_target_config(monkeypatch, name, value, message):
    monkeypatch.setattr(config, "TARGET_MODE", "random")
    monkeypatch.setattr(config, name, value)
    with pytest.raises(ValueError, match=message):
        targets.validate_target_config()


def test_target_choice_is_reproducible_and_independent_of_global_rng():
    candidates = targets.candidate_targets(10, (0, 0), 5)
    random.seed(1)
    first = [targets.evaluation_target(candidates, 43, episode) for episode in range(1, 20)]
    random.seed(2)
    second = [targets.evaluation_target(candidates, 43, episode) for episode in range(1, 20)]
    assert first == second
    assert len(set(first)) > 1
    before = random.getstate()
    first_rng, second_rng = targets.training_target_rng(42), targets.training_target_rng(42)
    assert ([first_rng.choice(candidates) for _ in range(10)]
            == [second_rng.choice(candidates) for _ in range(10)])
    # Target 선택은 환경 Noise가 쓰는 전역 random 상태를 바꾸지 않는다.
    assert random.getstate() == before


def test_reset_moves_target_only_when_given():
    env = experiment.build_environment()
    env.reset()
    assert env.target_position == (5, 5)
    env.reset((3, 1))
    assert env.target_position == (3, 1)
    assert env.agent_position == env.agent_start and env.step_count == 0
    env.reset()
    assert env.target_position == (3, 1)
    with pytest.raises(ValueError, match="inside"):
        env.reset((6, 0))


def test_state_does_not_contain_target_position():
    env = experiment.build_environment()
    rssi = env.reset((4, 2))
    state = experiment.make_state(rssi, None, config.NONE_ACTION)
    assert len(state) == 3 and (4, 2) not in state


def test_split_training_and_evaluation_use_their_own_cells(monkeypatch):
    monkeypatch.setattr(config, "TARGET_MODE", "split")
    seen = {"train": [], "eval": []}
    real_run = comparison.run_episode

    def record(env, agent, training=True, **kwargs):
        seen["train" if training else "eval"].append(kwargs["target_position"])
        return real_run(env, agent, training=training, **kwargs)

    monkeypatch.setattr(comparison, "run_episode", record)
    agent = comparison.train_q_learning_agent(5, seed=0)
    results = comparison.evaluate_q_learning(agent, 5, seed=1)
    assert len(seen["train"]) == 5 and len(seen["eval"]) == 5
    assert all(targets.checkerboard_split(cell) == "train" for cell in seen["train"])
    assert all(targets.checkerboard_split(cell) == "eval" for cell in seen["eval"])
    assert [(row["target_x"], row["target_y"]) for row in results] == seen["eval"]
