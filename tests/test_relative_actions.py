"""상대 행동 변환, RelativeActionEnv의 heading 관리, 상대 행동 Q-Learning 학습·비교 연결을 검증한다."""

import random

import numpy as np
import pytest

import config
from src import compare_algorithms as two
from src import compare_three_algorithms as three
from src import env_factory
from src.relative_actions import (
    CLOCKWISE,
    RELATIVE_ACTIONS,
    REVERSE,
    STRAIGHT,
    TURN_45,
    TURN_90,
    TURN_135,
    RelativeActionEnv,
    to_absolute,
)


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    for name, value in {
        "GRID_SIZE": 6, "AGENT_START": (0, 0), "TARGET_POSITION": (5, 5),
        "MAX_STEPS": 30, "NOISE_STD": 2.0, "NUM_EPISODES": 4,
        "NUM_EVAL_EPISODES": 4, "RANDOM_SEED": 42, "LOG_INTERVAL": 0,
        "TARGET_MODE": "random", "TARGET_MIN_DISTANCE": 2,
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_clockwise_contains_all_eight_directions_once():
    assert sorted(CLOCKWISE) == sorted(config.ACTIONS)


@pytest.mark.parametrize("heading", CLOCKWISE)
def test_straight_and_reverse(heading):
    assert to_absolute(heading, STRAIGHT) == heading
    reverse = to_absolute(heading, REVERSE)
    assert to_absolute(reverse, REVERSE) == heading
    assert reverse != heading


def test_turn_examples_from_right():
    assert to_absolute(config.RIGHT, TURN_45, 1) == config.DOWN_RIGHT
    assert to_absolute(config.RIGHT, TURN_45, -1) == config.UP_RIGHT
    assert to_absolute(config.RIGHT, TURN_90, 1) == config.DOWN
    assert to_absolute(config.RIGHT, TURN_90, -1) == config.UP
    assert to_absolute(config.RIGHT, TURN_135, 1) == config.DOWN_LEFT
    assert to_absolute(config.RIGHT, TURN_135, -1) == config.UP_LEFT
    assert to_absolute(config.RIGHT, REVERSE) == config.LEFT


def test_invalid_relative_action_or_side_is_rejected():
    with pytest.raises(ValueError):
        to_absolute(config.UP, 5)
    with pytest.raises(ValueError):
        to_absolute(config.UP, TURN_45, 0)


class FakeEnv:
    def __init__(self):
        self.moves = []
        self.agent_start = (0, 0)

    def reset(self, target_position=None):
        return -60.0

    def move(self, action):
        self.moves.append(action)
        return False


def test_reset_sets_random_heading_and_attributes_delegate():
    env = RelativeActionEnv(FakeEnv(), seed=1)
    assert env.reset() == -60.0
    assert env.heading in CLOCKWISE
    assert env.agent_start == (0, 0)


def test_move_before_reset_and_invalid_action_raise():
    env = RelativeActionEnv(FakeEnv(), seed=1)
    with pytest.raises(RuntimeError):
        env.move(STRAIGHT)
    env.reset()
    with pytest.raises(ValueError):
        env.move(7)


def test_move_passes_absolute_direction_and_updates_heading():
    inner = FakeEnv()
    env = RelativeActionEnv(inner, seed=3)
    env.reset()
    env.heading = config.RIGHT
    env.move(STRAIGHT)
    env.move(REVERSE)
    assert inner.moves == [config.RIGHT, config.LEFT]
    assert env.heading == config.LEFT


@pytest.mark.parametrize("action", [TURN_45, TURN_90, TURN_135])
def test_turns_use_both_sides(action):
    env = RelativeActionEnv(FakeEnv(), seed=0)
    env.reset()
    seen = set()
    for _ in range(100):
        env.heading = config.RIGHT
        env.move(action)
        seen.add(env.heading)
    assert seen == {to_absolute(config.RIGHT, action, 1), to_absolute(config.RIGHT, action, -1)}


def test_wrapper_rng_does_not_touch_global_random():
    env = RelativeActionEnv(FakeEnv(), seed=0)
    state = random.getstate()
    env.reset()
    for action in RELATIVE_ACTIONS:
        env.move(action)
    assert random.getstate() == state


def test_blocked_move_still_updates_heading_on_real_environment():
    random.seed(0)
    env = RelativeActionEnv(env_factory.build_environment(), seed=0)
    env.reset()
    env.heading = config.UP          # (0, 0)에서 위로는 막힘
    assert env.move(STRAIGHT) is True
    assert env.heading == config.UP
    assert env.agent_position == (0, 0)


def test_relative_training_uses_five_actions_and_relative_previous_actions():
    agent = two.train_q_learning_agent(3, seed=42, action_mode="relative")
    assert agent.num_actions == len(RELATIVE_ACTIONS)
    assert {state[2] for state in agent.q_table} <= {config.NONE_ACTION, *RELATIVE_ACTIONS}


def test_absolute_training_is_unchanged_by_new_option():
    default = two.train_q_learning_agent(3, seed=42)
    explicit = two.train_q_learning_agent(3, seed=42, action_mode="absolute")
    assert default.q_table.keys() == explicit.q_table.keys()
    for state in default.q_table:
        np.testing.assert_array_equal(default.q_table[state], explicit.q_table[state])


def test_invalid_action_mode_is_rejected():
    with pytest.raises(ValueError):
        two.train_q_learning_agent(1, seed=42, action_mode="polar")


def test_compare_with_relative_q_learning_saves_four_algorithms(tmp_path):
    algorithms = ("q_learning", "q_learning_relative", "greedy", "random")
    result = three.compare_three_algorithms(
        grid_size=6, max_steps=30, train_episodes=3, eval_episodes=3, runs=1,
        target_mode="random", target_min_distance=2, output_dir=tmp_path,
        plot=True, algorithms=algorithms,
    )
    assert {row["algorithm"] for row in result["summary"]} == set(algorithms)
    relative = [row for row in result["runs"] if row["algorithm"] == "q_learning_relative"]
    assert relative[0]["train_episodes"] == 3
    assert set(result["paths"]["algorithms"]) == set(algorithms)
    assert (tmp_path / "three_algorithm_paths.png").exists()


@pytest.mark.parametrize("algorithms", [(), ("greedy", "greedy"), ("unknown",)])
def test_invalid_algorithm_lists_are_rejected(algorithms, tmp_path):
    with pytest.raises(ValueError):
        three.compare_three_algorithms(
            grid_size=6, max_steps=30, train_episodes=1, eval_episodes=1, runs=1,
            output_dir=tmp_path, plot=False, algorithms=algorithms,
        )
