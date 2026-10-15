"""Random 베이스라인의 행동 선택, 재현성 및 독립성을 검증한다."""

import builtins
import random
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from src.random_agent import RandomAgent


@pytest.mark.parametrize("seed", [None, 0, config.RANDOM_SEED])
def test_creation(seed):
    agent = RandomAgent(seed=seed)

    assert isinstance(agent, RandomAgent)


def test_creation_without_seed():
    assert isinstance(RandomAgent(), RandomAgent)


def test_choose_action_returns_only_configured_directions():
    agent = RandomAgent(seed=42)
    valid_actions = {config.UP, config.DOWN, config.LEFT, config.RIGHT}

    for _ in range(1000):
        action = agent.choose_action()
        assert isinstance(action, int)
        assert action in valid_actions
        assert action != config.NONE_ACTION


def test_same_seed_reproduces_action_sequence():
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=42)

    assert [agent1.choose_action() for _ in range(100)] == [
        agent2.choose_action() for _ in range(100)
    ]


def test_fixed_different_seeds_produce_different_sequences():
    # Fixed seeds make this deterministic; arbitrary seeds need not differ.
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=43)

    assert [agent1.choose_action() for _ in range(100)] != [
        agent2.choose_action() for _ in range(100)
    ]


def test_choose_action_needs_no_state_rssi_reward_or_q_table():
    agent = RandomAgent(seed=42)

    assert agent.choose_action() in {config.UP, config.DOWN, config.LEFT, config.RIGHT}
    assert not hasattr(agent, "q_table")


def test_agent_does_not_change_global_random_state():
    before = random.getstate()
    agent = RandomAgent(seed=42)
    for _ in range(100):
        agent.choose_action()

    assert random.getstate() == before


def test_action_sequence_is_independent_of_global_noise_and_other_agents():
    agent = RandomAgent(seed=42)
    reference = RandomAgent(seed=42)
    other = RandomAgent(seed=43)
    before = random.getstate()
    try:
        random.seed(999)
        for _ in range(100):
            random.random()
            random.gauss(0.0, 2.0)
            other.choose_action()
            assert agent.choose_action() == reference.choose_action()
    finally:
        random.setstate(before)


def test_import_and_selection_without_learning_or_state_reward_dependencies():
    original_import = builtins.__import__
    forbidden = {"q_learning", "q_learning_stub", "state_reward"}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if forbidden.intersection(name.split(".")) or forbidden.intersection(fromlist or ()):
            raise AssertionError(f"Unexpected dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    module_path = Path(__file__).resolve().parents[1] / "src" / "random_agent.py"
    with patch("builtins.__import__", side_effect=guarded_import):
        namespace = runpy.run_path(str(module_path))
        agent = namespace["RandomAgent"](seed=42)
        for _ in range(10):
            assert agent.choose_action() in {
                config.UP, config.DOWN, config.LEFT, config.RIGHT,
            }
