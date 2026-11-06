"""Greedy Search의 방향 유지·변경 규칙, Seed 재현성, Episode 실행기와 CSV 저장을 검증한다."""

import csv
import random

import pytest

import config
from src import greedy_experiment
from src.greedy_agent import GreedySearchAgent
from src.state_reward import _cfg


TAU = _cfg("TREND_THRESHOLD")


@pytest.fixture(autouse=True)
def small_config(monkeypatch):
    python_state = random.getstate()
    for name, value in {
        "GRID_SIZE": 5, "AGENT_START": (0, 0), "TARGET_POSITION": (4, 4),
        "MAX_STEPS": 20, "NOISE_STD": 2.0, "NUM_EVAL_EPISODES": 4,
        "RANDOM_SEED": 7, "LOG_INTERVAL": 0,
    }.items():
        monkeypatch.setattr(config, name, value)
    try:
        yield
    finally:
        random.setstate(python_state)


def test_reset_picks_one_of_eight_directions_and_choose_keeps_it():
    agent = GreedySearchAgent(seed=1)
    direction = agent.reset()
    assert direction in config.ACTIONS
    assert agent.choose_action() == agent.choose_action() == direction


def test_choose_action_before_reset_initializes_direction():
    agent = GreedySearchAgent(seed=1)
    assert agent.choose_action() in config.ACTIONS


def test_keeps_direction_only_when_rssi_increases_more_than_threshold():
    agent = GreedySearchAgent(seed=3)
    direction = agent.reset()
    assert agent.observe(-60.0 + TAU + 0.1, -60.0) is True
    assert agent.choose_action() == direction


@pytest.mark.parametrize("delta", [TAU, 0.0, -TAU, -10.0])
def test_changes_direction_when_rssi_does_not_increase_enough(delta):
    agent = GreedySearchAgent(seed=3)
    direction = agent.reset()
    assert agent.observe(-60.0 + delta, -60.0) is False
    new_direction = agent.choose_action()
    assert new_direction != direction
    assert new_direction in config.ACTIONS


def test_changes_direction_when_blocked_even_if_rssi_increases():
    agent = GreedySearchAgent(seed=3)
    direction = agent.reset()
    assert agent.observe(-50.0, -60.0, blocked=True) is False
    assert agent.choose_action() != direction


def test_new_direction_covers_all_other_seven_directions():
    agent = GreedySearchAgent(seed=0)
    seen = set()
    for _ in range(500):
        agent.direction = config.UP
        agent.observe(-60.0, -60.0)
        seen.add(agent.direction)
    assert seen == set(config.ACTIONS) - {config.UP}


def test_same_seed_reproduces_direction_sequence():
    def sequence(seed):
        agent = GreedySearchAgent(seed=seed)
        values = [agent.reset()]
        for delta in (-1.0, 5.0, 0.0, -3.0, 4.0):
            agent.observe(-60.0 + delta, -60.0)
            values.append(agent.choose_action())
        return values

    assert sequence(11) == sequence(11)


def test_agent_rng_does_not_touch_global_random():
    state = random.getstate()
    agent = GreedySearchAgent(seed=5)
    agent.reset()
    agent.observe(-60.0, -60.0)
    assert random.getstate() == state


class ScriptedEnv:
    """정해진 RSSI를 돌려주는 최소 환경. Target 정보는 Agent에 노출하지 않는다."""

    def __init__(self, rssi_values, blocked_values, max_steps):
        self.rssi_values = list(rssi_values)
        self.blocked_values = list(blocked_values)
        self.max_steps = max_steps
        self.actions = []

    def reset(self, target_position=None):
        self.step_count = 0
        self.current = self.rssi_values.pop(0)
        return self.current

    def move(self, action):
        self.actions.append(action)
        self.step_count += 1
        self.current = self.rssi_values.pop(0)
        return self.blocked_values.pop(0)

    def get_rssi(self):
        return self.current

    def is_done(self):
        return self.step_count >= self.max_steps

    def get_info(self):
        return {"success": False, "step_count": self.step_count, "distance": 3.0}


def test_episode_feeds_consecutive_rssi_and_blocked_to_agent():
    env = ScriptedEnv([-70.0, -65.0, -66.0, -60.0], [False, False, True], max_steps=3)
    agent = GreedySearchAgent(seed=2)
    calls = []
    original = agent.observe

    def spy(current, previous, blocked=False):
        calls.append((current, previous, blocked))
        return original(current, previous, blocked)

    agent.observe = spy
    result = greedy_experiment.run_greedy_episode(env, agent)
    assert calls == [(-65.0, -70.0, False), (-66.0, -65.0, False), (-60.0, -66.0, True)]
    # 첫 이동은 +5 dB 증가 → 같은 방향 유지
    assert env.actions[0] == env.actions[1]
    assert result == {"success": False, "steps": 3, "final_distance": 3.0}


def test_episode_on_real_environment_ends_with_consistent_result():
    random.seed(0)
    env = greedy_experiment.build_environment()
    result = greedy_experiment.run_greedy_episode(env, GreedySearchAgent(seed=0))
    assert 1 <= result["steps"] <= config.MAX_STEPS
    assert result["success"] == (result["final_distance"] == 0.0)


def test_experiment_saves_csv_restores_global_random_and_is_reproducible(tmp_path):
    state = random.getstate()
    first = greedy_experiment.run_greedy_experiment(3, seed=9, output_path=tmp_path / "a.csv")
    assert random.getstate() == state
    second = greedy_experiment.run_greedy_experiment(3, seed=9, output_path=tmp_path / "b.csv")
    assert first == second
    with (tmp_path / "a.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["episode"]) for row in rows] == [1, 2, 3]


@pytest.mark.parametrize("episodes", [0, -1, True, 1.5])
def test_experiment_rejects_invalid_episode_count(episodes, tmp_path):
    with pytest.raises(ValueError):
        greedy_experiment.run_greedy_experiment(episodes, output_path=tmp_path / "x.csv")
