"""Q-Learning과 전체 실험의 Seed 기반 재현성을 검증한다."""

import random
from unittest.mock import patch

import numpy as np
import pytest

import config
from src import experiment
from src.q_learning import QLearningAgent


TEST_SEED = 2026
TRAIN_EPISODES = 6
EVAL_EPISODES = 4
MAX_STEPS = 12
TARGET_POSITION = (3, 3)


@pytest.fixture(autouse=True)
def preserve_global_rng_state():
    """각 테스트가 Python과 NumPy 전역 RNG 상태를 외부에 남기지 않게 한다."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _build_small_environment():
    """실제 구성 함수를 사용해 실행 시간이 짧은 환경을 만든다."""
    with (
        patch.object(config, "MAX_STEPS", MAX_STEPS),
        patch.object(config, "TARGET_POSITION", TARGET_POSITION),
    ):
        return experiment.build_environment()


def _build_small_experiment(seed):
    """작은 실제 환경과 지정 Seed의 Q-Learning Agent를 만든다."""
    env = _build_small_environment()
    with patch.object(config, "RANDOM_SEED", seed):
        agent = experiment.build_agent()
    return env, agent


def _train_with_seed(seed):
    """experiment.train과 같은 순서로 실제 Episode를 짧게 학습한다."""
    random.seed(seed)
    env, agent = _build_small_experiment(seed)
    results = []

    for episode in range(1, TRAIN_EPISODES + 1):
        results.append({"episode": episode, **experiment.run_episode(env, agent)})
        agent.decay_epsilon()

    return agent, results


def _train_and_evaluate_with_seed(seed):
    """학습된 동일 Agent를 별도 실제 환경에서 평가한다."""
    agent, _ = _train_with_seed(seed)
    random.seed(seed + 1)
    eval_env = _build_small_environment()
    return experiment.evaluate_agent(eval_env, agent, EVAL_EPISODES)


def test_action_sequence_reproducible_with_same_seed():
    """같은 Seed의 Agent 두 개가 같은 epsilon-greedy 행동 순서를 만든다."""
    state = (2, 0, -1)
    agent1 = QLearningAgent(epsilon=0.65, seed=TEST_SEED)
    agent2 = QLearningAgent(epsilon=0.65, seed=TEST_SEED)

    actions1 = [agent1.choose_action(state) for _ in range(100)]
    actions2 = [agent2.choose_action(state) for _ in range(100)]

    assert actions1 == actions2


def test_q_table_reproducible_with_same_seed():
    """같은 조건의 실제 학습 두 번이 동일한 최종 Q-table을 만든다."""
    agent1, _ = _train_with_seed(TEST_SEED)
    agent2, _ = _train_with_seed(TEST_SEED)

    assert agent1.q_table.keys() == agent2.q_table.keys()
    for state in agent1.q_table:
        np.testing.assert_array_equal(
            agent1.q_table[state],
            agent2.q_table[state],
        )
    assert agent1.epsilon == agent2.epsilon


def test_training_results_reproducible_with_same_seed():
    """같은 조건에서 Episode별 성공, Step, Reward, 거리가 모두 일치한다."""
    _, results1 = _train_with_seed(TEST_SEED)
    _, results2 = _train_with_seed(TEST_SEED)

    assert results1 == results2


def test_evaluation_results_reproducible_with_same_seed():
    """같은 학습 이력 뒤 greedy 평가 집계 결과가 일치한다."""
    evaluation1 = _train_and_evaluate_with_seed(TEST_SEED)
    evaluation2 = _train_and_evaluate_with_seed(TEST_SEED)

    assert evaluation1 == evaluation2


def test_agent_rng_is_independent_from_global_random():
    """전역 random 소비 순서가 Agent 전용 RNG의 행동 순서를 바꾸지 않는다."""
    state = (2, 0, -1)
    agent1 = QLearningAgent(epsilon=0.65, seed=TEST_SEED)
    actions1 = [agent1.choose_action(state) for _ in range(100)]

    random.seed(999)
    agent2 = QLearningAgent(epsilon=0.65, seed=TEST_SEED)
    actions2 = []
    for _ in range(100):
        random.random()
        random.gauss(0.0, 2.0)
        actions2.append(agent2.choose_action(state))

    assert actions1 == actions2
