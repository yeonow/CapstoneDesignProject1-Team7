"""Standalone assert tests: run with python test_environment.py."""

import math
from unittest.mock import call, patch

from environment import RSSIGridEnv


def make_env(
    agent_start: tuple[int, int] = (0, 0),
    target_position: tuple[int, int] = (3, 4),
    max_steps: int = 10,
    noise_std: float = 0.0,
) -> RSSIGridEnv:
    # Temporary simulation test values, not established Wi-Fi measurements.
    # Real parameters will be calibrated using Raspberry Pi / Wi-Fi adapter data.
    return RSSIGridEnv(
        grid_size=6,
        max_steps=max_steps,
        agent_start=agent_start,
        target_position=target_position,
        reference_rssi=-40.0,
        reference_distance=1.0,
        path_loss_exponent=2.0,
        noise_std=noise_std,
    )


def test_creation() -> None:
    with patch("environment.random.gauss") as noise:
        env = make_env()
        assert env.current_rssi is None
        assert env.get_rssi() is None
        assert env.agent_position == env.agent_start == (0, 0)
        assert env.target_position == (3, 4)
        assert env.step_count == 0
        assert noise.call_count == 0


def test_reset() -> None:
    env = make_env()
    env.reset()
    env.move(3)
    with patch("environment.random.gauss", return_value=0.0) as noise:
        rssi = env.reset()
        assert env.agent_position == env.agent_start
        assert env.step_count == 0
        assert env.target_position == (3, 4)
        assert isinstance(rssi, float)
        assert rssi == env.get_rssi()
        assert math.isclose(rssi, -40.0 - 20.0 * math.log10(5.0))
        assert noise.call_count == 1


def test_four_directions() -> None:
    for action, expected in [(0, (2, 1)), (1, (2, 3)), (2, (1, 2)), (3, (3, 2))]:
        env = make_env(agent_start=(2, 2))
        env.reset()
        env.move(action)
        assert env.agent_position == expected
        assert env.step_count == 1


def test_grid_boundaries() -> None:
    for start, action in [((2, 0), 0), ((2, 5), 1), ((0, 2), 2), ((5, 2), 3)]:
        env = make_env(agent_start=start, noise_std=2.0)
        with patch("environment.random.gauss", side_effect=[0.0, 1.0]) as noise:
            before = env.reset()
            env.move(action)
            assert env.agent_position == start
            assert env.step_count == 1
            assert env.get_rssi() == before + 1.0
            assert noise.call_count == 2


def test_invalid_actions() -> None:
    env = make_env()
    env.reset()
    before = env.get_info()
    with patch("environment.random.gauss") as noise:
        for action in (-1, 4, 99):
            try:
                env.move(action)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Action {action} did not raise ValueError")
            assert env.get_info() == before
        assert noise.call_count == 0


def test_distance() -> None:
    env = make_env(agent_start=(0, 0), target_position=(3, 4))
    assert math.isclose(env._calculate_distance(), 5.0)
    assert math.isclose(env.get_info()["distance"], 5.0)


def test_rssi_sampling() -> None:
    env = make_env(target_position=(5, 0), noise_std=2.0)
    with patch("environment.random.gauss", side_effect=[1.0, -2.0]) as noise:
        initial = env.reset()
        assert math.isclose(initial, -40.0 - 20.0 * math.log10(5.0) + 1.0)
        assert initial == env.get_rssi() == env.get_rssi()
        assert noise.call_count == 1
        env.move(3)
        updated = env.get_rssi()
        assert isinstance(updated, float)
        assert math.isclose(updated, -40.0 - 20.0 * math.log10(4.0) - 2.0)
        assert updated != initial
        assert updated == env.get_rssi() == env.get_rssi()
        assert noise.call_count == 2
        assert noise.call_args_list == [call(0.0, 2.0), call(0.0, 2.0)]


def test_noiseless_rssi_trend() -> None:
    env = make_env(target_position=(5, 0), noise_std=0.0)
    previous = env.reset()
    for _ in range(3):  # Distances 4, 3, 2 all exceed reference_distance.
        env.move(3)
        current = env.get_rssi()
        assert env._calculate_distance() > env.reference_distance
        assert current > previous
        previous = current


def test_target_reached() -> None:
    env = make_env(target_position=(1, 0))
    initial = env.reset()
    assert not env.is_done()
    env.move(3)
    info = env.get_info()
    assert info["success"] is True
    assert info["done"] is True
    assert info["distance"] == 0.0
    assert math.isfinite(env.get_rssi())
    assert env.get_rssi() == initial == env.reference_rssi
    env.reset()
    assert not env.is_done()
    assert env.get_info()["success"] is False
    coincident = make_env(agent_start=(1, 1), target_position=(1, 1))
    assert coincident.reset() == coincident.reference_rssi
    assert coincident.is_done()


def test_max_steps() -> None:
    env = make_env(target_position=(5, 5), max_steps=2)
    env.reset()
    env.move(3)
    assert not env.is_done()
    env.move(3)
    info = env.get_info()
    assert info["step_count"] == info["max_steps"] == 2
    assert info["success"] is False
    assert info["done"] is True
    env.move(3)
    assert env.step_count > env.max_steps and env.is_done()


def test_get_info() -> None:
    env = make_env()
    with patch("environment.random.gauss") as noise:
        assert env.get_info()["current_rssi"] is None
        assert noise.call_count == 0
    env.reset()
    env.move(3)
    with patch("environment.random.gauss") as noise:
        info = env.get_info()
        expected = {
            "agent_position": (1, 0),
            "target_position": (3, 4),
            "distance": math.sqrt(20.0),
            "step_count": 1,
            "max_steps": 10,
            "current_rssi": env.get_rssi(),
            "success": False,
            "done": env.is_done(),
        }
        assert expected.keys() <= info.keys()
        for key, value in expected.items():
            assert info[key] == value, key
        assert info == env.get_info()
        assert noise.call_count == 0
        info["step_count"] = 999
        assert env.step_count == 1


if __name__ == "__main__":
    test_creation()
    test_reset()
    test_four_directions()
    test_grid_boundaries()
    test_invalid_actions()
    test_distance()
    test_rssi_sampling()
    test_noiseless_rssi_trend()
    test_target_reached()
    test_max_steps()
    test_get_info()
    print("All environment tests passed.")
