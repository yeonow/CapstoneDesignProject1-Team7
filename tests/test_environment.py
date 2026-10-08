"""환경 기능을 따로 확인하는 assert 테스트다. 루트에서 python tests/test_environment.py로 실행한다."""

import math
import sys
from pathlib import Path
from unittest.mock import call, patch

# 파일을 직접 실행해도 src를 찾을 수 있도록 이 파일 기준의 프로젝트 루트를 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment import RSSIGridEnv

def make_env(
    agent_start: tuple[int, int] = (0, 0),
    target_position: tuple[int, int] = (3, 4),
    max_steps: int = 10,
    noise_std: float = 0.0,
) -> RSSIGridEnv:
    """각 테스트가 같은 임시 RSSI 설정으로 환경을 만들도록 돕는다."""
    # 아래 숫자는 동작 확인용 임시값이며 실제 Wi-Fi 측정값이 아니다.
    # 프로젝트 설정은 Raspberry Pi와 Wi-Fi Adapter 실측 후 보정할 예정이다.
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
    """객체만 만들었을 때 RSSI가 None이고, 아직 Noise도 생성하지 않았는지 확인한다."""
    with patch("src.environment.random.gauss") as noise:
        env = make_env()
        assert env.current_rssi is None
        assert env.get_rssi() is None
        assert env.agent_position == env.agent_start == (0, 0)
        assert env.target_position == (3, 4)
        assert env.step_count == 0
        assert noise.call_count == 0


def test_reset() -> None:
    """움직인 뒤 다시 시작하면 위치와 Step 수가 복원되고 초기 RSSI를 한 번 만드는지 확인한다."""
    env = make_env()
    env.reset()
    env.move(3)
    with patch("src.environment.random.gauss", return_value=0.0) as noise:
        rssi = env.reset()
        assert env.agent_position == env.agent_start
        assert env.step_count == 0
        assert env.target_position == (3, 4)
        assert isinstance(rssi, float)
        assert rssi == env.get_rssi()
        assert math.isclose(rssi, -40.0 - 20.0 * math.log10(5.0))
        assert noise.call_count == 1


def test_four_directions() -> None:
    """Action 0, 1, 2, 3이 각각 위, 아래, 왼쪽, 오른쪽으로 한 칸 이동하는지 확인한다."""
    for action, expected in [(0, (2, 1)), (1, (2, 3)), (2, (1, 2)), (3, (3, 2))]:
        env = make_env(agent_start=(2, 2))
        env.reset()
        assert env.move(action) is False
        assert env.agent_position == expected
        assert env.get_info()["blocked"] is False
        assert env.step_count == 1


def test_grid_boundaries() -> None:
    """네 경계 밖으로 나가지 않으며, 막혀도 Step 수와 RSSI 측정은 갱신되는지 확인한다."""
    for start, action in [((2, 0), 0), ((2, 5), 1), ((0, 2), 2), ((5, 2), 3)]:
        env = make_env(agent_start=start, noise_std=2.0)
        # Noise를 순서대로 0, 1로 정해 위치가 같아도 새 측정이 일어났는지 확인한다.
        with patch("src.environment.random.gauss", side_effect=[0.0, 1.0]) as noise:
            before = env.reset()
            assert env.move(action) is True
            assert env.agent_position == start
            assert env.get_info()["blocked"] is True
            assert env.step_count == 1
            assert env.get_rssi() == before + 1.0
            assert noise.call_count == 2


def test_invalid_actions() -> None:
    """잘못된 Action은 ValueError를 내고, 위치나 Step 수, RSSI는 바꾸지 않는지 확인한다."""
    env = make_env()
    env.reset()
    before = env.get_info()
    with patch("src.environment.random.gauss") as noise:
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
    """거리를 알고 있는 (0, 0)과 (3, 4)를 사용해 직선거리가 5인지 확인한다."""
    env = make_env(agent_start=(0, 0), target_position=(3, 4))
    assert math.isclose(env._calculate_distance(), 5.0)
    assert math.isclose(env.get_info()["distance"], 5.0)


def test_rssi_sampling() -> None:
    """reset과 move는 각각 한 번 측정하고, 같은 Step의 반복 조회는 같은 값을 주는지 확인한다."""
    env = make_env(target_position=(5, 0), noise_std=2.0)
    # 임의의 Noise 대신 정해진 값을 써서 계산 결과와 생성 횟수를 확실히 비교한다.
    with patch("src.environment.random.gauss", side_effect=[1.0, -2.0]) as noise:
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
    """Noise가 없을 때 Target에 가까워지면 RSSI가 더 강해지는지 확인한다."""
    env = make_env(target_position=(5, 0), noise_std=0.0)
    previous = env.reset()
    # 기준 거리 이하에서는 RSSI가 같을 수 있으므로 거리 4, 3, 2 구간만 비교한다.
    for _ in range(3):
        env.move(3)
        current = env.get_rssi()
        assert env._calculate_distance() > env.reference_distance
        assert current > previous
        previous = current


def test_target_reached() -> None:
    """Target에 도착하면 성공과 종료가 모두 True이고, 거리 0에서도 RSSI 계산이 되는지 확인한다."""
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
    # 처음부터 같은 위치에 있어도 reset에서 log10(0) 오류가 나면 안 된다.
    coincident = make_env(agent_start=(1, 1), target_position=(1, 1))
    assert coincident.reset() == coincident.reference_rssi
    assert coincident.is_done()


def test_max_steps() -> None:
    """Target에 못 도착해도 최대 Step이면 success=False, done=True로 끝나는지 확인한다."""
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
    """평가용 정보가 현재 환경과 일치하며, 정보를 읽는 동안 새 Noise를 만들지 않는지 확인한다."""
    env = make_env()
    with patch("src.environment.random.gauss") as noise:
        assert env.get_info()["current_rssi"] is None
        assert noise.call_count == 0
    env.reset()
    env.move(3)
    with patch("src.environment.random.gauss") as noise:
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
        # 반환된 평가 정보를 수정해도 실제 환경의 Step 수는 바뀌지 않아야 한다.
        info["step_count"] = 999
        assert env.step_count == 1

def test_reproducibility_with_same_seed() -> None:
    """같은 Seed, 환경 설정, Action 순서에서는 같은 결과가 재현되는지 확인한다."""
    import random

    actions = [3, 3, 1, 1, 2, 0]

    random.seed(42)
    env1 = make_env(noise_std=2.0)
    rssi1 = [env1.reset()]

    for action in actions:
        env1.move(action)
        rssi1.append(env1.get_rssi())

    random.seed(42)
    env2 = make_env(noise_std=2.0)
    rssi2 = [env2.reset()]

    for action in actions:
        env2.move(action)
        rssi2.append(env2.get_rssi())

    assert rssi1 == rssi2
    assert env1.agent_position == env2.agent_position
    assert env1.step_count == env2.step_count

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