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
    **channel_settings,
) -> RSSIGridEnv:
    """각 테스트가 같은 임시 RSSI 설정으로 환경을 만들도록 돕는다."""
    # 아래 숫자는 동작 확인용 임시값이며 실제 Wi-Fi 측정값이 아니다.
    # 프로젝트 설정은 Raspberry Pi와 Wi-Fi Adapter 실측 후 보정할 예정이다.
    settings = dict(
        grid_size=6,
        max_steps=max_steps,
        agent_start=agent_start,
        target_position=target_position,
        cell_size_m=1.0,
        carrier_frequency_hz=2.4e9,
        tx_power_dbm=0.0,
        eta_los=2.0,
        eta_nlos=20.0,
        path_loss_exponent=2.0,
        multipath_fading=1.0,
        noise_std=noise_std,
        min_distance_m=1.0,
        noise_floor_dbm=-90.0,  # 논문 제공값이 아닌 테스트용 수신기 잡음 전력 가정
        channel_state="los",
    )
    settings.update(channel_settings)
    return RSSIGridEnv(**settings)


def expected_received_power(env, distance_m, omega=0.0):
    """테스트 기준은 논문의 선형 식 (2)로 구해 실제 dB 구현과 독립적으로 비교한다."""
    eta = env.eta_los if env.channel_state == "los" else env.eta_nlos
    gain = (
        eta ** -1
        * env.multipath_fading ** 2
        * 10.0 ** (omega / 10.0)
        * (299_792_458.0 / (4.0 * math.pi * env.carrier_frequency_hz * distance_m))
        ** env.path_loss_exponent
    )
    return env.tx_power_dbm + 10.0 * math.log10(gain)


def expected_rssi(env, distance_m, omega=0.0):
    """식 (1)에 해당하는 신호+잡음 전력을 선형 단위로 합해 RSSI를 검증한다."""
    rx_power = expected_received_power(env, distance_m, omega)
    return 10.0 * math.log10(10.0 ** (rx_power / 10.0) + 10.0 ** (env.noise_floor_dbm / 10.0))


def test_creation() -> None:
    """객체만 만들었을 때 RSSI가 None이고, 아직 Noise도 생성하지 않았는지 확인한다."""
    with patch("src.environment.random.gauss") as noise:
        env = make_env()
        assert env.current_rx_power is None
        assert env.current_snr is None
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
        assert math.isclose(env.current_rx_power, expected_received_power(env, 5.0))
        assert math.isclose(rssi, expected_rssi(env, 5.0))
        assert env.current_snr == env.current_rx_power - env.noise_floor_dbm
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
            before_rx_power = env.current_rx_power
            assert env.move(action) is True
            assert env.agent_position == start
            assert env.get_info()["blocked"] is True
            assert env.step_count == 1
            # shadow fading은 P_rx에 더해지며, 최종 RSSI에는 식 (1)의 보정도 적용된다.
            assert env.current_rx_power == before_rx_power + 1.0
            assert env.get_rssi() > before
            distance_m = max(env._calculate_distance() * env.cell_size_m, env.min_distance_m)
            assert math.isclose(env.get_rssi(), expected_rssi(env, distance_m, omega=1.0))
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
        assert math.isclose(env.current_rx_power, expected_received_power(env, 5.0, omega=1.0))
        assert math.isclose(initial, expected_rssi(env, 5.0, omega=1.0))
        assert initial == env.get_rssi() == env.get_rssi()
        assert noise.call_count == 1
        env.move(3)
        updated = env.get_rssi()
        assert isinstance(updated, float)
        assert math.isclose(env.current_rx_power, expected_received_power(env, 4.0, omega=-2.0))
        assert math.isclose(updated, expected_rssi(env, 4.0, omega=-2.0))
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
        assert env._calculate_distance() * env.cell_size_m > env.min_distance_m
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
    assert env.get_rssi() == initial
    assert math.isclose(initial, expected_rssi(env, env.min_distance_m))
    env.reset()
    assert not env.is_done()
    assert env.get_info()["success"] is False
    # 처음부터 같은 위치에 있어도 reset에서 log10(0) 오류가 나면 안 된다.
    coincident = make_env(agent_start=(1, 1), target_position=(1, 1))
    assert math.isclose(
        coincident.reset(), expected_rssi(coincident, coincident.min_distance_m)
    )
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
        assert env.get_info()["current_rx_power"] is None
        assert env.get_info()["current_snr"] is None
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
            "current_rx_power": env.current_rx_power,
            "current_snr": env.current_snr,
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


def test_equation_2_matches_linear_gain() -> None:
    """LoS/NLoS와 비기본 파라미터에서도 dB 결과가 선형 식 (2)와 같은지 확인한다."""
    for state in ("los", "nlos"):
        env = make_env(
            cell_size_m=2.5, carrier_frequency_hz=915e6, tx_power_dbm=13.0,
            eta_los=2.5, eta_nlos=12.0, path_loss_exponent=2.7,
            multipath_fading=0.7, noise_std=3.0, channel_state=state,
        )
        for omega in (-3.2, 4.3):
            with patch("src.environment.random.gauss", return_value=omega) as noise:
                env.reset()
                assert math.isclose(env.current_rx_power, expected_received_power(env, 12.5, omega))
                assert math.isclose(env.get_rssi(), expected_rssi(env, 12.5, omega))
                noise.assert_called_once_with(0.0, 3.0)
        # 채널은 m로 계산해도 기존 Ground Truth distance는 Grid 단위를 유지한다.
        assert env.get_info()["distance"] == 5.0


def test_channel_parameter_effects() -> None:
    """단위 변환, 주파수, 송신 전력, nu^2, eta의 부호와 배율을 검증한다."""
    baseline = make_env()
    baseline.reset()
    initial = baseline.current_rx_power
    expected_loss = -10.0 * baseline.path_loss_exponent * math.log10(2.0)
    # 식 (2)의 손실/이득 배율은 식 (1) 적용 전 P_rx에서 확인해야 한다.
    for override, difference in (
        ({"cell_size_m": 2.0}, expected_loss),
        ({"carrier_frequency_hz": 4.8e9}, expected_loss),
        ({"tx_power_dbm": 7.0}, 7.0),
        ({"multipath_fading": 2.0}, 20.0 * math.log10(2.0)),
        ({"channel_state": "nlos"}, -10.0),
    ):
        env = make_env(**override)
        env.reset()
        assert math.isclose(env.current_rx_power - initial, difference)


def test_minimum_distance_in_meters() -> None:
    """거리 0과 최소 거리 미만의 양수 거리 모두 m 단위 하한으로 계산한다."""
    for target in ((0, 0), (1, 0), (5, 0)):
        env = make_env(target_position=target, cell_size_m=0.1, min_distance_m=0.5)
        assert math.isclose(env.reset(), expected_rssi(env, 0.5))
        assert math.isclose(env.current_rx_power, expected_received_power(env, 0.5))
        assert math.isfinite(env.get_rssi())


def test_channel_state_independent_of_blocked() -> None:
    """경계 충돌은 LoS를 NLoS로 바꾸지 않으며 NLoS에서도 이동은 가능하다."""
    for state in ("los", "nlos"):
        env = make_env(channel_state=state)
        initial = env.reset()
        assert env.move(0) is True
        assert env.get_info()["blocked"] is True
        assert env._get_channel_state() == state
        assert env.get_rssi() == initial  # noise_std=0, 위치와 채널 상태 모두 동일
        assert env.move(3) is False
        assert env._get_channel_state() == state
        assert env.get_info()["blocked"] is False


def test_channel_state_hook() -> None:
    """별도 상태 메서드를 교체하면 이동 로직 수정 없이 채널 손실이 바뀐다."""
    env = make_env()
    env.reset()
    initial = env.current_rx_power
    with patch.object(env, "_get_channel_state", return_value="nlos") as state:
        # 같은 위치에서 상태만 바꿔 eta_nlos가 실제 계산에 쓰이는지 검증한다.
        env.reset()
        assert math.isclose(env.current_rx_power - initial, -10.0)
        state.assert_called_once_with()
        assert env.get_info()["blocked"] is False


def test_invalid_channel_parameters() -> None:
    """거리/채널 계수의 로그 오류와 비정상 shadow 표준편차를 생성 시 차단한다."""
    cases = []
    for name in (
        "cell_size_m", "min_distance_m", "carrier_frequency_hz", "eta_los",
        "eta_nlos", "path_loss_exponent", "multipath_fading",
    ):
        cases.extend((name, value) for value in (0.0, -1.0, math.inf, math.nan))
    cases.extend(("noise_std", value) for value in (-1.0, math.inf, math.nan))
    cases.extend(("tx_power_dbm", value) for value in (math.inf, math.nan))
    cases.extend(("noise_floor_dbm", value) for value in (math.inf, -math.inf, math.nan))
    cases.append(("channel_state", "unknown"))
    for name, value in cases:
        with patch("src.environment.random.gauss") as noise:
            try:
                make_env(**{name: value})
            except ValueError as error:
                assert name in str(error)
            else:
                raise AssertionError(f"Invalid {name} was accepted: {value}")
            noise.assert_not_called()


def test_equation_1_recovers_received_power() -> None:
    """양수/0/음수 SNR에서 식 (1)의 정방향 계산으로 원래 P_rx를 복원한다."""
    env = make_env(noise_floor_dbm=-90.0)
    for rx_power in (-40.0, -90.0, -120.0):
        with patch.object(env, "_calculate_channel_gain_db", return_value=rx_power - env.tx_power_dbm) as gain:
            rssi = env.reset()
            gain.assert_called_once_with()
        assert env.current_rx_power == rx_power
        assert env.current_snr == rx_power - env.noise_floor_dbm
        restored = rssi - 10.0 * math.log10(1.0 + 10.0 ** (-env.current_snr / 10.0))
        assert math.isclose(restored, rx_power, rel_tol=0.0, abs_tol=1e-10)
        assert rssi > rx_power
        assert rssi > env.noise_floor_dbm
        if env.current_snr == 0.0:
            assert math.isclose(rssi - rx_power, 10.0 * math.log10(2.0))


def test_channel_measurements_cached_together() -> None:
    """reset/일반 이동/경계 이동마다 한 번 측정하고 세 저장값을 함께 갱신한다."""
    env = make_env(noise_std=2.0)
    snapshots = []
    with (
        patch("src.environment.random.gauss", side_effect=[0.0, 1.0, -2.0, 3.0]) as noise,
        patch.object(env, "_calculate_channel_gain_db", wraps=env._calculate_channel_gain_db) as gain,
    ):
        for measure in (env.reset, lambda: env.move(0), lambda: env.move(3), env.reset):
            measure()
            snapshot = env.get_info()
            for _ in range(3):
                assert env.get_info() == snapshot
                assert env.get_rssi() == snapshot["current_rssi"]
            assert snapshot["current_rx_power"] == env.current_rx_power
            assert snapshot["current_snr"] == env.current_snr
            assert env.current_snr == env.current_rx_power - env.noise_floor_dbm
            assert all(isinstance(snapshot[key], float) for key in (
                "current_rx_power", "current_snr", "current_rssi",
            ))
            snapshots.append(snapshot)
            assert gain.call_count == noise.call_count == len(snapshots)
    assert snapshots[1]["blocked"] is True
    assert snapshots[2]["blocked"] is False
    assert snapshots[3]["step_count"] == 0
    assert snapshots[3]["agent_position"] == env.agent_start
    assert len({snapshot["current_rx_power"] for snapshot in snapshots}) == 4
    assert len({snapshot["current_rssi"] for snapshot in snapshots}) == 4


def test_noise_floor_changes_snr_and_rssi_only() -> None:
    """수신기 잡음 가정은 식 (2)의 신호 전력을 변경하지 않고 식 (1)에만 반영한다."""
    quiet = make_env(noise_floor_dbm=-100.0)
    noisy = make_env(noise_floor_dbm=-60.0)
    quiet.reset()
    noisy.reset()
    assert quiet.current_rx_power == noisy.current_rx_power
    assert math.isclose(quiet.current_snr - noisy.current_snr, 40.0)
    assert noisy.get_rssi() > quiet.get_rssi()


def test_equation_1_extreme_snr_stays_finite() -> None:
    """큰 음수 SNR에서도 지수 overflow 없이 잡음 전력에 수렴한다."""
    env = make_env(noise_floor_dbm=-90.0)
    for snr in (-4000.0, 4000.0):
        rx_power = env.noise_floor_dbm + snr
        with patch.object(env, "_calculate_channel_gain_db", return_value=rx_power - env.tx_power_dbm):
            rssi = env.reset()
        assert math.isfinite(rssi)
        assert env.current_snr == snr
        assert math.isclose(rssi, max(rx_power, env.noise_floor_dbm))


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
    test_reproducibility_with_same_seed()
    test_equation_2_matches_linear_gain()
    test_channel_parameter_effects()
    test_minimum_distance_in_meters()
    test_channel_state_independent_of_blocked()
    test_channel_state_hook()
    test_invalid_channel_parameters()
    test_equation_1_recovers_received_power()
    test_channel_measurements_cached_together()
    test_noise_floor_changes_snr_and_rssi_only()
    test_equation_1_extreme_snr_stays_finite()
    print("All environment tests passed.")
