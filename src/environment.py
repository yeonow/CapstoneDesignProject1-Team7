import math
import random
from typing import Optional


class RSSIGridEnv:
    """Q-Learning이 탐색을 연습할 가상 2D 공간이다.

    Agent는 실종자를 찾는 탐색 주체이며, Target은 사전에 지정한
    스마트폰 Wi-Fi Hotspot의 위치다. 여기서는 이동, 거리, 가상 RSSI,
    종료 여부만 관리하고 State, Reward, Q-Learning 학습은 다른 파일에서 맡는다.
    Agent와 Target 좌표는 내부 계산용 실제 위치 정보(Ground Truth)이며,
    Q-Learning에게 직접 알려주는 State에 넣지 않는다.
    """

    def __init__(
        self,
        grid_size: int,
        max_steps: int,
        agent_start: tuple[int, int],
        target_position: tuple[int, int],
        cell_size_m: float,
        carrier_frequency_hz: float,
        tx_power_dbm: float,
        eta_los: float,
        eta_nlos: float,
        path_loss_exponent: float,
        multipath_fading: float,
        noise_std: float,
        min_distance_m: float,
        noise_floor_dbm: float,
        channel_state: str = "los",
    ) -> None:
        """가상 공간과 RSSI 설정을 준비한다. 객체 생성 자체는 RSSI 측정이 아니다.

        grid_size는 가상 공간 한 변의 칸 수, max_steps는 한 Episode에서
        Action을 수행할 수 있는 최대 횟수다. 경계에 막힌 Action도 횟수에 포함한다.
        agent_start는 처음 시작할 위치, agent_position은 현재 Agent 위치,
        target_position은 Target Wi-Fi 신호원이 있다고 가정한 위치다.
        좌표는 (x, y)로 관리하고, step_count에는 지금까지 수행한 Action 수를 기록한다.

        cell_size_m는 한 칸의 길이(m), min_distance_m는 채널 계산의 최소 거리(m)다.
        carrier_frequency_hz는 반송파 주파수(fc), tx_power_dbm은 송신 전력(dBm)이다.
        eta_los/eta_nlos는 LoS/NLoS 추가 경로 손실 계수(선형 단위, dB 아님),
        path_loss_exponent는 경로 손실 지수(kappa)다.
        multipath_fading은 다중경로 진폭 계수(nu)로, 현재는 고정 양수 입력을 쓴다.
        noise_std는 shadow fading(omega)의 dB 표준편차이며 0이면 변동이 없다.
        noise_floor_dbm은 SNR 계산에 사용하는 수신기 잡음 전력(dBm) 가정이다.
        논문에서 직접 주어진 값이 아닌 Simulation용 추가 가정으로 실측 후 보정한다.
        shadow fading의 변동 폭인 noise_std와는 별개의 값이다.
        channel_state는 현재 고정된 "los" 또는 "nlos"이며 Grid 경계 충돌과 무관하다.
        물리 파라미터는 실제 Wi-Fi 실측값이 아닌 Simulation용 임시 설정이다.
        이후 Raspberry Pi + Wi-Fi Adapter 실측으로 파라미터와 fading 특성을 보정한다.
        current_rx_power는 원하는 신호의 수신 전력 P_rx(dBm), current_snr는
        신호 대 잡음 전력비 SNR(dB), current_rssi는 식 (1)에 따른 신호와 잡음을
        포함한 수신 강도 RSSI(dBm)다. 세 값 모두 측정 전에는 None이다.
        """
        # 식 (2)의 나눗셈과 로그가 정의되도록 채널 입력만 검증한다.
        for name, value in {
            "cell_size_m": cell_size_m,
            "carrier_frequency_hz": carrier_frequency_hz,
            "eta_los": eta_los,
            "eta_nlos": eta_nlos,
            "path_loss_exponent": path_loss_exponent,
            "multipath_fading": multipath_fading,
            "min_distance_m": min_distance_m,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than 0")
        if not math.isfinite(tx_power_dbm):
            raise ValueError("tx_power_dbm must be finite")
        if not math.isfinite(noise_floor_dbm):
            raise ValueError("noise_floor_dbm must be finite")
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if channel_state not in ("los", "nlos"):
            raise ValueError('channel_state must be "los" or "nlos"')

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_start = agent_start
        self.agent_position = self.agent_start
        self.target_position = target_position
        self.step_count = 0
        self.cell_size_m = cell_size_m
        self.carrier_frequency_hz = carrier_frequency_hz
        self.tx_power_dbm = tx_power_dbm
        self.eta_los = eta_los
        self.eta_nlos = eta_nlos
        self.path_loss_exponent = path_loss_exponent
        self.multipath_fading = multipath_fading
        self.noise_std = noise_std
        self.min_distance_m = min_distance_m
        self.noise_floor_dbm = noise_floor_dbm
        self.channel_state = channel_state
        self.current_rx_power: float | None = None
        self.current_snr: float | None = None
        self.current_rssi: float | None = None
        self.last_move_blocked = False

    def reset(self) -> float:
        """Episode를 처음부터 다시 시작할 때 Agent를 시작 위치로 되돌린다.

        Step 수를 0으로 초기화하고, 시작 위치에서 Raw RSSI를 한 번 새로 측정한다.
        동일 측정의 P_rx, SNR, RSSI를 함께 저장하고 RSSI를 반환한다.
        """
        self.agent_position = self.agent_start
        self.step_count = 0
        self.last_move_blocked = False
        self.current_rssi = self._generate_rssi()
        return self.current_rssi

    def move(self, action: int) -> bool:
        """Q-Learning이 선택한 방향으로 Agent를 한 칸 움직인다.

        Action은 0=위(y-1), 1=아래(y+1), 2=왼쪽(x-1), 3=오른쪽(x+1)이다.
        Grid 밖으로 나가려 하면 위치는 그대로 둔다. 그래도 Action은 수행했으므로
        step_count를 늘리고 같은 위치에서 RSSI를 한 번 새로 측정한다.
        잘못된 Action 번호는 ValueError를 발생시킨다.
        경계에 막혀 제자리에 있었으면 True(blocked), 실제로 이동했으면 False를 반환한다.
        """
        x, y = self.agent_position

        if action == 0:  # 위
            y -= 1
        elif action == 1:  # 아래
            y += 1
        elif action == 2:  # 왼쪽
            x -= 1
        elif action == 3:  # 오른쪽
            x += 1
        else:
            raise ValueError("action must be 0 (UP), 1 (DOWN), 2 (LEFT), or 3 (RIGHT)")

        blocked = not (0 <= x < self.grid_size and 0 <= y < self.grid_size)
        if not blocked:
            self.agent_position = (x, y)

        self.step_count += 1
        self.last_move_blocked = blocked
        self.current_rssi = self._generate_rssi()
        return blocked

    def _calculate_distance(self) -> float:
        """Agent와 Target이 얼마나 떨어져 있는지 내부 위치 정보로 계산한다.

        이동한 경로 길이가 아닌 직선거리(Euclidean Distance)를 사용하며,
        반환 단위는 기존과 같은 Grid 칸 수이며, 채널 계산에서만 m로 변환한다.
        """
        agent_x, agent_y = self.agent_position
        target_x, target_y = self.target_position
        return math.hypot(target_x - agent_x, target_y - agent_y)

    def _get_channel_state(self) -> str:
        """현재는 설정된 LoS/NLoS를 사용하고, 추후 동적 채널 상태 모델로 교체한다.

        blocked는 이동 경계 충돌이지 전파 차폐가 아니므로 여기서 사용하지 않는다.
        논문의 고도/지형 기반 LoS 확률 모델은 현재 2D Grid에 도입하지 않는다.
        """
        return self.channel_state

    def _calculate_channel_gain_db(self) -> float:
        """논문의 식 (2)를 참고하여 거리, 주파수, LoS/NLoS, shadow fading,
        multipath fading을 반영한 채널 이득을 한 번 계산한다.

        기존 단순 Log-distance 모델의 기준 RSSI 대신 주파수와 전파 손실 요인을
        명시적으로 조절하기 위해 채널 모델을 변경했다. 참고: Catch Me If You Can:
        Deep Meta-RL for Search-and-Rescue Using LoRa UAV Networks, 식 (2).
        논문의 linear-scale 식 (2)를 RSSI/dBm 계산과 연결하기 쉽도록 dB 형태로
        변환해 구현했다. h = eta^-1 * nu^2 * 10^(omega/10) * (c/(4*pi*fc*d))^kappa.
        따라서 gain_db = -10*log10(eta) + 20*log10(nu) + omega
                          + 10*kappa*log10(c/(4*pi*fc*d)).

        이는 논문의 LoRa/UAV 환경 전체를 재현하는 모델이 아닌 2D Simulation용
        적용이다. 실제 Wi-Fi에 맞는 값과 fading 특성은 실측 후 보정해야 한다.
        """
        # d는 두 신호 지점 사이 거리(m)다. 기존 Grid 좌표/평가 인터페이스를
        # 유지하면서 물리식에 필요한 단위만 변환한다. 거리 0의 나눗셈/로그를
        # 피하고 근거리 모델 발산을 제한하도록 양수 최소 거리를 적용한다.
        distance_m = self._calculate_distance() * self.cell_size_m
        effective_distance = max(distance_m, self.min_distance_m)
        channel_state = self._get_channel_state()
        # eta는 LoS/NLoS의 추가 경로 손실 계수다. 선형 손실이므로 dB에서는 뺀다.
        if channel_state == "los":
            eta = self.eta_los
        elif channel_state == "nlos":
            eta = self.eta_nlos
        else:
            raise ValueError('channel state must be "los" or "nlos"')
        # omega는 지형/장애물에 의한 shadow fading(dB)을 단순화한 Gaussian 값이다.
        # RSSI 측정당 한 번만 샘플링하며, noise_std가 그 표준편차다.
        omega = random.gauss(0.0, self.noise_std)
        # nu는 다중경로 진폭 계수여서 nu^2의 dB 변환은 20*log10(nu)다.
        # 현재 고정 입력을 사용하며, 임의의 Rayleigh/Rician 분포를 가정하지 않는다.
        nu = self.multipath_fading
        kappa = self.path_loss_exponent  # 거리/주파수 감쇠의 지수
        fc = self.carrier_frequency_hz  # 반송파 주파수(Hz)
        c = 299_792_458.0  # 빛의 속도(m/s); 실험 조정값이 아닌 물리 상수
        # 작은 선형 gain을 먼저 만들지 않고 로그의 합으로 계산해 underflow를 피한다.
        return (
            -10.0 * math.log10(eta)
            + 20.0 * math.log10(nu)
            + omega
            + 10.0 * kappa * (
                math.log10(c) - math.log10(4.0 * math.pi)
                - math.log10(fc) - math.log10(effective_distance)
            )
        )

    def _generate_rssi(self) -> float:
        """채널을 한 번 샘플링해 P_rx, SNR, RSSI를 함께 저장하고 RSSI를 반환한다.

        논문의 식 (1)은 RSSI와 SNR로 Received Power를 계산하지만,
        본 Simulation에서는 식 (2)로 P_rx를 먼저 생성하므로 RSSI에 대해
        식을 정리해 역방향으로 사용한다:
        RSSI = P_rx + 10*log10(1 + 10^(-SNR/10)).
        """
        # 식 (2)의 P_rx = P_tx * h 계산은 유지한다. P_rx는 원하는 신호 전력이고,
        # SNR은 이 신호와 noise floor의 전력비이므로 dB 단위에서는 차로 계산한다.
        channel_gain_db = self._calculate_channel_gain_db()
        received_power_dbm = self.tx_power_dbm + channel_gain_db
        snr_db = received_power_dbm - self.noise_floor_dbm
        # 식 (1)의 보정항을 동치 형태로 계산한다. 큰 음수 SNR에서 10^(-SNR/10)의
        # overflow를 피하고, 높은 SNR의 작은 보정값은 log1p로 정밀도를 유지한다.
        correction_db = max(-snr_db, 0.0) + (10.0 / math.log(10.0)) * math.log1p(
            10.0 ** (-abs(snr_db) / 10.0)
        )
        rssi_dbm = received_power_dbm + correction_db
        # 신호+잡음의 강도인 RSSI를 P_rx와 구분하되, 같은 Step의 State/Reward와
        # 디버깅 정보가 동일한 채널 샘플을 보도록 세 값을 함께 저장한다.
        self.current_rx_power = received_power_dbm
        self.current_snr = snr_db
        self.current_rssi = rssi_dbm
        return self.current_rssi

    def get_rssi(self) -> Optional[float]:
        """새 RSSI를 만들지 않고 저장된 current_rssi만 돌려준다. 측정 전에는 None이다.

        한 번 이동한 뒤 RSSI는 딱 한 번 만들고 current_rssi에 저장한다.
        외부의 State와 Reward 계산은 이 값을 함께 사용한다. 조회할 때마다 새로
        만들면 Noise 때문에 Reward는 -62.1, State는 -64.0처럼 다른 값을 볼 수 있다.
        같은 이동 결과를 같은 RSSI로 판단하도록 여기서는 저장값만 반환한다.
        """
        return self.current_rssi

    def is_done(self) -> bool:
        """Target 위치에 도착했거나 최대 Step 수에 도달하면 Episode를 끝낸다."""
        return (
            self.agent_position == self.target_position
            or self.step_count >= self.max_steps
        )

    def get_info(self) -> dict[str, object]:
        """Simulation 진행 상황을 확인하고 실험 결과 저장이나 디버깅에 쓸 정보를 준다.

        Agent 위치, Target 위치, 거리 등은 평가용 실제 정보(Ground Truth)이며,
        Q-Learning State에는 넣지 않는다.
        success는 Target에 실제로 도착했는지, done은 Episode가 끝났는지를 뜻한다.
        Target 도착 시에는 success=True, done=True이고,
        도착하지 못한 채 최대 Step으로 끝나면 success=False, done=True다.
        blocked는 직전 Action이 Grid 경계에 막혀 제자리에 있었는지를 뜻한다.
        P_rx(dBm), SNR(dB), RSSI(dBm)는 같은 측정에서 저장된 값만 반환한다.
        """
        return {
            "agent_position": self.agent_position,
            "target_position": self.target_position,
            "distance": self._calculate_distance(),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_rx_power": self.current_rx_power,
            "current_snr": self.current_snr,
            "current_rssi": self.current_rssi,
            "success": self.agent_position == self.target_position,
            "done": self.is_done(),
            "blocked": self.last_move_blocked,
        }
