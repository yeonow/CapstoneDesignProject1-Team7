import math
import random


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
        reference_rssi: float,
        reference_distance: float,
        path_loss_exponent: float,
        noise_std: float,
    ) -> None:
        """가상 공간과 RSSI 설정을 준비한다. 객체 생성 자체는 RSSI 측정이 아니다.

        grid_size는 가상 공간 한 변의 칸 수, max_steps는 한 Episode에서
        Action을 수행할 수 있는 최대 횟수다. 경계에 막힌 Action도 횟수에 포함한다.
        agent_start는 처음 시작할 위치, agent_position은 현재 Agent 위치,
        target_position은 Target Wi-Fi 신호원이 있다고 가정한 위치다.
        좌표는 (x, y)로 관리하고, step_count에는 지금까지 수행한 Action 수를 기록한다.

        reference_rssi는 기준 거리에서의 RSSI 값이고, reference_distance는
        RSSI 계산의 기준 거리(Grid 단위, 양수)다. path_loss_exponent는
        거리가 멀어질수록 RSSI가 얼마나 감소하는지 조절한다.
        noise_std는 같은 위치에서도 RSSI가 달라지는 정도(표준편차, 0 이상)다.
        RSSI 설정은 실제 측정값이 아닌 Simulation용 임시값으로, 실측 후 보정한다.
        current_rssi에는 현재 Step의 RSSI를 저장하며, 측정 전에는 None이다.
        """
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.agent_start = agent_start
        self.agent_position = self.agent_start
        self.target_position = target_position
        self.step_count = 0
        self.reference_rssi = reference_rssi
        self.reference_distance = reference_distance
        self.path_loss_exponent = path_loss_exponent
        self.noise_std = noise_std
        self.current_rssi: float | None = None

    def reset(self) -> float:
        """Episode를 처음부터 다시 시작할 때 Agent를 시작 위치로 되돌린다.

        Step 수를 0으로 초기화하고, 시작 위치에서 Raw RSSI를 한 번 새로 측정한다.
        이 값을 current_rssi에 저장한 뒤 반환한다.
        """
        self.agent_position = self.agent_start
        self.step_count = 0
        self.current_rssi = self._generate_rssi()
        return self.current_rssi

    def move(self, action: int) -> None:
        """Q-Learning이 선택한 방향으로 Agent를 한 칸 움직인다.

        Action은 0=위(y-1), 1=아래(y+1), 2=왼쪽(x-1), 3=오른쪽(x+1)이다.
        Grid 밖으로 나가려 하면 위치는 그대로 둔다. 그래도 Action은 수행했으므로
        step_count를 늘리고 같은 위치에서 RSSI를 한 번 새로 측정한다.
        잘못된 Action 번호는 ValueError를 발생시킨다.
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

        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.agent_position = (x, y)

        self.step_count += 1
        self.current_rssi = self._generate_rssi()

    def _calculate_distance(self) -> float:
        """Agent와 Target이 얼마나 떨어져 있는지 내부 위치 정보로 계산한다.

        이동한 경로 길이가 아닌 직선거리(Euclidean Distance)를 사용하며,
        이 거리는 현재 위치의 가상 RSSI를 만들 때 쓰인다.
        """
        agent_x, agent_y = self.agent_position
        target_x, target_y = self.target_position
        return math.hypot(target_x - agent_x, target_y - agent_y)

    def _generate_rssi(self) -> float:
        """현재 Agent와 Target 사이 거리로 가상의 Raw RSSI를 한 번 만든다.

        거리에 따른 감쇠를 적용해 가까울수록 강하고 멀수록 약한 RSSI를 만든다.
        기준 거리 이하에서는 기본 RSSI가 같으며, Noise로 측정값이 달라질 수 있다.

        실제 데이터가 없어 Log-distance Path Loss 형태의 임시 모델을 사용한다.
        Raspberry Pi와 Wi-Fi Adapter로 측정한 뒤 파라미터와 Noise를 보정할 예정이다.
        """
        distance = self._calculate_distance()
        # 같은 위치에서는 거리가 0이지만 log10(0)은 계산할 수 없다.
        # RSSI 계산에 쓸 거리는 최소 reference_distance 이상으로 둔다.
        effective_distance = max(distance, self.reference_distance)
        base_rssi = self.reference_rssi - 10 * self.path_loss_exponent * math.log10(
            effective_distance / self.reference_distance
        )
        # 실제 RSSI는 같은 위치에서도 달라질 수 있어 Gaussian Noise로 흉내 낸다.
        # Noise 크기도 실측 전까지는 Simulation용 임시값이다.
        noise = random.gauss(0.0, self.noise_std)
        return base_rssi + noise

    def get_rssi(self) -> float | None:
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
        """
        return {
            "agent_position": self.agent_position,
            "target_position": self.target_position,
            "distance": self._calculate_distance(),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_rssi": self.current_rssi,
            "success": self.agent_position == self.target_position,
            "done": self.is_done(),
        }
