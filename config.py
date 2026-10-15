# =========================
# Q-Learning Parameters
# =========================

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.9

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


# =========================
# Reward Parameters
# =========================

REWARD_UP = 1.0
REWARD_KEEP = 0.0
REWARD_DOWN = -1.0
MOVE_COST = 0.1

# Target에 도착한 마지막 이동에 추가되는 성공 보상
TERMINAL_REWARD = 10.0

# Grid 경계에 막혀 제자리에 있었던 Action 처리
# 기본값은 일반 Action과 동일하게 처리한다. (Step +1, RSSI 재측정, Previous Action = 시도한 Action)
BLOCKED_APPLY_MOVE_COST = True   # False: 실제로 이동한 경우에만 MOVE_COST 적용
BLOCKED_TREND_KEEP = False       # True: 막혔을 때 Noise와 관계없이 Trend를 KEEP으로 처리
BLOCKED_PENALTY = 0.0            # 막힌 Action에 추가로 주는 penalty (0 이상)


# =========================
# Environment Parameters
# =========================

GRID_SIZE = 10
MAX_STEPS = 100


# =========================
# Action
# =========================

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# 첫 이동 전에는 Previous Action이 없음
NONE_ACTION = -1



# =========================
# Simulation Parameters
# =========================

# Simulation용 임시값
# 실제 RSSI 측정 후 보정 예정
AGENT_START = (0, 0)
TARGET_POSITION = (9, 9)

# 논문 식 (2)의 물리 파라미터를 조절하기 위한 Simulation용 임시값이다.
# 실제 Wi-Fi 실측값이 아니며 Raspberry Pi + Wi-Fi Adapter 측정 후 보정한다.
CELL_SIZE_M = 1.0               # Grid 한 칸을 실제 거리 1 m로 가정 -> 나중에 더 크게 하거나 해야 됨
MIN_DISTANCE_M = 1.0             
CARRIER_FREQUENCY_HZ = 2.4e9    # 2.4 GHz Wi-Fi를 가정한 값(스마트폰 Hotspot이 5 GHz를 쓰면: 5e9 로 수정)
TX_POWER_DBM = 0.0              # 실제 스마트폰 Wi-Fi 송신전력 측정/추정값이 아닌 Simulation용 임시값
# 식 (1)에 필요한 SNR을 만들기 위한 수신기 잡음 전력 가정(dBm)이다.
# 논문에서 직접 주어진 값이 아니며 Raspberry Pi + Wi-Fi Adapter 실측 후 보정한다.
# NOISE_STD(shadow fading 변동 폭)와 달리 신호 대 잡음 전력비의 기준으로 쓴다.
NOISE_FLOOR_DBM = -90.0
ETA_LOS = 2.0                   # 선형 추가 손실 계수; 논문의 eta_nlos > eta_los > 1에 맞춘 임시값
ETA_NLOS = 20.0
PATH_LOSS_EXPONENT = 2.0        # κ: 자유공간에 가까운 거리 감쇠를 가정하는 출발값
MULTIPATH_FADING = 1.0          # nu: 현재 고정 진폭 계수, 1이면 추가 multipath 영향 없음
NOISE_STD = 2.0                 # omega: Gaussian shadow fading의 표준편차(dB)
CHANNEL_STATE = "los"           # 향후 동적 LoS/NLoS 모델로 확장; 이동 blocked와 무관


# =========================
# Experiment Parameters
# =========================

NUM_EPISODES = 500
NUM_EVAL_EPISODES = 100
LOG_INTERVAL = 50
RANDOM_SEED = 42
RESULTS_DIR = "results"
