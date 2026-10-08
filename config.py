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

REFERENCE_RSSI = -40.0
REFERENCE_DISTANCE = 1.0
PATH_LOSS_EXPONENT = 2.0
NOISE_STD = 2.0


# =========================
# Experiment Parameters
# =========================

NUM_EPISODES = 500
NUM_EVAL_EPISODES = 100
LOG_INTERVAL = 50
RANDOM_SEED = 42
RESULTS_DIR = "results"
