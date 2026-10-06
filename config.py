# =========================
# Q-Learning Parameters
# =========================

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.9

EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

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
LOG_INTERVAL = 50
RANDOM_SEED = 42
RESULTS_DIR = "results"