"""Q-Learning과 Random 탐색이 함께 쓰는 RSSI Grid 환경 생성 함수.

두 알고리즘이 같은 config로 환경을 만들도록 생성 코드를 한 곳에 둔다.
Random 실행은 Q-Learning/State/Reward 모듈 없이도 동작해야 하므로
이 모듈은 environment와 config만 import한다.
"""

import config
from src.environment import RSSIGridEnv


def build_environment(noise_std=None):
    """config 설정과 선택적 Noise override로 환경을 생성한다."""
    return RSSIGridEnv(
        grid_size=config.GRID_SIZE,
        max_steps=config.MAX_STEPS,
        agent_start=config.AGENT_START,
        target_position=config.TARGET_POSITION,
        cell_size_m=config.CELL_SIZE_M,
        carrier_frequency_hz=config.CARRIER_FREQUENCY_HZ,
        tx_power_dbm=config.TX_POWER_DBM,
        eta_los=config.ETA_LOS,
        eta_nlos=config.ETA_NLOS,
        path_loss_exponent=config.PATH_LOSS_EXPONENT,
        multipath_fading=config.MULTIPATH_FADING,
        noise_std=config.NOISE_STD if noise_std is None else noise_std,
        min_distance_m=config.MIN_DISTANCE_M,
        noise_floor_dbm=config.NOISE_FLOOR_DBM,
        channel_state=config.CHANNEL_STATE,
    )
