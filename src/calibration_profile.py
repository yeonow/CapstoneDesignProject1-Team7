"""실측 프로필을 명시적으로 선택한다. 분석 라이브러리는 학습 시 import하지 않는다."""
import json
import math
from pathlib import Path

from src.calibration_data import EXPECTED_HASHES, ROOT, source_hashes


def load_profile(path):
    if path is None:
        raise ValueError("Calibrated mode requires RSSI_CALIBRATION_FILE")
    profile = json.loads((ROOT / Path(path)).read_text(encoding="utf-8"))
    if profile.get("schema_version") != 1 or profile.get("source_sha256") != EXPECTED_HASHES:
        raise ValueError("Unrecognized calibration schema or source provenance")
    if source_hashes() != EXPECTED_HASHES:
        raise ValueError("Calibration source files have changed")
    positive = ("path_loss_exponent", "carrier_frequency_hz")
    for key in (*positive, "reference_rx_power_dbm", "noise_floor_dbm"):
        value = profile.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid calibration parameter: {key}")
        if key in positive and value <= 0:
            raise ValueError(f"Calibration parameter must be positive: {key}")
    if profile.get("reference_distance_m") != 1.0 or profile.get("channel_state") != "los":
        raise ValueError("Only 1m reference LoS calibration is supported")
    return profile
