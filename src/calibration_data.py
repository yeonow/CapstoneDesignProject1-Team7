"""사용자가 확정한 측정 CSV의 읽기 전용 로더. 수치를 보정하거나 제거하지 않는다."""
import csv
import hashlib
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/rssi_calibration_20261125"
DISTANCES = (1, 2, 3, 5, 10, 15)
EXPECTED_HASHES = {
    "scenario1_provisional.csv": "1d869a86bae7da19feadd2afa4bb9039f5e47b2a279023c3a75ff38d30e40844",
    "scenario1_strict_provisional.csv": "beecdf1aba68311218e980c6d1bb0099b7f41b79f4825ebfd36f48f92bdeba95",
    "scenario2.csv": "db40a41ce0af3002543cb20134b85085dd2f9ea6f6494618352fe898bb0d47aa",
    "measurement_audit.csv": "c5c76d2425db3b0c0bd53db71707d1879122d9d01e66a66e197e7103322bdb33",
}


def source_hashes(directory=DATA):
    return {name: hashlib.sha256((Path(directory) / name).read_bytes()).hexdigest()
            for name in EXPECTED_HASHES}


def read_measurements(path, scenario=1):
    fields = ("distance_m", "sample_no", "rssi_dbm") if scenario == 1 else (
        "repeat", "direction", "measurement_order", "distance_m", "rssi_dbm")
    if scenario not in (1, 2):
        raise ValueError("scenario must be 1 or 2")
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        if not set(fields) <= set(headers) or len(headers) != len(set(headers)):
            raise ValueError("Missing or duplicate measurement columns")
        rows = []
        for line, raw in enumerate(reader, 2):
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f"Malformed CSV row {line}")
            row = {}
            for field in fields:
                if field == "direction":
                    if raw[field] not in ("approach", "away"):
                        raise ValueError(f"Invalid direction at row {line}")
                    row[field] = raw[field]
                    continue
                try:
                    value = float(raw[field])
                except ValueError:
                    raise ValueError(f"Missing or invalid {field} at row {line}") from None
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {field} at row {line}")
                if field in ("sample_no", "repeat", "measurement_order"):
                    if not value.is_integer() or value < 1:
                        raise ValueError(f"Invalid integer {field} at row {line}")
                    value = int(value)
                row[field] = value
            if row["distance_m"] <= 0 or not -128 <= row["rssi_dbm"] < 0:
                raise ValueError(f"Distance/RSSI out of range at row {line}")
            rows.append(row)
    if not rows:
        raise ValueError("Empty measurement CSV")
    return rows
