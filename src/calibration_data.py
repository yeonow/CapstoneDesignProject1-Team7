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


def validate_dataset(directory=DATA):
    """최종 90개, strict 89개, 이동 24개와 감사 기록을 원본 그대로 대조한다."""
    from experiments.field_measurement.analysis.trend import ordered_legs

    directory = Path(directory)
    hashes = source_hashes(directory)
    if hashes != EXPECTED_HASHES:
        raise ValueError("Source SHA-256 mismatch")
    primary = read_measurements(directory / "scenario1_provisional.csv")
    strict = read_measurements(directory / "scenario1_strict_provisional.csv")
    motion = read_measurements(directory / "scenario2.csv", 2)
    expected = {(d, n) for d in DISTANCES for n in range(1, 16)}
    keys = [(r["distance_m"], r["sample_no"]) for r in primary]
    if len(keys) != 90 or set(keys) != expected:
        raise ValueError("Expected 15 unique samples at each distance")
    identity = lambda r: (r["distance_m"], r["sample_no"], r["rssi_dbm"])
    primary_set, strict_set = set(map(identity, primary)), set(map(identity, strict))
    if len(strict) != 89 or len(strict_set) != 89 or not strict_set < primary_set:
        raise ValueError("Strict data must exclude exactly one unchanged primary sample")
    legs = list(ordered_legs(motion))  # 기존 순서/중복/구간 검증 재사용
    with (directory / "measurement_audit.csv").open(encoding="utf-8-sig", newline="") as file:
        audit = list(csv.DictReader(file))
    selected = [r for r in audit if r["selected"] == "1"]
    def audit_identity(row):
        if row["scenario"] == "1":
            return (1, float(row["distance_m"]), int(row["sample_no"]), float(row["rssi_dbm"]))
        return (2, int(row["repeat"]), row["direction"], int(row["measurement_order"]),
                float(row["distance_m"]), float(row["rssi_dbm"]))
    observed = [audit_identity(r) for r in selected]
    expected_selected = {(1, *identity(r)) for r in primary} | {
        (2, r["repeat"], r["direction"], r["measurement_order"], r["distance_m"], r["rssi_dbm"])
        for r in motion}
    if len(observed) != 114 or set(observed) != expected_selected:
        raise ValueError("Selected audit rows disagree with measurements")
    excluded = list(primary_set - strict_set)
    low_quality = [r for r in selected if r["status"] != "ok"]
    if len(low_quality) != 1 or audit_identity(low_quality[0]) != (1, *excluded[0]):
        raise ValueError("Strict exclusion must match selected low-quality audit row")
    return {"hashes": hashes, "primary_count": len(primary), "strict_count": len(strict),
            "motion_count": len(motion), "transition_count": sum(len(v) - 1 for _, _, v in legs),
            "audit_count": len(audit), "selected_count": len(selected),
            "excluded_from_strict": excluded, "selected_quality_warnings": low_quality,
            "frequency_mhz": sorted({float(r["frequency_mhz"]) for r in selected}),
            "policy": "Final 90 including 15 m; strict 89 sensitivity only; scenario2 trend only"}
