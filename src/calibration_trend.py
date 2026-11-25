"""시나리오 2만 사용한 Trend 탐색. State/Reward의 기본 임계값을 변경하지 않는다."""
from experiments.field_measurement.analysis.trend import ordered_legs, transitions, accuracy
from experiments.field_measurement.collect import DIRECTIONS
from src.state_reward import _cfg


def trend_report(rows):
    if _cfg("TREND_THRESHOLD") != 2.0:
        raise ValueError("This measured comparison requires the original 2 dB threshold")
    legs = list(ordered_legs(rows))
    measured = [r for repeat, direction, values in legs
                for r in transitions(repeat, direction, DIRECTIONS[direction], values)]
    summary = [{"repeat": repeat, **item} for repeat in (1, 2, "all")
               for item in accuracy([r for r in measured if repeat == "all" or r["repeat"] == repeat])]
    exploration = []
    # 같은 20개 차분을 재사용하므로 독립 검증/최적 임계값 선택이 아니다.
    for threshold in (0.0, 1.0, 2.0, 3.0, 4.0):
        changed = []
        for row in measured:
            delta = row["delta_db"]
            predicted = "up" if delta > threshold else "down" if delta < -threshold else "keep"
            changed.append({**row, "predicted": predicted, "correct": int(predicted == row["expected"])})
        exploration.extend({"threshold_db": threshold, **item} for item in accuracy(changed))
    return {"transitions": measured, "accuracy": summary, "exploratory_thresholds": exploration,
            "retained_threshold_db": 2.0,
            "decision": "Keep 2 dB: only 20 dependent differences, two repeats; no independent tuning set."}
