from src.calibration_data import DATA, read_measurements
from src.calibration_trend import trend_report


def test_trend_uses_twenty_within_leg_differences_and_keeps_threshold():
    result = trend_report(read_measurements(DATA / "scenario2.csv", 2))
    assert len(result["transitions"]) == 20
    assert len(result["accuracy"]) == 9
    assert result["retained_threshold_db"] == 2
    for row in result["transitions"]:
        assert row["delta_db"] == row["current_rssi_dbm"] - row["previous_rssi_dbm"]
        assert row["to_order"] - row["from_order"] == 1
    expected = [dict(threshold_db=2.0, **{k: v for k, v in r.items() if k != "repeat"})
                for r in result["accuracy"] if r["repeat"] == "all"]
    assert [r for r in result["exploratory_thresholds"] if r["threshold_db"] == 2] == expected
