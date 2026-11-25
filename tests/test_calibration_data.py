"""원본 해시 확인 및 가상 CSV 오류 검증; 실측값 자체는 변경하지 않는다."""
import pytest

from src.calibration_data import validate_dataset


def test_final_dataset_matches_audit():
    report = validate_dataset()
    assert (report["primary_count"], report["strict_count"], report["transition_count"]) == (90, 89, 20)
    assert report["selected_count"] == 114
    assert report["selected_quality_warnings"][0]["valid_packet_count"] == "9"


def test_changed_source_is_rejected(tmp_path):
    import shutil
    from src.calibration_data import DATA
    for source in DATA.glob("*.csv"):
        shutil.copyfile(source, tmp_path / source.name)
    path = tmp_path / "scenario1_provisional.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_dataset(tmp_path)
from src.calibration_data import DATA, EXPECTED_HASHES, read_measurements, source_hashes


def test_final_source_hashes_and_load():
    assert source_hashes() == EXPECTED_HASHES
    assert len(read_measurements(DATA / "scenario1_provisional.csv")) == 90
    assert len(read_measurements(DATA / "scenario2.csv", 2)) == 24


@pytest.mark.parametrize("value", ["", "NaN", "inf", "broken", "1"])
def test_invalid_rssi(tmp_path, value):
    path = tmp_path / "synthetic.csv"
    path.write_text(f"distance_m,sample_no,rssi_dbm\n1,1,{value}\n")
    with pytest.raises(ValueError):
        read_measurements(path)
