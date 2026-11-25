"""원본 해시 확인 및 가상 CSV 오류 검증; 실측값 자체는 변경하지 않는다."""
import pytest
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
