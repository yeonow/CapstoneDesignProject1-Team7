"""임시 파일만 사용하는 그림 출처/덮어쓰기 방지 테스트."""

import hashlib
import json

import pytest

from src.graph_artifacts import capture_inputs, reserve_output, write_manifest


def test_manifest_tracks_input_code_options_and_output(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_bytes(b"synthetic test data")
    inputs = capture_inputs([source])
    output = reserve_output(tmp_path / "graphs", "pre_calibration", "test-1")
    artifact = output / "example.txt"
    artifact.write_bytes(b"test output")
    path = write_manifest(output, inputs, {"error_bars": "sample_sd"}, [artifact], [__file__])
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["inputs"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["options"]["error_bars"] == "sample_sd"
    assert result["git"]["sha"] is None or len(result["git"]["sha"]) == 40
    assert result["source_files"] and result["artifacts"][0]["path"] == str(artifact.resolve())
    with pytest.raises(FileExistsError):
        write_manifest(output, inputs, {}, [artifact])
    with pytest.raises(FileExistsError):
        reserve_output(tmp_path / "graphs", "pre_calibration", "test-1")
    assert reserve_output(tmp_path / "graphs", "post_calibration", "test-1") != output


def test_changed_input_or_external_artifact_is_rejected(tmp_path):
    source = tmp_path / "synthetic.csv"
    source.write_bytes(b"before")
    inputs = capture_inputs([source])
    output = reserve_output(tmp_path / "graphs", "pre_calibration", "test")
    source.write_bytes(b"after")
    with pytest.raises(ValueError, match="입력 파일"):
        write_manifest(output, inputs, {}, [])
    with pytest.raises(ValueError, match="산출물"):
        write_manifest(output, capture_inputs([source]), {}, [source])
    assert not (output / "graph_manifest.json").exists()


@pytest.mark.parametrize("name", ["..", "../outside", "a/b", "a\\b", "", "/absolute"])
def test_output_name_cannot_escape_phase(tmp_path, name):
    with pytest.raises(ValueError):
        reserve_output(tmp_path, "pre_calibration", name)
