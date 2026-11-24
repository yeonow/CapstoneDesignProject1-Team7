import csv
import json
import random
import statistics
import sys
import zipfile

import pytest

from experiments.field_measurement import collect
from experiments.field_measurement.analysis import analyze_field_data as analysis
from experiments.field_measurement.analysis import prepare_field_data as preparation
from experiments.field_measurement.analysis.trend import transitions

from test_collection import BSSID, beacon


@pytest.mark.parametrize("content,message", [
    (b"", "Invalid CSV headers"),
    (b"rssi_dbm,rssi_dbm\n-55,-70\n", "Invalid CSV headers"),
    (b"distance_m,\n1,-55\n", "Invalid CSV headers"),
    (b'distance_m,rssi_dbm\n1,"-55\n', "Malformed CSV"),
])
def test_raw_and_prepared_csv_reject_invalid_structure(tmp_path, content, message):
    path = tmp_path / "broken.csv"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message):
        preparation.read_csv(preparation.CaptureDirectory(tmp_path), path.name)
    with pytest.raises(ValueError, match=message):
        analysis.load_csv(path)


def source_run(folder, run, distance, count, intercept, day):
    directory = folder / run
    directory.mkdir(parents=True)
    packets, summary = [], []
    for index in range(1, count + 1):
        context = {
            "run_id": run, "distance_m": distance, "sample_no": index,
            "capture_start": f"2026-01-{day:02d}T00:00:{index:02d}+00:00",
        }
        rows = [beacon(6, n + 1, n / 2, intercept - n % 3 - index % 2) for n in range(10)]
        packets.extend({**context, **row, "used_for_stats": 1} for row in rows)
        name = f"sample_{index:02d}.pcapng"
        (directory / name).write_bytes(b"synthetic placeholder")
        summary.append({
            **context, "duration_s": 5, "bssid": BSSID, "frequency_mhz": 2437,
            "packet_count": 10, "valid_packet_count": 10, "status": "ok", "pcapng": name,
            "rssi_dbm": statistics.median(int(row["radiotap.dbm_antsignal"]) for row in rows),
        })
    preparation.write_csv(directory / "packets.csv", collect.PACKET_COLUMNS, packets)
    preparation.write_csv(directory / "summary.csv", collect.SUMMARY_COLUMNS, summary)
    (directory / "completed.json").write_text(json.dumps({
        "completed_at": f"2026-01-{day:02d}T00:01:00+00:00", "power_mask_end": 0,
    }), encoding="utf-8")
    return directory


@pytest.fixture
def dataset(tmp_path):
    base = tmp_path / "raw" / "ch6"
    static = base / "scenario1"
    for distance in collect.DISTANCES:
        source_run(static, f"d{distance}", distance, 15, -30 - distance, 1)
    source_run(static, "d15_later", 15, 15, -70, 2)
    (static / "interrupted").mkdir()
    (static / "interrupted" / "summary.csv").write_text("incomplete", encoding="utf-8")
    for repeat in (1, 2):
        for direction, distances in collect.DIRECTIONS.items():
            for order, distance in enumerate(distances, 1):
                folder = base / "trend" / f"repeat_{repeat}" / direction / f"{order:02d}" / "scenario1"
                source_run(folder, "run", distance, 1, -30 - distance * 3, 2 + repeat)
    return base.parent


@pytest.mark.parametrize("field,value,error", [
    ("radiotap.channel.freq", "2452", "FCS, frequency"),
    ("wlan.ds.current_channel", "8", "Conflicting Beacon channel"),
    ("capture_start", "2026-01-03T00:00:00+00:00", "Packet context"),
    ("radiotap.dbm_antsignal", "nan", "FCS, frequency"),
])
def test_invalid_source_packets(dataset, field, value, error):
    path = dataset / "ch6/scenario1/d1/packets.csv"
    rows = analysis.load_csv(path)
    rows[0][field] = value
    preparation.write_csv(path, collect.PACKET_COLUMNS, rows)
    with pytest.raises(ValueError, match=error):
        preparation.prepare(preparation.CaptureDirectory(dataset), "last", channel=6)


def test_unused_conflicting_beacon_invalidates_entire_sample(dataset):
    path = dataset / "ch6/scenario1/d1/packets.csv"
    rows = analysis.load_csv(path)
    rows.append({**rows[0], **beacon(channel=9, number=11), "used_for_stats": "0"})
    preparation.write_csv(path, collect.PACKET_COLUMNS, rows)
    with pytest.raises(ValueError, match="Conflicting Beacon channel"):
        preparation.prepare(preparation.CaptureDirectory(dataset), "last", channel=6)


@pytest.mark.parametrize("change,message", [
    ("empty_summary", "missing or mismatched run IDs"),
    ("wrong_run", "missing or mismatched run IDs"),
    ("early_completion", "Completion predates capture"),
    ("stale_aggregate", "Aggregate summary disagrees"),
])
def test_completed_run_provenance(dataset, change, message):
    scenario = dataset / "ch6/scenario1"
    summary = scenario / "d1/summary.csv"
    rows = analysis.load_csv(summary)
    if change == "empty_summary":
        preparation.write_csv(summary, collect.SUMMARY_COLUMNS, [])
    elif change == "wrong_run":
        rows[0]["run_id"] = "different"
        preparation.write_csv(summary, collect.SUMMARY_COLUMNS, rows)
    elif change == "early_completion":
        summary.with_name("completed.json").write_text(json.dumps({
            "completed_at": "2026-01-01T00:00:00+00:00", "power_mask_end": 0,
        }), encoding="utf-8")
    else:
        preparation.write_csv(scenario / "summary_all.csv", collect.SUMMARY_COLUMNS, rows)
    with pytest.raises(ValueError, match=message):
        preparation.prepare(preparation.CaptureDirectory(dataset), "last", channel=6)


def test_primary_analysis_can_run_before_15m_is_available(dataset, tmp_path, monkeypatch):
    static = dataset / "ch6/scenario1"
    for run in ("d15", "d15_later"):
        (static / run / "completed.json").unlink()
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", str(dataset), "--output", str(prepared), "--channel", "6"])
    preparation.main()
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["provisional_15m_run"] is None
    source = analysis.load_csv(prepared / "scenario1_strict_provisional.csv")
    assert len(source) == 75
    assert list(analysis.qualified_values(source, False)) == list(collect.DISTANCES[:-1])
    with pytest.raises(ValueError, match="No qualified provisional 15 m samples"):
        analysis.qualified_values(source, True)
    (static / "d10" / "completed.json").unlink()
    with pytest.raises(ValueError, match="every planned distance"):
        preparation.prepare(preparation.CaptureDirectory(dataset), "first", channel=6)


def test_movement_requires_chronological_capture_order(dataset):
    run = dataset / "ch6/trend/repeat_1/away/01/scenario1/run"
    for filename, columns in (("summary.csv", collect.SUMMARY_COLUMNS),
                              ("packets.csv", collect.PACKET_COLUMNS)):
        path = run / filename
        rows = analysis.load_csv(path)
        for row in rows:
            row["capture_start"] = "2026-01-02T00:00:01+00:00"
        preparation.write_csv(path, columns, rows)
    with pytest.raises(ValueError, match="capture timestamps are out of order"):
        preparation.prepare(preparation.CaptureDirectory(dataset), "first", channel=6)


def test_pipeline_with_directory_and_zip(dataset, tmp_path, monkeypatch):
    raw_zip = tmp_path / "raw.zip"
    with zipfile.ZipFile(raw_zip, "w") as archive:
        for entry in dataset.rglob("*"):
            if entry.is_file():
                archive.write(entry, f"capture/{entry.relative_to(dataset).as_posix()}")
    with zipfile.ZipFile(raw_zip) as archive:
        static, motion, audit = preparation.prepare(archive, "last", channel=6)
    assert (len(static), len(motion), len(audit)) == (90, 24, 129)
    assert {row["run_id"] for row in static if row["distance_m"] == "15"} == {"d15_later"}

    prepared = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", str(dataset), "--output", str(prepared), "--channel", "6"])
    preparation.main()
    assert len(analysis.load_csv(prepared / "scenario1_primary.csv")) == 75
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["analyze", str(prepared), "--output", str(output), "--draws", "20"])
    random_state = random.getstate()
    analysis.main()
    assert random.getstate() == random_state
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["path_loss"]["count"] == 75
    assert report["trend_accuracy"][-1]["total"] == 20
    assert report["trend_accuracy"][-1]["correct"] == 20
    assert report["simulation"]["draws"] == 20
    assert len(analysis.load_csv(output / "simulation_distance_comparison.csv")) == 5
    for name in ("distance_rssi", "gaussian_noise", "movement_trend", "simulation_comparison"):
        assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")


def test_log_distance_and_threshold_boundaries():
    fit, residuals = analysis.path_loss({1: [-30, -30], 10: [-50, -50], 100: [-70, -70]})
    assert fit["exponent"] == pytest.approx(2)
    assert fit["reference_rssi_dbm"] == pytest.approx(-30)
    assert max(map(abs, residuals)) == pytest.approx(0)
    motion = list(transitions(1, "approach", (10, 9, 8, 7), [-65, -63, -60, -62]))
    assert [row["predicted"] for row in motion] == ["keep", "up", "keep"]


def test_analysis_rejects_changed_csv_before_writing_results(dataset, tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", str(dataset), "--output", str(prepared), "--channel", "6"])
    preparation.main()
    with (prepared / "scenario2.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["analyze", str(prepared), "--output", str(output)])
    with pytest.raises(ValueError, match="do not match metadata"):
        analysis.main()
    assert not output.exists()


def test_analysis_uses_the_bytes_it_hashed(dataset, tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", str(dataset), "--output", str(prepared), "--channel", "6"])
    preparation.main()
    source = prepared / "scenario1_strict_provisional.csv"
    hashes = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))["analysis_input_sha256"]
    read_bytes = type(source).read_bytes

    def read_and_replace(path):
        content = read_bytes(path)
        if path.name == "scenario2.csv":
            source.write_bytes(b"replacement after the initial read")
        return content

    monkeypatch.setattr(type(source), "read_bytes", read_and_replace)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["analyze", str(prepared), "--output", str(output), "--draws", "20"])
    analysis.main()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["path_loss"]["count"] == 75
    assert report["scenario1_sha256"] == hashes[source.name]
    assert source.read_bytes() == b"replacement after the initial read"
