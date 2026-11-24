#!/usr/bin/env python3
"""Validate and prepare field experiment datasets from a private ZIP or capture directory."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import re
import shutil
import statistics
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.field_measurement.analysis"

from ..beacons import decode, sample_values
from ..collect import DIRECTIONS, DISTANCES, DURATION, MIN_PACKETS
from ..setup.radio import frequency


class CaptureDirectory:
    def __init__(self, root):
        self.root = root

    def namelist(self):
        return [path.relative_to(self.root).as_posix() for path in self.root.rglob("*")
                if path.is_file()]

    def open(self, path):
        return (self.root / path).open("rb")

    def testzip(self):
        return None


def read_csv(archive, path):
    with archive.open(path) as raw:
        rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))
    require(all(None not in row and None not in row.values() for row in rows),
            f"Malformed CSV: {path}")
    return rows


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(archive, fifteen_meter_run, channel=9, verify_captures=False):
    members = archive.namelist()
    names = set(members)
    require(len(names) == len(members), "Duplicate archive paths")
    require(archive.testzip() is None, "ZIP integrity check failed")
    pattern = re.compile(rf"(?P<root>(?:.*/)?ch{channel}/)scenario1/"
                         r"(?:summary_all\.csv|[^/]+/completed\.json)")
    roots = {match["root"] for n in names if (match := pattern.fullmatch(n))}
    require(len(roots) == 1, f"Expected one scenario 1 dataset at channel {channel}")
    root, = roots
    audit = []
    validated = set()
    bssids = set()

    def parse_time(text):
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        require(stamp.utcoffset() is not None, "Capture timestamps need a UTC offset")
        return stamp

    def capture_time(row):
        return parse_time(row["capture_start"])

    def summaries(path):
        base = path.removesuffix("summary_all.csv")
        completed = sorted(n for n in names if n.startswith(base)
                           and n[len(base):].count("/") == 1 and n.endswith("/completed.json"))
        completed_rows = {}
        for marker in completed:
            run = PurePosixPath(marker).parent.name
            with archive.open(marker) as raw:
                completion = json.load(raw)
            require(isinstance(completion, dict)
                    and isinstance(completion.get("completed_at"), str)
                    and type(completion.get("power_mask_end")) is int,
                    f"Invalid completion marker: {marker}")
            power = completion["power_mask_end"]
            require(0 <= power <= 0xFFFFFFFF and not power & 1,
                    f"Invalid completion power state: {marker}")
            rows = read_csv(archive, marker.removesuffix("completed.json") + "summary.csv")
            require(rows and all(row["run_id"] == run for row in rows),
                    f"Completed run summary has missing or mismatched run IDs: {marker}")
            finished = parse_time(completion["completed_at"])
            require(all(capture_time(row) <= finished for row in rows),
                    f"Completion predates capture: {marker}")
            completed_rows[run] = rows
        if path in names:
            aggregate = read_csv(archive, path)
            if completed_rows:
                grouped = defaultdict(list)
                for row in aggregate:
                    grouped[row["run_id"]].append(row)
                require(dict(grouped) == completed_rows,
                        f"Aggregate summary disagrees with completed runs: {path}")
            return aggregate
        require(completed_rows, f"No completed capture at {base}")
        return [row for rows in completed_rows.values() for row in rows]

    def verify_capture(path, bssid, expected):
        with tempfile.TemporaryDirectory(prefix="field-rssi-") as directory:
            local = Path(directory) / "sample.pcapng"
            with archive.open(path) as source, local.open("wb") as target:
                shutil.copyfileobj(source, target)
            rows = decode(local, bssid)
        values = sample_values(rows, bssid, channel)
        require(values is not None, f"Conflicting Beacon channel in PCAPNG: {path}")
        observed = [(r["frame.number"], value) for r, value in zip(rows, values) if value is not None]
        require(observed == expected, f"Decoded PCAPNG differs from packet CSV: {path}")

    def verify_summary(path, rows):
        """Check aggregate vs per-run summary, CSV counts and representative RSSI."""
        base = path.rsplit("/", 1)[0]
        groups = defaultdict(list)
        for row in rows:
            groups[row["run_id"]].append(row)
        for run, agg in groups.items():
            require(re.fullmatch(r"[A-Za-z0-9_.-]+", run) and run not in {".", ".."},
                    "Invalid run ID")
            folder = f"{base}/{run}"
            require(folder not in validated, f"Duplicate run folder: {folder}")
            source = read_csv(archive, f"{folder}/summary.csv")
            require(agg == source, f"Summary mismatch: {folder}")
            packets = defaultdict(list)
            for p in read_csv(archive, f"{folder}/packets.csv"):
                require(p["run_id"] == run, f"Packet run mismatch: {folder}")
                packets[p["sample_no"]].append(p)
            require(set(packets) <= {r["sample_no"] for r in source},
                    f"Unexpected packet sample: {folder}")
            captures = set()
            for row in source:
                capture_time(row)
                require(int(row["duration_s"]) == DURATION
                        and int(row["frequency_mhz"]) == frequency(channel),
                        f"Unexpected experiment conditions: {folder}")
                bssid = row["bssid"].lower()
                require(re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", bssid),
                        "Invalid hotspot BSSID")
                bssids.add(bssid)
                capture = row["pcapng"]
                require(PurePosixPath(capture).name == capture and "\\" not in capture
                        and capture.endswith(".pcapng") and capture not in captures,
                        f"Invalid or duplicate capture filename: {folder}")
                captures.add(capture)
                capture_path = f"{folder}/{capture}"
                require(capture_path in names, f"Missing capture: {capture_path}")
                p = packets[row["sample_no"]]
                require(len({int(x["frame.number"]) for x in p}) == len(p),
                        f"Duplicate packet numbers: {folder}")
                good = []
                values = sample_values(p, bssid, channel)
                require(values is not None, f"Conflicting Beacon channel in packet CSV: {folder}")
                for packet, value in zip(p, values):
                    require(all(packet[k] == row[k] for k in
                                ("sample_no", "capture_start", "distance_m")),
                            f"Packet context mismatch: {folder}")
                    require(packet["used_for_stats"] == str(int(value is not None)),
                            f"Packet FCS, frequency, RSSI or Beacon channel mismatch: {folder}")
                    if value is not None:
                        good.append((packet["frame.number"], value))
                if verify_captures:
                    verify_capture(capture_path, bssid, good)
                require(len(p) == int(row["packet_count"]), f"Packet count mismatch: {folder}")
                require(len(good) == int(row["valid_packet_count"]), f"Valid count mismatch: {folder}")
                if good:
                    med = statistics.median(value for _, value in good)
                    require(med == float(row["rssi_dbm"]), f"RSSI median mismatch: {folder}")
                else:
                    require(not row["rssi_dbm"], f"Unexpected nonempty RSSI: {folder}")
                expected = "ok" if len(good) >= MIN_PACKETS else (
                    "low_packet_count" if good else "no_valid_packets")
                require(row["status"] == expected, f"Status mismatch: {folder}")
            validated.add(folder)

    def audit_row(row, scenario, path, selected, reason, **extra):
        return {"scenario": scenario, **extra, **{k: row[k] for k in (
            "distance_m", "sample_no", "rssi_dbm", "status", "packet_count",
            "valid_packet_count", "run_id", "capture_start", "duration_s", "frequency_mhz")},
                "selected": int(selected), "selection_reason": reason,
                "source_summary": path}

    path1 = root + "scenario1/summary_all.csv"
    rows1 = summaries(path1)
    verify_summary(path1, rows1)
    runs = defaultdict(lambda: defaultdict(list))
    for row in rows1:
        runs[float(row["distance_m"])][row["run_id"]].append(row)
    required_distances = set(DISTANCES) - {15}
    require(required_distances <= runs.keys() <= set(DISTANCES),
            "Scenario 1 requires every planned distance from 1 to 10 m; 15 m is optional")
    scenario1 = []
    for distance in DISTANCES:
        if distance not in runs:
            continue
        candidates = runs[distance]
        require(distance == 15 or len(candidates) == 1,
                f"Unexpected run count at {distance} m; review manually")
        ordered = sorted(candidates, key=lambda k: (min(map(capture_time, candidates[k])), k))
        chosen = ordered[-1] if distance == 15 and fifteen_meter_run == "last" else ordered[0]
        for run in ordered:
            group = sorted(candidates[run], key=lambda r: int(r["sample_no"]))
            require([int(r["sample_no"]) for r in group] == list(range(1, 16)),
                    f"Incomplete or duplicate measurement numbers: {run}")
            for row in group:
                selected = run == chosen
                reason = "selected" if selected else "alternate_15m_run"
                if selected and distance == 15:
                    reason = "provisional_15m_" + fifteen_meter_run
                a = audit_row(row, 1, path1, selected, reason)
                audit.append(a)
                if selected:
                    scenario1.append(a)

    scenario2 = []
    previous_start = None
    for repeat in (1, 2):
        for direction, distances in DIRECTIONS.items():
            for order, distance in enumerate(distances, 1):
                path = root + f"trend/repeat_{repeat}/{direction}/{order:02d}/scenario1/summary_all.csv"
                rows = summaries(path)
                verify_summary(path, rows)
                require(all(float(r["distance_m"]) == distance and r["sample_no"] == "1" for r in rows),
                        f"Trend point mismatch: {path}")
                # Use the first successful attempt chronologically. Preserve all attempts.
                ok = sorted((r for r in rows if r["status"] == "ok"),
                            key=lambda r: (capture_time(r), r["run_id"]))
                require(len(ok) >= 1, f"No qualified RSSI at {path}; review before analysis")
                chosen = ok[0]
                started = capture_time(chosen)
                require(previous_start is None or previous_start <= started,
                        f"Scenario 2 capture timestamps are out of order at {path}")
                previous_start = started
                for row in rows:
                    selected = row is chosen
                    a = audit_row(row, 2, path, selected,
                                  "first_ok_attempt" if selected else "unselected_attempt",
                                  repeat=repeat, direction=direction, measurement_order=order)
                    audit.append(a)
                    if selected:
                        scenario2.append(a)
    evidence = re.compile(r"^repeat_[12]/(?:approach|away)/0[1-6]/scenario1/"
                          r"(?:summary_all\.csv|[^/]+/completed\.json)$")
    require(all(evidence.fullmatch(n[len(root + "trend/"):]) for n in names
                if n.startswith(root + "trend/")
                and n.endswith(("/summary_all.csv", "/completed.json"))),
            "Unexpected trend directories; review manually")
    require(len(bssids) == 1, "Multiple hotspot BSSIDs in the selected channel dataset")
    require(len(scenario1) == 15 * len(runs) and len(scenario2) == 24, "Unexpected dataset size")
    return scenario1, scenario2, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original experiment ZIP or capture directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory")
    parser.add_argument("--15m-run", dest="fifteen_meter_run", choices=("first", "last"), default="first",
                        help="Select the first or last 15 m run; document the physical location")
    parser.add_argument("--channel", type=int, choices=range(1, 14), default=9)
    parser.add_argument("--verify-captures", action="store_true",
                        help="Re-decode PCAPNG files with TShark and compare valid frames")
    args = parser.parse_args()
    if args.archive.is_dir():
        scenario1, scenario2, audit = prepare(CaptureDirectory(args.archive), args.fifteen_meter_run,
                                              args.channel, args.verify_captures)
        source_sha256 = None
    else:
        with zipfile.ZipFile(args.archive) as archive:
            scenario1, scenario2, audit = prepare(archive, args.fifteen_meter_run,
                                                  args.channel, args.verify_captures)
        source_sha256 = file_sha256(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "scenario1_provisional.csv",
              ["distance_m", "sample_no", "rssi_dbm"], scenario1)
    write_csv(args.output / "scenario2.csv",
              ["repeat", "direction", "measurement_order", "distance_m", "rssi_dbm"], scenario2)
    write_csv(args.output / "measurement_audit.csv", ["scenario", "repeat", "direction", "measurement_order",
              "distance_m", "sample_no", "rssi_dbm", "status", "packet_count", "valid_packet_count",
              "run_id", "capture_start", "duration_s", "frequency_mhz", "selected", "selection_reason",
              "source_summary"], audit)
    # Strict subset is useful for sensitivity analysis; retain the original sample numbers.
    strict = [r for r in scenario1 if r["status"] == "ok"]
    write_csv(args.output / "scenario1_strict_provisional.csv",
              ["distance_m", "sample_no", "rssi_dbm"], strict)
    primary = [r for r in strict if float(r["distance_m"]) != 15]
    provisional_run = (args.fifteen_meter_run
                       if any(float(r["distance_m"]) == 15 for r in scenario1) else None)
    write_csv(args.output / "scenario1_primary.csv",
              ["distance_m", "sample_no", "rssi_dbm"], primary)
    (args.output / "metadata.json").write_text(json.dumps({
        "archive_sha256": source_sha256, "channel": args.channel, "duration_s": DURATION,
        "min_packets": MIN_PACKETS, "verified_pcapng": args.verify_captures,
        "provisional_15m_run": provisional_run, "primary_count": len(primary),
        "analysis_input_sha256": {
            name: file_sha256(args.output / name)
            for name in ("scenario1_strict_provisional.csv", "scenario2.csv")
        },
    }, indent=2), encoding="utf-8")
    fifteen_meter_status = f"{provisional_run} (provisional)" if provisional_run else "unavailable"
    print(f"Scenario 1: {len(scenario1)} prepared rows, {len(strict)} status=ok; 15 m run: {fifteen_meter_status}")
    print(f"Scenario 2: {len(scenario2)} qualified points")
    print(f"Audit: {len(audit)} attempts, {sum(int(r['valid_packet_count']) for r in audit)} valid packet rows")
    print(f"All statuses: {dict(Counter(r['status'] for r in audit))}")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
