#!/usr/bin/env python3
"""Prepare the two RSSI field-experiment datasets without modifying raw records.

Input: the ZIP containing ch9/scenario1 and ch9/trend. Uses only Python's
standard library. Run with --15m-run first or last to explicitly select a
complete 15 m run. The latter remains provisional until its location is verified.
"""
import argparse
import csv
import io
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

DISTANCES = (1, 2, 3, 5, 10, 15)
DIRECTIONS = {"approach": (10, 9, 8, 7, 6, 5),
              "away": (5, 6, 7, 8, 9, 10)}


def read_csv(archive, path):
    with archive.open(path) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(archive, fifteen_meter_run):
    names = set(archive.namelist())
    require(archive.testzip() is None, "ZIP integrity check failed")
    # Accept both a directly zipped directory and a ZIP with an outer folder.
    candidates = [n for n in names if n.endswith("ch9/scenario1/summary_all.csv")]
    require(len(candidates) == 1, "Expected exactly one ch9/scenario1/summary_all.csv")
    root = candidates[0][:-len("scenario1/summary_all.csv")]
    audit = []
    validated = set()

    def verify_summary(path, rows):
        """Check aggregate vs per-run summary, CSV counts and representative RSSI."""
        base = path.rsplit("/", 1)[0]
        groups = defaultdict(list)
        for row in rows:
            groups[row["run_id"]].append(row)
        for run, agg in groups.items():
            require(PurePosixPath(run).name == run, "Invalid run ID")
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
            for row in source:
                require(row["duration_s"] == "5" and row["frequency_mhz"] == "2452",
                        f"Unexpected experiment conditions: {folder}")
                require(f"{folder}/{row['pcapng']}" in names,
                        f"Missing capture: {folder}/{row['pcapng']}")
                p = packets[row["sample_no"]]
                good = [x for x in p if x["used_for_stats"] == "1"]
                require(len(p) == int(row["packet_count"]), f"Packet count mismatch: {folder}")
                require(len(good) == int(row["valid_packet_count"]), f"Valid count mismatch: {folder}")
                if good:
                    require(all(x["wlan.fcs.status"] == "1" for x in good),
                            f"Invalid FCS in used packets: {folder}")
                    med = statistics.median(float(x["radiotap.dbm_antsignal"]) for x in good)
                    require(med == float(row["rssi_dbm"]), f"RSSI median mismatch: {folder}")
                else:
                    require(not row["rssi_dbm"], f"Unexpected nonempty RSSI: {folder}")
                expected = "ok" if len(good) >= 10 else (
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
    rows1 = read_csv(archive, path1)
    verify_summary(path1, rows1)
    runs = defaultdict(lambda: defaultdict(list))
    for row in rows1:
        runs[float(row["distance_m"])][row["run_id"]].append(row)
    require(set(runs) == set(DISTANCES), "Unexpected distance set")
    scenario1 = []
    for distance in DISTANCES:
        candidates = runs[distance]
        require(len(candidates) == (2 if distance == 15 else 1),
                f"Unexpected run count at {distance} m; review manually")
        ordered = sorted(candidates, key=lambda k: min(r["capture_start"] for r in candidates[k]))
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
    expected_paths = set()
    for repeat in (1, 2):
        for direction, distances in DIRECTIONS.items():
            for order, distance in enumerate(distances, 1):
                path = root + f"trend/repeat_{repeat}/{direction}/{order:02d}/scenario1/summary_all.csv"
                expected_paths.add(path)
                require(path in names, f"Missing trend point: {path}")
                rows = read_csv(archive, path)
                verify_summary(path, rows)
                require(all(float(r["distance_m"]) == distance and r["sample_no"] == "1" for r in rows),
                        f"Trend point mismatch: {path}")
                # Use the first successful attempt chronologically. Preserve all attempts.
                ok = sorted((r for r in rows if r["status"] == "ok"), key=lambda r: r["capture_start"])
                require(len(ok) >= 1, f"No qualified RSSI at {path}; review before analysis")
                chosen = ok[0]
                for row in rows:
                    selected = row is chosen
                    a = audit_row(row, 2, path, selected,
                                  "first_ok_attempt" if selected else "unselected_attempt",
                                  repeat=repeat, direction=direction, measurement_order=order)
                    audit.append(a)
                    if selected:
                        scenario2.append(a)
    found = {n for n in names if n.startswith(root + "trend/") and n.endswith("/summary_all.csv")}
    require(found == expected_paths, "Unexpected trend directories; review manually")
    require(len(scenario1) == 90 and len(scenario2) == 24, "Unexpected dataset size")
    return scenario1, scenario2, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original experiment ZIP")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory")
    parser.add_argument("--15m-run", dest="fifteen_meter_run", choices=("first", "last"), required=True,
                        help="Select the first or last 15 m run; document the physical location")
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        scenario1, scenario2, audit = prepare(archive, args.fifteen_meter_run)
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
    print(f"Scenario 1: {len(scenario1)} planned rows, {len(strict)} status=ok; 15 m run: {args.fifteen_meter_run} (provisional)")
    print(f"Scenario 2: {len(scenario2)} qualified points")
    print(f"Audit: {len(audit)} attempts, {sum(int(r['valid_packet_count']) for r in audit)} valid packet rows")
    print(f"All statuses: {dict(Counter(r['status'] for r in audit))}")
    print(f"Outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
