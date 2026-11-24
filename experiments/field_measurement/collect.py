"""Collect private, 5-second Beacon samples from a specified 2.4 GHz hotspot."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import uuid

from .beacons import FIELDS, decode, sample_values
from .setup.radio import Radio, frequency, require_linux_root

DISTANCES = (1, 2, 3, 5, 10, 15)
DIRECTIONS = {"approach": (10, 9, 8, 7, 6, 5), "away": (5, 6, 7, 8, 9, 10)}
DURATION = 5
MIN_PACKETS = 10
CONTEXT = ("run_id", "distance_m", "sample_no", "capture_start")
PACKET_COLUMNS = (*CONTEXT, "used_for_stats", *FIELDS)
SUMMARY_COLUMNS = (*CONTEXT, "duration_s", "bssid", "frequency_mhz", "packet_count",
                   "valid_packet_count", "rssi_dbm", "status", "pcapng")


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def covers_sample_window(accepted):
    if len(accepted) < MIN_PACKETS:
        return False
    try:
        times = [float(row["frame.time_relative"]) for row, _ in accepted]
    except ValueError:
        return False
    return (all(math.isfinite(elapsed) and elapsed >= 0 for elapsed in times)
            and all(previous <= current for previous, current in zip(times, times[1:]))
            and times[-1] - times[0] >= DURATION - 1)


def capture(radio, channel, bssid, path, duration):
    before = radio.health(channel)
    command = [
        "tshark", "-i", radio.interface, "-n", "-a", f"duration:{duration}",
        "-f", f"type mgt subtype beacon and wlan addr2 {bssid}", "-F", "pcapng", "-w", "-",
    ]
    with path.open("xb") as output, subprocess.Popen(command, stdout=output) as process:
        try:
            for _ in range(2 * (duration + 5)):
                try:
                    result = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    radio.health(channel)
            else:
                raise RuntimeError("TShark live capture timed out")
        finally:
            if process.poll() is None:
                process.kill()
        if radio.health(channel) & ~before & (1 << 16):
            raise RuntimeError("Undervoltage occurred during capture")
        if result:
            raise subprocess.CalledProcessError(result, command)
    return decode(path, bssid)


def discover(radio, args, probe_dir, hint):
    for channel in (hint, *(c for c in args.channels if c != hint)):
        try:
            radio.tune(channel)
        except subprocess.CalledProcessError:
            radio.health()
            continue
        rows = capture(radio, channel, args.bssid, probe_dir / f"{uuid.uuid4().hex}.pcapng", 1)
        values = sample_values(rows, args.bssid, channel)
        if values and sum(value is not None for value in values) >= 2:
            return channel
    raise RuntimeError("No Beacon-verified hotspot channel found for the specified BSSID")


def csv_writer(stack, path, fields):
    stream = stack.enter_context(path.open("x", newline="", encoding="utf-8"))
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    return writer, stream


def private_output(output):
    base = output.resolve()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    if base.stat().st_mode & 0o077:
        raise RuntimeError("Choose a private output directory with permissions 0700")
    return base


def run(args):
    from contextlib import ExitStack

    radio = Radio(args.interface)
    base = private_output(args.output)
    probe_dir = base / "channel_probes"
    probe_dir.mkdir(exist_ok=True, mode=0o700)
    with radio.monitor():
        power_before = radio.health()
        if power_before & (1 << 16):
            print("Raspberry Pi reports undervoltage earlier in this boot.", flush=True)
        channel = discover(radio, args, probe_dir, args.channel)
        print(f"Beacon-verified channel {channel} (requested {args.channel})", flush=True)
        root = base / f"ch{channel}"
        if args.direction:
            root = root / "trend" / f"repeat_{args.repeat}" / args.direction / f"{args.order:02d}"
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:10]
        folder = root / "scenario1" / run_id
        folder.mkdir(parents=True, mode=0o700)
        metadata = {"interface": args.interface, "channel": channel, "requested_channel": args.channel,
                    "duration_s": DURATION, "min_packets": MIN_PACKETS, "planned_count": args.count,
                    "location_status": args.location_status, "power_mask_start": power_before,
                    "bssid": args.bssid, "distance_m": args.distance, "run_id": run_id}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with ExitStack() as stack:
            packets, packet_file = csv_writer(stack, folder / "packets.csv", PACKET_COLUMNS)
            summary, summary_file = csv_writer(stack, folder / "summary.csv", SUMMARY_COLUMNS)
            for index in range(1, args.count + 1):
                for attempt in (1, 2):
                    name = f"sample_{index:02d}_attempt_{attempt}.pcapng"
                    started = timestamp()
                    rows = capture(radio, channel, args.bssid, folder / name, DURATION)
                    signals = sample_values(rows, args.bssid, channel)
                    accepted = [(row, value) for row, value in zip(rows, signals or [])
                                if value is not None]
                    if covers_sample_window(accepted):
                        break
                    detected = discover(radio, args, probe_dir, channel)
                    if detected != channel:
                        raise RuntimeError(f"Hotspot moved from channel {channel} to {detected}. "
                                           "Start a fresh run and keep channels separate.")
                else:
                    raise RuntimeError(f"Sample {index} lacked enough valid Beacons across the window")
                context = dict(zip(CONTEXT, (run_id, args.distance, index, started)))
                packets.writerows({**context, **row, "used_for_stats": int(value is not None)}
                                  for row, value in zip(rows, signals))
                packet_file.flush()
                summary.writerow({**context, "duration_s": DURATION, "bssid": args.bssid,
                                  "frequency_mhz": frequency(channel), "packet_count": len(rows),
                                  "valid_packet_count": len(accepted), "status": "ok", "pcapng": name,
                                  "rssi_dbm": statistics.median(value for _, value in accepted)})
                summary_file.flush()
                print(f"Saved sample {index}/{args.count}", flush=True)
        power_after = radio.health(channel)
        if power_after & ~power_before & (1 << 16):
            raise RuntimeError("Undervoltage occurred during the run")
    # A capture is ready for preprocessing only after the adapter is restored.
    completion = folder / "completed.json"
    temporary = completion.with_suffix(".tmp")
    temporary.write_text(json.dumps({"power_mask_end": power_after, "completed_at": timestamp()}),
                         encoding="utf-8")
    temporary.replace(completion)


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--bssid", type=str.lower, required=True)
    parser.add_argument("--channel", type=int, choices=range(1, 14), required=True)
    parser.add_argument("--scan-channels", type=int, nargs="+", choices=range(1, 14))
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction", choices=DIRECTIONS)
    parser.add_argument("--repeat", type=int, choices=(1, 2))
    parser.add_argument("--order", type=int, choices=range(1, 7))
    parser.add_argument("--location-status", choices=("provisional", "verified"))
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", args.bssid):
        parser.error("Invalid hotspot BSSID")
    if args.direction or args.repeat or args.order:
        if not (args.direction and args.repeat and args.order):
            parser.error("Supply direction, repeat and order together")
        if args.distance != DIRECTIONS[args.direction][args.order - 1]:
            parser.error("Distance does not match the movement protocol")
        args.count = 1
    else:
        if args.distance not in DISTANCES:
            parser.error("Distance must be one of 1, 2, 3, 5, 10, 15")
        args.count = 15
    args.location_status = args.location_status or ("provisional" if args.distance == 15 else "verified")
    args.channels = tuple(dict.fromkeys(args.scan_channels or range(1, 14)))
    return args


def detach(argv):
    unit = "rssi-field-" + uuid.uuid4().hex[:10]
    subprocess.run([
        "systemd-run", f"--unit={unit}", "--property=Type=exec", "--property=KillMode=mixed",
        f"--working-directory={Path.cwd()}", sys.executable, "-m", __spec__.name,
        *(arg.replace("$", "$$") for arg in argv if arg != "--detach"),
    ], check=True)
    print(f"Status: systemctl status {unit}; log: journalctl -u {unit}")


def terminate(_signum, _frame):
    raise KeyboardInterrupt


def main():
    args = arguments(sys.argv[1:])
    try:
        require_linux_root()
        if args.detach:
            detach(sys.argv[1:])
        else:
            os.umask(0o077)
            signal.signal(signal.SIGTERM, terminate)
            run(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
