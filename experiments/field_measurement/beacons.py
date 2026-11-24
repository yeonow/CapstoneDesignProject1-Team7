"""Decode target Beacons and enforce a common per-frame RSSI policy."""

import csv
import io
import subprocess

from .setup.radio import frequency

FIELDS = [
    "frame.number", "frame.time_relative", "wlan.bssid",
    "radiotap.channel.freq", "radiotap.dbm_antsignal",
    "radiotap.flags.fcs", "radiotap.flags.badfcs", "wlan.fcs",
    "wlan.fcs.status", "wlan.ds.current_channel", "wlan.ht.info.primarychannel",
]


def decode(path, bssid):
    command = [
        "tshark", "-r", str(path), "-n", "-o", "wlan.check_checksum:TRUE",
        "-Y", f"wlan.fc.type_subtype == 0x0008 && wlan.bssid == {bssid}",
        "-T", "fields", "-E", "header=y", "-E", "separator=,", "-E", "occurrence=a",
        "-E", "aggregator=;",
    ]
    for field in FIELDS:
        command.extend(("-e", field))
    result = subprocess.check_output(command, text=True, timeout=30)
    reader = csv.DictReader(io.StringIO(result))
    if reader.fieldnames != FIELDS:
        raise ValueError("Unexpected TShark columns")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed TShark CSV")
    return rows


def good_fcs(row, bssid):
    return (row["wlan.bssid"].lower() == bssid
            and row["wlan.fcs.status"] == "1"
            and row["radiotap.flags.fcs"].lower() in {"true", "1"}
            and row["radiotap.flags.badfcs"].lower() in {"false", "0", ""})


def advertised_channels(row):
    return {value.strip() for field in ("wlan.ds.current_channel", "wlan.ht.info.primarychannel")
            for value in row.get(field, "").split(";") if value.strip()}


def beacon_rssi(row, bssid, channel):
    if not (good_fcs(row, bssid) and advertised_channels(row) == {str(channel)}):
        return None
    try:
        if float(row["radiotap.channel.freq"]) == frequency(channel):
            # Preserve the legacy first-antenna reading while collecting all DS/HT IEs.
            value = int(row["radiotap.dbm_antsignal"].split(";", 1)[0])
            if -128 <= value < 0:
                return value
    except ValueError:
        pass
    return None


def sample_values(rows, bssid, channel):
    """Reject an entire capture when a valid target Beacon advertises another channel."""
    if any(good_fcs(row, bssid) and advertised_channels(row) - {str(channel)} for row in rows):
        return None
    return [beacon_rssi(row, bssid, channel) for row in rows]
