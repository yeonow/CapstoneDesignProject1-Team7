from contextlib import contextmanager
import csv
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.field_measurement import collect
from experiments.field_measurement.beacons import FIELDS, beacon_rssi
from experiments.field_measurement.setup import radio

BSSID = "02:00:00:00:00:01"


def beacon(channel=9, number=1, time=0, rssi=-60):
    return dict(zip(FIELDS, map(str, (number, time, BSSID, radio.frequency(channel), rssi,
                                     "True", "False", "0x12345678", 1, channel, channel))))


@pytest.mark.parametrize("field,value", [
    ("wlan.fcs.status", "2"), ("radiotap.flags.fcs", "0"), ("radiotap.flags.badfcs", "1"),
    ("radiotap.channel.freq", "2447"), ("wlan.ds.current_channel", "9;8"),
    ("wlan.ht.info.primarychannel", "8"), ("radiotap.dbm_antsignal", "nan"),
    ("radiotap.dbm_antsignal", "0"), ("wlan.bssid", "02:00:00:00:00:02"),
])
def test_frame_policy(field, value):
    assert beacon_rssi(beacon(), BSSID, 9) == -60
    row = {**beacon(), field: value}
    assert beacon_rssi(row, BSSID, 9) is None


def test_conflicting_beacon_invalidates_capture():
    assert collect.sample_values([beacon(), beacon(channel=6)], BSSID, 9) is None


def test_multi_antenna_frames_keep_legacy_first_signal():
    row = {**beacon(), "radiotap.dbm_antsignal": "-60;-56;-59"}
    assert beacon_rssi(row, BSSID, 9) == -60


@pytest.mark.parametrize("time", ["", "nan", "inf", -1, 0.25, 9])
def test_invalid_timestamps_do_not_qualify_a_capture(time):
    rows = [(beacon(time=index / 2), -60) for index in range(10)]
    assert collect.covers_sample_window(rows)
    rows[3][0]["frame.time_relative"] = str(time)
    assert not collect.covers_sample_window(rows)


def test_compact_packet_burst_does_not_cover_a_full_sample():
    rows = [(beacon(time=index / 10), -60) for index in range(10)]
    assert not collect.covers_sample_window(rows)


def test_adapter_identity_and_power(tmp_path, monkeypatch):
    adapter = tmp_path / "wlan1" / "device"
    adapter.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="USB"):
        radio.Radio("wlan1", tmp_path)
    for name, value in {"idVendor": "abcd", "idProduct": "ef01",
                        "busnum": "1", "devnum": "3"}.items():
        (adapter / name).write_text(value)
    device = radio.Radio("wlan1", tmp_path)
    monkeypatch.setattr(radio, "command", lambda *args: "throttled=0x10000")
    monkeypatch.setattr(device, "state", lambda: ("monitor", True, (9, 2452)))
    assert device.health(9) == 0x10000
    with pytest.raises(RuntimeError, match="mistuned"):
        device.health(6)
    monkeypatch.setattr(radio, "command", lambda *args: "throttled=0x10001")
    with pytest.raises(RuntimeError, match="undervoltage"):
        device.health()
    (adapter / "devnum").write_text("4")
    with pytest.raises(RuntimeError, match="identity changed"):
        device.health()
    (adapter / "devnum").unlink()
    with pytest.raises(OSError):
        device.health()


def test_monitor_restores_adapter_on_capture_failure(tmp_path, monkeypatch):
    device = radio.Radio.__new__(radio.Radio)
    device.interface, device.device = "wlan1", "usb"
    calls = []
    monkeypatch.setattr(radio, "LOCK_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "fcntl",
                        SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(device, "health", lambda: 0)
    monkeypatch.setattr(device, "usb_device", lambda: "usb")
    monkeypatch.setattr(device, "state", lambda: ("managed", False, None))
    monkeypatch.setattr(device, "ip", lambda *args: calls.append(("ip", *args)))

    def invoke(*args):
        calls.append(args)
        return "30 (disconnected)"

    def iw(*args):
        calls.append(("iw", *args))
        return "Not connected.\n"

    monkeypatch.setattr(radio, "command", invoke)
    monkeypatch.setattr(device, "iw", iw)
    with pytest.raises(RuntimeError, match="capture failed"):
        with device.monitor():
            raise RuntimeError("capture failed")
    assert calls[-8:] == [
        ("nmcli", "device", "set", "wlan1", "managed", "no"),
        ("ip", "down"), ("iw", "set", "type", "monitor"), ("ip", "up"),
        ("ip", "down"), ("iw", "set", "type", "managed"), ("ip", "down"),
        ("nmcli", "device", "set", "wlan1", "managed", "yes"),
    ]


@pytest.mark.parametrize("blocker,message", [
    ("lock", "Another field capture owns this adapter"),
    ("association", "no active connection"),
    ("network_manager", "connection or transition"),
])
def test_monitor_refuses_busy_interfaces_without_changing_them(tmp_path, monkeypatch, blocker, message):
    device = radio.Radio.__new__(radio.Radio)
    device.interface = "wlan1"
    calls = []

    def lock(*args):
        if blocker == "lock":
            raise BlockingIOError

    def iw(*args):
        calls.append(("iw", *args))
        return "Connected to target" if blocker == "association" else "Not connected."

    def invoke(*args):
        calls.append(args)
        return "40 (connecting)"

    monkeypatch.setattr(radio, "LOCK_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(flock=lock, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(radio, "command", invoke)
    monkeypatch.setattr(device, "health", lambda: 0)
    monkeypatch.setattr(device, "state", lambda: ("managed", True, None))
    monkeypatch.setattr(device, "iw", iw)
    monkeypatch.setattr(device, "ip", lambda *args: calls.append(("ip", *args)))
    with pytest.raises(RuntimeError, match=message), device.monitor():
        pytest.fail("A busy interface was selected")

    allowed = [("iw", "link"), ("nmcli", "-g", "GENERAL.STATE", "device", "show", "wlan1")]
    assert calls == allowed[:{"lock": 0, "association": 1, "network_manager": 2}[blocker]]


def options(tmp_path):
    return ["--interface", "wlan1", "--bssid", BSSID.upper(), "--channel", "9",
            "--scan-channels", "6", "9", "--distance", "10", "--output", str(tmp_path),
            "--direction", "approach", "--repeat", "1", "--order", "1"]


def test_reject_wrong_motion_distance(tmp_path):
    argv = options(tmp_path)
    argv[-1] = "2"
    with pytest.raises(SystemExit):
        collect.arguments(argv)


def test_detached_service_has_no_ssh_terminal(tmp_path, monkeypatch):
    argv = options(tmp_path)
    argv[argv.index("--output") + 1] = str(tmp_path / "$private data")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    collect.detach([*argv, "--detach"])
    command, = commands
    assert command[0] == "systemd-run"
    assert "--property=KillMode=mixed" in command
    assert "--detach" not in command
    assert command[command.index("--output") + 1].endswith("$$private data")


@pytest.mark.parametrize("failure", [None, "channel_change", "undervoltage", "restore", "short_window"])
def test_collection_with_fake_radio(tmp_path, monkeypatch, capsys, failure):
    args = collect.arguments(options(tmp_path))
    tuned = []
    power_masks = iter((0, 0x10000 if failure == "undervoltage" else 0))
    @contextmanager
    def monitor():
        yield
        assert not list(tmp_path.rglob("completed.json"))
        if failure == "restore":
            raise RuntimeError("Adapter restoration failed")

    adapter = SimpleNamespace(monitor=monitor, tune=tuned.append, health=lambda *_: next(power_masks))
    monkeypatch.setattr(collect, "Radio", lambda interface: adapter)
    monkeypatch.setattr(collect, "private_output", lambda output: output)

    def capture(_radio, channel, _bssid, path, duration):
        path.touch()
        # Requested channel 9 is stale. In the failure case the hotspot returns
        # there during the five-second sample.
        if duration == 5 and failure == "channel_change":
            return [beacon(channel=9)]
        if duration == 5 and failure == "short_window" and path.stem.endswith("_attempt_1"):
            return [beacon(channel, index + 1, index / 10) for index in range(10)]
        actual = 9 if failure == "channel_change" and any(tmp_path.rglob("sample*.pcapng")) else 6
        if channel == actual:
            return [beacon(channel, index + 1, index / 2, -60 - index) for index in range(10)]
        return []

    monkeypatch.setattr(collect, "capture", capture)
    if failure and failure != "short_window":
        error = {"channel_change": "from channel 6 to 9", "undervoltage": "Undervoltage occurred",
                 "restore": "Adapter restoration failed"}[failure]
        with pytest.raises(RuntimeError, match=error):
            collect.run(args)
        assert not list(tmp_path.rglob("completed.json"))
    else:
        collect.run(args)
        completed, = tmp_path.rglob("completed.json")
        with (completed.parent / "summary.csv").open(newline="", encoding="utf-8") as stream:
            row, = list(csv.DictReader(stream))
        assert row["frequency_mhz"] == "2437"
        assert row["rssi_dbm"] == "-64.5"
        if failure == "short_window":
            assert row["pcapng"].endswith("_attempt_2.pcapng")
            assert (completed.parent / "sample_01_attempt_1.pcapng").is_file()
    logs = capsys.readouterr().err
    if failure == "short_window":
        assert "attempt 1/2: invalid timestamps or insufficient time coverage" in logs
    if failure == "channel_change":
        assert "conflicting channels advertised by target Beacons" in logs
    assert tuned[:2] == [9, 6]
