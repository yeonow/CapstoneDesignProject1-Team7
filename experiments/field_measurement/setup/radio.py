"""Manage an explicitly selected Raspberry Pi USB monitor adapter."""

import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path


def command(*args, timeout=15):
    return subprocess.check_output(args, text=True, timeout=timeout)


def frequency(channel):
    return 2407 + 5 * channel


class Radio:
    def __init__(self, interface, sysfs=Path("/sys/class/net")):
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,14}", interface):
            raise ValueError("Invalid interface name")
        self.interface = interface
        self.sysfs = sysfs
        self.device = self.usb_device()

    def usb_device(self):
        try:
            device = (self.sysfs / self.interface / "device").resolve(strict=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Adapter {self.interface} is missing or disconnected") from exc
        for parent in (device, *device.parents):
            if (parent / "idVendor").is_file() and (parent / "idProduct").is_file():
                return (parent, (parent / "busnum").read_text().strip(),
                        (parent / "devnum").read_text().strip())
        raise RuntimeError(f"{self.interface} is not backed by a USB adapter")

    def ip(self, *args):
        return command("ip", "link", "set", "dev", self.interface, *args)

    def iw(self, *args):
        return command("iw", "dev", self.interface, *args)

    def state(self):
        info = self.iw("info")
        mode = re.search(r"^\s*type\s+(\w+)\s*$", info, re.M)
        channel = re.search(r"channel\s+(\d+)\s+\((\d+)\s*MHz\)", info)
        link, = json.loads(command("ip", "-j", "link", "show", "dev", self.interface))
        if not mode:
            raise RuntimeError("Cannot determine wireless interface mode")
        return mode[1], "UP" in link["flags"], (
            tuple(map(int, channel.groups())) if channel else None
        )

    def power(self):
        result = re.fullmatch(r"throttled=(0x[0-9a-f]+)",
                              command("vcgencmd", "get_throttled").strip(), re.I)
        if not result:
            raise RuntimeError("Cannot read Raspberry Pi undervoltage status")
        value = int(result[1], 16)
        # Bit 0 is current undervoltage, bit 16 is sticky since boot.
        if value & 1:
            raise RuntimeError(f"Active undervoltage (get_throttled=0x{value:x})")
        return value

    def health(self, channel=None):
        if self.usb_device() != self.device:
            raise RuntimeError("USB adapter identity changed")
        power = self.power()
        mode, up, tuned = self.state()
        if channel is not None and (mode, up, tuned) != (
            "monitor", True, (channel, frequency(channel))
        ):
            raise RuntimeError(f"Monitor interface is down or mistuned (expected channel {channel})")
        return power

    def tune(self, channel):
        self.iw("set", "channel", str(channel), "HT20")
        return self.health(channel)

    @contextmanager
    def monitor(self):
        """Temporarily take only a dedicated, unassociated USB adapter."""
        import fcntl

        lock_dir = Path("/run/lock/rssi-field")
        lock_dir.mkdir(mode=0o700, exist_ok=True)
        with (lock_dir / self.interface).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another field capture owns this adapter") from exc
            self.health()
            mode, up, tuned = self.state()
            if mode not in {"managed", "monitor"}:
                raise RuntimeError(f"Unsupported initial interface mode: {mode}")
            if mode == "managed" and not self.iw("link").startswith("Not connected"):
                raise RuntimeError("Select a dedicated USB adapter with no active connection")
            nm_state = command("nmcli", "-g", "GENERAL.STATE", "device", "show", self.interface)
            managed = not nm_state.startswith("10 ")
            if managed and not nm_state.startswith(("20 ", "30 ")):
                raise RuntimeError("NetworkManager reports a connection or transition on the adapter")
            try:
                if managed:
                    command("nmcli", "device", "set", self.interface, "managed", "no")
                if mode != "monitor":
                    self.ip("down")
                    self.iw("set", "type", "monitor")
                self.ip("up")
                yield
            finally:
                if self.usb_device() != self.device:
                    raise RuntimeError("Cannot restore the original USB adapter")
                try:
                    if mode != "monitor":
                        self.ip("down")
                        self.iw("set", "type", mode)
                    elif tuned:
                        self.iw("set", "freq", str(tuned[1]))
                    self.ip("up" if up else "down")
                finally:
                    if managed:
                        command("nmcli", "device", "set", self.interface, "managed", "yes")


def require_linux_root():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError("Run this command with sudo on Raspberry Pi OS")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interface")
    args = parser.parse_args()
    try:
        radio = Radio(args.interface)
        print(f"USB bus/device: {radio.device[1:]}, state: {radio.state()}, "
              f"get_throttled=0x{radio.health():x}")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")
