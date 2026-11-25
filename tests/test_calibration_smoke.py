import json
import random

import pytest

import config
from src.calibration_data import DATA, EXPECTED_HASHES
from src.calibration_smoke import run_smoke, temporary_settings, paired_targets
from src.environment import RSSIGridEnv


def test_paired_smoke_repeats_and_uses_calibrated_learning(tmp_path, monkeypatch):
    profile = dict(schema_version=1, source_sha256=EXPECTED_HASHES, reference_distance_m=1,
                   channel_state="los", reference_rx_power_dbm=-60, path_loss_exponent=1.6,
                   carrier_frequency_hz=2452e6, noise_floor_dbm=-90)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    settings = {k: v for k, v in vars(config).items() if k.isupper()}
    rng = random.getstate()
    seen = []
    original = RSSIGridEnv._generate_rssi
    def track(env):
        seen.append(env.calibrated_reference_rx_power_dbm)
        return original(env)
    monkeypatch.setattr(RSSIGridEnv, "_generate_rssi", track)
    first = run_smoke(path, tmp_path / "first", 1, 2)
    second = run_smoke(path, tmp_path / "second", 1, 2)
    assert first == second
    assert random.getstate() == rng
    assert settings == {k: v for k, v in vars(config).items() if k.isupper()}
    assert seen.count(-60) > 2 and seen.count(None) > 2
    assert first["paired_targets_and_seeds_verified"]
    with pytest.raises(FileExistsError):
        run_smoke(path, tmp_path / "first", 1, 2)


def test_smoke_uses_paper_chebyshev_target_distance(tmp_path):
    manifest = run_smoke(DATA / "calibration_profile.json", tmp_path / "smoke", 1, 1)
    for mode in ("baseline", "calibrated"):
        settings = manifest["settings"][mode]
        assert settings["RSSI_MODEL"] == mode
        assert (settings["GRID_SIZE"], settings["AGENT_START"], settings["TARGET_MODE"],
                settings["TARGET_MIN_DISTANCE"], settings["TARGET_DISTANCE_METRIC"]) == (
                    20, (0, 0), "split", 5, "chebyshev")


def test_settings_restore_on_failure():
    previous = config.RSSI_MODEL
    with pytest.raises(RuntimeError):
        with temporary_settings(RSSI_MODEL="calibrated"):
            raise RuntimeError("intentional")
    assert config.RSSI_MODEL == previous


def test_pairing_mismatch_fails():
    rows = [dict(run=1, episode=1, algorithm=a, train_seed=42, eval_seed=43, target_x=x, target_y=9)
            for a, x in (("q_learning", 5), ("random", 6))]
    with pytest.raises(ValueError, match="mismatch"):
        paired_targets({"episodes": rows})
