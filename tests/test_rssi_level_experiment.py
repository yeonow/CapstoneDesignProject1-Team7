"""세 RSSI Level 설정 비교 실행기: 설정 구분, 메타데이터, 출력 보호, CLI 기본값."""
import json
import random
import sys

import pytest

import config
from src import rssi_level_experiment as level_experiment


ORIGINAL = [-80.0, -70.0, -60.0, -50.0]
DIAGNOSIS_ADJUSTED = [-89.04901890526463, -81.25897503062048,
                      -73.46893115597632, -65.67888728133218]


def test_three_settings_record_thresholds_and_share_targets(tmp_path):
    before_config = {k: v for k, v in vars(config).items() if k.isupper()}
    before_random = random.getstate()
    output = tmp_path / "levels"
    manifest = level_experiment.run_level_experiment(output, train_episodes=2, eval_episodes=3,
                                                     runs=1)
    assert random.getstate() == before_random
    assert {k: v for k, v in vars(config).items() if k.isupper()} == before_config
    settings = manifest["settings"]
    assert list(settings) == ["baseline_original", "calibrated_original", "calibrated_adjusted"]
    assert [settings[name]["RSSI_MODEL"] for name in settings] == ["baseline", "calibrated", "calibrated"]
    assert settings["baseline_original"]["rssi_thresholds"] == ORIGINAL
    assert settings["calibrated_original"]["rssi_thresholds"] == ORIGINAL
    assert settings["calibrated_adjusted"]["rssi_thresholds"] == DIAGNOSIS_ADJUSTED
    assert settings["calibrated_adjusted"]["level_mapping"]["adjusted_thresholds"] == DIAGNOSIS_ADJUSTED
    assert settings["baseline_original"]["level_mapping"] is None
    for entry in settings.values():
        assert entry["trend_threshold_db"] == 2.0
        assert (entry["config"]["GRID_SIZE"], entry["config"]["MAX_STEPS"], entry["config"]["NOISE_STD"],
                entry["config"]["TARGET_MODE"], entry["config"]["TARGET_DISTANCE_METRIC"],
                entry["config"]["TARGET_MIN_DISTANCE"]) == (20, 400, 2.0, "split", "chebyshev", 5)
    assert manifest["status"] == "completed" and manifest["paired_targets_and_seeds_verified"]
    assert set(manifest["mean_success_rate"]) == set(settings)
    saved = json.loads((output / "level_manifest.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(manifest))
    for name in settings:
        assert (output / name / "algorithm_comparison_episodes.csv").exists()


def test_existing_output_and_unknown_setting_are_refused(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "past.csv").write_text("past", encoding="utf-8")
    with pytest.raises(FileExistsError):
        level_experiment.run_level_experiment(existing, train_episodes=1, eval_episodes=1, runs=1)
    assert [p.name for p in existing.iterdir()] == ["past.csv"]
    with pytest.raises(ValueError, match="settings"):
        level_experiment.run_level_experiment(tmp_path / "new", settings=["baseline_adjusted"],
                                              train_episodes=1, eval_episodes=1, runs=1)
    assert not (tmp_path / "new").exists()


def test_single_setting_can_run_alone(tmp_path):
    manifest = level_experiment.run_level_experiment(
        tmp_path / "adjusted", settings=["calibrated_adjusted"], train_episodes=1,
        eval_episodes=2, runs=1)
    assert list(manifest["settings"]) == ["calibrated_adjusted"]
    assert sorted(p.name for p in (tmp_path / "adjusted").iterdir()) == [
        "calibrated_adjusted", "level_manifest.json"]


def test_cli_defaults_are_final_counts_without_running(monkeypatch, tmp_path):
    captured = {}

    def capture(output, **kwargs):
        captured.update(output=output, **kwargs)
        return {"mean_success_rate": {}}

    monkeypatch.setattr(level_experiment, "run_level_experiment", capture)
    monkeypatch.setattr(sys, "argv", ["rssi_level_experiment", "--output", str(tmp_path / "x")])
    level_experiment.main()
    assert (captured["train_episodes"], captured["eval_episodes"], captured["runs"],
            captured["seed"]) == (3000, 100, 100, 42)
    assert captured["settings"] == ["baseline_original", "calibrated_original", "calibrated_adjusted"]
