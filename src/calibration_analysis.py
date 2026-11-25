"""최종 실측 분석과 보정 프로필 저장: python -m src.calibration_analysis --output NEW_DIR."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from src.calibration_data import DATA, ROOT, read_measurements, validate_dataset
from src.calibration_fit import fit_measurements
from src.calibration_trend import trend_report
from src.env_factory import build_environment
from experiments.field_measurement.analysis.prepare_field_data import write_csv


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def compare_curves(primary, profile, draws=500):
    """기존 환경 자체로 noise-free 예측과 가정한 2dB 변동을 계산한다."""
    if config.RSSI_MODEL != "baseline":
        raise ValueError("Run analysis from baseline config to preserve a clear comparison")
    if draws < 2:
        raise ValueError("At least two draws required")
    saved = random.getstate()
    rows = []
    try:
        for mode in ("baseline", "calibrated"):
            env = build_environment(noise_std=0)
            env.grid_size = 21
            env.agent_start = (0, 0)
            env.cell_size_m = 1
            if mode == "calibrated":
                env.calibrated_reference_rx_power_dbm = profile["reference_rx_power_dbm"]
                env.path_loss_exponent = profile["path_loss_exponent"]
                env.carrier_frequency_hz = profile["carrier_frequency_hz"]
            for distance in (1, 2, 3, 5, 10, 15, 20):
                env.noise_std = 0
                predicted = env.reset(target_position=(distance, 0))
                rx = env.current_rx_power
                env.noise_std = config.NOISE_STD
                random.seed(42 + distance)  # 두 모드의 동일 거리에서 같은 Gaussian 표본
                values = [env.reset() for _ in range(draws)]
                measured = [r["rssi_dbm"] for r in primary if r["distance_m"] == distance]
                rows.append({"mode": mode, "distance_m": distance, "observed": bool(measured),
                             "measured_mean_dbm": statistics.mean(measured) if measured else None,
                             "measured_sd_db": statistics.stdev(measured) if measured else None,
                             "noise_free_rssi_dbm": predicted, "noise_free_rx_dbm": rx,
                             "simulated_mean_dbm": statistics.mean(values),
                             "simulated_sd_db": statistics.stdev(values),
                             "assumed_noise_std_db": env.noise_std, "draws": draws})
    finally:
        random.setstate(saved)
    return rows


def run_analysis(output_dir=ROOT / "results/rssi_calibration_20261125/analysis"):
    quality = validate_dataset()
    primary = read_measurements(DATA / "scenario1_provisional.csv")
    strict = read_measurements(DATA / "scenario1_strict_provisional.csv")
    motion = read_measurements(DATA / "scenario2.csv", 2)
    fitted = fit_measurements(primary, config.NOISE_FLOOR_DBM)
    sensitivity = fit_measurements(strict, config.NOISE_FLOOR_DBM)
    profile = {"schema_version": 1, "source_sha256": quality["hashes"],
               **{k: fitted[k] for k in ("reference_rx_power_dbm", "path_loss_exponent",
                                        "reference_distance_m", "noise_floor_dbm")},
               "carrier_frequency_hz": quality["frequency_mhz"][0] * 1e6,
               "channel_state": "los", "measurement_range_m": [1, 15],
               "noise_policy": "Keep existing 2 dB assumption, not inferred spatial shadow variance",
               "limitations": fitted["limits"]}
    comparison = compare_curves(primary, profile)
    trend = trend_report(motion)
    rmse = {}
    for mode in ("baseline", "calibrated"):
        prediction = {r["distance_m"]: r["noise_free_rssi_dbm"] for r in comparison if r["mode"] == mode}
        rmse[mode] = math.sqrt(statistics.mean((r["rssi_dbm"] - prediction[r["distance_m"]]) ** 2
                                              for r in primary))
    output = ROOT / Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # 기존 결과/부분 실행도 덮어쓰지 않는다.
    json_write(output / "profile.json", profile)
    report = {"quality": quality, "primary_fit": fitted, "strict_sensitivity": sensitivity,
              "rmse_against_all_90_db": rmse, "trend": trend,
              "scope": "In-sample model calibration, not validation of generalization or localization performance"}
    json_write(output / "report.json", report)
    for name, rows in (("distance_statistics", fitted["distance_statistics"]),
                       ("model_comparison", comparison), ("trend_transitions", trend["transitions"]),
                       ("trend_accuracy", trend["accuracy"]),
                       ("trend_threshold_exploration", trend["exploratory_thresholds"])):
        write_csv(output / f"{name}.csv", list(rows[0]), rows)
    plot_comparison(comparison, output)
    plot_motion(motion, output)
    print(json.dumps({"output": str(output), "rmse_db": rmse,
                      "reference_rx_power_dbm": fitted["reference_rx_power_dbm"],
                      "exponent": fitted["path_loss_exponent"]}, indent=2))
    return report


def plot_comparison(rows, output):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    observed = [r for r in rows if r["mode"] == "baseline" and r["observed"]]
    ax.errorbar([r["distance_m"] for r in observed], [r["measured_mean_dbm"] for r in observed],
                yerr=[r["measured_sd_db"] for r in observed], fmt="o", capsize=4,
                label="Measured mean +/- repeat SD (15 per distance)")
    for mode in ("baseline", "calibrated"):
        curve = [r for r in rows if r["mode"] == mode]
        ax.plot([r["distance_m"] for r in curve], [r["noise_free_rssi_dbm"] for r in curve],
                marker=".", label=f"{mode}: noise-free RSSI")
    ax.axvspan(15, 20, color="gray", alpha=.12, label="Extrapolation: no data above 15 m")
    ax.set(xlabel="Distance (m)", ylabel="RSSI (dBm)", xticks=[1, 2, 3, 5, 10, 15, 20],
           title="Scenario 1 calibration (20 m: hotspot disconnected, unobserved)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"distance_model_comparison.{suffix}", dpi=300)
    plt.close(fig)


def plot_motion(rows, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, repeat in zip(axes, (1, 2)):
        for direction in ("approach", "away"):
            leg = sorted((r for r in rows if r["repeat"] == repeat and r["direction"] == direction),
                         key=lambda r: r["measurement_order"])
            ax.plot([r["measurement_order"] for r in leg], [r["rssi_dbm"] for r in leg],
                    marker="o", label=f"{direction}: " + ",".join(str(int(r["distance_m"])) for r in leg) + " m")
        ax.set(title=f"Scenario 2: Repeat {repeat}", xlabel="Measurement order", xticks=range(1, 7))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("RSSI (dBm)")
    fig.savefig(output / "movement_trends.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/rssi_calibration_20261125/analysis")
    run_analysis(parser.parse_args().output)
