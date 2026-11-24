"""Analyze qualified field RSSI medians and generate CSV, JSON and PNG reports."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from src.state_reward import _cfg

from .prepare_field_data import require, write_csv
from .trend import analyze_trends, ordered_legs
from ..collect import DISTANCES


def qualified_values(source, include_provisional):
    groups = defaultdict(list)
    keys = set()
    for row in load_csv(source):
        distance, number = int(float(row["distance_m"])), int(row["sample_no"])
        value = float(row["rssi_dbm"])
        require(distance in DISTANCES and distance == float(row["distance_m"])
                and 1 <= number <= 15 and (distance, number) not in keys
                and math.isfinite(value) and -128 <= value < 0,
                "Invalid or duplicate qualified Scenario 1 sample")
        keys.add((distance, number))
        if include_provisional or distance != 15:
            groups[distance].append(value)
    distances = DISTANCES if include_provisional else DISTANCES[:-1]
    require(set(groups) == set(distances), "Scenario 1 lacks qualified samples at a required distance")
    return dict(sorted(groups.items()))


def load_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(all(None not in row and None not in row.values() for row in rows),
            f"Malformed file: {path.name}")
    return rows


def distance_statistics(groups):
    for distance, values in groups.items():
        lower, upper = np.percentile(values, [25, 75])
        yield {
            "distance_m": distance, "count": len(values), "mean_dbm": np.mean(values),
            "median_dbm": np.median(values), "std_db": np.std(values, ddof=1) if len(values) > 1 else None,
            "q1_dbm": lower, "q3_dbm": upper, "min_dbm": min(values), "max_dbm": max(values),
        }


def path_loss(groups):
    distances = np.repeat(list(groups), [len(values) for values in groups.values()])
    measured = np.concatenate(list(groups.values()))
    logarithm = np.log10(distances)
    fit = stats.linregress(logarithm, measured)
    predicted = fit.intercept + fit.slope * logarithm
    residuals = measured - predicted
    dof = len(measured) - 2
    half_width = stats.t.ppf(0.975, dof) * fit.stderr / 10
    result = {
        "count": len(measured), "reference_distance_m": 1, "reference_rssi_dbm": fit.intercept,
        "exponent": -fit.slope / 10, "exponent_95_ci": [-fit.slope / 10 - half_width,
                                                       -fit.slope / 10 + half_width],
        "r_squared": fit.rvalue ** 2, "rmse_db": np.sqrt(np.mean(residuals ** 2)),
        "residual_std_db": np.sqrt(np.sum(residuals ** 2) / dof),
    }
    return result, residuals


def gaussian_fit(values, dof):
    values = np.asarray(values)
    sigma = np.sqrt(np.sum((values - np.mean(values)) ** 2) / dof) if dof > 0 else None
    shapiro = stats.shapiro(values) if len(values) >= 3 and sigma else None
    return {"count": len(values), "mean_db": np.mean(values), "sigma_db": sigma,
            "shapiro_w": float(shapiro.statistic) if shapiro else None,
            "shapiro_p": float(shapiro.pvalue) if shapiro else None}


def figures(groups, fit, residuals, within, within_fit, output):
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.boxplot(list(groups.values()), positions=list(groups), widths=0.55)
    grid = np.linspace(min(groups), max(groups), 200)
    ax.plot(grid, fit["reference_rssi_dbm"] - 10 * fit["exponent"] * np.log10(grid),
            label="Log-distance fit")
    ax.set(xlabel="Distance (m)", ylabel="5-second median RSSI (dBm)")
    ax.legend()
    fig.savefig(output / "distance_rssi.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    axes[0].hist(within, bins="auto", density=True, alpha=0.6, label="Within-distance")
    sigma = within_fit["sigma_db"]
    if sigma:
        grid = np.linspace(min(within), max(within), 200)
        axes[0].plot(grid, stats.norm.pdf(grid, within_fit["mean_db"], sigma), label="Gaussian")
    axes[0].set(xlabel="Deviation (dB)", ylabel="Density")
    axes[0].legend()
    for ax, values, title in zip(axes[1:], (within, residuals),
                                 ("Within-distance Q–Q", "Path-loss residual Q–Q")):
        stats.probplot(values, dist="norm", plot=ax)
        ax.set_title(title)
        ax.set_ylabel("Ordered deviations (dB)")
    fig.savefig(output / "gaussian_noise.png", dpi=160)
    plt.close(fig)


def analyze_static(source, output, include_provisional):
    groups = qualified_values(source, include_provisional)
    summary = list(distance_statistics(groups))
    fit, residuals = path_loss(groups)
    within = np.concatenate([np.asarray(values) - np.mean(values) for values in groups.values()])
    within_fit = gaussian_fit(within, len(within) - len(groups))
    write_csv(output / "distance_statistics.csv", list(summary[0]), summary)
    figures(groups, fit, residuals, within, within_fit, output)
    return groups, {
        "path_loss": fit,
        "gaussian": {"within_distance": within_fit,
                     "path_loss_residual": gaussian_fit(residuals, len(residuals) - 2)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-provisional", action="store_true")
    args = parser.parse_args()
    source = args.prepared / "scenario1_strict_provisional.csv"
    motion = args.prepared / "scenario2.csv"
    legs = list(ordered_legs(load_csv(motion)))
    args.output.mkdir(parents=True, exist_ok=True)
    _, report = analyze_static(source, args.output, args.include_provisional)
    report.update({
        "trend_threshold_db": _cfg("TREND_THRESHOLD"),
        "trend_accuracy": analyze_trends(legs, args.output),
        "includes_provisional_15m": args.include_provisional,
        "scenario1_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scenario2_sha256": hashlib.sha256(motion.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    })
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
