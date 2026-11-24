"""Compare field RSSI against the repository's configured simulation."""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

from src.env_factory import build_environment

from ..collect import DIRECTIONS
from .prepare_field_data import write_csv
from .trend import accuracy, transitions


def compare(groups, field_accuracy, output, seed, draws):
    environment = build_environment()
    environment.agent_start = (0, 0)
    # Keep the configured cell size and channel model. Extend the analysis grid
    # to cover the measured distances, which can exceed the training grid.
    environment.grid_size = max(environment.grid_size, math.ceil(max(groups) / environment.cell_size_m) + 1)

    def measure(distance):
        return environment.reset((distance / environment.cell_size_m, 0))

    state = random.getstate()
    try:
        random.seed(seed)
        samples = {distance: [measure(distance) for _ in range(draws)] for distance in groups}
        random.seed(f"{seed}:movement")
        simulated_accuracy = {direction: [] for direction in (*DIRECTIONS, "all")}
        for _ in range(draws):
            steps = [row for repeat in (1, 2) for direction, distances in DIRECTIONS.items()
                     for row in transitions(repeat, direction, distances,
                                            [measure(distance) for distance in distances])]
            for row in accuracy(steps):
                simulated_accuracy[row["direction"]].append(row["accuracy"])
    finally:
        random.setstate(state)

    comparison = []
    for distance, measured in groups.items():
        simulated = samples[distance]
        lower, upper = np.percentile(simulated, [2.5, 97.5])
        field_mean, sim_mean = np.mean(measured), np.mean(simulated)
        comparison.append({
            "distance_m": distance, "measured_count": len(measured), "simulated_count": draws,
            "measured_mean_dbm": field_mean,
            "measured_std_db": np.std(measured, ddof=1) if len(measured) > 1 else None,
            "simulated_mean_dbm": sim_mean, "simulated_std_db": np.std(simulated, ddof=1),
            "simulated_p025_dbm": lower, "simulated_p975_dbm": upper,
            "bias_sim_minus_field_db": sim_mean - field_mean,
        })
    write_csv(output / "simulation_distance_comparison.csv", list(comparison[0]), comparison)
    fig, (ax, trend_ax) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    ax.plot(list(groups), [r["measured_mean_dbm"] for r in comparison], "o-", label="Field mean")
    ax.plot(list(groups), [r["simulated_mean_dbm"] for r in comparison], "s-", label="Simulation mean")
    ax.fill_between(list(groups), [r["simulated_p025_dbm"] for r in comparison],
                    [r["simulated_p975_dbm"] for r in comparison], alpha=0.2,
                    label="Simulation central 95%")
    ax.set(xlabel="Distance (m)", ylabel="RSSI (dBm)")
    ax.legend()
    if 15 in groups:
        ax.set_title("15 m data are provisional")
    labels = list(simulated_accuracy)
    positions = range(1, len(labels) + 1)
    trend_ax.boxplot(list(simulated_accuracy.values()), positions=positions, showfliers=False)
    trend_ax.scatter(positions, [row["accuracy"] for row in field_accuracy],
                     label="Field accuracy", zorder=3)
    trend_ax.set(xticks=positions, xticklabels=labels, ylabel="Direction accuracy",
                 ylim=(-0.04, 1.04), title="Simulated trend accuracy")
    trend_ax.legend()
    fig.savefig(output / "simulation_comparison.png", dpi=160)
    plt.close(fig)

    distribution = {}
    for observed, (direction, values) in zip(field_accuracy, simulated_accuracy.items()):
        lower, upper = np.percentile(values, [2.5, 97.5])
        distribution[direction] = {"field_accuracy": observed["accuracy"],
                                   "mean_accuracy": np.mean(values), "p025": lower, "p975": upper}
    parameters = {name: getattr(environment, name) for name in (
        "cell_size_m", "carrier_frequency_hz", "tx_power_dbm", "eta_los", "eta_nlos",
        "path_loss_exponent", "multipath_fading", "noise_std", "min_distance_m",
        "noise_floor_dbm", "channel_state",
    )}
    biases = np.array([row["bias_sim_minus_field_db"] for row in comparison])
    return {"seed": seed, "draws": draws, "parameters": parameters,
            "mean_curve_rmse_db": np.sqrt(np.mean(biases ** 2)),
            "mean_bias_sim_minus_field_db": np.mean(biases),
            "trend_accuracy_distribution": distribution}
