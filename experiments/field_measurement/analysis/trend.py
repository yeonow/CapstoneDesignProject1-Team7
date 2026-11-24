"""Evaluate adjacent RSSI changes within each independently measured movement leg."""

from collections import defaultdict
import math

import matplotlib.pyplot as plt
import numpy as np

from src.state_reward import TREND_DOWN, TREND_KEEP, TREND_UP, get_rssi_trend

from ..collect import DIRECTIONS
from .prepare_field_data import require, write_csv

LABELS = {TREND_DOWN: "down", TREND_KEEP: "keep", TREND_UP: "up"}
EXPECTED = {"approach": "up", "away": "down"}


def ordered_legs(rows):
    legs = defaultdict(dict)
    for row in rows:
        repeat, direction = int(row["repeat"]), row["direction"]
        order, distance, rssi = int(row["measurement_order"]), float(row["distance_m"]), float(row["rssi_dbm"])
        require(repeat in (1, 2) and direction in DIRECTIONS and 1 <= order <= 6,
                "Unknown movement leg or measurement order")
        require(distance == DIRECTIONS[direction][order - 1] and math.isfinite(rssi)
                and -128 <= rssi < 0, "Invalid movement distance or RSSI")
        points = legs[repeat, direction]
        require(order not in points, "Duplicate movement point")
        points[order] = rssi
    require(set(legs) == {(r, d) for r in (1, 2) for d in DIRECTIONS},
            "Expected both directions for each repeat")
    for repeat in (1, 2):
        for direction in DIRECTIONS:
            points = legs[repeat, direction]
            require(set(points) == set(range(1, 7)), "Incomplete movement leg")
            yield repeat, direction, [points[order] for order in range(1, 7)]


def transitions(repeat, direction, distances, values):
    for index, (before, after) in enumerate(zip(values, values[1:]), 1):
        prediction = LABELS[get_rssi_trend(after, before)]
        yield {
            "repeat": repeat, "direction": direction, "from_order": index, "to_order": index + 1,
            "from_distance_m": distances[index - 1], "to_distance_m": distances[index],
            "previous_rssi_dbm": before, "current_rssi_dbm": after, "delta_db": after - before,
            "expected": EXPECTED[direction], "predicted": prediction,
            "correct": int(prediction == EXPECTED[direction]),
        }


def accuracy(rows):
    group = defaultdict(list)
    for row in rows:
        group[row["direction"]].append(row)
    for direction in (*DIRECTIONS, "all"):
        items = rows if direction == "all" else group[direction]
        total = len(items)
        correct = sum(row["correct"] for row in items)
        yield {
            "direction": direction, "correct": correct, "total": total, "accuracy": correct / total,
            **{label: sum(row["predicted"] == label for row in items) for label in LABELS.values()},
        }


def analyze_trends(legs, output):
    rows = [row for repeat, direction, values in legs
            for row in transitions(repeat, direction, DIRECTIONS[direction], values)]
    aggregate = list(accuracy(rows))
    by_repeat = [{"repeat": repeat, **row} for repeat in (1, 2)
                 for row in accuracy([item for item in rows if item["repeat"] == repeat])]
    report = [{"repeat": "all", **row} for row in aggregate] + by_repeat
    write_csv(output / "trend_transitions.csv", list(rows[0]), rows)
    write_csv(output / "trend_accuracy.csv", list(report[0]), report)

    fig, (path_ax, matrix_ax) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for repeat, direction, values in legs:
        path_ax.plot(range(1, 7), values, marker="o", label=f"{repeat} {direction}")
    path_ax.set(xticks=range(1, 7), xlabel="Measurement order within leg",
                ylabel="5-second median RSSI (dBm)")
    path_ax.legend()
    labels = list(LABELS.values())
    matrix = np.array([[row[label] for label in labels] for row in aggregate[:-1]])
    matrix_ax.imshow(matrix, cmap="Blues", vmin=0)
    matrix_ax.set(xticks=range(len(labels)), xticklabels=labels,
                  yticks=range(len(DIRECTIONS)), yticklabels=list(DIRECTIONS),
                  xlabel="Predicted RSSI trend", ylabel="Actual movement")
    for y, row in enumerate(matrix):
        for x, count in enumerate(row):
            matrix_ax.text(x, y, str(count), ha="center", va="center",
                           color="white" if count > matrix.max() / 2 else "black")
    fig.savefig(output / "movement_trend.png", dpi=160)
    plt.close(fig)
    return aggregate
