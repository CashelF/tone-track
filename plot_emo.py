#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_emo_csv(path):
    times = []
    arousal = []
    dominance = []
    valence = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = float(row["start_time"])
            end = float(row["end_time"])
            t = 0.5 * (start + end)  # midpoint of the interval

            times.append(t)
            arousal.append(float(row["arousal"]))
            dominance.append(float(row["dominance"]))
            valence.append(float(row["valence"]))

    return times, arousal, dominance, valence


def main():
    parser = argparse.ArgumentParser(
        description="Plot arousal, dominance, valence over time."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to CSV file with columns: start_time,end_time,arousal,dominance,valence",
    )
    args = parser.parse_args()

    times, arousal, dominance, valence = load_emo_csv(args.csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Arousal
    axes[0].plot(times, arousal, marker="o")
    axes[0].set_ylabel("Arousal")
    axes[0].grid(True, alpha=0.3)

    # Dominance
    axes[1].plot(times, dominance, marker="o")
    axes[1].set_ylabel("Dominance")
    axes[1].grid(True, alpha=0.3)

    # Valence
    axes[2].plot(times, valence, marker="o")
    axes[2].set_ylabel("Valence")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Emotion Trajectory Over Time", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
