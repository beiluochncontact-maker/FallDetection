import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FEATURE_COLUMNS, OUTPUT_DIR, SAMPLING_RATE
from visualization.base import BaseVisualizer

# Alignment window around onset (falls) / center crop (ADL), in samples @ 100 Hz
PRE_SAMPLES = 100   # 1.0 s before
POST_SAMPLES = 200  # 2.0 s after
SEGMENT_LEN = PRE_SAMPLES + POST_SAMPLES
MAX_TRIALS = 200
RNG_SEED = 42


def _magnitude(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = [f"{prefix}X", f"{prefix}Y", f"{prefix}Z"]
    data = df[cols].to_numpy(dtype=np.float64)
    return np.linalg.norm(data, axis=1)


def _pad_or_slice(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    """Extract [start, start+length) with edge padding if out of bounds."""
    end = start + length
    n = len(arr)
    if start >= 0 and end <= n:
        return arr[start:end].copy()

    out = np.empty(length, dtype=np.float64)
    for i in range(length):
        j = start + i
        if j < 0:
            out[i] = arr[0] if n else 0.0
        elif j >= n:
            out[i] = arr[-1] if n else 0.0
        else:
            out[i] = arr[j]
    return out


def _extract_mags(df: pd.DataFrame, align: int) -> np.ndarray:
    """Return shape (3, SEGMENT_LEN): acc_mag, gyr_mag, euler_mag."""
    start = align - PRE_SAMPLES
    acc = _pad_or_slice(_magnitude(df, "Acc"), start, SEGMENT_LEN)
    gyr = _pad_or_slice(_magnitude(df, "Gyr"), start, SEGMENT_LEN)
    euler = _pad_or_slice(_magnitude(df, "Euler"), start, SEGMENT_LEN)
    return np.stack([acc, gyr, euler], axis=0)


def _collect_segments(dataset: dict, is_fall: bool, rng: random.Random) -> np.ndarray:
    """Stack segments -> (n_trials, 3, SEGMENT_LEN)."""
    items = [v for v in dataset.values() if bool(v["is_fall"]) == is_fall]
    if len(items) > MAX_TRIALS:
        items = rng.sample(items, MAX_TRIALS)

    segments = []
    for item in items:
        df = pd.read_csv(item["csv_path"])
        if not all(c in df.columns for c in FEATURE_COLUMNS):
            continue
        if is_fall:
            align = int(item["onset"])
        else:
            align = len(df) // 2
        segments.append(_extract_mags(df, align))

    if not segments:
        return np.empty((0, 3, SEGMENT_LEN), dtype=np.float64)
    return np.stack(segments, axis=0)


class DataVisualizer(BaseVisualizer):

    def __init__(self):
        super().__init__(Path(OUTPUT_DIR) / "figures" / "data")

    def plot_fall_vs_normal_meanstd(self, dataset):
        rng = random.Random(RNG_SEED)
        fall = _collect_segments(dataset, is_fall=True, rng=rng)
        normal = _collect_segments(dataset, is_fall=False, rng=rng)

        if fall.size == 0 or normal.size == 0:
            print("[Warn] skip fall_vs_normal_meanstd: empty fall or normal set")
            return

        t = (np.arange(SEGMENT_LEN) - PRE_SAMPLES) / float(SAMPLING_RATE)
        titles = [
            "Acc magnitude",
            "Gyr magnitude",
            "Euler magnitude",
        ]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        for i, ax in enumerate(axes):
            for segs, label, color in (
                (fall, f"Fall (n={len(fall)}, @onset)", "#c0392b"),
                (normal, f"Normal (n={len(normal)}, @center)", "#2980b9"),
            ):
                mean = segs[:, i, :].mean(axis=0)
                std = segs[:, i, :].std(axis=0)
                ax.plot(t, mean, color=color, label=label, linewidth=1.5)
                ax.fill_between(
                    t,
                    mean - std,
                    mean + std,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                )

            ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_ylabel(titles[i])
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time relative to align (s)")
        fig.suptitle(
            "Fall vs Normal — mean ± std "
            f"(window {PRE_SAMPLES / SAMPLING_RATE:.1f}s before / "
            f"{POST_SAMPLES / SAMPLING_RATE:.1f}s after)"
        )

        self.save(fig, "fall_vs_normal_meanstd.png")

    def visualize_all(self, dataset):
        print("\nGenerating Data Visualization...")
        self.plot_fall_vs_normal_meanstd(dataset)
        print("Finished.\n")
