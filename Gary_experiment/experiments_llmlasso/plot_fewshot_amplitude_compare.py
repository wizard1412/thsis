#!/usr/bin/env python3
"""
Compare two corrected tap-amplitude methods against the old one.

Methods:
  A. Old              : peak_value / max(peak_values)
  B. Global range     : (peak_value - min_valley) / (max_peak - min_valley)
  C. Peak - adj valley: (peak - mean(adj_valleys)), then divided by its own max
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from run_feature_extraction_with_ratings import denoise_signal

FPS = 240

EXAMPLES = [
    (0, "553661824_20200629_Hands_L_7-_R.mp4FT.csv"),
    (1, "445817794_20200720_Hands_R_12-_L.mp4FT.csv"),
    (2, "879885247_20200608_Hands_L_11_R.mp4FT.csv"),
    (3, "546503158_20200310_Hands_L_9_R.mp4FT.csv"),
]

CSV_DIR = Path("./csv_files")
OUT = Path("./results/fewshot_amplitude_compare.png")


def analyze(csv_path):
    data = pd.read_csv(csv_path)
    distance = np.sqrt(
        (data['X_thumb'] - data['X_index']) ** 2 +
        (data['Y_thumb'] - data['Y_index']) ** 2
    ).values
    sig = denoise_signal(distance)

    dn = (sig - sig.min()) / (sig.max() - sig.min() + 1e-10)
    peaks, _ = find_peaks(dn, height=0.2, distance=int(FPS * 0.15), prominence=0.15, width=1)
    valleys, _ = find_peaks(-dn, distance=int(FPS * 0.15), prominence=0.15, width=1)
    if len(peaks) == 0 or len(valleys) == 0:
        return None

    peak_vals = sig[peaks]
    valley_vals = sig[valleys]
    t_peaks = peaks / FPS

    # A. old
    a_old = peak_vals / peak_vals.max()

    # B. global range (valleys -> 0, peaks -> 1)
    rng = peak_vals.max() - valley_vals.min()
    b_global = (peak_vals - valley_vals.min()) / (rng + 1e-10)

    # C. peak - adjacent valley (mean of left & right valley within range)
    c_raw = np.empty(len(peaks))
    for i, p in enumerate(peaks):
        left  = valleys[valleys < p]
        right = valleys[valleys > p]
        lv = sig[left[-1]]  if len(left)  else None
        rv = sig[right[0]]  if len(right) else None
        if lv is not None and rv is not None:
            v = (lv + rv) / 2
        elif lv is not None:
            v = lv
        else:
            v = rv
        c_raw[i] = sig[p] - v
    c_peakval = c_raw / c_raw.max()

    return {
        "t_peaks": t_peaks,
        "a_old":    a_old,
        "b_global": b_global,
        "c_peakval": c_peakval,
        "n_peaks":   len(peaks),
        "n_valleys": len(valleys),
    }


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.flatten()
    for ax, (score, csv_name) in zip(axes, EXAMPLES):
        r = analyze(CSV_DIR / csv_name)
        if r is None:
            continue
        ax.plot(r["t_peaks"], r["a_old"],     "o-", color="tab:gray",   label="A. Old (peak/max_peak)",  alpha=0.8)
        ax.plot(r["t_peaks"], r["b_global"],  "s-", color="tab:blue",   label="B. Global range",         alpha=0.9)
        ax.plot(r["t_peaks"], r["c_peakval"], "^-", color="tab:red",    label="C. Peak - adj valley",    alpha=0.9)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Score {score}  (peaks={r['n_peaks']}, valleys={r['n_valleys']})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized amplitude")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)

        print(f"\nScore {score}:")
        print(f"  A (old)         : {np.round(r['a_old'], 3).tolist()}")
        print(f"  B (global)      : {np.round(r['b_global'], 3).tolist()}")
        print(f"  C (peak-valley) : {np.round(r['c_peakval'], 3).tolist()}")

    fig.suptitle("Tap Amplitude Normalization Methods Comparison", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
