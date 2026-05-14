#!/usr/bin/env python3
"""
Corrected tap-amplitude plot for few-shot examples.

Normalization (per professor's feedback):
  - Identify all peaks and valleys first.
  - Global rescale: (x - min_valley) / (max_peak - min_valley)
    so the lowest valley sits at 0 and the highest peak sits at 1.
  - Per-tap amplitude = normalized signal value AT the peak (now a true
    ratio relative to full open/close range), so the smallest tap can
    approach 0 if the patient barely lifts the finger.
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
OUT_SIGNAL = Path("./results/fewshot_amplitude_corrected_signal.png")
OUT_TAPS   = Path("./results/fewshot_amplitude_corrected_taps.png")


def extract_normalized(csv_path):
    data = pd.read_csv(csv_path)
    distance = np.sqrt(
        (data['X_thumb'] - data['X_index']) ** 2 +
        (data['Y_thumb'] - data['Y_index']) ** 2
    ).values
    sig = denoise_signal(distance)

    # temp normalization only for peak detection parameters
    dn = (sig - sig.min()) / (sig.max() - sig.min() + 1e-10)
    peaks, _ = find_peaks(
        dn, height=0.2, distance=int(FPS * 0.15), prominence=0.15, width=1
    )
    valleys, _ = find_peaks(
        -dn, distance=int(FPS * 0.15), prominence=0.15, width=1
    )
    if len(peaks) == 0 or len(valleys) == 0:
        return None

    max_peak   = sig[peaks].max()
    min_valley = sig[valleys].min()
    rng = max_peak - min_valley
    if rng <= 0:
        return None

    sig_norm = (sig - min_valley) / rng        # valleys→0, peaks→1 (globally)
    return {
        "t": np.arange(len(sig)) / FPS,
        "sig_norm": sig_norm,
        "peaks": peaks,
        "valleys": valleys,
    }


def plot_signal():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes = axes.flatten()
    for ax, (score, csv_name) in zip(axes, EXAMPLES):
        r = extract_normalized(CSV_DIR / csv_name)
        if r is None:
            continue
        ax.plot(r["t"], r["sig_norm"], color="tab:blue", linewidth=1.2, label="Signal")
        ax.plot(r["t"][r["peaks"]],   r["sig_norm"][r["peaks"]],   "o", color="tab:red",   ms=6, label="Peaks")
        ax.plot(r["t"][r["valleys"]], r["sig_norm"][r["valleys"]], "v", color="tab:green", ms=6, label="Valleys")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Score {score}  (peaks={len(r['peaks'])}, valleys={len(r['valleys'])})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized amplitude")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Few-shot: Corrected Normalization (valleys→0, peaks→1)", fontsize=13)
    fig.tight_layout()
    OUT_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SIGNAL, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_SIGNAL}")


def plot_taps():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    axes = axes.flatten()
    for ax, (score, csv_name) in zip(axes, EXAMPLES):
        r = extract_normalized(CSV_DIR / csv_name)
        if r is None:
            continue
        peak_amps = r["sig_norm"][r["peaks"]]
        t_peaks = r["t"][r["peaks"]]
        ax.plot(t_peaks, peak_amps, marker="o", linewidth=2, color="tab:blue")
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"Score {score}  (n={len(peak_amps)} taps)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized tap amplitude")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)

        print(f"Score {score}: peak_amps = {np.round(peak_amps, 3).tolist()}")

    fig.suptitle("Few-shot: Corrected Tap Amplitudes (ratio of full range)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_TAPS, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_TAPS}")


if __name__ == "__main__":
    plot_signal()
    plot_taps()
