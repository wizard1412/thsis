#!/usr/bin/env python3
"""Plot denoised distance signal with peaks and valleys for the 4 few-shot examples."""

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
OUT_PATH = Path("./results/fewshot_amplitude_peaks_valleys.png")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    axes = axes.flatten()

    for ax, (score, csv_name) in zip(axes, EXAMPLES):
        csv_path = CSV_DIR / csv_name
        data = pd.read_csv(csv_path)
        distance = np.sqrt(
            (data['X_thumb'] - data['X_index']) ** 2 +
            (data['Y_thumb'] - data['Y_index']) ** 2
        ).values
        sig = denoise_signal(distance)

        dn = (sig - sig.min()) / (sig.max() - sig.min() + 1e-10)
        peaks, _ = find_peaks(
            dn, height=0.2, distance=int(FPS * 0.15), prominence=0.15, width=1
        )
        valleys, _ = find_peaks(
            -dn, distance=int(FPS * 0.15), prominence=0.15, width=1
        )

        max_amp = sig[peaks].max() if len(peaks) else 1.0
        sig_norm = sig / max_amp
        t = np.arange(len(sig)) / FPS

        ax.plot(t, sig_norm, color="tab:blue", linewidth=1.2, label="Signal")
        ax.plot(t[peaks], sig_norm[peaks], "o", color="tab:red", markersize=6, label="Peaks")
        ax.plot(t[valleys], sig_norm[valleys], "v", color="tab:green", markersize=6, label="Valleys")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)

        ax.set_title(f"Score {score}  (peaks={len(peaks)}, valleys={len(valleys)})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized amplitude (ratio to max peak)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Few-shot Examples: Tap Signal with Peaks and Valleys", fontsize=13)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
