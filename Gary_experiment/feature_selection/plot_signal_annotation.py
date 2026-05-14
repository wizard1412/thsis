#!/usr/bin/env python3
"""
Annotated figure illustrating per-tap amplitude formula:
    A_k = d(peak_k) - [d(v_before) + d(v_after)] / 2

Shows d(peak_k), d(v_before), d(v_after), baseline, and A_k arrow
on the actual finger-tapping distance signal.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

FPS = 240
CSV_PATH = Path("./experiments_llm_lasso_amplitude/csv_files/546503158_20200310_Hands_R_9_L_redo.mp4FT.csv")
OUT_PATH = Path("./results/signal_annotation.png")

# Which tap index to annotate (0-based); None = auto-pick middle tap
ANNOTATE_TAP = None
# How many seconds around the annotated tap to show
WINDOW_SEC = 2.5


def denoise(signal, window=11, polyorder=3):
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window, polyorder)


def main():
    data = pd.read_csv(CSV_PATH)
    raw_dist = np.sqrt(
        (data["X_thumb"] - data["X_index"]) ** 2 +
        (data["Y_thumb"] - data["Y_index"]) ** 2
    ).values
    dist = denoise(raw_dist)
    t = np.arange(len(dist)) / FPS

    # ── Peak / valley detection ───────────────────────────────────────────────
    dn = (dist - dist.min()) / (dist.max() - dist.min() + 1e-10)
    peaks, _ = find_peaks(dn, height=0.2, distance=int(FPS * 0.15),
                          prominence=0.15, width=1)
    valleys, _ = find_peaks(-dn, distance=int(FPS * 0.10),
                            prominence=0.05, width=1)

    # ── Choose tap to annotate ────────────────────────────────────────────────
    tap_idx = ANNOTATE_TAP if ANNOTATE_TAP is not None else len(peaks) // 2
    tap_idx = max(1, min(tap_idx, len(peaks) - 2))  # need neighbours

    p = peaks[tap_idx]
    vb_candidates = valleys[valleys < p]
    va_candidates = valleys[valleys > p]
    if len(vb_candidates) == 0 or len(va_candidates) == 0:
        raise RuntimeError("Could not find adjacent valleys for chosen tap.")

    vb = vb_candidates[-1]   # valley before peak
    va = va_candidates[0]    # valley after peak

    d_peak  = dist[p]
    d_vb    = dist[vb]
    d_va    = dist[va]
    baseline = (d_vb + d_va) / 2.0

    # ── Crop time window ─────────────────────────────────────────────────────
    half = int(WINDOW_SEC * FPS)
    i0 = max(0, p - half)
    i1 = min(len(dist) - 1, p + half)
    sl = slice(i0, i1 + 1)

    t_crop    = t[sl]
    dist_crop = dist[sl]

    # Indices relative to crop
    def rel(i):
        return t[i]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_crop, dist_crop, color="steelblue", lw=2, zorder=3,
            label="Finger distance (denoised)")

    # Show all peaks / valleys in window
    for pk in peaks:
        if i0 <= pk <= i1:
            marker = "o" if pk != p else "o"
            color  = "tab:red" if pk == p else "lightcoral"
            ms     = 10 if pk == p else 6
            ax.plot(t[pk], dist[pk], marker, color=color, ms=ms, zorder=5,
                    label="Tap peaks" if pk == peaks[0] else "")
    for vk in valleys:
        if i0 <= vk <= i1:
            color = "tab:orange" if vk in (vb, va) else "lightgray"
            ms    = 9 if vk in (vb, va) else 5
            ax.plot(t[vk], dist[vk], "v", color=color, ms=ms, zorder=5,
                    label="Valleys" if vk == valleys[0] else "")

    # ── Opening phase: v_before → peak_k ────────────────────────────────────
    t_vb   = rel(vb)
    t_peak = rel(p)
    t_va   = rel(va)
    ax.axvspan(t_vb, t_peak, alpha=0.13, color="tab:green", zorder=2,
               label="Opening phase")
    ax.axvspan(t_peak, t_va, alpha=0.13, color="tab:purple", zorder=2,
               label="Closing phase")
    # Labels near top of each shaded region
    y_top = dist_crop.max()
    ax.text((t_vb + t_peak) / 2, y_top * 0.97,
            "Opening\nphase", fontsize=10, color="tab:green",
            ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="tab:green", alpha=0.7))
    ax.text((t_peak + t_va) / 2, y_top * 0.97,
            "Closing\nphase", fontsize=10, color="tab:purple",
            ha="center", va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="tab:purple", alpha=0.7))

    # ── Baseline dashed line between v_before and v_after ─────────────────────
    t_base_left  = t_vb
    t_base_right = t_va
    ax.hlines(baseline, t_base_left, t_base_right,
              colors="gray", linestyles="--", lw=1.5, zorder=4,
              label="Baseline = [d(valley_before)+d(valley_after)]/2")

    # ── A_k double-headed arrow (baseline → peak) ─────────────────────────────
    ax.annotate(
        "", xy=(t_peak, d_peak), xytext=(t_peak, baseline),
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
        zorder=6
    )
    ax.text(t_peak + 0.04, (d_peak + baseline) / 2,
            r"$A_k$", fontsize=15, fontweight="bold", color="black", va="center")

    # ── d(peak_k) label ───────────────────────────────────────────────────────
    ax.annotate(
        r"$d(\mathrm{peak}_k)$",
        xy=(t_peak, d_peak),
        xytext=(t_peak + 0.30, d_peak + (dist_crop.max() - dist_crop.min()) * 0.12),
        fontsize=12, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.5),
        ha="left"
    )

    # ── d(v_before) label ─────────────────────────────────────────────────────
    ax.annotate(
        r"$d(\mathrm{valley}_\mathrm{before})$",
        xy=(rel(vb), d_vb),
        xytext=(rel(vb) - 0.40, d_vb - (dist_crop.max() - dist_crop.min()) * 0.18),
        fontsize=12, color="tab:orange",
        arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.5),
        ha="right"
    )

    # ── d(v_after) label ──────────────────────────────────────────────────────
    ax.annotate(
        r"$d(\mathrm{valley}_\mathrm{after})$",
        xy=(rel(va), d_va),
        xytext=(rel(va) + 0.40, d_va - (dist_crop.max() - dist_crop.min()) * 0.18),
        fontsize=12, color="tab:orange",
        arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.5),
        ha="left"
    )

    # ── Baseline label ────────────────────────────────────────────────────────
    ax.text((t_base_left + t_base_right) / 2, baseline + 1.5,
            "baseline", fontsize=10, color="gray", ha="center", style="italic")

    # ── Formula box ───────────────────────────────────────────────────────────
    formula = (r"$A_k = d(\mathrm{peak}_k)"
               r" - \dfrac{d(\mathrm{valley}_\mathrm{before})+d(\mathrm{valley}_\mathrm{after})}{2}$")
    ax.text(0.02, 0.97, formula, transform=ax.transAxes,
            fontsize=13, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.9))

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Finger distance (px)", fontsize=12)
    ax.set_title("Per-tap Amplitude Definition", fontsize=13, fontweight="bold")
    ax.set_xlim(t_crop[0], t_crop[-1])

    # Legend (deduplicate)
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
