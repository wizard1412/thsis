"""
Focused Comparison: Fewshot vs Zeroshot & DeepSeek-R1 Scaling
==============================================================
Outputs (in focused_results/png/ and focused_results/pdf/):
  Analysis 1 - Fewshot vs Zeroshot:
    1a. fewshot_zeroshot_grouped     - Grouped bar per metric, paired by model family
    1b. fewshot_zeroshot_delta       - Delta (few - zero) heatmap, higher = fewshot better
    1c. fewshot_zeroshot_radar       - Radar chart for each model (few vs zero pair)
  Analysis 2 - DeepSeek-R1 Scaling:
    2a. deepseek_scaling_bar         - 32b vs 70b side-by-side bars per metric
    2b. deepseek_scaling_profile     - Line profile across metrics (interaction plot)
    2c. deepseek_scaling_delta       - Delta (70b - 32b) diverging bar
"""

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from math import pi

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUMMARY_CSV = "multi_model_results/summary_metrics.csv"
OUTPUT_DIR  = "focused_results"
DIR_PNG     = os.path.join(OUTPUT_DIR, "png")
DIR_PDF     = os.path.join(OUTPUT_DIR, "pdf")

PYTHON_BIN = r"C:\Users\user\miniconda3\envs\thsis\python.exe"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":  150,
})
sns.set_theme(style="whitegrid")

# Colour palette: fewshot = warm, zeroshot = cool
COLOR_FEW  = "#E07B39"   # orange
COLOR_ZERO = "#4A90D9"   # blue
COLOR_32B  = "#5B8DB8"
COLOR_70B  = "#C0392B"

MAX_SCORE = 4

METRICS      = ["MAE", "RMSE", "Corr", "Kappa", "WD"]
METRIC_LABEL = {
    "MAE":   "MAE ↓",
    "RMSE":  "RMSE ↓",
    "Corr":  "Correlation ↑",
    "Kappa": "Weighted Kappa ↑",
    "WD":    "Wasserstein Dist. ↓",
}
HIGHER_BETTER = {"Corr", "Kappa"}   # all others: lower is better

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(name, **kwargs):
    kw = dict(dpi=300, bbox_inches="tight")
    kw.update(kwargs)
    plt.savefig(os.path.join(DIR_PNG, f"{name}.png"), **kw)
    plt.savefig(os.path.join(DIR_PDF, f"{name}.pdf"), **kw)


def delta_sign(m, val_few, val_zero):
    """Return +1 if fewshot is better, -1 if zeroshot is better."""
    diff = val_few - val_zero
    if m in HIGHER_BETTER:
        return +1 if diff > 0 else -1
    else:
        return +1 if diff < 0 else -1  # lower is better → negative delta = few better


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load():
    df = pd.read_csv(SUMMARY_CSV)
    return df


def load_raw_models(folder="."):
    """Load per-session CSV files, same logic as multi_model_comparison.py."""
    import glob
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    models = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            ai_cols = [str(i) for i in range(10)]
            if not all(c in df.columns for c in ai_cols):
                continue
            # Extract model name
            base = os.path.splitext(os.path.basename(f))[0]
            if "scored_by_" in base:
                suffix = base.split("scored_by_", 1)[1]
                name = suffix.replace("_nothink", "").replace("_0227", "").replace("_0208", "")
            else:
                name = base
            df["ai_median"] = df[ai_cols].median(axis=1)
            if "label_Dr. Tan" in df.columns and "label_Dr. Chien" in df.columns:
                df["consensus"] = (df["label_Dr. Tan"] + df["label_Dr. Chien"]) / 2
            elif "average_score_Doctors" in df.columns:
                df["consensus"] = df["average_score_Doctors"]
            else:
                continue
            models[name] = df
        except Exception:
            pass
    return models


def _build_pair_matrix(scores_a, scores_b):
    mat = np.zeros((MAX_SCORE + 1, MAX_SCORE + 1))
    valid = 0
    for a, b in zip(scores_a, scores_b):
        if np.isnan(float(a)) or np.isnan(float(b)):
            continue
        r = int(np.clip(np.round(float(a)), 0, MAX_SCORE))
        c = int(np.clip(np.round(float(b)), 0, MAX_SCORE))
        mat[r, c] += 1
        valid += 1
    if valid > 0:
        mat = mat / valid * 100
    return mat, valid


def _plot_heatmap(ax, mat, title, xlabel, ylabel, cmap="YlGnBu"):
    sns.heatmap(mat, annot=True, fmt=".1f", cmap=cmap,
                vmin=0, vmax=max(mat.max(), 1),
                linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "%", "shrink": 0.8},
                ax=ax, square=True)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(range(MAX_SCORE + 1), fontsize=7)
    ax.set_yticklabels(range(MAX_SCORE + 1), fontsize=7, rotation=0)


# ===========================================================================
# Analysis 1: Fewshot vs Zeroshot
# ===========================================================================

PAIRS = [
    ("DeepSeek-R1 32b",   "deepseek-r1_32b_fewshot",   "deepseek-r1_32b_zeroshot"),
    ("DeepSeek-R1 70b",   "deepseek-r1_70b_fewshot",   "deepseek-r1_70b_zeroshot"),
    ("Qwen2.5 32b",       "qwen2.5_32b_fewshot",       "qwen2.5_32b_zeroshot"),
    ("Qwen2.5 72b",       "qwen2.5_72b_fewshot",       "qwen2.5_72b_zeroshot"),
    ("Qwen3 30b",         "qwen3_30b_fewshot",          "qwen3_30b_zeroshot"),
    ("Gemini 3 Flash",    "Gemini3_flash_fewshot",      "Gemini3_flash_zeroshot"),
]


def _get(df, model, metric):
    row = df[df.Model == model]
    if row.empty:
        return np.nan, np.nan, np.nan
    r = row.iloc[0]
    return r[metric], r.get(f"{metric}_CI_lo", np.nan), r.get(f"{metric}_CI_hi", np.nan)


# ---------------------------------------------------------------------------
# 1a. Grouped bar chart per metric
# ---------------------------------------------------------------------------

def plot_1a_grouped_bar(df):
    n_pairs   = len(PAIRS)
    n_metrics = len(METRICS)
    ncols = 3
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), sharey=False)
    axes_flat = axes.flatten()
    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    x = np.arange(n_pairs)
    bar_w = 0.35

    for ax, metric in zip(axes_flat, METRICS):
        vals_f, vals_z = [], []
        err_f_lo, err_f_hi = [], []
        err_z_lo, err_z_hi = [], []

        for _, fs, zs in PAIRS:
            vf, lo_f, hi_f = _get(df, fs, metric)
            vz, lo_z, hi_z = _get(df, zs, metric)
            vals_f.append(vf); err_f_lo.append(vf - lo_f); err_f_hi.append(hi_f - vf)
            vals_z.append(vz); err_z_lo.append(vz - lo_z); err_z_hi.append(hi_z - vz)

        vals_f  = np.array(vals_f)
        vals_z  = np.array(vals_z)
        yerr_f  = np.array([err_f_lo, err_f_hi])
        yerr_z  = np.array([err_z_lo, err_z_hi])

        bars_f = ax.bar(x - bar_w / 2, vals_f, bar_w, label="Fewshot",  color=COLOR_FEW,
                        yerr=yerr_f, capsize=4, error_kw=dict(elinewidth=1))
        bars_z = ax.bar(x + bar_w / 2, vals_z, bar_w, label="Zeroshot", color=COLOR_ZERO,
                        yerr=yerr_z, capsize=4, error_kw=dict(elinewidth=1))

        # Mark best bar with a star
        for i, (vf, vz) in enumerate(zip(vals_f, vals_z)):
            best_few = (metric in HIGHER_BETTER and vf >= vz) or (metric not in HIGHER_BETTER and vf <= vz)
            best_bar = bars_f[i] if best_few else bars_z[i]
            ax.text(best_bar.get_x() + best_bar.get_width() / 2,
                    best_bar.get_height() + (max(vals_f.max(), vals_z.max()) * 0.02),
                    "*", ha="center", va="bottom", fontsize=12, color="#2ecc71", fontweight="bold")

        labels = [p[0] for p in PAIRS]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(METRIC_LABEL[metric], fontweight="bold")
        ax.set_ylabel(metric)
        if ax == axes_flat[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("1a_fewshot_zeroshot_grouped")
    plt.close()
    print("  [1a] Grouped bar chart saved.")


# ---------------------------------------------------------------------------
# 1b. Delta heatmap  (few - zero, normalised so that + = fewshot better)
# ---------------------------------------------------------------------------

def plot_1b_delta_heatmap(df):
    family_names = [p[0] for p in PAIRS]
    delta_mat = np.zeros((len(PAIRS), len(METRICS)))

    for i, (_, fs, zs) in enumerate(PAIRS):
        for j, m in enumerate(METRICS):
            vf, _, _ = _get(df, fs, m)
            vz, _, _ = _get(df, zs, m)
            raw_delta = vf - vz
            # Normalise direction: + = fewshot better for ALL metrics
            if m in HIGHER_BETTER:
                delta_mat[i, j] = raw_delta   # higher few → positive = fewshot better
            else:
                delta_mat[i, j] = -raw_delta  # lower few → negative raw → flip sign

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_title("")

    vabs = np.nanmax(np.abs(delta_mat))
    im = ax.imshow(delta_mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")

    # Annotate with raw direction-adjusted delta
    for i in range(len(PAIRS)):
        for j, m in enumerate(METRICS):
            raw_vf, _, _ = _get(df, PAIRS[i][1], m)
            raw_vz, _, _ = _get(df, PAIRS[i][2], m)
            raw_diff = raw_vf - raw_vz
            text_color = "white" if abs(delta_mat[i, j]) > vabs * 0.65 else "black"
            ax.text(j, i, f"{raw_diff:+.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=10)
    ax.set_yticks(range(len(PAIRS)))
    ax.set_yticklabels(family_names, fontsize=10)
    plt.colorbar(im, ax=ax, label="Direction-adjusted Δ", shrink=0.85)

    plt.tight_layout()
    save_fig("1b_fewshot_zeroshot_delta")
    plt.close()
    print("  [1b] Delta heatmap saved.")


# ---------------------------------------------------------------------------
# 1c. Radar chart: for each model family, few vs zero
# ---------------------------------------------------------------------------

def plot_1c_radar(df):
    # Normalize each metric across all models (0-1, higher = better on radar)
    all_vals = {m: df[m].values for m in METRICS}
    norm_range = {}
    for m in METRICS:
        vmin, vmax = np.nanmin(all_vals[m]), np.nanmax(all_vals[m])
        norm_range[m] = (vmin, vmax)

    def normalize(m, v):
        vmin, vmax = norm_range[m]
        if vmax == vmin:
            return 0.5
        n = (v - vmin) / (vmax - vmin)
        return n if m in HIGHER_BETTER else 1 - n

    angles = [n / len(METRICS) * 2 * pi for n in range(len(METRICS))]
    angles += angles[:1]
    labels = [METRIC_LABEL[m] for m in METRICS]

    n_pairs = len(PAIRS)
    ncols = 3
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             subplot_kw=dict(polar=True))


    for idx, (name, fs, zs) in enumerate(PAIRS):
        r, c = divmod(idx, ncols)
        ax = axes[r][c] if nrows > 1 else axes[c]

        for model, color, label in [(fs, COLOR_FEW, "Fewshot"), (zs, COLOR_ZERO, "Zeroshot")]:
            vals = [normalize(m, _get(df, model, m)[0]) for m in METRICS]
            vals += vals[:1]
            ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label, markersize=5)
            ax.fill(angles, vals, alpha=0.10, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(name, fontsize=10, fontweight="bold", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c] if nrows > 1 else axes[c]
        ax.axis("off")

    plt.tight_layout()
    save_fig("1c_fewshot_zeroshot_radar")
    plt.close()
    print("  [1c] Radar chart saved.")


# ===========================================================================
# Analysis 2: DeepSeek-R1 Scaling (32b vs 70b)
# ===========================================================================

SCALING_PAIRS = [
    ("fewshot",  "deepseek-r1_32b_fewshot",  "deepseek-r1_70b_fewshot"),
    ("zeroshot", "deepseek-r1_32b_zeroshot", "deepseek-r1_70b_zeroshot"),
]


# ---------------------------------------------------------------------------
# 2a. Side-by-side bar chart: 32b vs 70b, split by fewshot / zeroshot
# ---------------------------------------------------------------------------

def plot_2a_scaling_bar(df):
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5), sharey=False)


    x      = np.array([0, 1])   # fewshot, zeroshot
    bar_w  = 0.3
    c_32   = COLOR_32B
    c_70   = COLOR_70B

    for ax, metric in zip(axes, METRICS):
        vals_32, vals_70 = [], []
        err_32, err_70   = [], []

        for prompt, m32, m70 in SCALING_PAIRS:
            v32, lo32, hi32 = _get(df, m32, metric)
            v70, lo70, hi70 = _get(df, m70, metric)
            vals_32.append(v32); err_32.append([v32 - lo32, hi32 - v32])
            vals_70.append(v70); err_70.append([v70 - lo70, hi70 - v70])

        vals_32 = np.array(vals_32)
        vals_70 = np.array(vals_70)
        err_32  = np.array(err_32).T
        err_70  = np.array(err_70).T

        bars_32 = ax.bar(x - bar_w / 2, vals_32, bar_w, color=c_32, label="32b",
                         yerr=err_32, capsize=4, error_kw=dict(elinewidth=1))
        bars_70 = ax.bar(x + bar_w / 2, vals_70, bar_w, color=c_70, label="70b",
                         yerr=err_70, capsize=4, error_kw=dict(elinewidth=1))

        # Star on best
        for i, (v32, v70) in enumerate(zip(vals_32, vals_70)):
            best_32 = (metric in HIGHER_BETTER and v32 >= v70) or \
                      (metric not in HIGHER_BETTER and v32 <= v70)
            best_bar = bars_32[i] if best_32 else bars_70[i]
            top = max(v32, v70) * 1.04
            ax.text(best_bar.get_x() + best_bar.get_width() / 2, top,
                    "*", ha="center", va="bottom", fontsize=13, color="#27ae60", fontweight="bold")

        # Annotate values
        for bar, v in list(zip(bars_32, vals_32)) + list(zip(bars_70, vals_70)):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["Fewshot", "Zeroshot"], fontsize=10)
        ax.set_title(METRIC_LABEL[metric], fontweight="bold")
        ax.set_ylabel(metric)
        if ax == axes[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    save_fig("2a_deepseek_scaling_bar")
    plt.close()
    print("  [2a] Scaling bar chart saved.")


# ---------------------------------------------------------------------------
# 2b. Interaction / profile plot across metrics
# ---------------------------------------------------------------------------

def plot_2b_scaling_profile(df):
    # Normalize for display (each metric 0-1 across all four models)
    all_models = [m for _, m32, m70 in SCALING_PAIRS for m in (m32, m70)]

    def norm_val(metric, val):
        vals = [_get(df, m, metric)[0] for m in all_models]
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax == vmin:
            return 0.5
        n = (val - vmin) / (vmax - vmin)
        return n if metric in HIGHER_BETTER else 1 - n

    x = np.arange(len(METRICS))
    styles = {
        "32b_fewshot":  dict(color=COLOR_32B,  ls="-",  marker="o", label="32b Fewshot"),
        "70b_fewshot":  dict(color=COLOR_70B,  ls="-",  marker="s", label="70b Fewshot"),
        "32b_zeroshot": dict(color=COLOR_32B,  ls="--", marker="o", label="32b Zeroshot"),
        "70b_zeroshot": dict(color=COLOR_70B,  ls="--", marker="s", label="70b Zeroshot"),
    }

    models_map = {
        "32b_fewshot":  "deepseek-r1_32b_fewshot",
        "70b_fewshot":  "deepseek-r1_70b_fewshot",
        "32b_zeroshot": "deepseek-r1_32b_zeroshot",
        "70b_zeroshot": "deepseek-r1_70b_zeroshot",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("DeepSeek-R1: Metric Profile (32b vs 70b × Fewshot vs Zeroshot)\n"
                 "(Normalized: outer = better)",
                 fontsize=13, fontweight="bold")

    for key, style in styles.items():
        vals = [norm_val(m, _get(df, models_map[key], m)[0]) for m in METRICS]
        ax.plot(x, vals, lw=2, markersize=8, **style)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=10)
    ax.set_ylabel("Normalized score (higher = better)", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.set_tick_params(labelsize=9)

    plt.tight_layout()
    save_fig("2b_deepseek_scaling_profile")
    plt.close()
    print("  [2b] Scaling profile plot saved.")


# ---------------------------------------------------------------------------
# 2c. Diverging bar: Δ (70b - 32b), direction-adjusted
# ---------------------------------------------------------------------------

def plot_2c_scaling_delta(df):
    prompts     = ["Fewshot", "Zeroshot"]
    model_pairs = [("deepseek-r1_32b_fewshot", "deepseek-r1_70b_fewshot"),
                   ("deepseek-r1_32b_zeroshot","deepseek-r1_70b_zeroshot")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)


    x = np.arange(len(METRICS))
    for ax, prompt, (m32, m70) in zip(axes, prompts, model_pairs):
        deltas = []
        for metric in METRICS:
            v32 = _get(df, m32, metric)[0]
            v70 = _get(df, m70, metric)[0]
            raw  = v70 - v32
            adj  = raw if metric in HIGHER_BETTER else -raw
            deltas.append(adj)

        colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
        bars = ax.barh(x, deltas, color=colors, edgecolor="gray", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=1)

        for bar, d, metric in zip(bars, deltas, METRICS):
            v32 = _get(df, m32, metric)[0]
            v70 = _get(df, m70, metric)[0]
            raw_diff = v70 - v32
            xpos = d + (0.01 if d >= 0 else -0.01)
            ha   = "left" if d >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    f"{raw_diff:+.3f}", va="center", ha=ha, fontsize=9, fontweight="bold")

        ax.set_yticks(x)
        ax.set_yticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=10)
        ax.set_title(f"{prompt}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Direction-adjusted Δ (70b − 32b)")
        ax.invert_yaxis()

    plt.tight_layout()
    save_fig("2c_deepseek_scaling_delta")
    plt.close()
    print("  [2c] Scaling delta plot saved.")


# ===========================================================================
# Analysis 3: Reasoning Distillation Effect
#   deepseek-r1_32b  =  Qwen2.5-32b base + RL reasoning distillation
#   deepseek-r1_70b  =  Llama-3-70b  base + RL reasoning distillation
#   → compare same-size base vs distilled, isolating reasoning training effect
# ===========================================================================

DISTILL_PAIRS = [
    # (label, base_few, distill_few, base_zero, distill_zero)
    ("32b  (Qwen2.5 base)",
     "qwen2.5_32b_fewshot",  "deepseek-r1_32b_fewshot",
     "qwen2.5_32b_zeroshot", "deepseek-r1_32b_zeroshot"),
]


# ---------------------------------------------------------------------------
# 3a. 2×2 interaction plot: prompt × model (base vs distilled), per metric
# ---------------------------------------------------------------------------

def plot_3a_distill_interaction(df):
    """
    For each metric: two lines (Qwen2.5-32b base, DeepSeek-R1-32b distilled)
    across two x-positions (Zeroshot → Fewshot).
    Classic interaction plot to reveal the crossing effect.
    """
    n_metrics = len(METRICS)
    ncols = 3
    nrows = math.ceil(n_metrics / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 5 * nrows),
        sharey=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten() if nrows > 1 else list(axes) if ncols > 1 else [axes]

    # Hide extra subplots
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    COLOR_BASE     = "#4A90D9"   # blue  = Qwen2.5 base
    COLOR_DISTILL  = "#E07B39"   # orange = DeepSeek-R1 distilled

    x = [0, 1]
    x_labels = ["Zeroshot", "Fewshot"]

    for ax, metric in zip(axes_flat, METRICS):
        for color, m_zero, m_few, label, marker in [
            (COLOR_BASE,    "qwen2.5_32b_zeroshot",    "qwen2.5_32b_fewshot",
             "Qwen2.5-32b\n(base)", "o"),
            (COLOR_DISTILL, "deepseek-r1_32b_zeroshot", "deepseek-r1_32b_fewshot",
             "DeepSeek-R1-32b\n(distilled)", "s"),
        ]:
            vz = _get(df, m_zero, metric)[0]
            vf = _get(df, m_few,  metric)[0]
            lo_z, hi_z = _get(df, m_zero, metric)[1], _get(df, m_zero, metric)[2]
            lo_f, hi_f = _get(df, m_few,  metric)[1], _get(df, m_few,  metric)[2]

            ax.plot(x, [vz, vf], color=color, lw=2.5, marker=marker,
                    markersize=9, label=label)
            ax.errorbar([0], [vz], yerr=[[vz - lo_z], [hi_z - vz]],
                        fmt="none", color=color, capsize=5, elinewidth=1.2)
            ax.errorbar([1], [vf], yerr=[[vf - lo_f], [hi_f - vf]],
                        fmt="none", color=color, capsize=5, elinewidth=1.2)

            # Annotate values
            ax.text(0 - 0.06, vz, f"{vz:.3f}", ha="right", va="center",
                    fontsize=8.5, color=color, fontweight="bold")
            ax.text(1 + 0.06, vf, f"{vf:.3f}", ha="left",  va="center",
                    fontsize=8.5, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(METRIC_LABEL[metric], fontweight="bold")
        ax.set_ylabel(metric)
        if ax == axes_flat[0]:
            ax.legend(fontsize=8.5, loc="best")

    save_fig("3a_distill_interaction")
    plt.close()
    print("  [3a] Distillation interaction plot saved.")


# ---------------------------------------------------------------------------
# 3b. Delta heatmap: Δ(distill - base), by prompt condition
# ---------------------------------------------------------------------------

def plot_3b_distill_delta(df):
    """
    Rows = prompt condition (Fewshot / Zeroshot)
    Cols = metrics
    Values = direction-adjusted Δ (distilled - base); green = distillation helps
    """
    conditions = [
        ("Fewshot",  "qwen2.5_32b_fewshot",  "deepseek-r1_32b_fewshot"),
        ("Zeroshot", "qwen2.5_32b_zeroshot", "deepseek-r1_32b_zeroshot"),
    ]

    delta_mat = np.zeros((2, len(METRICS)))
    raw_mat   = np.zeros((2, len(METRICS)))

    for i, (cond, base, distill) in enumerate(conditions):
        for j, m in enumerate(METRICS):
            vb = _get(df, base,    m)[0]
            vd = _get(df, distill, m)[0]
            raw = vd - vb
            raw_mat[i, j] = raw
            delta_mat[i, j] = raw if m in HIGHER_BETTER else -raw  # + = distillation better

    vabs = np.nanmax(np.abs(delta_mat))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_title(
        "Δ (DeepSeek-R1-32b − Qwen2.5-32b): direction-adjusted\n"
        "Green = Reasoning distillation helps  |  Red = Base model better",
        fontsize=12, fontweight="bold", pad=14
    )

    im = ax.imshow(delta_mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")

    for i in range(2):
        for j, m in enumerate(METRICS):
            raw = raw_mat[i, j]
            text_color = "white" if abs(delta_mat[i, j]) > vabs * 0.65 else "black"
            ax.text(j, i, f"{raw:+.3f}", ha="center", va="center",
                    fontsize=12, color=text_color, fontweight="bold")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_LABEL[m] for m in METRICS], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Fewshot", "Zeroshot"], fontsize=11)
    plt.colorbar(im, ax=ax, label="Direction-adjusted Δ", shrink=0.85)

    plt.tight_layout()
    save_fig("3b_distill_delta")
    plt.close()
    print("  [3b] Distillation delta heatmap saved.")


# ---------------------------------------------------------------------------
# 3c. Summary table figure: 4 models × key metrics
# ---------------------------------------------------------------------------

def plot_3c_distill_summary_table(df):
    """Compact table: Qwen2.5-32b vs DeepSeek-R1-32b, few vs zero, with CI."""
    models_ordered = [
        ("Qwen2.5-32b  (base) — Fewshot",   "qwen2.5_32b_fewshot"),
        ("Qwen2.5-32b  (base) — Zeroshot",  "qwen2.5_32b_zeroshot"),
        ("DeepSeek-R1-32b  (distilled) — Fewshot",  "deepseek-r1_32b_fewshot"),
        ("DeepSeek-R1-32b  (distilled) — Zeroshot", "deepseek-r1_32b_zeroshot"),
    ]
    col_labels = ["Model", "MAE ↓", "RMSE ↓", "Correlation ↑", "Kappa ↑", "WD ↓", "Stability ↓"]

    cell_data = []
    for label, model in models_ordered:
        row = df[df.Model == model].iloc[0]
        cell_data.append([
            label,
            row["MAE_fmt"], row["RMSE_fmt"], row["Corr_fmt"],
            row["Kappa_fmt"], row["WD_fmt"],
            f"{row['Stability']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(20, 3.5))
    ax.axis("off")
    ax.set_title(
        "Reasoning Distillation: Qwen2.5-32b (base) vs DeepSeek-R1-32b (distilled) — with 95% Bootstrap CI",
        fontsize=13, fontweight="bold", pad=18
    )

    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.0)

    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row shading: Qwen=blue tint, DeepSeek=orange tint
    for i in range(4):
        color = "#d6eaf8" if i < 2 else "#fdebd0"
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(color)

    plt.tight_layout()
    save_fig("3c_distill_summary_table")
    plt.close()
    print("  [3c] Distillation summary table saved.")


# ===========================================================================
# Analysis 4: Key Models Only — focused per-session & model-vs-consensus
# ===========================================================================

KEY_MODELS = [
    ("DeepSeek-R1-32b\nFewshot",  "deepseek-r1_32b_fewshot"),
    ("DeepSeek-R1-70b\nFewshot",  "deepseek-r1_70b_fewshot"),
    ("Qwen2.5-32b\nZeroshot",     "qwen2.5_32b_zeroshot"),
    ("DeepSeek-R1-70b\nZeroshot", "deepseek-r1_70b_zeroshot"),
]


def plot_4a_key_per_session(models):
    """Per-session overlay for the 4 key models only."""
    first_df = list(models.values())[0]
    gt = first_df["consensus"].values
    n_sessions = len(gt)
    x = np.arange(1, n_sessions + 1)

    fig, ax = plt.subplots(figsize=(max(12, n_sessions * 0.32), 5))
    ax.plot(x, gt, "rs-", label="Ground Truth (Consensus)", lw=2, markersize=6, zorder=10)

    colors = ["#E07B39", "#C0392B", "#4A90D9", "#2980B9"]
    for (label, model_key), color in zip(KEY_MODELS, colors):
        if model_key not in models:
            continue
        median_scores = models[model_key]["ai_median"].values
        ax.plot(x, median_scores, "o-", label=label.replace("\n", " "),
                color=color, lw=1.5, markersize=5, alpha=0.85)

    ax.set_xlabel("Session")
    ax.set_ylabel("Score")
    ax.set_title("Per-Session Score: Key Models vs Ground Truth",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x[::max(1, n_sessions // 15)])
    ax.set_ylim(-0.3, MAX_SCORE + 0.5)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    plt.tight_layout()
    save_fig("4a_key_per_session")
    plt.close()
    print("  [4a] Key model per-session plot saved.")


def plot_4b_key_vs_consensus(models):
    """Model vs consensus heatmaps for key models + human baseline."""
    sample_df = list(models.values())[0]
    has_doctors = ("label_Dr. Tan" in sample_df.columns and
                   "label_Dr. Chien" in sample_df.columns)

    n_plots = len(KEY_MODELS) + (1 if has_doctors else 0)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows), squeeze=False)


    plot_idx = 0
    for label, model_key in KEY_MODELS:
        if model_key not in models:
            continue
        r, c = divmod(plot_idx, ncols)
        ax = axes[r][c]
        mat, nv = _build_pair_matrix(
            models[model_key]["ai_median"].values,
            models[model_key]["consensus"].values)
        _plot_heatmap(ax, mat, f"{label.replace(chr(10), ' ')}\nvs Consensus (n={nv})",
                      "Consensus", label.replace("\n", " "), cmap="Purples")
        plot_idx += 1

    if has_doctors:
        r, c = divmod(plot_idx, ncols)
        ax = axes[r][c]
        mat, nv = _build_pair_matrix(
            sample_df["label_Dr. Tan"].values.astype(float),
            sample_df["label_Dr. Chien"].values.astype(float))
        _plot_heatmap(ax, mat, f"Two Neurologists\n(n={nv})",
                      "Dr. Chien", "Dr. Tan", cmap="OrRd")
        plot_idx += 1

    for idx in range(plot_idx, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig("4b_key_vs_consensus")
    plt.close()
    print("  [4b] Key model vs consensus heatmaps saved.")


def plot_4c_key_summary_table(df):
    """Compact summary table for key models only, with 95% CI."""
    col_labels = ["Model", "MAE ↓", "RMSE ↓", "Correlation ↑", "Kappa ↑", "WD ↓", "Stability ↓"]
    cell_data = []
    for label, model_key in KEY_MODELS:
        row = df[df.Model == model_key]
        if row.empty:
            continue
        r = row.iloc[0]
        cell_data.append([
            label.replace("\n", " "),
            r["MAE_fmt"], r["RMSE_fmt"], r["Corr_fmt"],
            r["Kappa_fmt"], r["WD_fmt"], f"{r['Stability']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(20, 3.0))
    ax.axis("off")
    ax.set_title("Key Model Performance Summary — 95% Bootstrap CI",
                 fontsize=13, fontweight="bold", pad=18)

    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.22, 0.13, 0.13, 0.13, 0.13, 0.13, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # Header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight best per metric column
    highlight = [
        ("MAE",   False), ("RMSE",  False), ("Corr", True),
        ("Kappa", True),  ("WD",    False),
    ]
    for col_idx, (metric, higher) in enumerate(highlight):
        vals = [df[df.Model == mk].iloc[0][metric] for _, mk in KEY_MODELS
                if not df[df.Model == mk].empty]
        best_i = int(np.nanargmax(vals) if higher else np.nanargmin(vals))
        table[(best_i + 1, col_idx + 1)].set_facecolor("#d5f5e3")

    plt.tight_layout()
    save_fig("4c_key_summary_table")
    plt.close()
    print("  [4c] Key model summary table saved.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    for d in [OUTPUT_DIR, DIR_PNG, DIR_PDF]:
        os.makedirs(d, exist_ok=True)

    df = load()
    print(f"Loaded {len(df)} models from {SUMMARY_CSV}")
    models = load_raw_models(".")
    print(f"Loaded {len(models)} raw model CSVs\n")

    print("=== Analysis 1: Fewshot vs Zeroshot ===")
    plot_1a_grouped_bar(df)
    plot_1b_delta_heatmap(df)
    plot_1c_radar(df)

    print("\n=== Analysis 2: DeepSeek-R1 Scaling ===")
    plot_2a_scaling_bar(df)
    plot_2b_scaling_profile(df)
    plot_2c_scaling_delta(df)

    print("\n=== Analysis 3: Reasoning Distillation (Qwen2.5-32b base vs DeepSeek-R1-32b distilled) ===")
    plot_3a_distill_interaction(df)
    plot_3b_distill_delta(df)
    plot_3c_distill_summary_table(df)

    print("\n=== Analysis 4: Key Models Focused Plots ===")
    plot_4a_key_per_session(models)
    plot_4b_key_vs_consensus(models)
    plot_4c_key_summary_table(df)

    print(f"\nAll outputs saved to '{OUTPUT_DIR}/'.")
    for sub in ("png", "pdf"):
        sub_dir = os.path.join(OUTPUT_DIR, sub)
        if os.path.isdir(sub_dir):
            print(f"  [{sub}/]")
            for f in sorted(os.listdir(sub_dir)):
                print(f"    - {f}")


if __name__ == "__main__":
    main()
