"""
Multi-Model Performance Comparison Script
==========================================
Generates comprehensive visualizations and tables for comparing
multiple AI models against human expert (doctor) ground truth.

Presentation styles adapted from thesis Tables 3.3-3.11 and Figures 3.2-3.5,
extended for multi-model comparison.

Outputs (in multi_model_results/png/ and multi_model_results/pdf/):
  1. summary_table        - Metrics table with 95% bootstrap CI (like Table 3.10/3.11)
  2. grouped_barplot       - Grouped bar chart per metric (like Figure 3.3)
  3. per_session_lineplot  - Per-session score overlay (like Figure 3.4/3.5)
  4. score_distribution    - Score distribution per model (like Figure 3.2)
  5. radar_chart           - Multi-dimensional radar comparison (NEW)
  6. metrics_heatmap       - Models x Metrics heatmap (NEW)
  7. wd_by_scoregroup      - WD broken down by ground truth score (like Table 3.7)
  8. pairwise_tests        - Pairwise KS test p-values (like Table 3.8/3.9)
"""

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, cohen_kappa_score, recall_score,
                             accuracy_score, precision_score, f1_score, roc_auc_score)
from scipy.stats import wasserstein_distance, ks_2samp, spearmanr
from math import pi
import hashlib
import json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "multi_model_results"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
SCORE_RANGE = range(5)  # 0-4
MAX_SCORE = 4
CACHE_VERSION = "v2"  # bump when metrics change to invalidate stale caches

# Plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})
sns.set_theme(style="whitegrid")

DIR_PNG = os.path.join(OUTPUT_DIR, "png")
DIR_PDF = os.path.join(OUTPUT_DIR, "pdf")


def save_fig(filename_no_ext, png_only=False, **kwargs):
    """Save current figure to both png/ and pdf/ subdirectories.

    Pass png_only=True to skip the PDF (e.g. for very large figures that
    would exceed available memory during PDF rasterization).
    """
    kw = dict(dpi=300, bbox_inches="tight")
    kw.update(kwargs)
    plt.savefig(os.path.join(DIR_PNG, f"{filename_no_ext}.png"), **kw)
    if not png_only:
        plt.savefig(os.path.join(DIR_PDF, f"{filename_no_ext}.pdf"), **kw)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _file_hash(filepath):
    """MD5 hash of file contents for cache invalidation."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _data_hash(arr):
    """Hash a numpy array for cache keying."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _load_json_cache(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_json_cache(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _to_native(v):
    """Convert numpy scalars / NaN to JSON-serializable Python types."""
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer, np.bool_)):
        return int(v)
    return v  # str, bool, None


def _restore_rows(rows):
    """Convert None back to np.nan in rows loaded from JSON cache."""
    return [{k: (np.nan if v is None else v) for k, v in row.items()}
            for row in rows]


def extract_model_name(filename):
    """Extract a readable model name from the CSV filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    if "scored_by_" in base:
        suffix = base.split("scored_by_", 1)[1]
        # Clean up common patterns
        name = suffix.replace("_nothink", "").replace("_0227", "").replace("_0208", "")
        return name
    return base


def load_all_models(folder="."):
    """Load all CSV files and return ({model_name: DataFrame}, {model_name: filepath})."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    models = {}
    filepaths = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            ai_cols = [str(i) for i in range(10)]
            if not all(c in df.columns for c in ai_cols):
                continue
            name = extract_model_name(f)
            # Compute per-row summaries
            df["ai_median"] = df[ai_cols].median(axis=1)
            df["ai_mean"] = df[ai_cols].mean(axis=1)
            df["ai_std"] = df[ai_cols].std(axis=1)
            # Consensus (average of two doctors)
            if "label_Dr. Tan" in df.columns and "label_Dr. Chien" in df.columns:
                df["consensus"] = (df["label_Dr. Tan"] + df["label_Dr. Chien"]) / 2
            elif "average_score_Doctors" in df.columns:
                df["consensus"] = df["average_score_Doctors"]
            else:
                continue
            models[name] = df
            filepaths[name] = os.path.abspath(f)
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    return models, filepaths


def _icc31(y_true, y_pred):
    """ICC(3,1) — two-way mixed, single measures, consistency.

    Formula: (MSR - MSE) / (MSR + (k-1)*MSE)
    where k=2 raters, MSR = between-subjects MS, MSE = residual MS.
    """
    k = 2
    n = len(y_true)
    ratings = np.column_stack([y_true, y_pred])  # shape (n, 2)
    grand_mean = ratings.mean()
    row_means  = ratings.mean(axis=1)
    col_means  = ratings.mean(axis=0)

    SSR = k * np.sum((row_means - grand_mean) ** 2)
    SSC = n * np.sum((col_means - grand_mean) ** 2)
    SST = np.sum((ratings - grand_mean) ** 2)
    SSE = SST - SSR - SSC

    MSR = SSR / (n - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    if MSR + (k - 1) * MSE == 0:
        return np.nan
    return (MSR - MSE) / (MSR + (k - 1) * MSE)


def compute_metrics(y_true, y_pred):
    """Compute MAE, Spearman ρ, Weighted Kappa, Adjacent Accuracy, ICC(3,1), and WD."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {k: np.nan for k in ["MAE", "Spearman", "Kappa", "AAcc", "ICC", "WD"]}

    # Round median predictions to nearest integer (valid MDS-UPDRS scores 0-4)
    yt_r = np.clip(np.round(yt).astype(int), 0, MAX_SCORE).astype(float)
    yp_r = np.clip(np.round(yp).astype(int), 0, MAX_SCORE).astype(float)

    mae      = mean_absolute_error(yt_r, yp_r)
    if np.unique(yt_r).size < 2 or np.unique(yp_r).size < 2:
        spearman = np.nan
    else:
        spearman = spearmanr(yt_r, yp_r).statistic
    try:
        kappa = cohen_kappa_score(
            yt_r.astype(int),
            yp_r.astype(int),
            weights="quadratic",
        )
    except Exception:
        kappa = np.nan
    acc  = (yt_r == yp_r).mean()
    aacc = (np.abs(yt_r - yp_r) <= 1).mean()
    icc  = _icc31(yt_r, yp_r)
    wd   = wasserstein_distance(yt_r, yp_r)
    return {"MAE": mae, "Spearman": spearman, "Kappa": kappa, "Acc": acc,
            "AAcc": aacc, "ICC": icc, "WD": wd}


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Bootstrap 95% CI for all metrics."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    records = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        records.append(compute_metrics(y_true[idx], y_pred[idx]))
    boot_df = pd.DataFrame(records)
    alpha = (1 - ci) / 2
    ci_dict = {}
    for col in boot_df.columns:
        lo = boot_df[col].quantile(alpha)
        hi = boot_df[col].quantile(1 - alpha)
        ci_dict[col] = (lo, hi)
    return ci_dict


def format_with_ci(val, ci_lo, ci_hi, decimals=2):
    """Format value with 95% CI like '0.57 [0.43, 0.74]'."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(val)} [{fmt.format(ci_lo)}, {fmt.format(ci_hi)}]"


# ---------------------------------------------------------------------------
# 1. Summary Table with Bootstrap CI  (like Table 3.10 / 3.11)
# ---------------------------------------------------------------------------

def _compute_acc(y_true, y_pred, max_score=MAX_SCORE):
    """Compute accuracy: fraction where round(pred) == round(true)."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = np.clip(np.round(y_true[mask]).astype(int), 0, max_score)
    yp = np.clip(np.round(y_pred[mask]).astype(int), 0, max_score)
    if len(yt) == 0:
        return np.nan
    return (yt == yp).mean()


def _compute_recall(y_true, y_pred, max_score=MAX_SCORE):
    """Compute macro-averaged recall across score classes 0..max_score."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = np.clip(np.round(y_true[mask]).astype(int), 0, max_score)
    yp = np.clip(np.round(y_pred[mask]).astype(int), 0, max_score)
    if len(yt) == 0:
        return np.nan
    labels = list(range(max_score + 1))
    return recall_score(yt, yp, labels=labels, average="macro", zero_division=0)


def _compute_binary_metrics(y_true, y_pred, threshold=1, max_score=MAX_SCORE):
    """Binarize scores (0 = Normal, >=threshold = Abnormal) and compute ACC, Recall, Precision, F1, AUC.

    Positive class = Abnormal (score >= threshold).
    AUC uses the continuous predicted score (before rounding) as the decision score.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = np.clip(np.round(y_true[mask]).astype(int), 0, max_score)
    yp_cont = y_pred[mask]                                          # continuous score for AUC
    yp = np.clip(np.round(yp_cont).astype(int), 0, max_score)
    if len(yt) == 0:
        return {k: np.nan for k in ["Bin_ACC", "Bin_Recall", "Bin_Precision", "Bin_F1", "AUC"]}
    yt_bin = (yt >= threshold).astype(int)
    yp_bin = (yp >= threshold).astype(int)
    try:
        auc = roc_auc_score(yt_bin, yp_cont)
    except Exception:
        auc = np.nan
    return {
        "Bin_ACC": accuracy_score(yt_bin, yp_bin),
        "Bin_Recall": recall_score(yt_bin, yp_bin, zero_division=0),
        "Bin_Precision": precision_score(yt_bin, yp_bin, zero_division=0),
        "Bin_F1": f1_score(yt_bin, yp_bin, zero_division=0),
        "AUC": auc,
    }


def _compute_baselines(models):
    """Compute Doctor baseline and Random baseline rows.

    Doctor baselines:
      - 'Doctor (Tan)':   uses Dr. Tan's score as the prediction vs consensus.
      - 'Doctor (Chien)': uses Dr. Chien's score as the prediction vs consensus.
    Random baseline:
      - 'Random (uniform)': uniformly random scores 0-4, averaged over 10,000 trials.
      - 'Random (proportional)': random scores sampled proportionally to the ground-truth
        distribution, averaged over 10,000 trials.

    Results are cached to CACHE_DIR/baselines.json keyed by a hash of the ground-truth
    data, so repeated runs skip the expensive random simulation entirely.
    """
    # --- Cache check ---
    cache_path = os.path.join(CACHE_DIR, "baselines.json")
    sample_df = list(models.values())[0]
    yt_for_hash = sample_df["consensus"].values.astype(float)
    data_key = _data_hash(yt_for_hash)
    cache = _load_json_cache(cache_path)
    if cache.get("data_hash") == data_key and cache.get("version") == CACHE_VERSION:
        print("  [cache] Baselines loaded from cache (skipping 20,000 random trials).")
        return _restore_rows(cache["rows"])

    N_RANDOM_TRIALS = 10_000
    rng = np.random.default_rng(0)

    sample_df = list(models.values())[0]
    yt = sample_df["consensus"].values.astype(float)
    yt_r = np.clip(np.round(yt).astype(int), 0, MAX_SCORE).astype(float)
    n = len(yt_r)

    baseline_rows = []

    # --- Doctor baselines ---
    for doc_col, doc_name in [("label_Dr. Tan", "Doctor (Tan)"),
                               ("label_Dr. Chien", "Doctor (Chien)")]:
        if doc_col not in sample_df.columns:
            continue
        yp = sample_df[doc_col].values.astype(float)
        metrics = compute_metrics(yt, yp)
        ci = bootstrap_ci(yt, yp)
        row = {"Model": doc_name, "is_baseline": True}
        for m in ["MAE", "Spearman", "Kappa", "Acc", "AAcc"]:
            row[m] = metrics[m]
            row[f"{m}_CI_lo"] = ci[m][0]
            row[f"{m}_CI_hi"] = ci[m][1]
            row[f"{m}_fmt"] = format_with_ci(metrics[m], ci[m][0], ci[m][1])
        row["Stability"] = 0.0  # doctors give deterministic scores
        row["ACC_Tan"] = np.nan
        row["ACC_Chien"] = np.nan
        row["ACC_Consensus"] = _compute_acc(yt, yp)
        row["Recall_Tan"] = np.nan
        row["Recall_Chien"] = np.nan
        row["Recall_Consensus"] = _compute_recall(yt, yp)
        bin_metrics = _compute_binary_metrics(yt, yp)
        row.update(bin_metrics)
        baseline_rows.append(row)

    # --- Random baselines ---
    for strategy, strat_name in [("uniform", "Random (uniform)"),
                                  ("proportional", "Random (proportional)")]:
        # Aggregate metrics over many random trials
        trial_metrics = []
        for _ in range(N_RANDOM_TRIALS):
            if strategy == "uniform":
                yp = rng.integers(0, MAX_SCORE + 1, size=n).astype(float)
            else:
                # Sample proportionally to the observed ground-truth distribution
                unique, counts = np.unique(yt_r, return_counts=True)
                probs = counts / counts.sum()
                yp = rng.choice(unique, size=n, p=probs).astype(float)
            trial_metrics.append(compute_metrics(yt_r, yp))

        trial_df = pd.DataFrame(trial_metrics)
        mean_m = trial_df.mean()
        lo_m = trial_df.quantile(0.025)
        hi_m = trial_df.quantile(0.975)

        row = {"Model": strat_name, "is_baseline": True}
        for m in ["MAE", "Spearman", "Kappa", "Acc", "AAcc"]:
            row[m] = mean_m[m]
            row[f"{m}_CI_lo"] = lo_m[m]
            row[f"{m}_CI_hi"] = hi_m[m]
            row[f"{m}_fmt"] = format_with_ci(mean_m[m], lo_m[m], hi_m[m])
        row["Stability"] = np.nan  # not applicable
        row["ACC_Tan"] = np.nan
        row["ACC_Chien"] = np.nan
        row["ACC_Consensus"] = mean_m.get("ACC", np.nan)
        row["Recall_Tan"] = np.nan
        row["Recall_Chien"] = np.nan
        row["Recall_Consensus"] = np.nan
        # Binary metrics for random
        trial_bin = []
        for _ in range(N_RANDOM_TRIALS):
            if strategy == "uniform":
                yp = rng.integers(0, MAX_SCORE + 1, size=n).astype(float)
            else:
                yp = rng.choice(unique, size=n, p=probs).astype(float)
            trial_bin.append(_compute_binary_metrics(yt_r, yp))
        bin_df = pd.DataFrame(trial_bin)
        for k in ["Bin_ACC", "Bin_Recall", "Bin_Precision", "Bin_F1"]:
            row[k] = bin_df[k].mean()
        baseline_rows.append(row)

    # --- Save to cache ---
    serializable = [{k: _to_native(v) for k, v in row.items()} for row in baseline_rows]
    _save_json_cache(cache_path, {"data_hash": data_key, "version": CACHE_VERSION, "rows": serializable})
    print("  [cache] Baseline results cached for future runs.")

    return baseline_rows


def generate_summary_table(models, filepaths=None):
    """Create a summary metrics table with 95% CI, saved as CSV + figure.

    Includes Doctor and Random baselines for reference.
    Bootstrap CI results are cached per model (keyed by CSV file hash) so
    only new or changed models require recomputation.
    """
    if filepaths is None:
        filepaths = {}

    ci_cache_path = os.path.join(CACHE_DIR, "model_bootstrap_ci.json")
    ci_cache = _load_json_cache(ci_cache_path)
    ci_cache_updated = False

    rows = []
    for name, df in models.items():
        yt = df["consensus"].values.astype(float)
        yp = df["ai_median"].values.astype(float)  # use median of 10 evals
        metrics = compute_metrics(yt, yp)

        # Bootstrap CI — use cache when the CSV hasn't changed
        fpath = filepaths.get(name, "")
        fhash = _file_hash(fpath) if fpath and os.path.exists(fpath) else ""
        cached_entry = ci_cache.get(name, {})
        _ci_metrics = ["MAE", "Spearman", "Kappa", "Acc", "AAcc"]
        if (fhash and cached_entry.get("file_hash") == fhash
                and all(m in cached_entry.get("ci", {}) for m in _ci_metrics)):
            raw_ci = cached_entry["ci"]
            ci = {m: (raw_ci[m][0], raw_ci[m][1]) for m in raw_ci}
            print(f"  [cache] Bootstrap CI for '{name}' loaded from cache.")
        else:
            ci = bootstrap_ci(yt, yp)
            ci_cache[name] = {
                "file_hash": fhash,
                "ci": {m: [float(ci[m][0]), float(ci[m][1])] for m in ci},
            }
            ci_cache_updated = True

        row = {"Model": name, "is_baseline": False}
        for m in ["MAE", "Spearman", "Kappa", "Acc", "AAcc"]:
            row[m] = metrics[m]
            row[f"{m}_CI_lo"] = ci[m][0]
            row[f"{m}_CI_hi"] = ci[m][1]
            row[f"{m}_fmt"] = format_with_ci(metrics[m], ci[m][0], ci[m][1])
        # ICC and WD (no CI for these)
        row["ICC"] = metrics["ICC"]
        row["WD"]  = metrics["WD"]
        # Stability (avg SD across 10 evaluations)
        row["Stability"] = df["ai_std"].mean()

        # Per-rater ACC / Recall and consensus ACC / Recall
        if "label_Dr. Tan" in df.columns:
            row["ACC_Tan"] = _compute_acc(
                df["label_Dr. Tan"].values.astype(float), yp)
            row["Recall_Tan"] = _compute_recall(
                df["label_Dr. Tan"].values.astype(float), yp)
        else:
            row["ACC_Tan"] = np.nan
            row["Recall_Tan"] = np.nan
        if "label_Dr. Chien" in df.columns:
            row["ACC_Chien"] = _compute_acc(
                df["label_Dr. Chien"].values.astype(float), yp)
            row["Recall_Chien"] = _compute_recall(
                df["label_Dr. Chien"].values.astype(float), yp)
        else:
            row["ACC_Chien"] = np.nan
            row["Recall_Chien"] = np.nan
        row["ACC_Consensus"] = _compute_acc(yt, yp)
        row["Recall_Consensus"] = _compute_recall(yt, yp)

        # Binary metrics (0 = Normal, >=1 = Abnormal) vs consensus
        bin_metrics = _compute_binary_metrics(yt, yp)
        row.update(bin_metrics)

        # Wasserstein Distance (overall score distribution)
        mask = ~(np.isnan(yt) | np.isnan(yp))
        yt_r = np.clip(np.round(yt[mask]).astype(int), 0, MAX_SCORE).astype(float)
        yp_r = np.clip(np.round(yp[mask]).astype(int), 0, MAX_SCORE).astype(float)
        row["WD"] = wasserstein_distance(yt_r, yp_r)

        rows.append(row)

    pruned_cache = {k: v for k, v in ci_cache.items() if k in models}
    if ci_cache_updated or len(pruned_cache) != len(ci_cache):
        _save_json_cache(ci_cache_path, pruned_cache)

    # --- Append baseline rows (Doctor + Random) ---
    baseline_rows = _compute_baselines(models)
    rows.extend(baseline_rows)

    summary = pd.DataFrame(rows)

    # CompositeScore = (1/3)(~MAE + ~Kappa + ~AAcc), min-max normalised, MAE inverted
    def _mm(s): lo, hi = s.min(), s.max(); return pd.Series(0.5, index=s.index) if hi == lo else (s - lo) / (hi - lo)
    summary["CompositeScore"] = (_mm(summary["MAE"].max() - summary["MAE"]) + _mm(summary["Kappa"]) + _mm(summary["AAcc"])) / 3.0

    # Save CSV
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics.csv"), index=False)

    # --- Render as figure (publication-quality table) ---
    col_labels = ["Model", "MAE ↓", "Spearman ρ ↑", "Kappa ↑", "Acc ↑", "AAcc ↑",
                  "Stability ↓", "ACC Tan ↑", "ACC Chien ↑", "ACC Cons. ↑",
                  "Composite ↑"]

    n = len(summary)
    fig_h = max(3, 1.5 + n * 0.55)
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")
    ax.set_title("Multi-Model Performance Summary with 95% Bootstrap CI",
                 fontsize=14, fontweight="bold", pad=18)

    cell_data = []
    for _, r in summary.iterrows():
        cell_data.append([
            r["Model"],
            r["MAE_fmt"], r["Spearman_fmt"], r["Kappa_fmt"], r["Acc_fmt"], r["AAcc_fmt"],
            f"{r['Stability']:.3f}",
            f"{r['ACC_Tan']:.1%}" if not np.isnan(r["ACC_Tan"]) else "N/A",
            f"{r['ACC_Chien']:.1%}" if not np.isnan(r["ACC_Chien"]) else "N/A",
            f"{r['ACC_Consensus']:.1%}",
            f"{r['CompositeScore']:.3f}",
        ])

    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.14, 0.09, 0.09, 0.09, 0.09, 0.09, 0.07, 0.08, 0.08, 0.08, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style baseline rows with distinct background
    for row_idx, (_, r) in enumerate(summary.iterrows()):
        if r.get("is_baseline", False):
            for j in range(len(col_labels)):
                table[(row_idx + 1, j)].set_facecolor("#f0e6d3")  # warm beige for baselines

    # Highlight best per column among AI models only (exclude baselines)
    ai_mask = ~summary["is_baseline"].values
    highlight_metrics = [
        ("MAE", False), ("Spearman", True), ("Kappa", True), ("Acc", True), ("AAcc", True),
        ("Stability", False),
        ("ACC_Tan", True), ("ACC_Chien", True), ("ACC_Consensus", True),
        ("CompositeScore", True),
    ]
    for col_idx, (metric, higher_better) in enumerate(highlight_metrics):
        vals = summary[metric].values.copy()
        vals[~ai_mask] = np.nan  # exclude baselines from best-highlighting
        if np.all(np.isnan(vals)):
            continue
        if higher_better:
            best_row = np.nanargmax(vals)
        else:
            best_row = np.nanargmin(vals)
        table[(best_row + 1, col_idx + 1)].set_facecolor("#d5f5e3")

    plt.tight_layout()
    save_fig("summary_table")
    plt.close()

    # --- Binary classification table (Normal vs Abnormal, threshold >= 1) ---
    bin_col_labels = ["Model", "Bin ACC ↑", "Bin Recall ↑", "Bin Precision ↑", "Bin F1 ↑"]

    fig_h2 = max(3, 1.5 + len(summary) * 0.55)
    fig, ax = plt.subplots(figsize=(12, fig_h2))
    ax.axis("off")
    ax.set_title("Binary Classification: Normal (0) vs Abnormal (≥1) — vs Consensus",
                 fontsize=14, fontweight="bold", pad=18)

    bin_cell_data = []
    for _, r in summary.iterrows():
        bin_cell_data.append([
            r["Model"],
            f"{r['Bin_ACC']:.1%}",
            f"{r['Bin_Recall']:.1%}",
            f"{r['Bin_Precision']:.1%}",
            f"{r['Bin_F1']:.1%}",
        ])

    bin_table = ax.table(
        cellText=bin_cell_data, colLabels=bin_col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.30, 0.15, 0.15, 0.15, 0.15],
    )
    bin_table.auto_set_font_size(False)
    bin_table.set_fontsize(10)
    bin_table.scale(1.0, 1.7)

    for j in range(len(bin_col_labels)):
        bin_table[(0, j)].set_facecolor("#2c3e50")
        bin_table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style baseline rows
    for row_idx, (_, r) in enumerate(summary.iterrows()):
        if r.get("is_baseline", False):
            for j in range(len(bin_col_labels)):
                bin_table[(row_idx + 1, j)].set_facecolor("#f0e6d3")

    # Highlight best per column among AI models only
    bin_highlight = ["Bin_ACC", "Bin_Recall", "Bin_Precision", "Bin_F1"]
    for col_idx, metric in enumerate(bin_highlight):
        vals = summary[metric].values.copy()
        vals[summary["is_baseline"].values] = np.nan
        if np.all(np.isnan(vals)):
            continue
        best_row = np.nanargmax(vals)
        bin_table[(best_row + 1, col_idx + 1)].set_facecolor("#d5f5e3")

    plt.tight_layout()
    save_fig("summary_table_binary")
    plt.close()

    print("  [1/8] Summary tables saved (multiclass + binary).")
    return summary


# ---------------------------------------------------------------------------
# 2. Grouped Bar Chart  (like Figure 3.3)
# ---------------------------------------------------------------------------

def generate_grouped_barplot(summary, models):
    """Grouped bar chart for each metric across models."""
    model_names = summary["Model"].tolist()
    n = len(model_names)

    metrics_to_plot = [
        ("MAE", "Mean Absolute Error ↓", "Reds_r"),
        ("Spearman", "Spearman ρ ↑", "Greens"),
        ("Kappa", "Weighted Kappa ↑", "Blues"),
        ("AAcc", "Adjacent Accuracy ↑", "Purples"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Also compute Human Baseline
    sample_df = list(models.values())[0]
    if "label_Dr. Tan" in sample_df.columns and "label_Dr. Chien" in sample_df.columns:
        human_metrics = compute_metrics(
            sample_df["label_Dr. Tan"].values.astype(float),
            sample_df["label_Dr. Chien"].values.astype(float),
        )
    else:
        human_metrics = None

    for idx, (metric, title, cmap) in enumerate(metrics_to_plot):
        ax = axes.flat[idx]
        vals = summary[metric].values
        ci_lo = summary[f"{metric}_CI_lo"].values
        ci_hi = summary[f"{metric}_CI_hi"].values
        yerr = np.array([vals - ci_lo, ci_hi - vals])

        colors = sns.color_palette(cmap, n)
        bars = ax.bar(range(n), vals, yerr=yerr, capsize=4,
                      color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric)

        # Human baseline horizontal line
        if human_metrics and not np.isnan(human_metrics.get(metric, np.nan)):
            ax.axhline(human_metrics[metric], color="red", linestyle="--",
                       linewidth=1.5, label="Human Baseline")
            ax.legend(fontsize=7, loc="best")

        # Annotate bar values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # (all 4 subplots used)

    plt.tight_layout()
    save_fig("grouped_barplot")
    plt.close()
    print("  [2/8] Grouped bar plot saved.")


# ---------------------------------------------------------------------------
# 3. Per-Session Line Plot  (like Figure 3.4 / 3.5)
# ---------------------------------------------------------------------------

def generate_per_session_lineplot(models):
    """Overlay model median scores vs ground truth for each session."""
    # Use first model's consensus as ground truth (same across all)
    first_df = list(models.values())[0]
    gt = first_df["consensus"].values
    n_sessions = len(gt)
    x = np.arange(1, n_sessions + 1)

    # --- Plot A: All models overlaid ---
    fig, ax = plt.subplots(figsize=(max(12, n_sessions * 0.3), 5))
    ax.plot(x, gt, "rs-", label="Ground Truth (Consensus)", linewidth=2,
            markersize=6, zorder=10)

    cmap = plt.colormaps["tab10"]
    for i, (name, df) in enumerate(models.items()):
        median_scores = df["ai_median"].values
        ax.plot(x, median_scores, "o-", label=name, color=cmap(i),
                linewidth=1.2, markersize=4, alpha=0.8)

    ax.set_xlabel("Session")
    ax.set_ylabel("Score")
    ax.set_title("Per-Session Score Comparison: All Models vs Ground Truth",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(-0.3, MAX_SCORE + 0.5)
    plt.tight_layout()
    save_fig("per_session_all")
    plt.close()

    # --- Plot B: One subplot per model (like Figure 3.4) ---
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, (name, df) in enumerate(models.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        median_scores = df["ai_median"].values
        ax.plot(x, gt, "rs-", label="Ground Truth", linewidth=1.5, markersize=5)
        ax.plot(x, median_scores, "bo-", label=f"{name}", linewidth=1.2, markersize=4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Session")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.3, MAX_SCORE + 0.5)
        ax.legend(fontsize=7)
        ax.set_xticks(x[::max(1, n_sessions // 15)])

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig("per_session_grid")
    plt.close()
    print("  [3/8] Per-session line plots saved.")


# ---------------------------------------------------------------------------
# 4. Score Distribution  (like Figure 3.2)
# ---------------------------------------------------------------------------

def generate_score_distribution(models):
    """Score distribution per ground truth score, for each model."""
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (name, df) in enumerate(models.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        gt_rounded = np.clip(np.round(df["consensus"].values).astype(int), 0, MAX_SCORE)
        pred_rounded = np.clip(np.round(df["ai_median"].values).astype(int), 0, MAX_SCORE)

        # Build counts per GT score
        data_for_plot = []
        for gt_score in SCORE_RANGE:
            mask = gt_rounded == gt_score
            if mask.sum() == 0:
                continue
            preds = pred_rounded[mask]
            for pred_score in SCORE_RANGE:
                count = (preds == pred_score).sum()
                match = "Match" if pred_score == gt_score else "Mismatch"
                data_for_plot.append({
                    "GT Score": gt_score,
                    "Predicted Score": pred_score,
                    "Count": count,
                    "Type": match,
                })

        if not data_for_plot:
            ax.set_title(name)
            continue

        plot_df = pd.DataFrame(data_for_plot)

        # Group by GT score, show stacked or grouped bars
        for gt_score in SCORE_RANGE:
            subset = plot_df[plot_df["GT Score"] == gt_score]
            if subset.empty:
                continue
            offset = gt_score * (MAX_SCORE + 2)
            for _, row in subset.iterrows():
                color = "#3498db" if row["Type"] == "Match" else "#2c3e50"
                ax.bar(offset + row["Predicted Score"], row["Count"],
                       color=color, edgecolor="white", width=0.8)

        # X-axis labels
        tick_positions = []
        tick_labels = []
        for gt_score in SCORE_RANGE:
            for ps in SCORE_RANGE:
                tick_positions.append(gt_score * (MAX_SCORE + 2) + ps)
                tick_labels.append(str(ps))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("Count")

        # Add GT score group labels
        for gt_score in SCORE_RANGE:
            center = gt_score * (MAX_SCORE + 2) + MAX_SCORE / 2
            ax.text(center, ax.get_ylim()[1] * 0.95, f"GT={gt_score}",
                    ha="center", fontsize=8, fontweight="bold", color="red")

    # Hide unused
    for idx in range(n_models, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#3498db", label="Match (pred=GT)"),
                       Patch(facecolor="#2c3e50", label="Mismatch")]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    save_fig("score_distribution")
    plt.close()
    print("  [4/8] Score distribution saved.")


# ---------------------------------------------------------------------------
# 5. Radar Chart (NEW - multi-dimensional model fingerprint)
# ---------------------------------------------------------------------------

def generate_radar_chart(summary):
    """Radar (spider) chart showing each model's normalized metric profile (AI models only)."""
    summary = summary[~summary["is_baseline"]].reset_index(drop=True)
    metrics = ["MAE", "Spearman", "Kappa", "AAcc"]
    labels = ["MAE ↓", "Spearman ρ ↑", "Kappa ↑", "AAcc ↑"]
    n_metrics = len(metrics)

    # Normalize: for all metrics, higher = better on radar
    # Invert MAE (lower is better → invert so higher = better on radar)
    normalized = {}
    for m in metrics:
        vals = summary[m].values
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmax == vmin:
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - vmin) / (vmax - vmin)
        if m == "MAE":
            norm = 1 - norm  # invert: lower raw → higher on radar
        normalized[m] = norm

    angles = [n / n_metrics * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_title("Multi-Model Performance Radar\n(outer = better)",
                 fontsize=13, fontweight="bold", pad=30)

    cmap = plt.colormaps["tab10"]
    for i, (_, row) in enumerate(summary.iterrows()):
        values = [normalized[m][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=cmap(i),
                label=row["Model"], markersize=5)
        ax.fill(angles, values, alpha=0.08, color=cmap(i))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    save_fig("radar_chart")
    plt.close()
    print("  [5/8] Radar chart saved.")


# ---------------------------------------------------------------------------
# 6. Metrics Heatmap  (NEW - dense Models × Metrics view)
# ---------------------------------------------------------------------------

def generate_metrics_heatmap(summary, models=None):
    """Enhanced heatmap: Models × Metrics with rank annotations, overall rank column,
    baseline rows (Doctor, Random), and model rows sorted by composite performance."""
    metrics = ["MAE", "Spearman", "Kappa", "AAcc", "Stability"]
    display_labels = ["MAE ↓", "Spearman ρ ↑", "Kappa ↑", "AAcc ↑", "Stability ↓"]
    lower_better = {"MAE", "Stability"}

    # Separate AI models from baselines
    ai_df = summary[~summary["is_baseline"]].copy().reset_index(drop=True)
    baseline_df = summary[summary["is_baseline"]].copy().reset_index(drop=True)
    df = ai_df  # rank only AI models

    # --- Compute per-column ranks (1 = best) among AI models ---
    rank_mat = np.zeros((len(df), len(metrics)), dtype=int)
    for j, m in enumerate(metrics):
        vals = df[m].values.astype(float)
        if m in lower_better:
            order = np.argsort(np.argsort(vals))          # rank ascending
        else:
            order = np.argsort(np.argsort(-vals))          # rank descending
        rank_mat[:, j] = order + 1

    # Overall rank = mean rank across metrics (lower = better overall)
    df["OverallRank"] = rank_mat.mean(axis=1)
    df = df.sort_values("OverallRank").reset_index(drop=True)

    # Re-derive rank_mat after sorting
    for j, m in enumerate(metrics):
        vals = df[m].values.astype(float)
        if m in lower_better:
            order = np.argsort(np.argsort(vals))
        else:
            order = np.argsort(np.argsort(-vals))
        rank_mat[:, j] = order + 1

    model_names = df["Model"].tolist()
    mat = df[metrics].values.astype(float)
    n_ai = len(df)

    # Append baseline rows (Doctor + Random) below AI models
    n_baselines = len(baseline_df)
    if n_baselines > 0:
        bl_vals = baseline_df[metrics].values.astype(float)
        mat = np.vstack([mat, bl_vals])
        model_names.extend(baseline_df["Model"].tolist())
        # Pad rank_mat with zeros for baseline rows (will be skipped in annotation)
        rank_mat = np.vstack([rank_mat, np.zeros((n_baselines, len(metrics)), dtype=int)])

    n_models, n_metrics = mat.shape

    # --- Normalize per column for color (0 = worst, 1 = best) ---
    mat_norm = np.zeros_like(mat)
    for j in range(n_metrics):
        col = mat[:, j]
        # Use only AI models for normalization range
        ai_col = col[:n_ai]
        vmin, vmax = np.nanmin(ai_col), np.nanmax(ai_col)
        if vmax > vmin:
            mat_norm[:, j] = (col - vmin) / (vmax - vmin)
        else:
            mat_norm[:, j] = 0.5
        if metrics[j] in lower_better:
            mat_norm[:, j] = 1 - mat_norm[:, j]
        mat_norm[:, j] = np.clip(mat_norm[:, j], 0, 1)

    # --- Figure ---
    fig_h = max(4, n_models * 0.85 + 2.5)
    fig_w = max(10, n_metrics * 1.5 + 3)
    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w + 2, fig_h),
        gridspec_kw={"width_ratios": [n_metrics, 1], "wspace": 0.05},
        constrained_layout=True,
    )
    ax, ax_rank = axes

    # ---- Main heatmap ----
    # Use seaborn heatmap for cleaner rendering; pass pre-normalised matrix
    sns.heatmap(
        mat_norm,
        ax=ax,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Normalized Score (1 = best)", "shrink": 0.75},
        annot=False,   # manual annotation below for richer text
        square=False,
    )

    # Manual cell annotations: "0.234\n(#1)"
    for i in range(n_models):
        is_baseline = i >= n_ai
        for j in range(n_metrics):
            val = mat[i, j]
            norm_val = mat_norm[i, j]
            text_color = "white" if norm_val < 0.25 or norm_val > 0.80 else "black"
            if is_baseline:
                text_color = "#444444"

            if np.isnan(val):
                label = "N/A"
            elif is_baseline:
                label = f"{val:.3f}"
            else:
                rank = rank_mat[i, j]
                label = f"{val:.3f}\n(#{rank})"

            # Star best AI value per column
            is_best = (not is_baseline) and (rank_mat[i, j] == 1)
            weight = "bold" if is_best else "normal"
            prefix = "[*] " if is_best else ""

            ax.text(j + 0.5, i + 0.5, prefix + label,
                    ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight=weight)

    ax.set_xticks(np.arange(n_metrics) + 0.5)
    ax.set_xticklabels(display_labels, fontsize=10, rotation=0)
    ax.set_yticks(np.arange(n_models) + 0.5)
    ax.set_yticklabels(model_names, fontsize=10, rotation=0)

    # Draw a separator line above the baseline rows
    if n_baselines > 0:
        ax.axhline(n_ai, color="#555555", linewidth=2, linestyle="--")

    ax.set_title("Model Performance Heatmap  (green = better,  [*] = best per metric)",
                 fontsize=13, fontweight="bold", pad=14)

    # ---- Overall Rank mini-bar ----
    # Only for AI models (exclude baselines)
    overall_ranks = df["OverallRank"].values  # already sorted
    rank_norm = (overall_ranks - overall_ranks.min()) / max(overall_ranks.max() - overall_ranks.min(), 1e-9)

    bar_data = np.zeros((n_models, 1))
    bar_data[:n_ai, 0] = 1 - rank_norm  # visual fill: higher = better rank
    for bi in range(n_baselines):
        bar_data[n_ai + bi, 0] = np.nan

    bar_norm = np.where(np.isnan(bar_data), 0, bar_data)
    sns.heatmap(
        bar_norm,
        ax=ax_rank,
        cmap="RdYlGn",
        vmin=0, vmax=1,
        linewidths=0.5, linecolor="white",
        cbar=False,
        annot=False,
    )
    for i in range(n_ai):
        ax_rank.text(0.5, i + 0.5, f"{overall_ranks[i]:.1f}",
                     ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white" if rank_norm[i] > 0.65 or rank_norm[i] < 0.2 else "black")
    for bi in range(n_baselines):
        ax_rank.text(0.5, n_ai + bi + 0.5, "—", ha="center", va="center",
                     fontsize=10, color="#888888")
    if n_baselines > 0:
        ax_rank.axhline(n_ai, color="#555555", linewidth=2, linestyle="--")

    ax_rank.set_xticks([0.5])
    ax_rank.set_xticklabels(["Avg\nRank ↓"], fontsize=9)
    ax_rank.set_yticks([])
    ax_rank.set_title("", pad=14)

    save_fig("metrics_heatmap")
    plt.close()
    print("  [6/8] Metrics heatmap saved.")


# ---------------------------------------------------------------------------
# 7. WD by Score Group  (like Table 3.7)
# ---------------------------------------------------------------------------

def generate_wd_by_scoregroup(models):
    """WD broken down by ground truth score group (0-4) for each model."""
    rows = []
    for name, df in models.items():
        gt = df["consensus"].values.astype(float)
        pred = df["ai_median"].values.astype(float)
        gt_rounded = np.clip(np.round(gt).astype(int), 0, MAX_SCORE)

        pred_r = np.clip(np.round(pred).astype(int), 0, MAX_SCORE).astype(float)
        gt_r = gt_rounded.astype(float)
        row = {"Model": name}
        row["WD_overall"] = wasserstein_distance(gt_r, pred_r)

        for score in SCORE_RANGE:
            mask = gt_rounded == score
            n_sessions = mask.sum()
            row[f"n_{score}"] = n_sessions
            if n_sessions >= 2:
                row[f"WD_{score}"] = wasserstein_distance(gt_r[mask], pred_r[mask])
            else:
                row[f"WD_{score}"] = np.nan
        rows.append(row)

    wd_df = pd.DataFrame(rows)

    # --- Render as table figure ---
    col_labels = ["Model", "WD overall"]
    for s in SCORE_RANGE:
        col_labels.append(f"Score {s}")

    cell_data = []
    for _, r in wd_df.iterrows():
        row_data = [r["Model"], f"{r['WD_overall']:.3f}"]
        for s in SCORE_RANGE:
            n = int(r[f"n_{s}"])
            wd = r[f"WD_{s}"]
            if np.isnan(wd):
                row_data.append(f"N ({n})")
            else:
                row_data.append(f"{wd:.2f} ({n})")
        cell_data.append(row_data)

    n = len(wd_df)
    fig_h = max(3, 1.5 + n * 0.55)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title("Wasserstein Distance by Ground Truth Score Group\n(count in parentheses)",
                 fontsize=13, fontweight="bold", pad=18)

    table = ax.table(
        cellText=cell_data, colLabels=col_labels,
        loc="center", cellLoc="center",
        colWidths=[0.22] + [0.13] * (len(col_labels) - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    plt.tight_layout()
    save_fig("wd_by_scoregroup")
    plt.close()

    # Also save CSV
    wd_df.to_csv(os.path.join(OUTPUT_DIR, "wd_by_scoregroup.csv"), index=False)
    print("  [7/8] WD by score group saved.")


# ---------------------------------------------------------------------------
# 8. Pairwise KS Tests  (like Table 3.8 / 3.9)
# ---------------------------------------------------------------------------

def generate_pairwise_tests(models):
    """Pairwise Kolmogorov-Smirnov test p-values between models' prediction errors."""
    model_names = list(models.keys())
    n = len(model_names)

    # Compute error distributions (prediction - ground truth)
    errors = {}
    for name, df in models.items():
        gt = df["consensus"].values.astype(float)
        pred = df["ai_median"].values.astype(float)
        errors[name] = pred - gt

    # Pairwise KS test
    pval_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            stat, pval = ks_2samp(errors[model_names[i]], errors[model_names[j]])
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval

    # --- Heatmap figure ---
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    ax.set_title("Pairwise KS Test p-values\n(error distribution comparison)",
                 fontsize=13, fontweight="bold", pad=15)

    # Mask diagonal
    mask = np.eye(n, dtype=bool)

    sns.heatmap(pval_matrix, annot=True, fmt=".3f",
                xticklabels=model_names, yticklabels=model_names,
                cmap="RdYlGn", vmin=0, vmax=1, mask=mask,
                linewidths=0.5, ax=ax, cbar_kws={"label": "p-value"})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    save_fig("pairwise_ks_test")
    plt.close()

    # --- Also save as formatted table ---
    pval_df = pd.DataFrame(pval_matrix, index=model_names, columns=model_names)
    pval_df.to_csv(os.path.join(OUTPUT_DIR, "pairwise_ks_pvalues.csv"))

    # Scientific notation table figure
    fig, ax = plt.subplots(figsize=(max(10, n * 2), max(3, 1.5 + n * 0.55)))
    ax.axis("off")
    ax.set_title("Pairwise KS Test p-values (scientific notation)",
                 fontsize=13, fontweight="bold", pad=18)

    cell_data = []
    for i in range(n):
        row_data = [model_names[i]]
        for j in range(n):
            if i == j:
                row_data.append("—")
            else:
                p = pval_matrix[i, j]
                if p < 0.001:
                    row_data.append(f"{p:.2e}")
                else:
                    row_data.append(f"{p:.3f}")
        cell_data.append(row_data)

    table = ax.table(
        cellText=cell_data,
        colLabels=["Model"] + model_names,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(n + 1):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=7)

    # Highlight significant cells (p < 0.05)
    for i in range(n):
        for j in range(n):
            if i != j and pval_matrix[i, j] < 0.05:
                table[(i + 1, j + 1)].set_facecolor("#fadbd8")

    plt.tight_layout()
    save_fig("pairwise_ks_table")
    plt.close()
    print("  [8/8] Pairwise KS tests saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# 9. Evaluation Pair Heatmaps  (like Figure 3.1)
# ---------------------------------------------------------------------------

def _build_pair_matrix(scores_a, scores_b, max_score=MAX_SCORE):
    """Build a percentage matrix from two score arrays (rounded to int)."""
    mat = np.zeros((max_score + 1, max_score + 1))
    valid = 0
    for a, b in zip(scores_a, scores_b):
        if np.isnan(a) or np.isnan(b):
            continue
        r = int(np.clip(np.round(a), 0, max_score))
        c = int(np.clip(np.round(b), 0, max_score))
        mat[r, c] += 1
        valid += 1
    if valid > 0:
        mat = mat / valid * 100
    return mat, valid


def _plot_heatmap(ax, mat, title, xlabel, ylabel, cmap="YlGnBu"):
    """Plot a single pair-heatmap on given axes."""
    sns.heatmap(mat, annot=True, fmt=".1f", cmap=cmap,
                vmin=0, vmax=max(mat.max(), 1),
                linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "%", "shrink": 0.8},
                ax=ax, square=True)
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(range(MAX_SCORE + 1), fontsize=7)
    ax.set_yticklabels(range(MAX_SCORE + 1), fontsize=7, rotation=0)


def generate_eval_pair_heatmaps(models):
    """
    Generate Figure-3.1-style heatmaps:
      A) Intra-model consistency: pairwise agreement among 10 eval runs
      B) Inter-model agreement:  model-A median vs model-B median
      C) Human baseline:         Dr. Tan vs Dr. Chien
    """
    ai_cols = [str(i) for i in range(10)]

    # ---- A) Intra-model consistency (one heatmap per model) ----
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows),
                             squeeze=False)

    for idx, (name, df) in enumerate(models.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # Collect all (eval_α, eval_β) pairs where α < β
        all_a, all_b = [], []
        for i in range(len(ai_cols)):
            for j in range(i + 1, len(ai_cols)):
                col_a, col_b = ai_cols[i], ai_cols[j]
                if col_a in df.columns and col_b in df.columns:
                    for _, row in df.iterrows():
                        a_val = row[col_a]
                        b_val = row[col_b]
                        if not (pd.isna(a_val) or pd.isna(b_val)):
                            all_a.append(float(a_val))
                            all_b.append(float(b_val))

        mat, n_pairs = _build_pair_matrix(
            np.array(all_a), np.array(all_b))
        _plot_heatmap(ax, mat, f"{name}\n(n={n_pairs} pairs)",
                      r"Score of Eval $\beta$", r"Score of Eval $\alpha$")

    # Hide unused
    for idx in range(n_models, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig("intra_model_heatmaps", png_only=True)
    plt.close()

    # ---- B) Inter-model agreement (model vs model median) ----
    model_names = list(models.keys())
    n = len(model_names)

    # Only upper triangle (avoid redundancy)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs_plot = len(pairs)
    ncols_b = min(3, n_pairs_plot)
    nrows_b = (n_pairs_plot + ncols_b - 1) // ncols_b

    fig, axes = plt.subplots(nrows_b, ncols_b,
                             figsize=(5 * ncols_b, 4.5 * nrows_b),
                             squeeze=False)

    for idx, (i, j) in enumerate(pairs):
        r, c = divmod(idx, ncols_b)
        ax = axes[r][c]
        name_a, name_b = model_names[i], model_names[j]
        scores_a = models[name_a]["ai_median"].values
        scores_b = models[name_b]["ai_median"].values
        mat, nv = _build_pair_matrix(scores_a, scores_b)
        _plot_heatmap(ax, mat, f"{name_a}\nvs {name_b} (n={nv})",
                      name_b, name_a, cmap="PuBuGn")

    for idx in range(n_pairs_plot, nrows_b * ncols_b):
        r, c = divmod(idx, ncols_b)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig("inter_model_heatmaps", png_only=True)
    plt.close()

    # ---- C) Human baseline + Model vs Consensus (combined) ----
    sample_df = list(models.values())[0]
    has_doctors = ("label_Dr. Tan" in sample_df.columns and
                   "label_Dr. Chien" in sample_df.columns)

    n_plots = n_models + (1 if has_doctors else 0)
    ncols_c = min(3, n_plots)
    nrows_c = (n_plots + ncols_c - 1) // ncols_c

    fig, axes = plt.subplots(nrows_c, ncols_c,
                             figsize=(5 * ncols_c, 4.5 * nrows_c),
                             squeeze=False)

    plot_idx = 0

    # Each model vs consensus
    for name, df in models.items():
        r, c = divmod(plot_idx, ncols_c)
        ax = axes[r][c]
        mat, nv = _build_pair_matrix(
            df["ai_median"].values, df["consensus"].values)
        _plot_heatmap(ax, mat, f"{name}\nvs Consensus (n={nv})",
                      "Consensus", name, cmap="Purples")
        plot_idx += 1

    # Human baseline
    if has_doctors:
        r, c = divmod(plot_idx, ncols_c)
        ax = axes[r][c]
        mat, nv = _build_pair_matrix(
            sample_df["label_Dr. Tan"].values.astype(float),
            sample_df["label_Dr. Chien"].values.astype(float))
        _plot_heatmap(ax, mat, f"Two Neurologists (n={nv})",
                      "Dr. Chien", "Dr. Tan", cmap="OrRd")
        plot_idx += 1

    for idx in range(plot_idx, nrows_c * ncols_c):
        r, c = divmod(idx, ncols_c)
        axes[r][c].axis("off")

    plt.tight_layout()
    save_fig("model_vs_consensus_heatmaps", png_only=True)
    plt.close()

    print("  [9/9] Evaluation pair heatmaps saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Create output directories (png and pdf separated)
    for d in [OUTPUT_DIR, DIR_PNG, DIR_PDF]:
        os.makedirs(d, exist_ok=True)

    # Load all models
    print("Loading CSV files...")
    models, filepaths = load_all_models(".")
    if not models:
        print("No valid CSV files found!")
        return

    print(f"Found {len(models)} models: {list(models.keys())}\n")

    # Generate all outputs
    summary = generate_summary_table(models, filepaths)
    generate_grouped_barplot(summary, models)
    generate_per_session_lineplot(models)
    generate_score_distribution(models)
    generate_radar_chart(summary)
    generate_metrics_heatmap(summary, models)
    generate_wd_by_scoregroup(models)
    generate_pairwise_tests(models)
    # generate_eval_pair_heatmaps(models)  # skipped: too slow with many models

    print(f"\nAll outputs saved to '{OUTPUT_DIR}/'.")
    print("Files generated:")
    for sub in ("png", "pdf"):
        sub_dir = os.path.join(OUTPUT_DIR, sub)
        if os.path.isdir(sub_dir):
            print(f"  [{sub}/]")
            for f in sorted(os.listdir(sub_dir)):
                print(f"    - {f}")


if __name__ == "__main__":
    main()
