"""
Feature Selection Comparison: MLP vs deepseek-r1:32b
=====================================================
Tests each traditional FS method's feature subset with two evaluators:
  1. MLP(8) LOOCV  (same as feature_selection_comparison.py)
  2. deepseek-r1:32b direct scoring via Ollama

For each FS method, the LLM receives only that method's selected features
for each patient and rates 0-4. Both evaluators are compared.
Individual per-eval scores are stored in cache; both mode and median
aggregations are computed and reported.

Usage:
  python feature_selection_comparison_with_llm.py
  python feature_selection_comparison_with_llm.py --llm-evals 3
  python feature_selection_comparison_with_llm.py --methods "Pearson" "LASSO (Standard)"
  python feature_selection_comparison_with_llm.py --skip-mlp
  python feature_selection_comparison_with_llm.py --skip-llm
  python feature_selection_comparison_with_llm.py --list-methods
"""

import os, json, warnings, time, re, requests, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
FEAT_CSV = "features_dataset_v2.csv"
FS_JSON  = "traditional_fs_results_v2/traditional_selection_results.json"

OUT_DIR  = "fs_results_llm"
PNG_DIR  = os.path.join(OUT_DIR, "png")
PDF_DIR  = os.path.join(OUT_DIR, "pdf")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

CACHE_FILE = os.path.join(OUT_DIR, "llm_score_cache.json")

# ─────────────────────────────────────────────────────────────────────────────
# LLM Config
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL          = "http://localhost:11434/api/generate"
LLM_MODEL           = "deepseek-r1:32b"
LLM_EVAL_TIMES      = 3
REQUEST_TIMEOUT     = 300
NUM_PREDICT         = 4096
DELAY_BETWEEN_CALLS = 2
MAX_RETRIES         = 3

# Few-shot examples: (integer_score, filename_prefix)
# Each prefix matches one or two rows in features_dataset_v2.csv
FEWSHOT_EXAMPLES = [
    (0, "553661824_20200629"),
    (1, "445817794_20200720"),
    (2, "879885247_20200608"),
    (3, "546503158_20200310"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MAX_SCORE = 4
N_BOOT    = 2000
RNG       = np.random.default_rng(42)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})
sns.set_theme(style="whitegrid")

FS_CATEGORY = {
    "Pearson":              "filter",
    "Spearman":             "filter",
    "Kendall":              "filter",
    "Mutual Info":          "filter",
    "F-regression":         "filter",
    "mRMR":                 "filter",
    "Distance Correlation": "filter",
    "HSIC":                 "filter",
    "Fisher Score":         "filter",
    "ReliefF":              "filter",
    "FCBF":                 "filter",
    "Variance Threshold":   "filter",
    "LASSO (Standard)":     "embedded",
    "ElasticNet":           "embedded",
    "Adaptive LASSO":       "embedded",
    "LARS":                 "embedded",
    "RF Importance":        "embedded",
    "Extra Trees":          "embedded",
    "Gradient Boosting":    "embedded",
    "XGBoost":              "embedded",
    "Permutation Importance": "embedded",
    "Stability Selection":  "embedded",
    "RFECV (RF)":           "wrapper",
    "RFECV (SVR)":          "wrapper",
    "RFECV (Lasso)":        "wrapper",
    "SFS Forward":          "wrapper",
    "SBS Backward":         "wrapper",
    "Boruta":               "wrapper",
}

CAT_COLOR = {
    "filter":   "#3498DB",
    "embedded": "#E67E22",
    "wrapper":  "#27AE60",
    "other":    "#95A5A6",
}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt template (based on experiments_llm/prompts/Prompt_fewshot_v2_dynamic.txt)
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_BASE = """The data provided below consists of extracted features derived from a subject performing the MDS-UPDRS Item 3.4 Finger Tapping test. The subject was instructed to tap the index finger on the thumb 10 times as quickly and as largely as possible.
The objective is to score the subject according to MDS-UPDRS item 3.4: FINGER TAPPING based on the selected 10 seconds of continuous finger tapping. The instruction and scoring criteria of MDS-UPDRS item 3.4: FINGER TAPPING are as follows:
"Instructions to the examiner: Each hand is tested separately. Demonstrate the task, but do not continue to perform the task while the patient is being tested. Instruct the patient to tap the index finger on the thumb 10 times as quickly AND as big as possible. Rate each side separately, evaluating speed, amplitude, hesitations, halts, and decrementing amplitude.
0: Normal: No problems.
1: Slight: Any of the following: a) the regular rhythm is broken with one or two interruptions or hesitations of the tapping movement; b) slight slowing; c) the amplitude decrements near the end of the 10 taps.
2: Mild: Any of the following: a) 3 to 5 interruptions during tapping; b) mild slowing; c) the amplitude decrements midway in the 10-tap sequence.
3: Moderate: Any of the following: a) more than 5 interruptions during tapping or at least one longer arrest (freeze) in ongoing movement; b) moderate slowing; c) the amplitude decrements starting after the 1st tap.
4: Severe: Cannot or can only barely perform the task because of slowing, interruptions, or decrements."
Below are the extracted features.
To determine the subject's score, analyze the provided features thoroughly and give a detailed motivation for the assigned score based on the MDS-UPDRS finger tapping instruction. Print the score using only one number on the last line of your answer, after the detailed motivation.

===================================
FEW-SHOT EXAMPLES
===================================
Below are examples for each score.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    drop_cols = {"hand", "filename", "Rating1", "Rating2", "Rating", "tap_amplitudes"}
    feat_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    X_full = df[feat_cols].values.astype(float)
    y      = df["Rating"].values.astype(float)
    print(f"  FEAT_CSV : {FEAT_CSV}")
    print(f"  Samples  : {len(df)}, Features: {len(feat_cols)}")
    return X_full, y, feat_cols, df


def load_feature_selections():
    with open(FS_JSON) as f:
        d = json.load(f)
    selections = {k: v for k, v in d["selections"].items() if "LLM-Lasso" not in k}
    print(f"  FS_JSON  : {FS_JSON}")
    print(f"  Methods  : {len(selections)}")
    return selections

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
def round_score(arr):
    return np.clip(np.floor(np.asarray(arr, dtype=float) + 0.5).astype(int), 0, MAX_SCORE)


def compute_metrics(y_true, y_pred):
    yt = round_score(np.array(y_true, dtype=float))
    yp = round_score(np.array(y_pred, dtype=float))
    mask = ~(np.isnan(yt.astype(float)) | np.isnan(yp.astype(float)))
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 5:
        return dict(MAE=np.nan, Spearman=np.nan, Kappa=np.nan, AAcc=np.nan)
    mae  = mean_absolute_error(yt, yp)
    sp   = spearmanr(yt, yp).statistic
    try:
        kappa = cohen_kappa_score(yt.astype(int), yp.astype(int), weights="quadratic")
    except Exception:
        kappa = np.nan
    aacc = (np.abs(yt - yp) <= 1).mean()
    return dict(MAE=mae, Spearman=sp, Kappa=kappa, AAcc=aacc)


def bootstrap_ci(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    records = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        records.append(compute_metrics(y_true[idx], y_pred[idx]))
    bdf = pd.DataFrame(records)
    result = {}
    for col in bdf.columns:
        result[f"{col}_CI_lo"] = bdf[col].quantile(0.025)
        result[f"{col}_CI_hi"] = bdf[col].quantile(0.975)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MLP LOOCV
# ─────────────────────────────────────────────────────────────────────────────
def mlp_loocv(X_sub, y):
    loo  = LeaveOneOut()
    mlp  = MLPRegressor(
        hidden_layer_sizes=(8,), activation="relu",
        solver="adam", alpha=1.0, max_iter=600, random_state=42,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   mlp),
    ])
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in loo.split(X_sub):
        try:
            pipe.fit(X_sub[train_idx], y[train_idx])
            preds[test_idx[0]] = pipe.predict(X_sub[test_idx])[0]
        except Exception:
            pass
    return preds

# ─────────────────────────────────────────────────────────────────────────────
# LLM prompt building
# ─────────────────────────────────────────────────────────────────────────────
def _format_feature_block(row, feat_names, indent="  "):
    lines = []
    for feat in feat_names:
        if feat in row.index:
            val = row[feat]
            if isinstance(val, float) and not np.isnan(val):
                val = round(float(val), 4)
            lines.append(f"{indent}{feat}: {val}")
    return "\n".join(lines)


def build_fewshot_text(df, feat_names):
    parts = []
    for score, prefix in FEWSHOT_EXAMPLES:
        mask = df["filename"].str.contains(prefix, na=False) if "filename" in df.columns else pd.Series([False]*len(df))
        if not mask.any():
            continue
        row = df[mask].iloc[0]
        parts.append(f"\n───────────────\nEXAMPLE FOR SCORE {score}\n───────────────")
        parts.append("STATISTICAL FEATURES:")
        parts.append(_format_feature_block(row, feat_names))
        parts.append(f"\nThis example was rated as SCORE {score}.\n")
    return "\n".join(parts)


def build_patient_prompt(row, feat_names, fewshot_text):
    subject_block  = "\n===================================\nSUBJECT TO SCORE\n==================================="
    subject_block += "\nSTATISTICAL FEATURES:\n"
    subject_block += _format_feature_block(row, feat_names)
    return PROMPT_BASE + fewshot_text + subject_block

# ─────────────────────────────────────────────────────────────────────────────
# LLM API
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  [Ollama] Cannot connect: {e}")
        return False


def call_ollama(prompt, retry=0):
    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0.7, "num_predict": NUM_PREDICT},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["response"]
    except (Exception, KeyboardInterrupt) as e:
        if retry < MAX_RETRIES - 1:
            time.sleep(5)
            return call_ollama(prompt, retry + 1)
        print(f"    [LLM ERROR] {type(e).__name__}: {e}")
        return None


def parse_score(response):
    if response is None:
        return None
    for line in reversed(response.strip().split("\n")):
        m = re.search(r"\b([0-4])\b", line)
        if m:
            return int(m.group(1))
    return None

# ─────────────────────────────────────────────────────────────────────────────
# LLM evaluation for one FS method
# ─────────────────────────────────────────────────────────────────────────────
def llm_evaluate_method(fs_name, df, feat_names, y, n_evals, cache):
    """Score all patients with LLM using feat_names.

    Cache stores raw per-patient trial scores (list of lists).
    Returns (preds_mode, preds_median) — both as float arrays.
    """
    cache_key = f"{fs_name}|{','.join(feat_names)}|n{n_evals}|v2"

    if cache_key in cache:
        print(f"    [cache hit]")
        raw = cache[cache_key]  # list of lists (per patient)
        return _aggregate(raw)

    fewshot_text = build_fewshot_text(df, feat_names)

    fewshot_mask = set()
    for _, prefix in FEWSHOT_EXAMPLES:
        matches = df[df["filename"].str.contains(prefix, na=False)].index.tolist() \
            if "filename" in df.columns else []
        fewshot_mask.update(matches)

    all_trial_scores = []
    for i, (_, row) in enumerate(df.iterrows()):
        flag = " [few-shot]" if i in fewshot_mask else ""
        patient_scores = []
        for e in range(n_evals):
            prompt = build_patient_prompt(row, feat_names, fewshot_text)
            resp   = call_ollama(prompt)
            s      = parse_score(resp)
            patient_scores.append(s)
            time.sleep(DELAY_BETWEEN_CALLS)

        valid = [s for s in patient_scores if s is not None]
        mode_val   = max(set(valid), key=valid.count) if valid else None
        median_val = float(np.median(valid))           if valid else None
        all_trial_scores.append(patient_scores)
        print(f"    [{i+1:2d}/{len(df)}]{flag} scores={patient_scores} "
              f"→ mode={mode_val}  median={median_val}")

    cache[cache_key] = all_trial_scores
    _save_cache(cache)
    return _aggregate(all_trial_scores)


def _aggregate(raw):
    """Convert raw per-patient score lists to (preds_mode, preds_median) arrays."""
    n = len(raw)
    preds_mode   = np.full(n, np.nan)
    preds_median = np.full(n, np.nan)
    for i, scores in enumerate(raw):
        valid = [s for s in scores if s is not None]
        if valid:
            preds_mode[i]   = max(set(valid), key=valid.count)
            preds_median[i] = float(np.median(valid))
    return preds_mode, preds_median

# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def run_comparison(X_full, y, feat_cols, df, selections, run_mlp, run_llm, n_evals):
    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    cache    = _load_cache() if run_llm else {}
    rows     = []

    for fs_name, feats in selections.items():
        idx = [feat_idx[f] for f in feats if f in feat_idx]
        if not idx:
            print(f"  [skip] {fs_name}: no valid features")
            continue
        X_sub    = X_full[:, idx]
        n_feats  = len(idx)
        category = FS_CATEGORY.get(fs_name, "other")

        print(f"\n  [{fs_name}]  n_features={n_feats}  category={category}")

        row = {"FS_Method": fs_name, "N_Features": n_feats, "Category": category}

        # ── MLP LOOCV ──────────────────────────────────────────────────────
        if run_mlp:
            print(f"    MLP LOOCV ...", end=" ", flush=True)
            preds_mlp = mlp_loocv(X_sub, y)
            m = compute_metrics(y, preds_mlp)
            ci = bootstrap_ci(y, preds_mlp)
            for k, v in m.items():
                row[f"MLP_{k}"] = v
            for k, v in ci.items():
                row[f"MLP_{k}"] = v
            print(f"MAE={m['MAE']:.3f}  Kappa={m['Kappa']:.3f}")

        # ── LLM scoring ───────────────────────────────────────────────────
        if run_llm:
            print(f"    LLM scoring ({n_evals}x per patient) ...")
            preds_mode, preds_median = llm_evaluate_method(
                fs_name, df, [f for f in feats if f in feat_idx],
                y, n_evals, cache
            )
            for agg_label, preds_llm in [("LLM_mode", preds_mode), ("LLM_median", preds_median)]:
                m  = compute_metrics(y, preds_llm)
                ci = bootstrap_ci(y, preds_llm)
                for k, v in m.items():
                    row[f"{agg_label}_{k}"] = v
                for k, v in ci.items():
                    row[f"{agg_label}_{k}"] = v
            coverage = (~np.isnan(preds_mode)).mean()
            row["LLM_Coverage"] = coverage
            print(f"    LLM mode:   MAE={compute_metrics(y,preds_mode)['MAE']:.3f}  "
                  f"Kappa={compute_metrics(y,preds_mode)['Kappa']:.3f}")
            print(f"    LLM median: MAE={compute_metrics(y,preds_median)['MAE']:.3f}  "
                  f"Kappa={compute_metrics(y,preds_median)['Kappa']:.3f}  Coverage={coverage:.0%}")

        rows.append(row)

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Normalise for composite
# ─────────────────────────────────────────────────────────────────────────────
def normalise(series, lower_is_better=False):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    norm = (series - lo) / (hi - lo)
    return (1 - norm) if lower_is_better else norm


def add_composite(df):
    df = df.copy()
    METRICS      = ["MAE", "Spearman", "Kappa", "AAcc"]
    LOWER_BETTER = {"MAE"}
    for prefix in ["MLP_", "LLM_"]:
        cols = [f"{prefix}{m}" for m in METRICS]
        if not all(c in df.columns for c in cols):
            continue
        for m in METRICS:
            df[f"{prefix}norm_{m}"] = normalise(df[f"{prefix}{m}"], m in LOWER_BETTER)
        df[f"{prefix}Composite"] = df[[f"{prefix}norm_{m}" for m in METRICS]].mean(axis=1)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(name):
    opts = dict(dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(PNG_DIR, f"{name}.png"), **opts)
    fig = plt.gcf()
    for ax in fig.get_axes():
        ax.set_title("")
    fig.suptitle("")
    plt.savefig(os.path.join(PDF_DIR, f"{name}.pdf"), **opts)
    plt.close()
    print(f"  Saved: {name}")


def plot_mlp_vs_llm_scatter(df, metric="Kappa"):
    mlp_col = f"MLP_{metric}"
    llm_col = f"LLM_{metric}"
    if mlp_col not in df.columns or llm_col not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [CAT_COLOR.get(FS_CATEGORY.get(fs, "other"), "#95A5A6")
              for fs in df["FS_Method"]]
    ax.scatter(df[mlp_col], df[llm_col], c=colors, s=80, edgecolors="white", linewidth=0.6)
    for _, row in df.iterrows():
        ax.annotate(row["FS_Method"], (row[mlp_col], row[llm_col]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6)

    lo = min(df[mlp_col].min(), df[llm_col].min()) - 0.05
    hi = max(df[mlp_col].max(), df[llm_col].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y=x")
    ax.set_xlabel(f"MLP {metric} ↑", fontsize=12)
    ax.set_ylabel(f"LLM ({LLM_MODEL}) {metric} ↑", fontsize=12)
    ax.set_title(f"MLP vs LLM: {metric} per FS method  (n=53, {LLM_MODEL})")
    ax.legend(fontsize=9)

    patches = [mpatches.Patch(color=c, label=k.capitalize())
               for k, c in CAT_COLOR.items() if k != "other"]
    fig.legend(handles=patches, title="FS Category", fontsize=9,
               loc="lower right", bbox_to_anchor=(1.01, 0.0))
    plt.tight_layout()
    save_fig(f"fig_mlp_vs_llm_{metric.lower()}")


def plot_ranked_comparison(df, metric="Kappa"):
    mlp_col = f"MLP_{metric}"
    llm_col = f"LLM_{metric}"
    has_mlp = mlp_col in df.columns
    has_llm = llm_col in df.columns
    if not has_mlp and not has_llm:
        return

    sort_col = mlp_col if has_mlp else llm_col
    sub = df.sort_values(sort_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(6, len(sub) * 0.3)))
    y_pos = np.arange(len(sub))
    bar_h = 0.35

    if has_mlp and has_llm:
        ax.barh(y_pos + bar_h/2, sub[mlp_col], bar_h, label="MLP(8)",
                color="#8E44AD", alpha=0.85, edgecolor="white")
        ax.barh(y_pos - bar_h/2, sub[llm_col], bar_h, label=LLM_MODEL,
                color="#27AE60", alpha=0.85, edgecolor="white")
    elif has_mlp:
        ax.barh(y_pos, sub[mlp_col], 0.6, label="MLP(8)",
                color="#8E44AD", alpha=0.85, edgecolor="white")
    else:
        ax.barh(y_pos, sub[llm_col], 0.6, label=LLM_MODEL,
                color="#27AE60", alpha=0.85, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["FS_Method"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(f"{metric} ↑", fontsize=11)
    ax.set_title(f"FS Methods Ranked by {metric}  –  MLP vs {LLM_MODEL}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    plt.tight_layout()
    save_fig(f"fig_ranked_{metric.lower()}")


def plot_heatmap_comparison(df):
    metrics = ["MAE", "Spearman", "Kappa", "AAcc"]
    evaluators = []
    if any(f"MLP_{m}" in df.columns for m in metrics):
        evaluators.append("MLP")
    if any(f"LLM_{m}" in df.columns for m in metrics):
        evaluators.append("LLM")

    n_panels = len(evaluators)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, max(6, len(df) * 0.25)))
    if n_panels == 1:
        axes = [axes]

    lower_better = {"MAE"}
    labels = {"MAE": "MAE ↓", "Spearman": "Spearman r ↑", "Kappa": "Kappa ↑", "AAcc": "AAcc ↑"}

    for ax, evaluator in zip(axes, evaluators):
        cols = [f"{evaluator}_{m}" for m in metrics if f"{evaluator}_{m}" in df.columns]
        pivot = df.set_index("FS_Method")[cols]
        pivot.columns = [c.replace(f"{evaluator}_", "") for c in pivot.columns]
        pivot = pivot.sort_values("Kappa" if "Kappa" in pivot.columns else pivot.columns[0],
                                  ascending=False)
        annot = pivot.applymap(lambda v: f"{v:.3f}" if pd.notna(v) else "–")
        sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn",
                    linewidths=0.5, linecolor="lightgray",
                    cbar_kws={"shrink": 0.6}, ax=ax)
        ax.set_title(f"{evaluator} ({'MLP(8) LOOCV' if evaluator == 'MLP' else LLM_MODEL})",
                     fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Feature Selection Method")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle(f"FS Methods – MLP vs {LLM_MODEL} Comparison  (n=53)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("fig_heatmap_comparison")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────
def print_table(df):
    has_mlp    = "MLP_MAE" in df.columns
    has_mode   = "LLM_mode_MAE" in df.columns
    has_median = "LLM_median_MAE" in df.columns
    print("\n" + "=" * 130)
    header = f"{'FS Method':<25} {'N':>4}"
    if has_mlp:
        header += f"  {'MLP_MAE':>8} {'MLP_Kap':>8}"
    if has_mode:
        header += f"  {'Mode_MAE':>9} {'Mode_Kap':>9}"
    if has_median:
        header += f"  {'Mdn_MAE':>8} {'Mdn_Kap':>8} {'Cov':>5}"
    print(header)
    print("=" * 130)

    sort_col = ("MLP_Kappa" if has_mlp else
                "LLM_mode_Kappa" if has_mode else
                "LLM_median_Kappa")
    sub = df.sort_values(sort_col, ascending=False) if sort_col in df.columns else df
    for _, row in sub.iterrows():
        line = f"  {row['FS_Method']:<23} {row['N_Features']:>4}"
        if has_mlp:
            line += (f"  {row.get('MLP_MAE', np.nan):>8.3f}"
                     f" {row.get('MLP_Kappa', np.nan):>8.3f}")
        if has_mode:
            line += (f"  {row.get('LLM_mode_MAE', np.nan):>9.3f}"
                     f" {row.get('LLM_mode_Kappa', np.nan):>9.3f}")
        if has_median:
            line += (f"  {row.get('LLM_median_MAE', np.nan):>8.3f}"
                     f" {row.get('LLM_median_Kappa', np.nan):>8.3f}"
                     f" {row.get('LLM_Coverage', np.nan):>5.0%}")
        print(line)
    print("=" * 130)

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FS Comparison: MLP vs deepseek-r1:7b")
    parser.add_argument("--methods",    nargs="+", metavar="METHOD",
                        help="FS method names to evaluate (default: all)")
    parser.add_argument("--llm-evals",  type=int,  default=LLM_EVAL_TIMES,
                        help=f"LLM evaluations per patient (default: {LLM_EVAL_TIMES})")
    parser.add_argument("--skip-mlp",   action="store_true", help="Skip MLP LOOCV")
    parser.add_argument("--skip-llm",   action="store_true", help="Skip LLM scoring")
    parser.add_argument("--list-methods", action="store_true", help="List available FS methods and exit")
    args = parser.parse_args()

    print("=" * 60)
    print("Feature Selection Comparison: MLP vs deepseek-r1:7b")
    print("=" * 60)

    print("\n[1] Loading data ...")
    X_full, y, feat_cols, df = load_data()
    selections = load_feature_selections()

    if args.list_methods:
        print("\nAvailable FS methods:")
        for name, feats in selections.items():
            print(f"  {name:<25} ({len(feats)} features)")
        return

    run_mlp = not args.skip_mlp
    run_llm = not args.skip_llm

    if run_llm and not check_ollama():
        print("  WARNING: Ollama not reachable. Skipping LLM evaluation.")
        run_llm = False

    if args.methods:
        missing = [m for m in args.methods if m not in selections]
        if missing:
            print(f"  ERROR: Unknown methods: {missing}")
            print(f"  Use --list-methods to see available options.")
            return
        selections = {k: v for k, v in selections.items() if k in args.methods}

    n_evals = args.llm_evals
    print(f"\n  Evaluators: {'MLP(8) LOOCV' if run_mlp else ''}"
          f"  {'+ ' if run_mlp and run_llm else ''}"
          f"{'LLM=' + LLM_MODEL + ' (' + str(n_evals) + 'x)' if run_llm else ''}")
    print(f"  Methods to run: {len(selections)}")

    print("\n[2] Running evaluation ...")
    results = run_comparison(X_full, y, feat_cols, df, selections,
                             run_mlp=run_mlp, run_llm=run_llm, n_evals=n_evals)
    results = add_composite(results)

    out_csv = os.path.join(OUT_DIR, "comparison_results.csv")
    results.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    print_table(results)

    print("\n[3] Generating figures ...")
    for metric in ["Kappa", "MAE", "Spearman", "AAcc"]:
        plot_mlp_vs_llm_scatter(results, metric=metric)
        plot_ranked_comparison(results, metric=metric)
    plot_heatmap_comparison(results)

    print(f"\nDone! → {PNG_DIR}/  and  {PDF_DIR}/")


if __name__ == "__main__":
    main()
