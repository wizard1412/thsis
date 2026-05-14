"""
Feature Selection Comparison
==============================
固定三個分類器（KNN-reg, SVM-rbf, RandomForest），
對 19 種特徵選擇方法的子集分別做 LOOCV，比較特徵選擇方法的效果。

資料: features_dataset.csv  (n=53, 66特徵)
特徵選擇結果: experiments_llm_lasso_deepseek70b/results/traditional_selection_results.json

輸出:
  fs_results/png/  &  fs_results/pdf/
  fs_results/loocv_results.csv
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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
# Feature dataset: prefer tap-seq extended version (has tap_amp_norm_01..20)
FEAT_CSV = "features_dataset_v2.csv"

# Feature selection results (traditional methods only; LLM-Lasso excluded)
FS_JSON = "traditional_fs_results_v2/traditional_selection_results.json"

OUT_DIR    = "fs_results_v2"
PNG_DIR    = os.path.join(OUT_DIR, "png")
PDF_DIR    = os.path.join(OUT_DIR, "pdf")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

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

# ─────────────────────────────────────────────────────────────────────────────
# 固定的三個分類器
# ─────────────────────────────────────────────────────────────────────────────
CLASSIFIERS = {
    "MLP(8)": MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        alpha=1.0,
        max_iter=600,
        random_state=42,
    ),
}

CLF_COLOR = {
    "MLP(8)": "#8E44AD",
}

# 特徵選擇方法的類別（用於顏色）
FS_CATEGORY = {
    # Filter
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
    # Embedded
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
    # Wrapper
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
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(name, **kw):
    opts = dict(dpi=300, bbox_inches="tight")
    opts.update(kw)
    plt.savefig(os.path.join(PNG_DIR, f"{name}.png"), **opts)
    fig = plt.gcf()
    fig.suptitle("")
    for ax in fig.get_axes():
        ax.set_title("")
    plt.savefig(os.path.join(PDF_DIR, f"{name}.pdf"), **opts)
    plt.close()
    print(f"  Saved: {name}")


def round_score(arr):
    return np.clip(np.floor(np.asarray(arr, dtype=float) + 0.5).astype(int), 0, MAX_SCORE)


def compute_metrics(y_true, y_pred):
    yt = round_score(np.array(y_true, dtype=float))
    yp = round_score(np.array(y_pred, dtype=float))
    mask = ~(np.isnan(yt.astype(float)) | np.isnan(yp.astype(float)))
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 5:
        return dict(MAE=np.nan, Spearman=np.nan, Kappa=np.nan, AAcc=np.nan)
    mae   = mean_absolute_error(yt, yp)
    sp    = spearmanr(yt, yp).statistic
    try:
        kappa = cohen_kappa_score(yt.astype(int), yp.astype(int), weights="quadratic")
    except Exception:
        kappa = np.nan
    aacc  = (np.abs(yt - yp) <= 1).mean()
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


def normalise(series, lower_is_better=False):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    norm = (series - lo) / (hi - lo)
    return (1 - norm) if lower_is_better else norm

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    drop_cols = {"hand", "filename", "Rating1", "Rating2", "Rating", "tap_amplitudes"}
    # Only keep numeric columns (guards against list/string columns like tap_amplitudes)
    feat_cols = [
        c for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    X_full = df[feat_cols].values.astype(float)
    y      = df["Rating"].values.astype(float)
    print(f"  FEAT_CSV : {FEAT_CSV}")
    print(f"  Columns  : {len(feat_cols)} numeric features")
    tap_cols = [c for c in feat_cols if c.startswith("tap_amp_norm_")]
    if tap_cols:
        print(f"  tap_amp_norm_* : {len(tap_cols)} sequence features included")
    return X_full, y, feat_cols


def load_feature_selections():
    with open(FS_JSON) as f:
        d = json.load(f)
    selections = d["selections"]   # {method_name: [feat1, feat2, ...]}
    # Drop any LLM-Lasso entries (LLM methods excluded from this comparison)
    selections = {k: v for k, v in selections.items() if "LLM-Lasso" not in k}
    print(f"  FS_JSON : {FS_JSON}")
    return selections

# ─────────────────────────────────────────────────────────────────────────────
# Run LOOCV
# ─────────────────────────────────────────────────────────────────────────────
def run_loocv(X_full, y, feat_cols, selections):
    loo = LeaveOneOut()
    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    rows = []

    for fs_name, feats in selections.items():
        # Map feature names → column indices
        idx = [feat_idx[f] for f in feats if f in feat_idx]
        if not idx:
            print(f"  [skip] {fs_name}: no valid features")
            continue
        X_sub = X_full[:, idx]
        n_feats = len(idx)

        for clf_name, clf in CLASSIFIERS.items():
            print(f"  LOOCV: {fs_name:25s} + {clf_name} ({n_feats} feats) ...")
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
                ("model",   clf),
            ])
            preds = np.full(len(y), np.nan)
            for train_idx, test_idx in loo.split(X_sub):
                try:
                    pipe.fit(X_sub[train_idx], y[train_idx])
                    preds[test_idx[0]] = pipe.predict(X_sub[test_idx])[0]
                except Exception:
                    pass

            metrics = compute_metrics(y, preds)
            ci      = bootstrap_ci(y, preds)
            rows.append({
                "FS_Method":  fs_name,
                "Classifier": clf_name,
                "N_Features": n_feats,
                "Category":   FS_CATEGORY.get(fs_name, "other"),
                **metrics, **ci,
            })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Add composite score
# ─────────────────────────────────────────────────────────────────────────────
METRICS      = ["MAE", "Spearman", "Kappa", "AAcc"]
LOWER_BETTER = {"MAE"}
METRIC_LABEL = {
    "MAE":      "MAE ↓",
    "Spearman": "Spearman r ↑",
    "Kappa":    "Weighted Kappa ↑",
    "AAcc":     "Adjacent Acc. ↑",
}

def add_composite(df):
    df = df.copy()
    for m in METRICS:
        df[f"norm_{m}"] = normalise(df[m], m in LOWER_BETTER)
    df["Composite"] = df[[f"norm_{m}" for m in METRICS]].mean(axis=1)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Fig A – Heatmap: FS method × Classifier  (one panel per metric)
# ─────────────────────────────────────────────────────────────────────────────
def figA_heatmap(df):
    import math
    ncols = 2
    nrows = math.ceil(len(METRICS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes_flat = axes.flatten()

    for ax, m in zip(axes_flat, METRICS):
        pivot = df.pivot(index="FS_Method", columns="Classifier", values=m)
        # Sort rows by mean across classifiers (best first)
        asc = m in LOWER_BETTER
        pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=asc).index]
        annot = pivot.applymap(lambda v: f"{v:.3f}" if pd.notna(v) else "–")
        cmap  = "RdYlGn_r" if m in LOWER_BETTER else "RdYlGn"
        sns.heatmap(pivot, annot=annot, fmt="", cmap=cmap,
                    linewidths=0.5, linecolor="lightgray",
                    cbar_kws={"label": m, "shrink": 0.6},
                    ax=ax)
        ax.set_title(METRIC_LABEL[m], fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Feature Selection Method")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")

    for k in range(len(METRICS), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("Feature Selection Methods × Classifiers – LOOCV Performance\n"
                 "(n=53; rows sorted by mean performance across classifiers)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("figA_fs_clf_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# Fig B – Ranked bar: FS method per classifier (composite score)
# ─────────────────────────────────────────────────────────────────────────────
def figB_ranked_bars(df):
    clf_names = list(CLASSIFIERS.keys())
    fig, axes = plt.subplots(1, len(clf_names), figsize=(7 * len(clf_names), 8))
    axes = np.atleast_1d(axes)

    for ax, clf_name in zip(axes, clf_names):
        sub = df[df["Classifier"] == clf_name].sort_values("Composite", ascending=True)
        colors = [CAT_COLOR.get(FS_CATEGORY.get(fs, "other"), "#95A5A6")
                  for fs in sub["FS_Method"]]
        bars = ax.barh(sub["FS_Method"], sub["Composite"],
                       color=colors, edgecolor="white", linewidth=0.4)
        for bar, val, n in zip(bars, sub["Composite"], sub["N_Features"]):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f} (n={n})", va="center", ha="left", fontsize=8)
        ax.set_title(clf_name, fontweight="bold")
        ax.set_xlabel("Composite Score (1=best)")
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()

    patches = [mpatches.Patch(color=c, label=k.capitalize())
               for k, c in CAT_COLOR.items()]
    fig.legend(handles=patches, title="FS Category", fontsize=10,
               loc="lower right", bbox_to_anchor=(1.01, 0.0))
    fig.suptitle("Feature Selection Methods Ranked by Composite Score\n"
                 "per Classifier  (LOOCV, n=53)  –  n = number of selected features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("figB_fs_ranked")

# ─────────────────────────────────────────────────────────────────────────────
# Fig C – Per-metric bar: compare FS methods, grouped by classifier
# ─────────────────────────────────────────────────────────────────────────────
def figC_per_metric(df):
    import math
    ncols = 2
    nrows = math.ceil(len(METRICS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes_flat = axes.flatten()

    clf_names = list(CLASSIFIERS.keys())
    n_fs   = df["FS_Method"].nunique()
    n_clf  = len(clf_names)
    x      = np.arange(n_fs)
    w      = 0.8 / max(n_clf, 1)
    offsets = (np.arange(n_clf) - (n_clf - 1) / 2) * w
    fs_order = (df.groupby("FS_Method")["Composite"].mean()
                  .sort_values(ascending=False).index.tolist())

    for ax, m in zip(axes_flat, METRICS):
        asc = m in LOWER_BETTER
        for i, clf_name in enumerate(clf_names):
            sub = df[df["Classifier"] == clf_name].set_index("FS_Method")
            vals = [sub.loc[fs, m] if fs in sub.index else np.nan for fs in fs_order]
            ax.bar(x + offsets[i], vals, w,
                   label=clf_name, color=CLF_COLOR[clf_name],
                   edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(fs_order, rotation=40, ha="right", fontsize=8)
        ax.set_title(METRIC_LABEL[m], fontweight="bold")
        ax.set_ylabel(m)
        if m == METRICS[0]:
            ax.legend(fontsize=9)

    for k in range(len(METRICS), len(axes_flat)):
        axes_flat[k].set_visible(False)

    fig.suptitle("Feature Selection Methods – Per-Metric Comparison\n"
                 f"(grouped bars = {' / '.join(clf_names)};  ordered by mean composite)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("figC_per_metric")

# ─────────────────────────────────────────────────────────────────────────────
# Fig D – N_features vs Kappa scatter (per classifier)
# ─────────────────────────────────────────────────────────────────────────────
def figD_nfeats_vs_kappa(df, kappa_thresh=0.5, nfeat_thresh=15):
    import matplotlib.lines as mlines
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    clf_names = list(CLASSIFIERS.keys())
    n_clf     = len(clf_names)

    # ── Auto-detect broken-axis gap in N_Features distribution ────────────
    all_nf = np.sort(df["N_Features"].unique())
    gaps   = [(all_nf[i+1] - all_nf[i], all_nf[i], all_nf[i+1])
              for i in range(len(all_nf) - 1)]
    if gaps:
        best_gap  = max(gaps, key=lambda g: g[0])
        nf_range  = float(all_nf[-1] - all_nf[0])
        use_break = best_gap[0] > max(5, nf_range * 0.25)
    else:
        use_break = False

    if use_break:
        _, break_lo, break_hi = best_gap
        main_xlim = (float(all_nf[0]) - 1.5, float(break_lo) + 1.5)
        out_xlim  = (float(break_hi) - 1.5,  float(all_nf[-1]) + 2.0)
        w_ratio   = [max(main_xlim[1] - main_xlim[0], 1),
                     max(out_xlim[1]  - out_xlim[0],  1)]

    fig   = plt.figure(figsize=((10 if use_break else 6) * n_clf, 5))
    outer = GridSpec(1, n_clf, figure=fig, wspace=0.4)

    legend_handles = [
        mlines.Line2D([0], [0], color="#E74C3C", ls="--", lw=1.4,
                      label=f"Kappa = {kappa_thresh}"),
        mlines.Line2D([0], [0], color="#E74C3C", ls=":",  lw=1.4,
                      label=f"N_features = {nfeat_thresh}"),
    ]

    for clf_i, clf_name in enumerate(clf_names):
        sub    = df[df["Classifier"] == clf_name].reset_index(drop=True)
        colors = [CAT_COLOR.get(FS_CATEGORY.get(fs, "other"), "#95A5A6")
                  for fs in sub["FS_Method"]]

        def _draw(ax, ann_lo=-np.inf, ann_hi=np.inf):
            ax.scatter(sub["N_Features"], sub["Kappa"], c=colors, s=90,
                       edgecolors="white", linewidth=0.6, zorder=3)
            for _, row in sub.iterrows():
                if ann_lo <= row["N_Features"] <= ann_hi:
                    ax.annotate(row["FS_Method"], (row["N_Features"], row["Kappa"]),
                                textcoords="offset points", xytext=(4, 3), fontsize=6.5)
            ax.axhline(kappa_thresh, color="#E74C3C", ls="--", lw=1.4, zorder=2)
            ax.axvline(nfeat_thresh, color="#E74C3C", ls=":",  lw=1.4, zorder=2)

        if use_break:
            inner = GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[clf_i],
                width_ratios=w_ratio, wspace=0.06,
            )
            ax1 = fig.add_subplot(inner[0])
            ax2 = fig.add_subplot(inner[1], sharey=ax1)

            _draw(ax1, ann_lo=main_xlim[0], ann_hi=main_xlim[1])
            _draw(ax2, ann_lo=out_xlim[0],  ann_hi=out_xlim[1])
            ax1.set_xlim(*main_xlim)
            ax2.set_xlim(*out_xlim)

            # Hide inner spines to form a visual break
            ax1.spines["right"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.tick_params(left=False, labelleft=False)

            # Draw ~~ diagonal break markers at the gap edges
            d = 0.018
            for _ax, xs in ((ax1, (1 - d, 1 + d)), (ax2, (-d, +d))):
                kw = dict(color="k", clip_on=False, lw=1.2,
                          transform=_ax.transAxes)
                _ax.plot(list(xs), [-d, +d],    **kw)
                _ax.plot(list(xs), [1 - d, 1 + d], **kw)

            ax1.set_xlabel("Number of Selected Features")
            ax1.set_ylabel("Weighted Kappa ↑")
            ax2.set_xlabel("")
            ax1.set_title(clf_name, fontweight="bold")
            ax1.legend(handles=legend_handles, loc="lower right", fontsize=8)
        else:
            ax = fig.add_subplot(outer[clf_i])
            _draw(ax)
            ax.set_xlabel("Number of Selected Features")
            ax.set_ylabel("Weighted Kappa ↑")
            ax.set_title(clf_name, fontweight="bold")
            ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    patches = [mpatches.Patch(color=c, label=k.capitalize())
               for k, c in CAT_COLOR.items()]
    fig.legend(handles=patches, title="FS Category", fontsize=9,
               loc="lower right", bbox_to_anchor=(1.01, 0.0))
    fig.suptitle("Number of Features vs Weighted Kappa\n",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("figD_nfeats_vs_kappa")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
def figE_feature_frequency(selections, results,
                           kappa_thresh=0.5, nfeat_thresh=15):
    """Bar chart of feature selection frequency among well-performing FS methods.

    Follows professor's suggestion: only count features from FS methods whose
    best result has Kappa > kappa_thresh AND N_Features < nfeat_thresh.

    Purple = selected by LLM-Lasso; colour intensity reflects frequency.
    """
    from collections import Counter
    import matplotlib.patches as mpatches

    # --- find FS methods that pass the performance filter ---
    # Take the best Kappa achieved by each FS method (across all classifiers)
    best_per_fs = (results.groupby("FS_Method")
                          .agg(best_kappa=("Kappa", "max"),
                               n_features=("N_Features", "first"))
                          .reset_index())
    passed = best_per_fs[
        (best_per_fs["best_kappa"] >= kappa_thresh) &
        (best_per_fs["n_features"] < nfeat_thresh)
    ]["FS_Method"].tolist()

    print(f"\n  Feature frequency filter: Kappa >= {kappa_thresh}, N_Features < {nfeat_thresh}")
    print(f"  {len(passed)}/{len(selections)} FS methods pass: {passed}")

    # --- count features among passing methods only ---
    counter = Counter()
    for method in passed:
        for f in selections.get(method, []):
            counter[f] += 1

    if not counter:
        print("  WARNING: No methods passed the filter; falling back to all methods.")
        for feats in selections.values():
            for f in feats:
                counter[f] += 1
        passed = list(selections.keys())

    n_methods  = len(passed)
    top_n      = min(25, len(counter))
    top        = counter.most_common(top_n)
    feats, counts = zip(*top)

    thr_high = max(1, round(n_methods * 0.5))   # ≥50% of passing methods
    thr_low  = max(1, round(n_methods * 0.25))  # ≥25%

    colors = [
        "#2980B9" if c >= thr_high else
        "#85C1E9" if c >= thr_low  else
        "#D6EAF8"
        for _, c in top
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(top_n), counts[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats[::-1], fontsize=10)
    ax.set_xlabel(
        f"Number of FS Methods Selecting This Feature\n"
        f"(filtered: Kappa ≥ {kappa_thresh} and N_Features < {nfeat_thresh}, "
        f"n = {n_methods} methods)",
        fontsize=10)
    ax.set_title(
        "Feature Selection Frequency Among Well-Performing FS Methods",
        fontweight="bold")
    ax.axvline(thr_high, color="#E74C3C", ls="--", lw=1.2,
               label=f">= {thr_high} methods (≥50%)")
    ax.axvline(thr_low,  color="#F39C12", ls="--", lw=1.2,
               label=f">= {thr_low} methods (≥25%)")
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=9)
    legend_handles = [
        mpatches.Patch(color="#2980B9", label=f">= {thr_high} methods (≥50%)"),
        mpatches.Patch(color="#85C1E9", label=f">= {thr_low} methods (≥25%)"),
        mpatches.Patch(color="#D6EAF8", label=f"< {thr_low} methods"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.set_xlim(0, n_methods + 3)
    plt.tight_layout()
    save_fig("figE_feature_frequency")


def plot_training_curves(X_full, y, feat_cols, selections, results_df, top_n=5):
    """Two-panel plot per method: training loss curve and LOO-CV MAE vs iterations.

    Trains an MLP with adam solver so loss_curve_ is available.
    LOO-CV MAE is evaluated at several iteration checkpoints.
    """
    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    checkpoints = [10, 25, 50, 100, 150, 200, 300, 400, 500]

    best_methods = (
        results_df.sort_values("Kappa", ascending=False)
        .drop_duplicates("FS_Method")
        .head(top_n)["FS_Method"]
        .tolist()
    )
    best_methods = [m for m in best_methods if m in selections]
    if not best_methods:
        print("  [skip] plot_training_curves: no methods with Kappa data")
        return

    fig, axes = plt.subplots(len(best_methods), 2,
                              figsize=(14, 4 * len(best_methods)))
    if len(best_methods) == 1:
        axes = axes.reshape(1, -1)

    loo = LeaveOneOut()

    for row_i, fs_name in enumerate(best_methods):
        feats = selections[fs_name]
        idx = [feat_idx[f] for f in feats if f in feat_idx]
        if not idx:
            continue
        X_sub = X_full[:, idx]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_sub)

        # ── Panel left: training loss curve with early stopping ──────────────
        mlp_full = MLPRegressor(
            hidden_layer_sizes=(8,), activation="relu",
            solver="adam", alpha=1.0, max_iter=600, random_state=42,
            early_stopping=False, n_iter_no_change=20, tol=1e-5,
        )
        mlp_full.fit(X_sc, y)
        converged_at = mlp_full.n_iter_
        axes[row_i, 0].plot(mlp_full.loss_curve_, color="#2980B9", lw=1.5)
        axes[row_i, 0].axvline(converged_at - 1, color="#E74C3C", lw=1.2, ls="--",
                                label=f"Converged @ iter {converged_at}")
        axes[row_i, 0].legend(fontsize=8)
        axes[row_i, 0].set_xlabel("Iteration")
        axes[row_i, 0].set_ylabel("Training Loss (MSE)")
        axes[row_i, 0].set_title(f"{fs_name}  [{len(idx)} features] — Training Loss")
        axes[row_i, 0].set_yscale("log")

        # ── Panel right: LOO-CV MAE at checkpoints up to convergence ─────────
        cv_checkpoints = [c for c in checkpoints if c <= converged_at]
        if converged_at not in cv_checkpoints:
            cv_checkpoints.append(converged_at)

        loo_maes = []
        for max_it in cv_checkpoints:
            preds = np.full(len(y), np.nan)
            for train_idx, test_idx in loo.split(X_sc):
                mlp = MLPRegressor(
                    hidden_layer_sizes=(8,), activation="relu",
                    solver="adam", alpha=1.0, max_iter=max_it, random_state=42,
                    n_iter_no_change=20, tol=1e-5,
                )
                try:
                    mlp.fit(X_sc[train_idx], y[train_idx])
                    preds[test_idx[0]] = mlp.predict(X_sc[test_idx])[0]
                except Exception:
                    pass
            valid = ~np.isnan(preds)
            yt = round_score(y[valid])
            yp = round_score(preds[valid])
            loo_maes.append(mean_absolute_error(yt, yp) if valid.any() else np.nan)

        axes[row_i, 1].plot(cv_checkpoints, loo_maes, "o-", color="#E67E22", lw=1.5,
                             markersize=5)
        axes[row_i, 1].axvline(converged_at, color="#E74C3C", lw=1.2, ls="--",
                                label=f"Converged @ iter {converged_at}")
        axes[row_i, 1].legend(fontsize=8)
        axes[row_i, 1].set_xlabel("Max Iterations")
        axes[row_i, 1].set_ylabel("LOO-CV MAE")
        axes[row_i, 1].set_title(f"{fs_name}  [{len(idx)} features] — LOO-CV MAE vs Iterations")
        axes[row_i, 1].grid(True, alpha=0.4)

    fig.suptitle(
        "MLP Training Verification  (ReLU, adam, 8 hidden units)\n"
        f"Top-{len(best_methods)} methods by Weighted Kappa",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()
    save_fig("figF_training_curves")


def print_table(df):
    print("\n" + "="*100)
    print(f"{'FS Method':<25} {'Classifier':<16} {'N':>4} {'MAE':>6} {'Spear':>6} {'Kappa':>6} {'AAcc':>6} {'Comp':>6}")
    print("="*100)
    for clf_name in CLASSIFIERS:
        sub = df[df["Classifier"] == clf_name].sort_values("Composite", ascending=False)
        print(f"\n--- {clf_name} ---")
        for _, row in sub.iterrows():
            print(f"  {row['FS_Method']:<23} {row['Classifier']:<16}"
                  f" {row['N_Features']:>4}"
                  f" {row['MAE']:>6.3f} {row['Spearman']:>6.3f}"
                  f" {row['Kappa']:>6.3f} {row['AAcc']:>6.3f}"
                  f" {row['Composite']:>6.3f}")
    print("="*100)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Feature Selection Comparison")
    print("=" * 60)

    print("\n[1] Loading data ...")
    X_full, y, feat_cols = load_data()
    selections = load_feature_selections()
    print(f"    {X_full.shape[0]} samples, {X_full.shape[1]} features")
    print(f"    {len(selections)} feature selection methods")
    print(f"    {len(CLASSIFIERS)} fixed classifiers: {list(CLASSIFIERS.keys())}")

    print("\n[2] Running LOOCV ...")
    results = run_loocv(X_full, y, feat_cols, selections)
    results = add_composite(results)
    results.to_csv(os.path.join(OUT_DIR, "loocv_results.csv"), index=False)
    print(f"  Saved: {OUT_DIR}/loocv_results.csv")

    print_table(results)

    print("\n[3] Generating figures ...")
    figA_heatmap(results)
    figB_ranked_bars(results)
    figC_per_metric(results)
    figD_nfeats_vs_kappa(results)
    figE_feature_frequency(selections, results)

    print("\n[4] Plotting training curves (top-5 methods by Kappa) ...")
    plot_training_curves(X_full, y, feat_cols, selections, results)

    print(f"\nDone! → {PNG_DIR}/  and  {PDF_DIR}/")
