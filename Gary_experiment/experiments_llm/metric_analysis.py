"""
Metric Orthogonality Analysis (experiments_llm edition)
=======================================================
Goal: find the minimal orthogonal evaluation-metric set for the
DeepSeek-R1-32B few-shot finger-tapping scoring experiments.

Unlike the original `result/metric_analysis.py` (which read pre-computed
summary CSVs with 6 metrics), this version derives the full metric set
directly from this folder's per-session predictions, so it always reflects
the latest experiment outputs.

Data source
-----------
results/<exp_prefix><n>/scored_by_deepseek-r1_32b.csv
  - columns "0".."9"        : the 10 repeated LLM scores per session
  - label_Dr. Tan / Chien   : the two raters' ground-truth scores
Each (model, n-features) config -> one data point with 6 metrics computed on
all valid sessions (approach B; pred = median of 10 reps).

Metrics (per config)
--------------------
  MAE       (lower better)   mean |pred_int - gt_int|
  Spearman  (higher better)  rank corr (pred_raw vs gt_cont)
  Kappa     (higher better)  quadratic-weighted Cohen's kappa
  AAcc      (higher better)  adjacent (+-1) accuracy
  ICC       (higher better)  ICC(3,1) consistency, pred vs gt
  WD        (lower better)   1-D Wasserstein distance between score distributions

Analysis (figures -> metric_analysis_results/{png,pdf})
  1. Pairwise Spearman correlation heatmap
  2. Metric scatter matrix
  3. Hierarchical-clustering dendrogram (redundant metric groups)
  4. PCA scree + loadings (minimal dimensionality)
  5. VIF (multicollinearity)
  6. Recommendation summary table (minimal orthogonal set)
  7. Exhaustive 3-metric subset selection (R^2 reconstruction of the rest)
"""

import os
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: F401
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths / config
# ─────────────────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
RESULTS_DIR = HERE / "results"
OUT_DIR     = HERE / "metric_analysis_results"
PNG_DIR     = OUT_DIR / "png"
PDF_DIR     = OUT_DIR / "pdf"
PNG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

SCORE_COLS = [str(i) for i in range(10)]
MAX_SCORE  = 4

MODEL_CONFIGS = {
    "ds32b_v2_amp":     {"exp_prefix": "fewshot_v2_ds32b_amp_freq",   "max_n": 15},
    "ds32b_v1_amp":     {"exp_prefix": "fewshot_v1_ds32b_amp_freq",   "max_n": 15},
    "ds32b_v1_llmfreq": {"exp_prefix": "fewshot_v1_ds32b_llmfreq",    "max_n": 10},
}
SCORED_CSV = "scored_by_deepseek-r1_32b.csv"

METRICS  = ["MAE", "Spearman", "Kappa", "AAcc", "ICC", "WD", "ACC"]
LOWER_BETTER = {"MAE", "WD"}
METRIC_LABEL = {
    "MAE":      "MAE ↓",
    "Spearman": "Spearman r ↑",
    "Kappa":    "Weighted κ ↑",
    "AAcc":     "Adjacent Acc. ↑",
    "ICC":      "ICC(3,1) ↑",
    "WD":       "Wasserstein ↓",
    "ACC":      "Exact Acc. ↑",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "figure.dpi":     150,
})


def save_fig(name):
    """PNG keeps the title; PDF has the title stripped (lab figure convention)."""
    fig = plt.gcf()
    fig.savefig(PNG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    # strip titles for the PDF version
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")
    fig.savefig(PDF_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def half_up(x):
    return np.clip(np.floor(np.asarray(x, float) + 0.5).astype(int), 0, MAX_SCORE)


def qwk(gt_int, pred_int):
    try:
        return float(cohen_kappa_score(gt_int, pred_int, weights="quadratic"))
    except Exception:
        return np.nan


def icc31(a, b):
    """ICC(3,1): two-way mixed, consistency, single measurement.
    a, b are the two 'raters' (prediction and ground truth) over n subjects.
    """
    M = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    n, k = M.shape
    if n < 3:
        return np.nan
    grand = M.mean()
    row_m = M.mean(axis=1)
    col_m = M.mean(axis=0)
    SSR = k * np.sum((row_m - grand) ** 2)
    SSC = n * np.sum((col_m - grand) ** 2)
    SST = np.sum((M - grand) ** 2)
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE
    return float((MSR - MSE) / denom) if denom != 0 else np.nan


def load_config(n, cfg):
    """Return (pred_raw, gt_cont) on all valid sessions, or None if missing."""
    csv = RESULTS_DIR / f"{cfg['exp_prefix']}{n}" / SCORED_CSV
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    pred_raw = np.nanmedian(
        df[SCORE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(), axis=1)
    tan   = df["label_Dr. Tan"].astype(float).to_numpy()
    chien = df["label_Dr. Chien"].astype(float).to_numpy()
    gt_cont = (tan + chien) / 2
    valid = ~np.isnan(gt_cont) & ~np.isnan(pred_raw)
    return pred_raw[valid], gt_cont[valid]


def metrics_for_config(pred_raw, gt_cont):
    pred_int = half_up(pred_raw)
    gt_int   = half_up(gt_cont)
    out = {
        "MAE":      float(np.mean(np.abs(pred_int - gt_int))),
        "ACC":      float(np.mean(pred_int == gt_int)),
        "AAcc":     float(np.mean(np.abs(pred_int - gt_int) <= 1)),
        "Kappa":    qwk(gt_int, pred_int),
        "WD":       float(wasserstein_distance(pred_int, gt_int)),
        "ICC":      icc31(pred_raw, gt_cont),
    }
    # Spearman needs variance in both vectors
    if np.std(pred_raw) > 0 and np.std(gt_cont) > 0:
        out["Spearman"] = float(stats.spearmanr(pred_raw, gt_cont).correlation)
    else:
        out["Spearman"] = np.nan
    return out


def load_data():
    rows = []
    for model, cfg in MODEL_CONFIGS.items():
        for n in range(1, cfg["max_n"] + 1):
            d = load_config(n, cfg)
            if d is None:
                continue
            pred_raw, gt_cont = d
            if len(gt_cont) < 5:
                continue
            m = metrics_for_config(pred_raw, gt_cont)
            rows.append({"Source": "LLM", "Method": f"{model}_n{n}",
                         "n_features": n, **m})
    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Pairwise Spearman correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig_corr_heatmap(df):
    sub = df[METRICS].dropna()
    n = len(sub)
    corr = sub.corr(method="spearman")

    pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i in corr.index:
        for j in corr.columns:
            if i != j:
                _, p = stats.spearmanr(sub[i], sub[j])
                pvals.loc[i, j] = p

    annot = corr.applymap(lambda v: f"{v:.2f}")
    for i in corr.index:
        for j in corr.columns:
            p = pvals.loc[i, j]
            s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if s:
                annot.loc[i, j] = f"{corr.loc[i, j]:.2f}{s}"

    labels = [METRIC_LABEL.get(m, m) for m in corr.columns]
    mask = np.triu(np.ones_like(corr), k=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=annot, fmt="", cmap="RdBu_r", vmin=-1, vmax=1,
                mask=mask.T, linewidths=0.5, linecolor="lightgray",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Spearman r", "shrink": 0.8}, ax=ax)
    ax.set_title(f"Pairwise Spearman Correlations Between Metrics\n"
                 f"LLM experiments (n={n} configs)  |  * p<0.05  ** p<0.01  *** p<0.001",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("fig1_corr_heatmap")
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Scatter matrix
# ─────────────────────────────────────────────────────────────────────────────
def fig_scatter_matrix(df):
    sub = df[METRICS].dropna().copy()
    n = len(sub)
    labels = [METRIC_LABEL.get(m, m) for m in METRICS]
    k = len(METRICS)
    fig, axes = plt.subplots(k, k, figsize=(14, 13))
    for i, mi in enumerate(METRICS):
        for j, mj in enumerate(METRICS):
            ax = axes[i][j]
            if i == j:
                ax.hist(sub[mi], bins=8, color="#3498DB", edgecolor="white")
                ax.set_xlabel(labels[i], fontsize=8)
            elif i > j:
                r, p = stats.spearmanr(sub[mj], sub[mi])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.scatter(sub[mj], sub[mi], s=30, alpha=0.7,
                           color="#E74C3C" if abs(r) > 0.9 else "#3498DB")
                ax.text(0.05, 0.93, f"r={r:.2f}{stars}", transform=ax.transAxes,
                        fontsize=8, va="top")
                z = np.polyfit(sub[mj], sub[mi], 1)
                xl = np.linspace(sub[mj].min(), sub[mj].max(), 50)
                ax.plot(xl, np.polyval(z, xl), "k--", lw=0.8, alpha=0.5)
            else:
                r, _ = stats.spearmanr(sub[mj], sub[mi])
                ax.set_facecolor(plt.cm.RdBu_r((r + 1) / 2))
                ax.text(0.5, 0.5, f"{r:.2f}", ha="center", va="center", fontsize=14,
                        fontweight="bold", color="white" if abs(r) > 0.6 else "black",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Metric Scatter Matrix (LLM experiments, n={n})\n"
                 "Lower-left: scatter + Spearman r  |  Upper-right: r (red = |r|>0.9)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("fig2_scatter_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Dendrogram
# ─────────────────────────────────────────────────────────────────────────────
def fig_dendrogram(df):
    keys = [m for m in METRICS if m in df.columns]
    k = len(keys)
    dist = np.zeros((k, k))
    for i, mi in enumerate(keys):
        for j, mj in enumerate(keys):
            if i == j:
                continue
            common = df[[mi, mj]].dropna()
            if len(common) >= 5:
                r, _ = stats.spearmanr(common[mi], common[mj])
                dist[i, j] = 1 - abs(r)
            else:
                dist[i, j] = 1.0
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist), method="complete")
    labels = [METRIC_LABEL.get(m, m) for m in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    dendrogram(Z, labels=labels, leaf_rotation=30, leaf_font_size=11,
               color_threshold=0.25, ax=ax)
    ax.set_ylabel("Distance  (1 − |Spearman r|)")
    ax.set_title("Hierarchical Clustering of Evaluation Metrics\n"
                 "Metrics in the same cluster are highly redundant", fontweight="bold")
    plt.tight_layout()
    save_fig("fig3_dendrogram")

    clusters = fcluster(Z, 0.25, criterion="distance")
    cdf = pd.DataFrame({"Metric": keys, "Cluster": clusters})
    print("\nMetric clusters (threshold = 0.25, |r| > 0.75):")
    for c in sorted(cdf["Cluster"].unique()):
        print(f"  Cluster {c}: {cdf[cdf.Cluster == c]['Metric'].tolist()}")
    return cdf


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – PCA
# ─────────────────────────────────────────────────────────────────────────────
def fig_pca(df):
    sub = df[METRICS].dropna().copy()
    n = len(sub)
    for m in LOWER_BETTER:
        if m in sub.columns:
            sub[m] = -sub[m]
    X = StandardScaler().fit_transform(sub)
    pca = PCA().fit(X)
    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(1, len(explained) + 1), explained, color="#3498DB", alpha=0.8, edgecolor="white")
    ax1.plot(range(1, len(explained) + 1), cumulative, "ro-", lw=1.5, ms=6)
    ax1.axhline(80, color="#27AE60", ls="--", lw=1, label="80% threshold")
    ax1.axhline(95, color="#E67E22", ls="--", lw=1, label="95% threshold")
    for i, e in enumerate(explained):
        ax1.text(i + 1, e + 0.5, f"{e:.1f}%", ha="center", fontsize=9)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Scree Plot – PCA of Metrics", fontweight="bold")
    ax1.legend(fontsize=9)

    loadings = pd.DataFrame(pca.components_.T,
                            index=[METRIC_LABEL.get(m, m) for m in METRICS],
                            columns=[f"PC{i+1}" for i in range(len(METRICS))])
    sns.heatmap(loadings.iloc[:, :4], annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, linewidths=0.4, linecolor="lightgray",
                cbar_kws={"label": "Loading", "shrink": 0.8}, ax=ax2)
    ax2.set_title("PCA Loadings (first 4 PCs)", fontweight="bold")
    fig.suptitle(f"PCA of Evaluation Metrics  —  LLM experiments (n={n}); "
                 "metrics standardised, MAE/WD inverted", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("fig4_pca")

    for i, (e, c) in enumerate(zip(explained, cumulative)):
        print(f"  PC{i+1}: {e:.1f}%  (cumulative {c:.1f}%)")
    print(f"  -> {int(np.argmax(cumulative >= 80)) + 1} PCs >=80%  |  "
          f"{int(np.argmax(cumulative >= 95)) + 1} PCs >=95%")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – VIF
# ─────────────────────────────────────────────────────────────────────────────
def fig_vif(df):
    sub = df[METRICS].dropna().copy()
    for m in LOWER_BETTER:
        if m in sub.columns:
            sub[m] = -sub[m]
    X = StandardScaler().fit_transform(sub)
    Xd = pd.DataFrame(X, columns=METRICS)
    vif = pd.DataFrame({
        "Metric": METRICS,
        "VIF":    [variance_inflation_factor(Xd.values, i) for i in range(Xd.shape[1])],
        "Label":  [METRIC_LABEL.get(m, m) for m in METRICS],
    }).sort_values("VIF", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#E74C3C" if v > 10 else "#E67E22" if v > 5 else "#27AE60" for v in vif["VIF"]]
    bars = ax.barh(vif["Label"], vif["VIF"], color=colors, edgecolor="white")
    for bar, val in zip(bars, vif["VIF"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=10)
    ax.axvline(5, color="#E67E22", ls="--", lw=1.2, label="VIF = 5 (moderate)")
    ax.axvline(10, color="#E74C3C", ls="--", lw=1.2, label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title(f"VIF – Multicollinearity Between Metrics  (n={len(sub)})\n"
                 "MAE/WD inverted so all = higher is better", fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("fig5_vif")

    print("\nVIF values:")
    for _, r in vif.iterrows():
        flag = "  <- SEVERE" if r.VIF > 10 else "  <- moderate" if r.VIF > 5 else ""
        print(f"  {r.Metric:10s}: {r.VIF:.2f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Recommendation summary table
# ─────────────────────────────────────────────────────────────────────────────
def fig_summary(corr, cdf):
    redundant = [(mi, mj, corr.loc[mi, mj])
                 for i, mi in enumerate(METRICS) for j, mj in enumerate(METRICS)
                 if j > i and abs(corr.loc[mi, mj]) > 0.90]
    dropped = {mj for _, mj, _ in redundant}

    rows = []
    for m in METRICS:
        rw = [f"{mj} (r={r:.2f})" if m == mi else f"{mi} (r={r:.2f})"
              for mi, mj, r in redundant if m in (mi, mj)]
        cl = str(cdf.loc[cdf.Metric == m, "Cluster"].values[0]) if m in cdf.Metric.values else "-"
        rows.append({"Metric": METRIC_LABEL.get(m, m), "Cluster": cl,
                     "Redundant with": ", ".join(rw) if rw else "—",
                     "Recommended": "NO" if m in dropped else "YES"})
    rdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, max(4, len(rdf) * 0.65 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=rdf.values, colLabels=rdf.columns, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.6)
    for j in range(len(rdf.columns)):
        tbl[0, j].set_facecolor("#2C3E50"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, row in rdf.iterrows():
        bg = "#D5F5E3" if row["Recommended"] == "YES" else "#FDFEFE"
        for j in range(len(rdf.columns)):
            tbl[i + 1, j].set_facecolor(bg)
    ax.set_title("Metric Recommendation Summary\n(green = recommended minimal orthogonal set)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    save_fig("fig6_summary_table")

    print("\n" + "=" * 70)
    print("RECOMMENDED MINIMAL METRIC SET")
    print("=" * 70)
    for m in METRICS:
        tag = "[YES]" if m not in dropped else "[NO] "
        print(f"  {tag} {METRIC_LABEL.get(m, m)}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Exhaustive 3-metric subset selection
# ─────────────────────────────────────────────────────────────────────────────
def fig_subset(df):
    sub = df[METRICS].dropna().copy()
    n = len(sub)
    for m in LOWER_BETTER:
        if m in sub.columns:
            sub[m] = -sub[m]
    Xd = pd.DataFrame(StandardScaler().fit_transform(sub), columns=METRICS)

    results = []
    for chosen in combinations(METRICS, 3):
        remaining = [m for m in METRICS if m not in chosen]
        Xin = Xd[list(chosen)].values
        r2 = {t: max(0.0, LinearRegression().fit(Xin, Xd[t].values).score(Xin, Xd[t].values))
              for t in remaining}
        results.append({"Subset": " + ".join(chosen), "mean_R2": float(np.mean(list(r2.values()))),
                        **{f"R2_{t}": v for t, v in r2.items()}})
    res = pd.DataFrame(results).sort_values("mean_R2", ascending=False).reset_index(drop=True)

    colors = ["#27AE60" if v >= 0.90 else "#F39C12" if v >= 0.75 else "#E74C3C" for v in res["mean_R2"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(res["Subset"][::-1], res["mean_R2"][::-1], color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, res["mean_R2"][::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.axvline(0.90, color="#27AE60", ls="--", lw=1.2, label="R²=0.90")
    ax.axvline(0.75, color="#F39C12", ls="--", lw=1.2, label="R²=0.75")
    ax.set_xlabel("Mean R² (predicting the remaining metrics)")
    ax.set_title(f"Exhaustive 3-Metric Subset Selection  —  LLM experiments (n={n})",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0, 1.08)
    plt.tight_layout()
    save_fig("fig7_subset_bar")

    print("\nTop-5 3-metric subsets (by mean R2):")
    for _, r in res.head(5).iterrows():
        print(f"  [{r['mean_R2']:.3f}]  {r['Subset']}")
    print(f"\nBest subset: {res.iloc[0]['Subset']}")
    res.to_csv(OUT_DIR / "subset_selection.csv", index=False)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Metric Orthogonality Analysis (experiments_llm)")
    print("=" * 60)

    print("\n[1] Deriving metrics from per-session predictions ...")
    df = load_data()
    print(f"  Configs (data points): {len(df)}")
    print(f"  Metrics: {METRICS}")
    df.to_csv(OUT_DIR / "per_config_metrics.csv", index=False)
    print(f"  Saved: per_config_metrics.csv")

    print("\n[2] Correlation heatmap ...")
    corr = fig_corr_heatmap(df)
    print("\n[3] Scatter matrix ...")
    fig_scatter_matrix(df)
    print("\n[4] Hierarchical clustering ...")
    cdf = fig_dendrogram(df)
    print("\n[5] PCA ...")
    fig_pca(df)
    print("\n[6] VIF ...")
    fig_vif(df)
    print("\n[7] Recommendation summary ...")
    fig_summary(corr, cdf)
    print("\n[8] 3-metric subset selection ...")
    fig_subset(df)

    print(f"\nDone! -> {PNG_DIR}/  and  {PDF_DIR}/")
