"""
Metric Orthogonality Analysis
==============================
目標：找出最小正交指標集合（Minimal Orthogonal Metric Set）

指標來源：
  - LLM 實驗 (multi_model_results/summary_metrics.csv)
    → 6 個指標: MAE, Spearman, Kappa, AAcc, Kendall, ICC
  - ML 固定 FS 實驗 (fixed_fs_results/loocv_results.csv)
    → 4 個指標: MAE, Spearman, Kappa, AAcc
  - ML 全特徵實驗 (all_results/ml_loocv_results.csv)
    → 5 個指標: MAE, Spearman, Kappa, AAcc, Bin_F1

分析步驟：
  1. Pairwise Spearman 相關矩陣（heat-map）
  2. 散佈圖矩陣（pair-plot）
  3. 階層式聚類 dendrogram（找冗餘群）
  4. PCA 解釋變異量（找最少維度）
  5. VIF 膨脹因子（找線性共線性）
  6. 結論：推薦最小指標集
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths & output
# ─────────────────────────────────────────────────────────────────────────────
LLM_CSV    = "multi_model_results/summary_metrics.csv"
ML_FS_CSV  = "fixed_fs_results/loocv_results.csv"
ML_ALL_CSV = "all_results/ml_loocv_results.csv"
OUT_DIR    = "metric_analysis_results"
PNG_DIR    = os.path.join(OUT_DIR, "png")
PDF_DIR    = os.path.join(OUT_DIR, "pdf")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.titlesize": 13,
    "figure.dpi":     150,
})

def save_fig(name):
    plt.savefig(os.path.join(PNG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(PDF_DIR, f"{name}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# Load & pool data
# ─────────────────────────────────────────────────────────────────────────────
SIX_METRICS  = ["MAE", "Spearman", "Kappa", "AAcc", "WD", "ICC", "ACC"]
FOUR_METRICS = ["MAE", "Spearman", "Kappa", "AAcc"]
LOWER_BETTER = {"MAE", "WD"}

METRIC_LABEL = {
    "MAE":      "MAE ↓",
    "Spearman": "Spearman r ↑",
    "Kappa":    "Weighted κ ↑",
    "AAcc":     "Adjacent Acc. ↑",
    "WD":       "Wasserstein Dist. ↓",
    "ICC":      "ICC(3,1) ↑",
    "ACC":      "Accuracy ↑",
    "Bin_F1":   "Binary F1 ↑",
}

def load_data():
    rows = []

    # LLM data – has all 6 metrics
    llm = pd.read_csv(LLM_CSV)
    llm = llm[~llm["is_baseline"].astype(str).str.lower().eq("true")]
    for _, r in llm.iterrows():
        row = {
            "Source": "LLM",
            "Method": r["Model"],
            **{m: r[m] for m in SIX_METRICS if m in r}
        }
        # ACC_Consensus is the exact accuracy column name in the LLM CSV
        if "ACC" not in row and "ACC_Consensus" in r:
            row["ACC"] = r["ACC_Consensus"]
        rows.append(row)

    # ML fixed-FS – 4 metrics
    ml_fs = pd.read_csv(ML_FS_CSV)
    for _, r in ml_fs.iterrows():
        rows.append({
            "Source": "ML-FS",
            "Method": f"{r['Predictor']} / {r['FS_Key']}",
            **{m: r[m] for m in FOUR_METRICS if m in r}
        })

    # ML all-features – 4+1 metrics
    if os.path.exists(ML_ALL_CSV):
        ml_all = pd.read_csv(ML_ALL_CSV)
        for _, r in ml_all.iterrows():
            rows.append({
                "Source": "ML-All",
                "Method": r["Method"],
                **{m: r[m] for m in FOUR_METRICS + ["Bin_F1"] if m in r}
            })

    df = pd.DataFrame(rows)
    # flip MAE and WD so all metrics are "higher = better"
    df["MAE_inv"] = -df["MAE"]
    if "WD" in df.columns:
        df["WD"] = -df["WD"]
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Pairwise Spearman correlation heatmap (LLM subset: all 6 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig1_corr_heatmap_llm(df):
    sub = df[df["Source"] == "LLM"][SIX_METRICS].dropna()
    n   = len(sub)
    corr = sub.corr(method="spearman")

    # compute p-values
    pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i in corr.index:
        for j in corr.columns:
            if i != j:
                _, p = stats.spearmanr(sub[i], sub[j])
                pvals.loc[i, j] = p

    annot = corr.applymap(lambda v: f"{v:.2f}")
    # add significance stars
    for i in corr.index:
        for j in corr.columns:
            s = ""
            p = pvals.loc[i, j]
            if p < 0.001: s = "***"
            elif p < 0.01: s = "**"
            elif p < 0.05: s = "*"
            if s:
                annot.loc[i, j] = f"{corr.loc[i,j]:.2f}{s}"

    labels = [METRIC_LABEL.get(m, m) for m in corr.columns]

    mask = np.triu(np.ones_like(corr), k=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=annot, fmt="", cmap="RdBu_r",
                vmin=-1, vmax=1,
                mask=mask.T,   # lower triangle only
                linewidths=0.5, linecolor="lightgray",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Spearman r", "shrink": 0.8},
                ax=ax)
    ax.set_title(f"Pairwise Spearman Correlations Between Metrics\n"
                 f"LLM experiments (n={n} models)  |  * p<0.05  ** p<0.01  *** p<0.001",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("fig1_corr_heatmap_llm")
    return corr, pvals

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Pairwise Spearman correlation heatmap (ML-FS: 4 metrics, large n)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_corr_heatmap_ml(df):
    sub = df[df["Source"] == "ML-FS"][FOUR_METRICS].dropna()
    n   = len(sub)
    corr = sub.corr(method="spearman")

    pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
    for i in corr.index:
        for j in corr.columns:
            if i != j:
                _, p = stats.spearmanr(sub[i], sub[j])
                pvals.loc[i, j] = p

    annot = corr.copy().astype(str)
    for i in corr.index:
        for j in corr.columns:
            s = ""
            p = pvals.loc[i, j]
            if p < 0.001: s = "***"
            elif p < 0.01: s = "**"
            elif p < 0.05: s = "*"
            annot.loc[i, j] = f"{corr.loc[i,j]:.2f}{s}"

    labels = [METRIC_LABEL.get(m, m) for m in corr.columns]

    mask = np.triu(np.ones_like(corr), k=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=annot, fmt="", cmap="RdBu_r",
                vmin=-1, vmax=1,
                mask=mask.T,
                linewidths=0.5, linecolor="lightgray",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Spearman r", "shrink": 0.8},
                ax=ax)
    ax.set_title(f"Pairwise Spearman Correlations Between Metrics\n"
                 f"ML experiments (n={n} method×FS combinations)  |  * p<0.05  ** p<0.01  *** p<0.001",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("fig2_corr_heatmap_ml")
    return corr, pvals

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Scatter matrix (LLM data, all 6 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_scatter_matrix(df):
    sub = df[df["Source"] == "LLM"][SIX_METRICS].dropna().copy()
    n   = len(sub)
    labels = [METRIC_LABEL.get(m, m) for m in SIX_METRICS]

    k = len(SIX_METRICS)
    fig, axes = plt.subplots(k, k, figsize=(14, 13))

    for i, (mi, li) in enumerate(zip(SIX_METRICS, labels)):
        for j, (mj, lj) in enumerate(zip(SIX_METRICS, labels)):
            ax = axes[i][j]
            if i == j:
                ax.hist(sub[mi], bins=8, color="#3498DB", edgecolor="white")
                ax.set_xlabel(li, fontsize=8)
            elif i > j:
                r, p = stats.spearmanr(sub[mj], sub[mi])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                ax.scatter(sub[mj], sub[mi], s=30, alpha=0.7,
                           color="#E74C3C" if abs(r) > 0.9 else "#3498DB")
                ax.text(0.05, 0.93, f"r={r:.2f}{stars}", transform=ax.transAxes,
                        fontsize=8, va="top")
                # regression line
                z = np.polyfit(sub[mj].dropna(), sub[mi].dropna(), 1)
                xl = np.linspace(sub[mj].min(), sub[mj].max(), 50)
                ax.plot(xl, np.polyval(z, xl), "k--", lw=0.8, alpha=0.5)
            else:
                r, _ = stats.spearmanr(sub[mj], sub[mi])
                color = plt.cm.RdBu_r((r + 1) / 2)
                ax.set_facecolor(color)
                ax.text(0.5, 0.5, f"{r:.2f}", ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if abs(r) > 0.6 else "black",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])

            if j == 0:
                ax.set_ylabel(li, fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Metric Scatter Matrix (LLM experiments, n={n})\n"
                 "Lower-left: scatter + Spearman r  |  Upper-right: r value (red = |r|>0.9)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("fig3_scatter_matrix")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Hierarchical clustering dendrogram of metrics
# ─────────────────────────────────────────────────────────────────────────────
def fig4_dendrogram(df):
    """Use 1 - |Spearman r| as distance between metrics.
    Pool all available data per metric pair.
    """
    all_metrics = SIX_METRICS + ["Bin_F1"]
    avail = {m: df[m].dropna().values for m in all_metrics if m in df.columns}
    avail_keys = list(avail.keys())

    # compute pairwise |Spearman r| using only rows where both metrics exist
    k = len(avail_keys)
    dist_mat = np.zeros((k, k))
    for i, mi in enumerate(avail_keys):
        for j, mj in enumerate(avail_keys):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                common = df[[mi, mj]].dropna()
                if len(common) >= 5:
                    r, _ = stats.spearmanr(common[mi], common[mj])
                    dist_mat[i, j] = 1 - abs(r)
                else:
                    dist_mat[i, j] = 1.0

    # ensure symmetry
    dist_mat = (dist_mat + dist_mat.T) / 2

    linkage_mat = linkage(squareform(dist_mat), method="complete")
    labels_nice = [METRIC_LABEL.get(m, m) for m in avail_keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    dend = dendrogram(linkage_mat, labels=labels_nice,
                      leaf_rotation=30, leaf_font_size=11,
                      color_threshold=0.25, ax=ax)
    ax.set_ylabel("Distance  (1 − |Spearman r|)")
    ax.set_title("Hierarchical Clustering of Evaluation Metrics\n"
                 "Metrics in the same cluster are highly redundant",
                 fontweight="bold")
    plt.tight_layout()
    save_fig("fig4_dendrogram")

    # annotate clusters at threshold 0.25
    clusters = fcluster(linkage_mat, 0.25, criterion="distance")
    cluster_df = pd.DataFrame({"Metric": avail_keys, "Cluster": clusters})
    print("\nMetric clusters (threshold = 0.25, |r| > 0.75):")
    for c in sorted(cluster_df["Cluster"].unique()):
        members = cluster_df[cluster_df["Cluster"] == c]["Metric"].tolist()
        print(f"  Cluster {c}: {members}")
    return cluster_df

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – PCA of metrics (LLM data, 6 metrics)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_pca(df):
    sub = df[df["Source"] == "LLM"][SIX_METRICS].dropna()
    n   = len(sub)

    # standardize (flip MAE & WD so all = higher is better)
    sub_norm = sub.copy()
    sub_norm["MAE"] = -sub_norm["MAE"]
    if "WD" in sub_norm.columns:
        sub_norm["WD"] = -sub_norm["WD"]

    scaler = StandardScaler()
    X = scaler.fit_transform(sub_norm)
    pca = PCA()
    pca.fit(X)

    explained = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # scree plot
    ax1.bar(range(1, len(explained)+1), explained, color="#3498DB", alpha=0.8, edgecolor="white")
    ax1.plot(range(1, len(explained)+1), cumulative, "ro-", lw=1.5, ms=6)
    ax1.axhline(80, color="#27AE60", ls="--", lw=1, label="80% threshold")
    ax1.axhline(95, color="#E67E22", ls="--", lw=1, label="95% threshold")
    for i, (e, c) in enumerate(zip(explained, cumulative)):
        ax1.text(i+1, e+0.5, f"{e:.1f}%", ha="center", fontsize=9)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Scree Plot – PCA of Metrics", fontweight="bold")
    ax1.legend(fontsize=9)

    # loadings heatmap
    loadings = pd.DataFrame(pca.components_.T,
                             index=[METRIC_LABEL.get(m, m) for m in SIX_METRICS],
                             columns=[f"PC{i+1}" for i in range(len(SIX_METRICS))])
    loadings_plot = loadings.iloc[:, :4]  # first 4 PCs

    sns.heatmap(loadings_plot, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, linewidths=0.4, linecolor="lightgray",
                cbar_kws={"label": "Loading", "shrink": 0.8}, ax=ax2)
    ax2.set_title("PCA Loadings (first 4 PCs)", fontweight="bold")

    fig.suptitle(f"Principal Component Analysis of Evaluation Metrics\n"
                 f"LLM experiments (n={n})  —  metrics standardised, MAE inverted",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("fig5_pca")

    # print summary
    for i, (e, c) in enumerate(zip(explained, cumulative)):
        print(f"  PC{i+1}: {e:.1f}%  (cumulative {c:.1f}%)")
    n_for_80 = np.argmax(cumulative >= 80) + 1
    n_for_95 = np.argmax(cumulative >= 95) + 1
    print(f"\n  -> {n_for_80} PCs explain >=80%  |  {n_for_95} PCs explain >=95%")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – VIF (Variance Inflation Factor) for core 4 metrics (ML data)
# ─────────────────────────────────────────────────────────────────────────────
def fig6_vif(df):
    # Use ML-FS data (large n = 104) to get stable VIF estimates
    sub = df[df["Source"] == "ML-FS"][FOUR_METRICS].dropna()
    sub_norm = sub.copy()
    sub_norm["MAE"] = -sub_norm["MAE"]

    X = StandardScaler().fit_transform(sub_norm)
    X_df = pd.DataFrame(X, columns=FOUR_METRICS)
    vif_data = {
        "Metric": FOUR_METRICS,
        "VIF":    [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])],
        "Label":  [METRIC_LABEL.get(m, m) for m in FOUR_METRICS],
    }
    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#E74C3C" if v > 10 else "#E67E22" if v > 5 else "#27AE60"
              for v in vif_df["VIF"]]
    bars = ax.barh(vif_df["Label"], vif_df["VIF"],
                   color=colors, edgecolor="white")
    for bar, val in zip(bars, vif_df["VIF"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=10)
    ax.axvline(5,  color="#E67E22", ls="--", lw=1.2, label="VIF = 5  (moderate)")
    ax.axvline(10, color="#E74C3C", ls="--", lw=1.2, label="VIF = 10 (severe)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("VIF Analysis – Multicollinearity Between Metrics\n"
                 "ML experiments (n=104);  MAE inverted so all = higher is better",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    save_fig("fig6_vif")

    print("\nVIF values:")
    for _, row in vif_df.iterrows():
        flag = "  ← SEVERE"   if row["VIF"] > 10 else \
               "  ← moderate" if row["VIF"] > 5  else ""
        print(f"  {row['Metric']:10s}: {row['VIF']:.2f}{flag}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Summary: Recommended metric set
# ─────────────────────────────────────────────────────────────────────────────
def fig7_summary(corr_llm, cluster_df, df):
    """Print and visualise the recommended minimal metric set."""
    # Derive redundant pairs (|r| > 0.90)
    redundant_pairs = []
    for i, mi in enumerate(SIX_METRICS):
        for j, mj in enumerate(SIX_METRICS):
            if j <= i: continue
            if mi not in corr_llm.index or mj not in corr_llm.columns: continue
            r = corr_llm.loc[mi, mj]
            if abs(r) > 0.90:
                redundant_pairs.append((mi, mj, r))

    # Build recommendation table
    all_metrics_ext = SIX_METRICS + ["Bin_F1"]
    avail_in_df = [m for m in all_metrics_ext if m in df.columns]

    rec_rows = []
    for m in avail_in_df:
        redundant_with = [
            f"{mj} (r={r:.2f})" if m == mi else f"{mi} (r={r:.2f})"
            for mi, mj, r in redundant_pairs
            if m == mi or m == mj
        ]
        # find which cluster
        cl = "–"
        if cluster_df is not None and m in cluster_df["Metric"].values:
            cl = str(cluster_df.loc[cluster_df["Metric"] == m, "Cluster"].values[0])

        rec_rows.append({
            "Metric":    METRIC_LABEL.get(m, m),
            "Cluster":   cl,
            "Redundant with": ", ".join(redundant_with) if redundant_with else "—",
            "Recommended": "YES" if m not in {mj for _, mj, r in redundant_pairs if abs(r) > 0.90} else "NO",
        })

    rec_df = pd.DataFrame(rec_rows)

    fig, ax = plt.subplots(figsize=(13, max(4, len(rec_df) * 0.65 + 1.5)))
    ax.axis("off")

    col_widths = [0.20, 0.08, 0.45, 0.12]
    tbl = ax.table(
        cellText=rec_df.values,
        colLabels=rec_df.columns,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # header style
    for j in range(len(rec_df.columns)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # highlight recommended rows
    for i, row in rec_df.iterrows():
        bg = "#D5F5E3" if row["Recommended"] == "YES" else "#FDFEFE"
        for j in range(len(rec_df.columns)):
            tbl[i+1, j].set_facecolor(bg)

    ax.set_title("Metric Recommendation Summary\n"
                 "(green = recommended in minimal orthogonal set)",
                 fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    save_fig("fig7_summary_table")

    print("\n" + "="*70)
    print("RECOMMENDED MINIMAL METRIC SET")
    print("="*70)
    recommended = [r["Metric"] for _, r in rec_df.iterrows() if r["Recommended"] == "YES"]
    for m in recommended:
        print(f"  [YES]  {m}")
    print("\nREDUNDANT (can be dropped):")
    dropped = [r["Metric"] for _, r in rec_df.iterrows() if r["Recommended"] != "YES"]
    for m in dropped:
        print(f"  [NO]   {m}")
    print("="*70)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 – Exhaustive 3-metric subset selection
# ─────────────────────────────────────────────────────────────────────────────
def fig8_subset_selection(df):
    """Try all C(6,3)=20 combinations of 3 metrics.

    For each subset S of size 3:
      - fit a linear regression from S to predict each of the remaining 3 metrics
      - record R² for each target
    Report: mean R² across all targets (how well 3 metrics explain the other 3),
    plus a heatmap of (subset × target metric) R².

    Intuition from PCA: PC1 explains ~78% with equal weights, so a 3-metric
    subset should be sufficient to reconstruct the rest with high R².
    """
    sub = df[df["Source"] == "LLM"][SIX_METRICS].dropna()
    n   = len(sub)

    # standardize (flip MAE & WD so all = higher is better)
    sub_norm = sub.copy()
    sub_norm["MAE"] = -sub_norm["MAE"]
    if "WD" in sub_norm.columns:
        sub_norm["WD"] = -sub_norm["WD"]
    X_all = StandardScaler().fit_transform(sub_norm)
    X_df  = pd.DataFrame(X_all, columns=SIX_METRICS)

    results = []
    for chosen in combinations(SIX_METRICS, 3):
        remaining = [m for m in SIX_METRICS if m not in chosen]
        X_in  = X_df[list(chosen)].values
        r2_per_target = {}
        for target in remaining:
            y = X_df[target].values
            reg = LinearRegression().fit(X_in, y)
            r2_per_target[target] = max(0.0, reg.score(X_in, y))  # clamp to 0
        mean_r2 = np.mean(list(r2_per_target.values()))
        results.append({
            "Subset":   " + ".join(chosen),
            "mean_R2":  mean_r2,
            **{f"R2_{t}": v for t, v in r2_per_target.items()},
        })

    res_df = pd.DataFrame(results).sort_values("mean_R2", ascending=False).reset_index(drop=True)

    # ── Fig 8a: bar chart of mean R² for all subsets ───────────────────────
    colors = ["#27AE60" if v >= 0.90 else "#F39C12" if v >= 0.75 else "#E74C3C"
              for v in res_df["mean_R2"]]
    fig, ax = plt.subplots(figsize=(9, 8))
    bars = ax.barh(res_df["Subset"][::-1], res_df["mean_R2"][::-1],
                   color=colors[::-1], edgecolor="white")
    for bar, val in zip(bars, res_df["mean_R2"][::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.axvline(0.90, color="#27AE60", ls="--", lw=1.2, label="R²=0.90")
    ax.axvline(0.75, color="#F39C12", ls="--", lw=1.2, label="R²=0.75")
    ax.set_xlabel("Mean R² (predicting the other 4 metrics)")
    ax.set_title(f"Exhaustive 3-Metric Subset Selection  —  LLM experiments (n={n})\n"
                 "Mean R² across all remaining metrics (higher = better subset)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.08)
    plt.tight_layout()
    save_fig("fig8a_subset_bar")

    # ── Fig 8b: heatmap of top-10 subsets × target metric R² ───────────────
    top10 = res_df.head(10)
    target_cols = [c for c in top10.columns if c.startswith("R2_")]
    target_labels = [METRIC_LABEL.get(c.replace("R2_", ""), c.replace("R2_", ""))
                     for c in target_cols]
    heatmap_data = top10.set_index("Subset")[target_cols].rename(
        columns=dict(zip(target_cols, target_labels))
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGn",
                vmin=0, vmax=1, linewidths=0.4, linecolor="lightgray",
                cbar_kws={"label": "R²", "shrink": 0.8}, ax=ax)
    ax.set_title(f"R² per Target Metric  —  Top-10 Subsets  (n={n})\n"
                 "How well does each 3-metric subset predict each remaining metric?",
                 fontweight="bold")
    ax.set_xlabel("Target metric (being predicted)")
    ax.set_ylabel("3-metric subset used as predictors")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    save_fig("fig8b_subset_heatmap")

    # print top 5
    print("\nTop-5 3-metric subsets (by mean R2):")
    for _, row in res_df.head(5).iterrows():
        print(f"  [{row['mean_R2']:.3f}]  {row['Subset']}")
    print(f"\nBest subset: {res_df.iloc[0]['Subset']}")
    return res_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Metric Orthogonality Analysis")
    print("=" * 60)

    print("\n[1] Loading data ...")
    df = load_data()
    print(f"  Total rows: {len(df)}")
    print(f"  Sources:    {df['Source'].value_counts().to_dict()}")
    print(f"  Columns:    {[c for c in df.columns if c not in ('Source','Method')]}")

    print("\n[2] Correlation heatmaps ...")
    corr_llm, pvals_llm = fig1_corr_heatmap_llm(df)
    corr_ml,  pvals_ml  = fig2_corr_heatmap_ml(df)

    print("\n[3] Scatter matrix (LLM) ...")
    fig3_scatter_matrix(df)

    print("\n[4] Hierarchical clustering ...")
    cluster_df = fig4_dendrogram(df)

    print("\n[5] PCA ...")
    fig5_pca(df)

    print("\n[6] VIF analysis ...")
    fig6_vif(df)

    print("\n[7] Summary table ...")
    fig7_summary(corr_llm, cluster_df, df)

    print("\n[8] Exhaustive 3-metric subset selection ...")
    fig8_subset_selection(df)

    print(f"\nDone! → {PNG_DIR}/  and  {PDF_DIR}/")
