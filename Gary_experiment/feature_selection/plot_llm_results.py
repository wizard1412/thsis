import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr

CACHE  = "fs_results_llm/llm_score_cache.json"
FEAT   = "features_dataset_v2.csv"
MLP    = "fs_results_v2/loocv_results.csv"
PNG    = "fs_results_llm/png"
PDF    = "fs_results_llm/pdf"
os.makedirs(PNG, exist_ok=True)
os.makedirs(PDF, exist_ok=True)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 150})
sns.set_theme(style="whitegrid")

FS_CATEGORY = {
    "Pearson": "filter", "Spearman": "filter", "Kendall": "filter",
    "Mutual Info": "filter", "F-regression": "filter", "mRMR": "filter",
    "Distance Correlation": "filter", "HSIC": "filter", "Fisher Score": "filter",
    "ReliefF": "filter", "FCBF": "filter", "Variance Threshold": "filter",
    "LASSO (Standard)": "embedded", "ElasticNet": "embedded", "Adaptive LASSO": "embedded",
    "LARS": "embedded", "RF Importance": "embedded", "Extra Trees": "embedded",
    "Gradient Boosting": "embedded", "XGBoost": "embedded",
    "Permutation Importance": "embedded", "Stability Selection": "embedded",
    "RFECV (RF)": "wrapper", "RFECV (SVR)": "wrapper", "RFECV (Lasso)": "wrapper",
    "SFS Forward": "wrapper", "SBS Backward": "wrapper", "Boruta": "wrapper",
}
CAT_COLOR = {"filter": "#3498DB", "embedded": "#E67E22", "wrapper": "#27AE60", "other": "#95A5A6"}

# ─── Load data ───────────────────────────────────────────────────────────────
df_feat = pd.read_csv(FEAT).dropna(subset=["Rating"]).reset_index(drop=True)
y_int   = np.clip(np.floor(df_feat["Rating"].values.astype(float) + 0.5).astype(int), 0, 4)

with open(CACHE) as f:
    cache = json.load(f)


def agg(raw):
    pm, pmed = [], []
    for p in raw:
        valid = [s for s in p if s is not None]
        pm.append(max(set(valid), key=valid.count) if valid else np.nan)
        pmed.append(float(np.median(valid)) if valid else np.nan)
    return np.array(pm, dtype=float), np.array(pmed, dtype=float)


def calc_metrics(yt, yp_raw):
    yp = np.array(yp_raw, dtype=float)
    mask = ~np.isnan(yp)
    yt2 = yt[mask]
    yp2 = np.clip(np.floor(yp[mask] + 0.5).astype(int), 0, 4)
    if len(yt2) < 5:
        return dict(MAE=np.nan, Kappa=np.nan, Spearman=np.nan, AAcc=np.nan)
    sp = spearmanr(yt2, yp2).statistic
    try:
        kappa = cohen_kappa_score(yt2, yp2, weights="quadratic")
    except Exception:
        kappa = np.nan
    return dict(MAE=mean_absolute_error(yt2, yp2), Kappa=kappa, Spearman=sp,
                AAcc=(np.abs(yt2 - yp2) <= 1).mean())


# Build LLM results from n3 cache
llm_rows = []
for key, raw in cache.items():
    if "|n3|v2" not in key:
        continue
    method = key.split("|")[0]
    if not raw or not isinstance(raw[0], list):
        continue
    pm, pmed = agg(raw)
    m_mode = calc_metrics(y_int, pm)
    m_med  = calc_metrics(y_int, pmed)
    llm_rows.append({
        "FS_Method": method,
        **{f"LLM_mode_{k}": v for k, v in m_mode.items()},
        **{f"LLM_med_{k}":  v for k, v in m_med.items()},
    })
llm = pd.DataFrame(llm_rows)

# Load MLP results
mlp_raw = pd.read_csv(MLP)[["FS_Method", "N_Features", "Category", "MAE", "Kappa", "Spearman", "AAcc"]]
mlp_raw.columns = ["FS_Method", "N_Features", "Category", "MLP_MAE", "MLP_Kappa", "MLP_Spearman", "MLP_AAcc"]

merged = mlp_raw.merge(llm, on="FS_Method").sort_values("MLP_Kappa", ascending=False)
print(f"Methods with both MLP and LLM results: {len(merged)}")


def save_fig(name):
    plt.savefig(f"{PNG}/{name}.png", dpi=300, bbox_inches="tight")
    for ax in plt.gcf().get_axes():
        ax.set_title("")
    plt.gcf().suptitle("")
    plt.savefig(f"{PDF}/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}")


# ── Fig 1: Ranked bar MLP vs LLM Kappa ──────────────────────────────────────
sub = merged.sort_values("MLP_Kappa", ascending=True)
fig, ax = plt.subplots(figsize=(8, max(6, len(sub) * 0.32)))
y_pos = np.arange(len(sub))
ax.barh(y_pos + 0.2, sub["MLP_Kappa"],      0.35, label="MLP(8)",
        color="#8E44AD", alpha=0.85, edgecolor="white")
ax.barh(y_pos - 0.2, sub["LLM_mode_Kappa"], 0.35, label="deepseek-r1:32b",
        color="#27AE60", alpha=0.85, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(sub["FS_Method"], fontsize=8)
ax.set_xlabel("Weighted Kappa ↑")
ax.set_title("MLP(8) LOOCV vs deepseek-r1:32b — Weighted Kappa per FS Method")
ax.legend(fontsize=9)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlim(-0.2, 0.9)
plt.tight_layout()
save_fig("fig1_ranked_kappa_mlp_vs_llm")

# ── Fig 2: Scatter MLP Kappa vs LLM Kappa ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
colors = [CAT_COLOR.get(FS_CATEGORY.get(m, "other"), "#95A5A6") for m in merged["FS_Method"]]
ax.scatter(merged["MLP_Kappa"], merged["LLM_mode_Kappa"],
           c=colors, s=90, edgecolors="white", zorder=3)
for _, row in merged.iterrows():
    ax.annotate(row["FS_Method"], (row["MLP_Kappa"], row["LLM_mode_Kappa"]),
                textcoords="offset points", xytext=(4, 3), fontsize=6.5)
lo = min(merged["MLP_Kappa"].min(), merged["LLM_mode_Kappa"].min()) - 0.05
hi = max(merged["MLP_Kappa"].max(), merged["LLM_mode_Kappa"].max()) + 0.05
ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="y = x  (equal)")
ax.set_xlabel("MLP(8) LOOCV  Weighted Kappa ↑", fontsize=11)
ax.set_ylabel("deepseek-r1:32b  Weighted Kappa ↑", fontsize=11)
ax.set_title("MLP vs LLM: Weighted Kappa\n(above diagonal = LLM wins)")
ax.legend(fontsize=9)
patches = [mpatches.Patch(color=c, label=k.capitalize())
           for k, c in CAT_COLOR.items() if k != "other"]
fig.legend(handles=patches, title="FS Category", fontsize=9,
           loc="lower right", bbox_to_anchor=(1.01, 0.0))
plt.tight_layout()
save_fig("fig2_scatter_kappa")

# ── Fig 3: Heatmap both evaluators ──────────────────────────────────────────
METS = ["Kappa", "MAE", "Spearman", "AAcc"]
sub2 = merged.set_index("FS_Method").sort_values("MLP_Kappa", ascending=False)
mlp_vals = sub2[["MLP_"    + m for m in METS]].copy()
llm_vals = sub2[["LLM_mode_" + m for m in METS]].copy()
mlp_vals.columns = METS
llm_vals.columns = METS

fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(sub2) * 0.28)))
for ax, data, title in zip(axes, [mlp_vals, llm_vals],
                            ["MLP(8) LOOCV", "deepseek-r1:32b (mode, 3×)"]):
    annot = data.applymap(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    display = data.copy()
    display["MAE"] = -display["MAE"]   # flip so green = good for all columns
    sns.heatmap(display, annot=annot, fmt="", cmap="RdYlGn", ax=ax,
                linewidths=0.5, linecolor="lightgray", cbar_kws={"shrink": 0.6})
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Feature Selection Method")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(["Kappa ↑", "MAE ↓", "Spearman ↑", "AAcc ↑"], fontsize=9)
fig.suptitle("MLP vs deepseek-r1:32b — All Metrics per FS Method  (n=53)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("fig3_heatmap_both")

# ── Fig 4: LLM mode vs median Kappa ─────────────────────────────────────────
sub3 = merged.sort_values("LLM_mode_Kappa", ascending=True)
fig, ax = plt.subplots(figsize=(8, max(6, len(sub3) * 0.32)))
y_pos = np.arange(len(sub3))
ax.barh(y_pos + 0.2, sub3["LLM_mode_Kappa"], 0.35, label="Mode",
        color="#27AE60", alpha=0.85, edgecolor="white")
ax.barh(y_pos - 0.2, sub3["LLM_med_Kappa"],  0.35, label="Median",
        color="#2980B9", alpha=0.85, edgecolor="white")
ax.set_yticks(y_pos)
ax.set_yticklabels(sub3["FS_Method"], fontsize=8)
ax.set_xlabel("Weighted Kappa ↑")
ax.set_title("deepseek-r1:32b — Mode vs Median Aggregation  (3 evals/patient)")
ax.legend(fontsize=9)
ax.axvline(0, color="gray", lw=0.8)
plt.tight_layout()
save_fig("fig4_mode_vs_median")

# ── Fig 5: N_features vs Kappa scatter ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, kcol, title in zip(
    axes,
    ["MLP_Kappa", "LLM_mode_Kappa"],
    ["MLP(8) LOOCV", "deepseek-r1:32b"]
):
    colors = [CAT_COLOR.get(FS_CATEGORY.get(m, "other"), "#95A5A6") for m in merged["FS_Method"]]
    ax.scatter(merged["N_Features"], merged[kcol], c=colors, s=80, edgecolors="white", zorder=3)
    for _, row in merged.iterrows():
        ax.annotate(row["FS_Method"], (row["N_Features"], row[kcol]),
                    textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    ax.axhline(0.5, color="#E74C3C", ls="--", lw=1.2, label="Kappa = 0.5")
    ax.set_xlabel("Number of Selected Features")
    ax.set_ylabel("Weighted Kappa ↑")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
patches = [mpatches.Patch(color=c, label=k.capitalize())
           for k, c in CAT_COLOR.items() if k != "other"]
fig.legend(handles=patches, title="FS Category", fontsize=9,
           loc="lower right", bbox_to_anchor=(1.01, 0.0))
fig.suptitle("Number of Features vs Weighted Kappa — MLP vs LLM",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("fig5_nfeats_vs_kappa")

# ── Fig 6: LLM only — ranked bar all 4 metrics ──────────────────────────────
sub_llm = merged.sort_values("LLM_mode_Kappa", ascending=True)
METS4   = ["Kappa", "MAE", "Spearman", "AAcc"]
LABELS4 = ["Weighted Kappa ↑", "MAE ↓", "Spearman r ↑", "Adjacent Acc ↑"]
LOWER   = {"MAE"}

fig, axes = plt.subplots(2, 2, figsize=(16, max(10, len(sub_llm) * 0.35)))
axes_flat = axes.flatten()
y_pos = np.arange(len(sub_llm))
colors = [CAT_COLOR.get(FS_CATEGORY.get(m, "other"), "#95A5A6") for m in sub_llm["FS_Method"]]

for ax, met, lab in zip(axes_flat, METS4, LABELS4):
    col = f"LLM_mode_{met}"
    sort_asc = met in LOWER
    order = sub_llm.sort_values(col, ascending=not sort_asc)
    c_ord = [CAT_COLOR.get(FS_CATEGORY.get(m, "other"), "#95A5A6") for m in order["FS_Method"]]
    bars = ax.barh(range(len(order)), order[col], color=c_ord, edgecolor="white", alpha=0.88)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["FS_Method"], fontsize=8)
    for bar, val, n in zip(bars, order[col], order["N_Features"]):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} (n={int(n)})", va="center", ha="left", fontsize=7)
    ax.set_xlabel(lab)
    ax.set_title(f"deepseek-r1:32b — {lab}", fontweight="bold")
    ax.set_xlim(0 if met != "MAE" else 0,
                (order[col].max() + 0.15) if met != "Kappa" else 0.85)

patches = [mpatches.Patch(color=c, label=k.capitalize())
           for k, c in CAT_COLOR.items() if k != "other"]
fig.legend(handles=patches, title="FS Category", fontsize=9,
           loc="lower right", bbox_to_anchor=(1.01, 0.0))
fig.suptitle("deepseek-r1:32b Direct Scoring — FS Methods Ranked  (n=53, 3 evals/patient)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("fig6_llm_only_all_metrics")

# ── Fig 7: LLM only — heatmap ────────────────────────────────────────────────
sub_h = merged.set_index("FS_Method").sort_values("LLM_mode_Kappa", ascending=False)
llm_h = sub_h[["LLM_mode_" + m for m in METS4]].copy()
llm_h.columns = METS4

fig, ax = plt.subplots(figsize=(7, max(6, len(llm_h) * 0.28)))
annot = llm_h.map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
display = llm_h.copy()
display["MAE"] = -display["MAE"]
sns.heatmap(display, annot=annot, fmt="", cmap="RdYlGn", ax=ax,
            linewidths=0.5, linecolor="lightgray", cbar_kws={"shrink": 0.6})
ax.set_title("deepseek-r1:32b — Performance per FS Method  (sorted by Kappa)",
             fontweight="bold")
ax.set_ylabel("Feature Selection Method")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
ax.set_xticklabels(["Kappa ↑", "MAE ↓", "Spearman ↑", "AAcc ↑"], fontsize=9)
plt.tight_layout()
save_fig("fig7_llm_only_heatmap")

# ── Fig 8: LLM only — N_features vs Kappa (broken x-axis) ───────────────────
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec

nf = merged["N_Features"].values
all_nf  = np.sort(np.unique(nf))
gaps    = [(all_nf[i+1]-all_nf[i], all_nf[i], all_nf[i+1])
           for i in range(len(all_nf)-1)]
best_gap = max(gaps, key=lambda g: g[0])
break_lo, break_hi = float(best_gap[1]), float(best_gap[2])

main_xlim = (float(all_nf[0]) - 1.5, break_lo + 1.5)
out_xlim  = (break_hi - 1.5,         float(all_nf[-1]) + 5.0)
w_ratio   = [max(main_xlim[1]-main_xlim[0], 1),
             max(out_xlim[1] -out_xlim[0],  1)]

colors = [CAT_COLOR.get(FS_CATEGORY.get(m, "other"), "#95A5A6") for m in merged["FS_Method"]]

fig = plt.figure(figsize=(11, 6))
gs  = GridSpec(1, 2, figure=fig, width_ratios=w_ratio, wspace=0.06)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)

def _draw(ax, ann_lo, ann_hi):
    ax.scatter(merged["N_Features"], merged["LLM_mode_Kappa"],
               c=colors, s=90, edgecolors="white", zorder=3)
    for _, row in merged.iterrows():
        if ann_lo <= row["N_Features"] <= ann_hi:
            ax.annotate(row["FS_Method"],
                        (row["N_Features"], row["LLM_mode_Kappa"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6.5)
    ax.axhline(0.5, color="#E74C3C", ls="--", lw=1.2, label="Kappa = 0.5")

_draw(ax1, ann_lo=main_xlim[0], ann_hi=main_xlim[1])
_draw(ax2, ann_lo=out_xlim[0],  ann_hi=out_xlim[1])
ax1.set_xlim(*main_xlim)
ax2.set_xlim(*out_xlim)

ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.tick_params(left=False, labelleft=False)

d = 0.018
for _ax, xs in ((ax1, (1-d, 1+d)), (ax2, (-d, +d))):
    kw = dict(color="k", clip_on=False, lw=1.2, transform=_ax.transAxes)
    _ax.plot(list(xs), [-d, +d],       **kw)
    _ax.plot(list(xs), [1-d, 1+d],     **kw)

ax1.set_xlabel("Number of Selected Features")
ax1.set_ylabel("Weighted Kappa ↑")
ax2.set_xlabel("")
ax1.legend(fontsize=9)
ax1.set_title("deepseek-r1:32b — Number of Features vs Kappa  (broken axis)")

patches = [mpatches.Patch(color=c, label=k.capitalize())
           for k, c in CAT_COLOR.items() if k != "other"]
fig.legend(handles=patches, title="FS Category", fontsize=9,
           loc="lower right", bbox_to_anchor=(1.01, 0.0))
plt.tight_layout()
save_fig("fig8_llm_only_nfeats_vs_kappa")

# ── Fig 9 & 10: Feature frequency (LLM-passing & MLP-passing) ───────────────
from collections import Counter

with open("traditional_fs_results_v2/traditional_selection_results.json") as f:
    fs_data = json.load(f)
selections = fs_data["selections"]

for evaluator, kappa_col, label, figname, nfeat_thresh in [
    ("LLM", "LLM_mode_Kappa", "deepseek-r1:32b", "fig9_llm_feature_frequency",  10),
    ("MLP", "MLP_Kappa",       "MLP(8) LOOCV",   "fig10_mlp_feature_frequency", 15),
]:
    kappa_thresh = 0.5

    passed = merged[
        (merged[kappa_col] >= kappa_thresh) &
        (merged["N_Features"] < nfeat_thresh)
    ]["FS_Method"].tolist()

    print(f"\n  [{evaluator}] Kappa >= {kappa_thresh}, N_Features < {nfeat_thresh}: "
          f"{len(passed)} methods pass: {passed}")

    counter = Counter()
    for method in passed:
        for feat in selections.get(method, []):
            counter[feat] += 1

    if not counter:
        print(f"  WARNING: No methods passed for {evaluator}; using all methods.")
        for feats in selections.values():
            for feat in feats:
                counter[feat] += 1
        passed = list(selections.keys())

    n_methods = len(passed)
    top_n     = min(25, len(counter))
    top       = counter.most_common(top_n)
    feats_top, counts_top = zip(*top)

    thr_high = max(1, round(n_methods * 0.5))
    thr_low  = max(1, round(n_methods * 0.25))

    bar_colors = [
        "#2980B9" if c >= thr_high else
        "#85C1E9" if c >= thr_low  else
        "#D6EAF8"
        for _, c in top
    ]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(range(top_n), counts_top[::-1],
                   color=bar_colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feats_top[::-1], fontsize=9)
    ax.set_xlabel(
        f"Number of FS Methods Selecting This Feature\n"
        f"(filter: Kappa ≥ {kappa_thresh}, N_Features < {nfeat_thresh}, "
        f"n = {n_methods} methods)",
        fontsize=10)
    ax.set_title(
        f"Feature Frequency Among Well-Performing FS Methods  [{label}]",
        fontweight="bold")
    ax.axvline(thr_high, color="#E74C3C", ls="--", lw=1.2)
    ax.axvline(thr_low,  color="#F39C12", ls="--", lw=1.2)
    for bar, cnt in zip(bars, counts_top[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=9)
    legend_handles = [
        mpatches.Patch(color="#2980B9", label=f">= {thr_high} methods (≥50%)"),
        mpatches.Patch(color="#85C1E9", label=f">= {thr_low}  methods (≥25%)"),
        mpatches.Patch(color="#D6EAF8", label=f"<  {thr_low}  methods"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.set_xlim(0, n_methods + 2)
    plt.tight_layout()
    save_fig(figname)

# ── Fig 11: FS method agreement — which methods pass LLM / MLP / both ───────
kappa_thresh     = 0.5
llm_nfeat_thresh = 10
mlp_nfeat_thresh = 15

llm_passed = set(merged[(merged["LLM_mode_Kappa"] >= kappa_thresh) &
                         (merged["N_Features"]     <  llm_nfeat_thresh)]["FS_Method"])
mlp_passed = set(merged[(merged["MLP_Kappa"]       >= kappa_thresh) &
                         (merged["N_Features"]      <  mlp_nfeat_thresh)]["FS_Method"])

all_methods = sorted(merged["FS_Method"].unique())
llm_flags   = [1 if m in llm_passed else 0 for m in all_methods]
mlp_flags   = [1 if m in mlp_passed else 0 for m in all_methods]

status = []
for m in all_methods:
    if m in llm_passed and m in mlp_passed:
        status.append("Both")
    elif m in llm_passed:
        status.append("LLM only")
    elif m in mlp_passed:
        status.append("MLP only")
    else:
        status.append("Neither")
status_color = {"Both": "#27AE60", "LLM only": "#2980B9",
                "MLP only": "#8E44AD", "Neither": "#BDC3C7"}

llm_kappas = [merged.loc[merged["FS_Method"]==m, "LLM_mode_Kappa"].values[0] for m in all_methods]
mlp_kappas = [merged.loc[merged["FS_Method"]==m, "MLP_Kappa"].values[0]       for m in all_methods]

sort_idx = sorted(range(len(all_methods)),
                  key=lambda i: (mlp_kappas[i] + llm_kappas[i]) / 2)
all_methods_s = [all_methods[i] for i in sort_idx]
llm_kappas_s  = [llm_kappas[i]  for i in sort_idx]
mlp_kappas_s  = [mlp_kappas[i]  for i in sort_idx]
status_s      = [status[i]       for i in sort_idx]
bar_colors    = [status_color[s] for s in status_s]

fig, ax = plt.subplots(figsize=(9, max(6, len(all_methods_s) * 0.32)))
y_pos = np.arange(len(all_methods_s))
ax.barh(y_pos + 0.2, mlp_kappas_s, 0.35, color="#8E44AD", alpha=0.75,
        edgecolor="white", label="MLP(8)")
ax.barh(y_pos - 0.2, llm_kappas_s, 0.35, color="#2980B9", alpha=0.75,
        edgecolor="white", label="deepseek-r1:32b")
ax.set_yticks(y_pos)
ax.set_yticklabels(all_methods_s, fontsize=8)
ax.axvline(kappa_thresh, color="#E74C3C", ls="--", lw=1.2,
           label=f"Kappa = {kappa_thresh}")
ax.set_xlabel("Weighted Kappa ↑")
ax.set_title(f"FS Method Agreement: MLP vs LLM  (Kappa={kappa_thresh}, LLM N<{llm_nfeat_thresh}, MLP N<{mlp_nfeat_thresh})")
ax.set_xlim(-0.2, 0.9)
status_handles = [mpatches.Patch(color=c, label=s)
                  for s, c in status_color.items()]
ax.legend(fontsize=9)
fig.legend(handles=status_handles, title="Passes threshold",
           fontsize=8, loc="lower right", bbox_to_anchor=(1.01, 0.0))
plt.tight_layout()
save_fig("fig11_method_agreement")

# ── Fig 12: Feature frequency side-by-side LLM vs MLP ───────────────────────
llm_counter = Counter()
mlp_counter = Counter()
for m in llm_passed:
    for feat in selections.get(m, []):
        llm_counter[feat] += 1
for m in mlp_passed:
    for feat in selections.get(m, []):
        mlp_counter[feat] += 1

all_feats = sorted(set(llm_counter) | set(mlp_counter),
                   key=lambda f: -(llm_counter.get(f, 0) + mlp_counter.get(f, 0)))
top_feats = all_feats[:25]

llm_counts = [llm_counter.get(f, 0) for f in top_feats]
mlp_counts = [mlp_counter.get(f, 0) for f in top_feats]

feat_status = []
for f in top_feats:
    lc, mc = llm_counter.get(f, 0), mlp_counter.get(f, 0)
    if lc > 0 and mc > 0:
        feat_status.append("both")
    elif lc > 0:
        feat_status.append("llm_only")
    else:
        feat_status.append("mlp_only")

fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(top_feats) * 0.35)),
                          sharey=True)
y_pos = np.arange(len(top_feats))

for ax, counts, label, color in [
    (axes[0], mlp_counts, f"MLP(8)  [{len(mlp_passed)} methods]",  "#8E44AD"),
    (axes[1], llm_counts, f"deepseek-r1:32b  [{len(llm_passed)} methods]", "#2980B9"),
]:
    bars = ax.barh(y_pos, counts, color=color, alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.set_xlabel("Times selected by passing methods")
    ax.set_title(label, fontweight="bold")
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=8)
    ax.invert_yaxis()

feat_handles = [
    mpatches.Patch(color="#27AE60", label="Selected by both"),
    mpatches.Patch(color="#2980B9", label="LLM only"),
    mpatches.Patch(color="#8E44AD", label="MLP only"),
]
fig.suptitle(
    f"Feature Frequency: LLM vs MLP  "
    f"(Kappa ≥ {kappa_thresh}, LLM N<{llm_nfeat_thresh}, MLP N<{mlp_nfeat_thresh})",
    fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig("fig12_feature_frequency_comparison")

print("\nAll figures saved to fs_results_llm/png/ and fs_results_llm/pdf/")
