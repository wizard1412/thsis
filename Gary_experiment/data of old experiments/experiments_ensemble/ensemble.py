"""
Ensemble: LLM-Lasso (fewshot) + ML models
==========================================
final_score = alpha * llm_score + (1 - alpha) * ml_score

流程:
1. 讀取特徵資料集 (features_dataset.csv)，使用 LLM-Lasso 選出的 6 個特徵
2. 讀取 LLM 評分 CSV，取 10 次 run 的中位數作為 LLM 預測值
3. 對共同樣本做 LOOCV，對多種 ML 模型分別預測
4. 對 alpha in [0, 1] 做 grid search，找最佳加權比例
5. 輸出結果表格與圖表

執行: cd experiments_ensemble/ && python ensemble.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.base import is_classifier
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
FEAT_CSV = "../feature_selection/features_dataset.csv"
FS_JSON  = "../feature_selection/experiments_llm_lasso_deepseek70b/results/traditional_selection_results.json"

LLM_CSVS = {
    "deepseek-r1_70b": "../result/scored_by_deepseek-r1_70b_llmlasso.csv",
    "qwen2.5_72b":     "../result/scored_by_qwen2.5_72b_llmlasso.csv",
}

ML_MODELS = {
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "RandomForest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "ExtraTrees":       ExtraTreesRegressor(n_estimators=100, random_state=42),
    "SVR":              SVR(kernel="rbf", C=1.0),
    "Ridge":            Ridge(alpha=1.0),
    "KNN":              KNeighborsRegressor(n_neighbors=5),
    "KNN-clf":          KNeighborsClassifier(n_neighbors=5),
}

FS_METHOD   = "LLM-Lasso"
RANDOM_SEED = 42
MAX_SCORE   = 4
N_BOOT      = 2000

OUT_DIR = "results"
PNG_DIR = os.path.join(OUT_DIR, "png")
PDF_DIR = os.path.join(OUT_DIR, "pdf")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

RNG = np.random.default_rng(RANDOM_SEED)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def round_score(arr):
    return np.clip(np.round(np.array(arr, dtype=float)), 0, MAX_SCORE).astype(int)


def compute_metrics(y_true, y_pred_raw):
    yt  = round_score(np.array(y_true, dtype=float))
    yp  = round_score(np.array(y_pred_raw, dtype=float))
    mask = ~(np.isnan(yt.astype(float)) | np.isnan(yp.astype(float)))
    yt, yp = yt[mask], yp[mask]
    if len(yt) < 5:
        return dict(MAE=np.nan, Spearman=np.nan, Kappa=np.nan, AAcc=np.nan)
    mae   = mean_absolute_error(yt, yp)
    sp    = spearmanr(yt, yp).statistic
    try:
        kappa = cohen_kappa_score(yt, yp, weights="quadratic")
    except Exception:
        kappa = np.nan
    aacc  = (np.abs(yt - yp) <= 1).mean()
    return dict(MAE=mae, Spearman=sp, Kappa=kappa, AAcc=aacc)


def bootstrap_ci(y_true, y_pred_raw, metric="Kappa"):
    y_true = np.array(y_true)
    y_pred_raw = np.array(y_pred_raw)
    n = len(y_true)
    vals = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred_raw[idx])
        vals.append(m[metric])
    vals = np.array(vals)
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


def normalize_fname(s):
    return str(s).strip().replace(".csv", "")


def save_fig(name):
    plt.savefig(os.path.join(PNG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(PDF_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.close()


def run_loocv(X, y, model):
    loo   = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    y_fit = np.round(y).astype(int) if is_classifier(model) else y
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])
    for train_idx, test_idx in loo.split(X):
        pipe.fit(X[train_idx], y_fit[train_idx])
        preds[test_idx] = pipe.predict(X[test_idx])
    return preds


# ─────────────────────────────────────────────────────────────
# Step 1: 讀取特徵資料集
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading feature dataset ...")
feat_df = pd.read_csv(FEAT_CSV)
feat_df["filename_key"] = feat_df["filename"].apply(normalize_fname)

drop_cols = {"hand", "filename", "filename_key", "Rating1", "Rating2", "Rating"}
feat_cols = [c for c in feat_df.columns if c not in drop_cols]
feat_df = feat_df.dropna(subset=["Rating"]).reset_index(drop=True)
print(f"  Samples: {len(feat_df)}, Features: {len(feat_cols)}")

# ─────────────────────────────────────────────────────────────
# Step 2: 讀取 LLM-Lasso 特徵
# ─────────────────────────────────────────────────────────────
with open(FS_JSON) as f:
    fs_data = json.load(f)
selected_features = fs_data["selections"][FS_METHOD]
valid_feats = [f for f in selected_features if f in feat_cols]
feat_idx = [feat_cols.index(f) for f in valid_feats]
print(f"\nFS method: {FS_METHOD} -> {len(valid_feats)} features: {valid_feats}")

# ─────────────────────────────────────────────────────────────
# Step 3: LOOCV for all LLM x ML combinations
# ─────────────────────────────────────────────────────────────
ALPHAS = np.round(np.arange(0.0, 1.05, 0.05), 2)

# results_all[(llm_name, ml_name)] = {result_df, y_true, llm_preds, ml_preds}
results_all = {}
# cache ML preds per (llm_name, ml_name)
ml_preds_cache = {}   # (llm_name, ml_name) -> preds array

for llm_name, llm_csv_path in LLM_CSVS.items():
    print(f"\n{'='*60}")
    print(f"LLM: {llm_name}")

    if not os.path.exists(llm_csv_path):
        print(f"  [SKIP] File not found: {llm_csv_path}")
        continue

    llm_raw = pd.read_csv(llm_csv_path)
    llm_raw["filename_key"] = llm_raw["filename"].apply(normalize_fname)
    score_cols = [str(i) for i in range(10) if str(i) in llm_raw.columns]
    if not score_cols:
        print(f"  [SKIP] No score columns (0~9) found.")
        continue

    llm_raw["llm_pred"] = llm_raw[score_cols].median(axis=1)
    llm_lookup = llm_raw.set_index("filename_key")["llm_pred"].to_dict()

    feat_df["llm_pred"] = feat_df["filename_key"].map(llm_lookup)
    df_merged = feat_df.dropna(subset=["llm_pred"]).reset_index(drop=True)
    print(f"  Overlapping samples: {len(df_merged)}")

    X_sub     = df_merged[feat_cols].values.astype(float)[:, feat_idx]
    y_true    = df_merged["Rating"].values.astype(float)
    llm_preds = df_merged["llm_pred"].values.astype(float)

    for ml_name, ml_model in ML_MODELS.items():
        print(f"  Running LOOCV: {ml_name} ...", end=" ", flush=True)
        ml_preds = run_loocv(X_sub, y_true, ml_model)
        ml_preds_cache[(llm_name, ml_name)] = ml_preds

        rows = []
        for alpha in ALPHAS:
            ensemble_preds = alpha * llm_preds + (1 - alpha) * ml_preds
            m = compute_metrics(y_true, ensemble_preds)
            rows.append({"alpha": alpha, **m})
        result_df = pd.DataFrame(rows)

        best_row  = result_df.loc[result_df["Kappa"].idxmax()]
        m_ml_only = compute_metrics(y_true, ml_preds)
        print(f"best alpha={best_row['alpha']:.2f}  Kappa={best_row['Kappa']:.3f}  "
              f"(ML only={m_ml_only['Kappa']:.3f})")

        results_all[(llm_name, ml_name)] = {
            "result_df": result_df,
            "y_true":    y_true,
            "llm_preds": llm_preds,
            "ml_preds":  ml_preds,
        }

# ─────────────────────────────────────────────────────────────
# Step 4: Alpha sweep 圖 (per LLM, all ML models)
# ─────────────────────────────────────────────────────────────
if results_all:
    metrics_sweep  = ["Kappa", "Spearman", "MAE"]
    titles_sweep   = ["Quadratic Weighted Kappa", "Spearman r", "MAE"]
    hb_sweep       = [True, True, False]
    colors_sweep   = plt.cm.tab10.colors

    for llm_name in LLM_CSVS:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, metric, title, hb in zip(axes, metrics_sweep, titles_sweep, hb_sweep):
            for i, ml_name in enumerate(ML_MODELS):
                key = (llm_name, ml_name)
                if key not in results_all:
                    continue
                df = results_all[key]["result_df"]
                ax.plot(df["alpha"], df[metric], marker="o", markersize=4,
                        label=ml_name, color=colors_sweep[i])
                best_idx = df[metric].idxmax() if hb else df[metric].idxmin()
                ax.axvline(df.loc[best_idx, "alpha"], color=colors_sweep[i],
                           linestyle="--", alpha=0.35)
            ax.set_xlabel("alpha  (LLM weight)")
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle(f"Ensemble alpha sweep  [{llm_name}]", fontsize=13, y=1.02)
        plt.tight_layout()
        save_fig(f"ensemble_alpha_sweep_{llm_name}")
        print(f"\nFigure saved: {PNG_DIR}/ensemble_alpha_sweep_{llm_name}.png")

    # Save full alpha-sweep CSV
    summary_rows = []
    for (llm_name, ml_name), data in results_all.items():
        for _, row in data["result_df"].iterrows():
            summary_rows.append({"llm": llm_name, "ml": ml_name, **row})
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "ensemble_results.csv"), index=False)
    print(f"Results saved: {OUT_DIR}/ensemble_results.csv")

# ─────────────────────────────────────────────────────────────
# Step 5: 三方對比 (Only LLM vs Only ML vs Best Ensemble)
#         per LLM x ML model
# ─────────────────────────────────────────────────────────────
if results_all:
    print(f"\n{'='*60}")
    print("Comparison: Only LLM  vs  Only ML  vs  Best Ensemble")
    print("=" * 60)

    comparison_rows = []

    for llm_name in LLM_CSVS:
        print(f"\n  === LLM: {llm_name} ===")
        print(f"  {'ML Model':<18}  {'System':<14}  {'MAE':>6}  {'Spearman':>9}  "
              f"{'Kappa':>7} [95% CI]{'':>8}  {'AAcc':>6}")
        print("  " + "-" * 82)

        for ml_name in ML_MODELS:
            key = (llm_name, ml_name)
            if key not in results_all:
                continue
            data      = results_all[key]
            result_df = data["result_df"]
            y_true    = data["y_true"]
            llm_preds = data["llm_preds"]
            ml_preds  = data["ml_preds"]

            best_idx   = result_df["Kappa"].idxmax()
            best_row   = result_df.loc[best_idx]
            best_alpha = best_row["alpha"]
            best_ens   = best_alpha * llm_preds + (1 - best_alpha) * ml_preds

            m_ml  = compute_metrics(y_true, ml_preds)
            m_llm = compute_metrics(y_true, llm_preds)
            m_ens = compute_metrics(y_true, best_ens)

            ci_lo_ml,  ci_hi_ml  = bootstrap_ci(y_true, ml_preds)
            ci_lo_llm, ci_hi_llm = bootstrap_ci(y_true, llm_preds)
            ci_lo_ens, ci_hi_ens = bootstrap_ci(y_true, best_ens)

            systems = [
                ("Only ML",          m_ml,  ci_lo_ml,  ci_hi_ml,  0.0),
                ("Only LLM",         m_llm, ci_lo_llm, ci_hi_llm, 1.0),
                (f"Ens a={best_alpha:.2f}", m_ens, ci_lo_ens, ci_hi_ens, best_alpha),
            ]

            for k, (sys_name, m, ci_lo, ci_hi, alpha_val) in enumerate(systems):
                ml_col = ml_name if k == 0 else ""
                print(f"  {ml_col:<18}  {sys_name:<14}  {m['MAE']:>6.3f}  "
                      f"{m['Spearman']:>9.3f}  {m['Kappa']:>7.3f} "
                      f"[{ci_lo:.3f},{ci_hi:.3f}]  {m['AAcc']:>6.3f}")
                comparison_rows.append({
                    "llm": llm_name, "ml": ml_name, "system": sys_name,
                    "alpha": alpha_val,
                    "MAE": m["MAE"], "Spearman": m["Spearman"],
                    "Kappa": m["Kappa"],
                    "Kappa_CI_lo": ci_lo, "Kappa_CI_hi": ci_hi,
                    "AAcc": m["AAcc"],
                })

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = os.path.join(OUT_DIR, "comparison_results.csv")
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nComparison table saved: {comp_csv}")

# ─────────────────────────────────────────────────────────────
# Step 6: 對比長條圖 — per LLM, group by ML model
# ─────────────────────────────────────────────────────────────
if results_all and comparison_rows:
    plot_metrics_c  = ["Kappa", "Spearman", "MAE", "AAcc"]
    plot_titles_c   = ["Quadratic Weighted Kappa", "Spearman r", "MAE", "Adjacent Accuracy"]
    hb_c            = [True, True, False, True]

    colors_sys = ["#4C72B0", "#DD8452", "#55A868"]   # ML, LLM, Ensemble
    hatches    = ["", "//", "xx"]
    sys_order  = ["Only ML", "Only LLM"]   # third is dynamic

    for llm_name in LLM_CSVS:
        sub = comp_df[comp_df["llm"] == llm_name]
        if sub.empty:
            continue

        ml_names = list(ML_MODELS.keys())
        n_sys    = 3
        bar_w    = 0.22
        group_gap = 0.4

        fig, axes = plt.subplots(1, len(plot_metrics_c),
                                 figsize=(5 * len(plot_metrics_c), 5))

        for ax, metric, title, hb in zip(axes, plot_metrics_c, plot_titles_c, hb_c):
            x_pos    = 0.0
            x_ticks  = []
            x_labels = []

            for ml_name in ml_names:
                rows_ml = sub[sub["ml"] == ml_name].sort_values("system")
                # order: Only ML, Only LLM, Ensemble
                def sys_sort(s):
                    if "Only ML"  == s: return 0
                    if "Only LLM" == s: return 1
                    return 2
                rows_ml = rows_ml.copy()
                rows_ml["_order"] = rows_ml["system"].map(sys_sort)
                rows_ml = rows_ml.sort_values("_order")

                group_center = x_pos + (n_sys - 1) * bar_w / 2
                x_ticks.append(group_center)
                x_labels.append(ml_name)

                for j, (_, row) in enumerate(rows_ml.iterrows()):
                    val = row[metric]
                    lbl = row["system"] if ml_name == ml_names[0] else ""
                    ax.bar(x_pos, val, width=bar_w,
                           color=colors_sys[j], hatch=hatches[j],
                           edgecolor="white", linewidth=0.8, label=lbl)
                    if metric == "Kappa":
                        ci_lo = row["Kappa_CI_lo"]
                        ci_hi = row["Kappa_CI_hi"]
                        ax.errorbar(x_pos, val,
                                    yerr=[[val - ci_lo], [ci_hi - val]],
                                    fmt="none", color="black",
                                    capsize=3, linewidth=1.0)
                    ax.text(x_pos, val + 0.005, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=6.5, rotation=90)
                    x_pos += bar_w

                x_pos += group_gap

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, fontsize=8, rotation=20, ha="right")
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)
            if hb:
                ax.set_ylim(0, min(1.25, ax.get_ylim()[1] * 1.2))
            else:
                ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

        axes[0].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Only ML vs Only LLM vs Best Ensemble  [{llm_name}]",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        save_fig(f"comparison_ml_llm_ensemble_{llm_name}")
        print(f"Comparison figure saved: {PNG_DIR}/comparison_ml_llm_ensemble_{llm_name}.png")

# ─────────────────────────────────────────────────────────────
# Step 7: ML-only 模型對比摘要表
# ─────────────────────────────────────────────────────────────
if comparison_rows:
    print(f"\n{'='*60}")
    print("ML-only model summary (alpha=0, no LLM):")
    print("=" * 60)
    ml_only_df = comp_df[comp_df["system"] == "Only ML"].copy()
    ml_only_df = ml_only_df[["llm", "ml", "MAE", "Spearman", "Kappa",
                              "Kappa_CI_lo", "Kappa_CI_hi", "AAcc"]]
    # ML preds are the same regardless of LLM, just show once per ml
    ml_summary = ml_only_df.drop_duplicates("ml").drop(columns="llm")
    print(ml_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_ens_df = comp_df[comp_df["system"].str.startswith("Ens")].copy()
    print(f"\n{'='*60}")
    print("Best ensemble summary (best alpha by Kappa):")
    print("=" * 60)
    print(best_ens_df[["llm", "ml", "alpha", "MAE", "Spearman",
                        "Kappa", "Kappa_CI_lo", "Kappa_CI_hi", "AAcc"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\nDone.")
