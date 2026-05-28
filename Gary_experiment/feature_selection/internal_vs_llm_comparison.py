"""
Internal Estimator vs LLM Scoring Comparison
=============================================
For each embedded and wrapper feature selection method, run LOO-CV with the
method's own internal estimator, then compare against the LLM direct-scoring
results already stored in fs_results_llm/comparison_results.csv.

Key question: does the LLM evaluator agree with the method's native model?

Internal estimators (same as wrapper_vs_mlp_comparison.py)
-----------------------------------------------------------
  Embedded:
    LASSO (Standard)     → LassoCV
    ElasticNet           → ElasticNetCV
    Adaptive LASSO       → LassoCV
    LARS                 → LassoLarsCV
    RF Importance        → RandomForestRegressor(500)
    Extra Trees          → ExtraTreesRegressor(500)
    Gradient Boosting    → GradientBoostingRegressor
    XGBoost              → XGBRegressor
    Permutation Imp.     → RandomForestRegressor(300)
    Stability Selection  → LassoCV
  Wrapper:
    RFECV (RF)           → OrdinalRFClassifier(200)
    RFECV (SVR)          → SVR(kernel=linear)
    RFECV (Lasso)        → Lasso(alpha=0.01)
    SFS Forward          → Ridge
    SBS Backward         → Ridge
    Boruta               → RandomForestRegressor(200)

LLM results source: fs_results_llm/comparison_results.csv
  Columns used: LLM_median_Kappa (and all other metrics + CIs)

Output directory: fs_results_llm_vs_internal/
  fig1_bar_kappa.pdf/png
  fig2_scatter_kappa.pdf/png
  fig3_all_metrics.pdf/png
  fig4_kappa_delta.pdf/png
  internal_vs_llm_metrics.csv
"""

import json, warnings
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
from sklearn.linear_model import (
    Lasso, LassoCV, LassoLarsCV, ElasticNetCV, Ridge,
)
from sklearn.svm import SVR
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
FEAT_CSV = HERE / "features_dataset_v2.csv"
FS_JSON  = HERE / "traditional_fs_results_v2" / "traditional_selection_results.json"
LLM_CSV  = HERE / "fs_results_llm" / "comparison_results.csv"
OUT_DIR  = HERE / "fs_results_llm_vs_internal"
OUT_DIR.mkdir(exist_ok=True)

MAX_SCORE   = 4
RANDOM_SEED = 42
RNG         = np.random.default_rng(RANDOM_SEED)

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})
sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# OrdinalRFClassifier (mirrors wrapper_vs_mlp_comparison.py)
# ─────────────────────────────────────────────────────────────────────────────
class OrdinalRFClassifier(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=200, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        y_binned  = np.clip(np.floor(np.asarray(y) + 0.5).astype(int), 0, 4)
        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.clf_.fit(X, y_binned)
        return self

    def predict(self, X):
        proba      = self.clf_.predict_proba(X)
        full_proba = np.zeros((len(X), 5))
        for i, c in enumerate(self.clf_.classes_):
            full_proba[:, c] = proba[:, i]
        return (full_proba * np.arange(5)).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Method → (category, estimator_factory)
# ─────────────────────────────────────────────────────────────────────────────
_lasso_est = lambda: LassoCV(cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=100)
_en_est    = lambda: ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=50,
)
_lars_est  = lambda: LassoLarsCV(cv=5, max_iter=500)
_rf500_est = lambda: RandomForestRegressor(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
_et500_est = lambda: ExtraTreesRegressor(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
_gb_est    = lambda: GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=3,
    subsample=0.8, random_state=RANDOM_SEED,
)
_xgb_est   = lambda: (XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, verbosity=0,
) if _HAS_XGB else None)
_rf300_est = lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
_rf200_est = lambda: RandomForestRegressor(n_estimators=200, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1)

FS_ESTIMATORS = {
    # Embedded
    "LASSO (Standard)":       ("embedded", _lasso_est),
    "ElasticNet":             ("embedded", _en_est),
    "Adaptive LASSO":         ("embedded", _lasso_est),
    "LARS":                   ("embedded", _lars_est),
    "RF Importance":          ("embedded", _rf500_est),
    "Extra Trees":            ("embedded", _et500_est),
    "Gradient Boosting":      ("embedded", _gb_est),
    "XGBoost":                ("embedded", _xgb_est),
    "Permutation Importance": ("embedded", _rf300_est),
    "Stability Selection":    ("embedded", _lasso_est),
    # Wrapper
    "RFECV (RF)":             ("wrapper",  lambda: OrdinalRFClassifier(
                                  n_estimators=200, max_depth=5, random_state=RANDOM_SEED)),
    "RFECV (SVR)":            ("wrapper",  lambda: SVR(kernel="linear", C=1.0)),
    "RFECV (Lasso)":          ("wrapper",  lambda: Lasso(alpha=0.01, max_iter=50000,
                                                         random_state=RANDOM_SEED)),
    "SFS Forward":            ("wrapper",  lambda: Ridge(alpha=1.0)),
    "SBS Backward":           ("wrapper",  lambda: Ridge(alpha=1.0)),
    "Boruta":                 ("wrapper",  _rf200_est),
}

CAT_COLOR = {
    "embedded": "#E67E22",
    "wrapper":  "#27AE60",
}
LLM_COLOR = "#8E44AD"


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    drop_cols = {"hand", "filename", "Rating1", "Rating2", "Rating", "tap_amplitudes"}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    X  = df[feat_cols].values.astype(float)
    y  = df["Rating"].values.astype(float)
    print(f"  Data: {X.shape[0]} samples, {len(feat_cols)} features")
    return X, y, feat_cols


def load_selections(feat_cols):
    with open(FS_JSON) as f:
        d = json.load(f)
    sels = d["selections"]
    fidx = {f: i for i, f in enumerate(feat_cols)}
    result = {}
    for method in FS_ESTIMATORS:
        feats = sels.get(method, [])
        idx   = [fidx[f] for f in feats if f in fidx]
        if idx:
            result[method] = idx
            print(f"  {method:25s}: {len(idx):3d} features")
        else:
            print(f"  {method:25s}: [skip]")
    return result


def load_llm_results():
    df = pd.read_csv(LLM_CSV)
    df = df.set_index("FS_Method")
    print(f"  LLM CSV: {LLM_CSV.name}  ({len(df)} methods)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
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
    mae   = mean_absolute_error(yt, yp)
    sp    = spearmanr(yt, yp).statistic
    try:
        kappa = cohen_kappa_score(yt.astype(int), yp.astype(int), weights="quadratic")
    except Exception:
        kappa = np.nan
    aacc  = (np.abs(yt - yp) <= 1).mean()
    return dict(MAE=mae, Spearman=sp, Kappa=kappa, AAcc=aacc)


# ─────────────────────────────────────────────────────────────────────────────
# LOO-CV with internal estimator
# ─────────────────────────────────────────────────────────────────────────────
def run_loocv(X_sub, y, estimator):
    loo   = LeaveOneOut()
    preds = np.full(len(y), np.nan)
    pipe  = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   estimator),
    ])
    for train_idx, test_idx in loo.split(X_sub):
        try:
            pipe.fit(X_sub[train_idx], y[train_idx])
            preds[test_idx[0]] = pipe.predict(X_sub[test_idx])[0]
        except Exception:
            pass
    return preds


# ─────────────────────────────────────────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────────────────────────────────────────
def run_all(X, y, feat_cols, llm_df):
    selections = load_selections(feat_cols)
    records    = []

    for method, idx in selections.items():
        cat, est_factory = FS_ESTIMATORS[method]
        internal_est     = est_factory()
        if internal_est is None:
            print(f"  [skip] {method}: estimator not available")
            continue

        if method not in llm_df.index:
            print(f"  [skip] {method}: not found in LLM results")
            continue

        X_sub   = X[:, idx]
        n_feats = len(idx)
        llm_row = llm_df.loc[method]

        print(f"\n  [{cat}] {method}  ({n_feats} features)")

        preds_i = run_loocv(X_sub, y, internal_est)
        m_i     = compute_metrics(y, preds_i)
        print(f"    Internal  Kappa={m_i['Kappa']:.3f}  MAE={m_i['MAE']:.3f}")
        print(f"    LLM(med)  Kappa={llm_row['LLM_median_Kappa']:.3f}  "
              f"MAE={llm_row['LLM_median_MAE']:.3f}")

        records.append({
            "Method":             method,
            "Category":           cat,
            "N_Features":         n_feats,
            # Internal estimator
            "Int_MAE":            m_i["MAE"],
            "Int_Spearman":       m_i["Spearman"],
            "Int_Kappa":          m_i["Kappa"],
            "Int_AAcc":           m_i["AAcc"],
            # LLM median
            "LLM_MAE":            llm_row["LLM_median_MAE"],
            "LLM_Spearman":       llm_row["LLM_median_Spearman"],
            "LLM_Kappa":          llm_row["LLM_median_Kappa"],
            "LLM_AAcc":           llm_row["LLM_median_AAcc"],
            "LLM_Kappa_CI_lo":    llm_row["LLM_median_Kappa_CI_lo"],
            "LLM_Kappa_CI_hi":    llm_row["LLM_median_Kappa_CI_hi"],
            "LLM_MAE_CI_lo":      llm_row["LLM_median_MAE_CI_lo"],
            "LLM_MAE_CI_hi":      llm_row["LLM_median_MAE_CI_hi"],
            # Delta
            "Delta_Kappa":        llm_row["LLM_median_Kappa"] - m_i["Kappa"],
            "Delta_MAE":          llm_row["LLM_median_MAE"]   - m_i["MAE"],
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "internal_vs_llm_metrics.csv", index=False)
    print(f"\n  Saved: internal_vs_llm_metrics.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(name):
    plt.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig = plt.gcf()
    fig.suptitle("")
    for ax in fig.get_axes():
        ax.set_title("")
    plt.savefig(OUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}.png / .pdf")


def sort_methods_by_internal_kappa(df):
    order = []
    for cat in ("embedded", "wrapper"):
        sub = df[df["Category"] == cat].sort_values("Int_Kappa", ascending=False)
        order.extend(sub["Method"].tolist())
    return order


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Grouped bar: Internal vs LLM(mode) vs LLM(median) — Kappa + MAE
# ─────────────────────────────────────────────────────────────────────────────
def _draw_bar_panel(ax, df, methods, x, width, metric, ci_lo_col, ci_hi_col):
    """Draw one grouped-bar panel for a single metric (Internal vs LLM median)."""
    n_emb = sum(1 for m in methods
                if df[df["Method"] == m]["Category"].values[0] == "embedded")

    for offset, col, label, color in [
        (-width / 2, f"Int_{metric}", "Internal estimator", None),
        (+width / 2, f"LLM_{metric}", "LLM (median)",       LLM_COLOR),
    ]:
        vals   = []
        colors = []
        err_lo = []
        err_hi = []
        for m in methods:
            row = df[df["Method"] == m].iloc[0]
            vals.append(row[col])
            colors.append(color if color else CAT_COLOR[row["Category"]])
            if label == "LLM (median)":
                err_lo.append(row[col] - row[ci_lo_col])
                err_hi.append(row[ci_hi_col] - row[col])
            else:
                err_lo.append(0)
                err_hi.append(0)

        ax.bar(x + offset, vals, width,
               color=colors if color is None else color,
               edgecolor="white", lw=0.5, alpha=0.9, label=label)
        if any(e > 0 for e in err_hi):
            ax.errorbar(x + offset, vals,
                        yerr=[err_lo, err_hi],
                        fmt="none", color="black", capsize=3, lw=1)

    ax.axvline(n_emb - 0.5, color="gray", lw=1.2, ls=":", alpha=0.8)
    ymax = ax.get_ylim()[1]
    ax.text(n_emb / 2 - 0.5,                          ymax * 0.97, "Embedded",
            ha="center", fontsize=9, color="gray", va="top")
    ax.text(n_emb + (len(methods) - n_emb) / 2 - 0.5, ymax * 0.97, "Wrapper",
            ha="center", fontsize=9, color="gray", va="top")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)


def fig1_bar_kappa_mae(df):
    methods = sort_methods_by_internal_kappa(df)
    x       = np.arange(len(methods))
    width   = 0.35

    fig, (ax_k, ax_m) = plt.subplots(2, 1, figsize=(14, 10))

    # — Kappa (higher is better) —
    _draw_bar_panel(ax_k, df, methods, x, width, "Kappa",
                    "LLM_Kappa_CI_lo", "LLM_Kappa_CI_hi")
    ax_k.set_ylabel("Weighted Kappa ↑")
    ax_k.set_ylim(bottom=min(0, ax_k.get_ylim()[0]) - 0.02)
    ax_k.set_title("Weighted Kappa  (↑ higher is better)", fontweight="bold")

    # — MAE (lower is better) —
    _draw_bar_panel(ax_m, df, methods, x, width, "MAE",
                    "LLM_MAE_CI_lo", "LLM_MAE_CI_hi")
    ax_m.set_ylabel("MAE ↓")
    ax_m.set_title("MAE  (↓ lower is better)", fontweight="bold")

    legend_handles = [
        mpatches.Patch(color=CAT_COLOR["embedded"], label="Internal – Embedded"),
        mpatches.Patch(color=CAT_COLOR["wrapper"],  label="Internal – Wrapper"),
        mpatches.Patch(color=LLM_COLOR,             label="LLM (median)"),
    ]
    fig.legend(handles=legend_handles, fontsize=9,
               loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("LOO-CV: Internal Estimator vs LLM Scoring (median)\n"
                 "(Error bars = 95% bootstrap CI for LLM)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("fig1_bar_kappa_mae")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Scatter: Internal Kappa vs LLM Kappa  (one dot per method)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_scatter_kappa(df):
    _, ax = plt.subplots(figsize=(7, 6))

    for _, row in df.iterrows():
        color   = CAT_COLOR[row["Category"]]
        marker  = "o" if row["Category"] == "embedded" else "s"
        xi      = row["Int_Kappa"]
        yi      = row["LLM_Kappa"]
        yerr_lo = yi - row["LLM_Kappa_CI_lo"]
        yerr_hi = row["LLM_Kappa_CI_hi"] - yi
        ax.errorbar(xi, yi, yerr=[[yerr_lo], [yerr_hi]],
                    fmt=marker, color=color, markersize=8,
                    ecolor=color, elinewidth=1, capsize=3,
                    alpha=0.85, markeredgecolor="white", markeredgewidth=0.5)
        ax.annotate(row["Method"].replace(" (", "\n("),
                    (xi, yi), textcoords="offset points",
                    xytext=(5, 3), fontsize=7)

    all_vals = pd.concat([df["Int_Kappa"], df["LLM_Kappa"]]).dropna()
    lo = all_vals.min() - 0.05
    hi = all_vals.max() + 0.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Internal Estimator — Weighted Kappa")
    ax.set_ylabel("LLM (median) — Weighted Kappa")
    ax.set_title("Internal Estimator Kappa vs LLM Scoring Kappa\n(LOO-CV, n=53)",
                 fontweight="bold")

    patches = [
        mpatches.Patch(color=CAT_COLOR["embedded"], label="Embedded (circle)"),
        mpatches.Patch(color=CAT_COLOR["wrapper"],  label="Wrapper (square)"),
    ]
    ax.legend(handles=patches, fontsize=10)
    plt.tight_layout()
    save_fig("fig2_scatter_kappa")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – All metrics: grouped bar per metric (MAE / Spearman / Kappa / AAcc)
# ─────────────────────────────────────────────────────────────────────────────
def fig3_all_metrics(df):
    METRICS = ["Kappa", "MAE", "Spearman", "AAcc"]
    METRIC_LABEL = {
        "Kappa":    "Weighted Kappa ↑",
        "MAE":      "MAE ↓",
        "Spearman": "Spearman r ↑",
        "AAcc":     "Adjacent Acc. ↑",
    }
    methods = sort_methods_by_internal_kappa(df)
    x       = np.arange(len(methods))
    width   = 0.26
    n_emb   = sum(1 for m in methods
                  if df[df["Method"] == m]["Category"].values[0] == "embedded")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)

    for ax_idx, metric in enumerate(METRICS):
        ax = axes[ax_idx // 2][ax_idx % 2]

        for offset, prefix, label, color in [
            (-width / 2, "Int_",  "Internal",    None),
            (+width / 2, "LLM_",  "LLM (median)", LLM_COLOR),
        ]:
            vals   = []
            colors = []
            for m in methods:
                row = df[df["Method"] == m].iloc[0]
                vals.append(row[f"{prefix}{metric}"])
                colors.append(color if color else CAT_COLOR[row["Category"]])

            ax.bar(x + offset, vals, width,
                   color=colors if color is None else color,
                   edgecolor="white", lw=0.5, alpha=0.9, label=label)

        ax.axvline(n_emb - 0.5, color="gray", lw=1, ls=":", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABEL[metric], fontweight="bold")
        ax.set_ylabel(metric)

    handles = [
        mpatches.Patch(color=CAT_COLOR["embedded"], label="Internal – Embedded"),
        mpatches.Patch(color=CAT_COLOR["wrapper"],  label="Internal – Wrapper"),
        mpatches.Patch(color=LLM_COLOR,             label="LLM (median)"),
    ]
    fig.legend(handles=handles, fontsize=10,
               loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.suptitle("Internal Estimator vs LLM Scoring (median) — All Metrics (LOO-CV, n=53)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("fig3_all_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Kappa delta: ΔKappa = LLM Kappa − Internal Kappa
# ─────────────────────────────────────────────────────────────────────────────
def _draw_delta_panel(ax, df, x_col, delta_col, xlabel, ylabel, title):
    """Draw one delta scatter panel."""
    for _, row in df.iterrows():
        color  = CAT_COLOR[row["Category"]]
        marker = "o" if row["Category"] == "embedded" else "s"
        ax.scatter(row[x_col], row[delta_col],
                   color=color, s=90, marker=marker,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(row["Method"].replace(" (", "\n("),
                    (row[x_col], row[delta_col]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)

    ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(xlim, 0, ylim[1], color=LLM_COLOR,              alpha=0.04)
    ax.fill_between(xlim, ylim[0], 0, color=CAT_COLOR["embedded"],  alpha=0.04)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")


def fig4_delta(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    panels = [
        (axes[0], "Int_Kappa", "Delta_Kappa",
         "Internal Kappa", "ΔKappa  =  LLM − Internal",
         "ΔKappa  [LLM median]\nAbove 0 → LLM better"),
        (axes[1], "Int_MAE",   "Delta_MAE",
         "Internal MAE",   "ΔMAE  =  LLM − Internal",
         "ΔMAE  [LLM median]\nBelow 0 → LLM better  (lower MAE)"),
    ]

    for ax, x_col, delta_col, xlabel, ylabel, title in panels:
        _draw_delta_panel(ax, df, x_col, delta_col, xlabel, ylabel, title)

    patches = [
        mpatches.Patch(color=CAT_COLOR["embedded"], label="Embedded (circle)"),
        mpatches.Patch(color=CAT_COLOR["wrapper"],  label="Wrapper (square)"),
    ]
    fig.legend(handles=patches, fontsize=10,
               loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.suptitle("Internal Estimator vs LLM (median) — Delta Analysis\n"
                 "Kappa: Δ > 0 → LLM better  |  MAE: Δ < 0 → LLM better",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig("fig4_delta")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("Internal Estimator vs LLM Scoring Comparison")
    print("=" * 62)

    print("\n[1/6] Loading data...")
    X, y, feat_cols = load_data()

    print("\n[2/6] Loading LLM results...")
    llm_df = load_llm_results()

    print("\n[3/6] Running LOO-CV with internal estimators...")
    df = run_all(X, y, feat_cols, llm_df)

    print("\n[4/6] Fig 1 – Grouped bar (Kappa)...")
    fig1_bar_kappa_mae(df)

    print("\n[5/6] Fig 2 – Scatter (Internal Kappa vs LLM Kappa)...")
    fig2_scatter_kappa(df)

    print("\n[6/6] Fig 3 – All metrics + Fig 4 – Kappa delta...")
    fig3_all_metrics(df)
    fig4_delta(df)

    print("\n" + "=" * 62)
    print("Done. Results in:", OUT_DIR)
    print("=" * 62)
    print("\nSummary:")
    summary = df[["Method", "Category", "N_Features",
                  "Int_Kappa", "LLM_Kappa", "Delta_Kappa",
                  "Int_MAE",   "LLM_MAE",   "Delta_MAE"]].copy()
    summary = summary.sort_values("Int_Kappa", ascending=False)
    print(summary.to_string(index=False))
