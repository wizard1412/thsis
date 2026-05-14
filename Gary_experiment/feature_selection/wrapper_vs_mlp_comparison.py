"""
Internal Estimator vs MLP Prediction Comparison
=================================================
For each embedded and wrapper feature selection method, run LOO-CV twice:
  1. Using the method's own internal estimator
  2. Using MLP(8) with the exact same selected features

Key question: does a method with poor internal predictions still select
features that let MLP perform well?  A method ranked highly by MLP but
having poor internal predictions may be selecting noise-correlated features.

Internal estimators used
------------------------
  Embedded:
    LASSO (Standard)     → LassoCV
    ElasticNet           → ElasticNetCV
    Adaptive LASSO       → LassoCV  (final reweighted step)
    LARS                 → LassoLarsCV
    RF Importance        → RandomForestRegressor(500 trees)
    Extra Trees          → ExtraTreesRegressor(500 trees)
    Gradient Boosting    → GradientBoostingRegressor
    XGBoost              → XGBRegressor
    Permutation Imp.     → RandomForestRegressor(300 trees)
    Stability Selection  → LassoCV
  Wrapper:
    RFECV (RF)           → RandomForestRegressor(200 trees)
    RFECV (SVR)          → SVR(kernel=linear)
    RFECV (Lasso)        → Lasso(alpha=0.01)
    SFS Forward          → Ridge
    SBS Backward         → Ridge
    Boruta               → RandomForestRegressor(200 trees)

Output directory: wrapper_vs_mlp_results/
  figW1_scatter_grid_embedded.pdf/png
  figW1_scatter_grid_wrapper.pdf/png
  figW2_residual_lines_embedded.pdf/png
  figW2_residual_lines_wrapper.pdf/png
  figW3_metric_summary.pdf/png
  figW4_kappa_delta.pdf/png
  internal_vs_mlp_preds.csv
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
from sklearn.neural_network import MLPRegressor
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


class OrdinalRFClassifier(BaseEstimator, RegressorMixin):
    """RandomForestClassifier that outputs E[class] = Σ c·P(c) as a continuous score.

    Faithfully reproduces the estimator used inside rfecv_rf_select():
    trains on binned (0–4 integer) labels but predicts a continuous expected
    value via predict_proba(), enabling fair metric comparison with MLP.
    """
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
        proba      = self.clf_.predict_proba(X)           # shape (n, |classes|)
        full_proba = np.zeros((len(X), 5))
        for i, c in enumerate(self.clf_.classes_):        # handle missing classes
            full_proba[:, c] = proba[:, i]
        return (full_proba * np.arange(5)).sum(axis=1)    # E[class]

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths / Config
# ─────────────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
FEAT_CSV = HERE / "features_dataset_v2.csv"
FS_JSON  = HERE / "traditional_fs_results_v2" / "traditional_selection_results.json"
OUT_DIR  = HERE / "wrapper_vs_mlp_results"
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
# Method → internal estimator + category
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

# (category, estimator_factory)
FS_ESTIMATORS = {
    # Embedded
    "LASSO (Standard)":       ("embedded", _lasso_est),
    "ElasticNet":             ("embedded", _en_est),
    "Adaptive LASSO":         ("embedded", _lasso_est),   # final Lasso step
    "LARS":                   ("embedded", _lars_est),
    "RF Importance":          ("embedded", _rf500_est),
    "Extra Trees":            ("embedded", _et500_est),
    "Gradient Boosting":      ("embedded", _gb_est),
    "XGBoost":                ("embedded", _xgb_est),
    "Permutation Importance": ("embedded", _rf300_est),
    "Stability Selection":    ("embedded", _lasso_est),
    # Wrapper
    "RFECV (RF)":             ("wrapper",  lambda: OrdinalRFClassifier(
                                  n_estimators=200, max_depth=5,
                                  random_state=RANDOM_SEED)),
    "RFECV (SVR)":            ("wrapper",  lambda: SVR(kernel="linear", C=1.0)),
    "RFECV (Lasso)":          ("wrapper",  lambda: Lasso(alpha=0.01, max_iter=50000,
                                                         random_state=RANDOM_SEED)),
    "SFS Forward":            ("wrapper",  lambda: Ridge(alpha=1.0)),
    "SBS Backward":           ("wrapper",  lambda: Ridge(alpha=1.0)),
    "Boruta":                 ("wrapper",  _rf200_est),
}

MLP_EST = lambda: MLPRegressor(
    hidden_layer_sizes=(8,), activation="relu", solver="adam",
    alpha=1.0, max_iter=600, random_state=RANDOM_SEED,
)

CAT_COLOR = {
    "embedded": "#E67E22",
    "wrapper":  "#27AE60",
}
EST_COLOR = {
    "internal": None,        # filled per-method from CAT_COLOR
    "mlp":      "#8E44AD",
}
EST_MARKER = {"internal": "o", "mlp": "s"}
EST_LABEL  = {"internal": "Internal estimator", "mlp": "MLP(8)"}

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
    sels  = d["selections"]
    fidx  = {f: i for i, f in enumerate(feat_cols)}
    result = {}
    for method in FS_ESTIMATORS:
        feats = sels.get(method, [])
        idx   = [fidx[f] for f in feats if f in fidx]
        if idx:
            result[method] = idx
            print(f"  {method:25s}: {len(idx):3d} features")
        else:
            print(f"  {method:25s}: [skip — no valid features in dataset]")
    return result


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
# LOO-CV
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
def run_all(X, y, feat_cols):
    selections = load_selections(feat_cols)
    records    = []
    pred_rows  = []

    for method, idx in selections.items():
        cat, est_factory = FS_ESTIMATORS[method]
        internal_est     = est_factory()
        if internal_est is None:
            print(f"  [skip] {method}: estimator not available (xgboost?)")
            continue
        X_sub   = X[:, idx]
        n_feats = len(idx)

        print(f"\n  [{cat}] {method}  ({n_feats} features)")

        # 1) Internal estimator
        preds_i = run_loocv(X_sub, y, internal_est)
        m_i     = compute_metrics(y, preds_i)
        print(f"    Internal  Kappa={m_i['Kappa']:.3f}  MAE={m_i['MAE']:.3f}")

        # 2) MLP(8)
        preds_m = run_loocv(X_sub, y, MLP_EST())
        m_m     = compute_metrics(y, preds_m)
        print(f"    MLP(8)    Kappa={m_m['Kappa']:.3f}  MAE={m_m['MAE']:.3f}")

        for est_key, met in [("internal", m_i), ("mlp", m_m)]:
            records.append({
                "Method":     method,
                "Category":   cat,
                "Estimator":  EST_LABEL[est_key],
                "N_Features": n_feats,
                **met,
            })

        yt_int = round_score(y)
        yi_int = round_score(preds_i)
        ym_int = round_score(preds_m)

        for i in range(len(y)):
            pred_rows.append({
                "Sample":                i,
                "y_true":               y[i],
                "Method":               method,
                "Category":             cat,
                "y_hat_internal":       preds_i[i],
                "y_hat_mlp":            preds_m[i],
                # raw: 取整前的差值（連續）
                "residual_raw_internal": preds_i[i] - y[i],
                "residual_raw_mlp":      preds_m[i] - y[i],
                # int: 各自取整後相減（與 metrics 一致）
                "residual_int_internal": int(yi_int[i]) - int(yt_int[i]),
                "residual_int_mlp":      int(ym_int[i]) - int(yt_int[i]),
            })

    df_metrics = pd.DataFrame(records)
    df_preds   = pd.DataFrame(pred_rows)
    df_preds.to_csv(OUT_DIR / "internal_vs_mlp_preds.csv", index=False)
    print(f"\n  Saved: internal_vs_mlp_preds.csv")
    return df_metrics, df_preds


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


def jitter(arr, scale=0.07):
    return arr + RNG.uniform(-scale, scale, size=len(arr))


# ─────────────────────────────────────────────────────────────────────────────
# Fig W1 – LOO Scatter Grid  (split by category)
# ─────────────────────────────────────────────────────────────────────────────
def figW1_scatter_grid(df_metrics, df_preds, category):
    methods = [m for m, (c, _) in FS_ESTIMATORS.items()
               if c == category and m in df_preds["Method"].unique()]
    if not methods:
        return
    n     = len(methods)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows),
                             squeeze=False)
    cat_col = CAT_COLOR[category]

    for ax_idx, method in enumerate(methods):
        ax  = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df_preds[df_preds["Method"] == method]
        if sub.empty:
            ax.set_visible(False)
            continue

        for est_key, col, color in [
            ("internal", "y_hat_internal", cat_col),
            ("mlp",      "y_hat_mlp",      EST_COLOR["mlp"]),
        ]:
            ax.scatter(jitter(sub["y_true"].values),
                       jitter(sub[col].values),
                       color=color, marker=EST_MARKER[est_key],
                       alpha=0.65, s=40, label=EST_LABEL[est_key],
                       edgecolors="white", linewidths=0.4)

        lim = [-0.3, MAX_SCORE + 0.3]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xticks(range(MAX_SCORE + 1))
        ax.set_yticks(range(MAX_SCORE + 1))
        ax.set_xlabel("True score (y_true)")
        ax.set_ylabel("Predicted score (ŷ)")

        row_i = df_metrics[(df_metrics["Method"] == method) &
                            (df_metrics["Estimator"] == EST_LABEL["internal"])]
        row_m = df_metrics[(df_metrics["Method"] == method) &
                            (df_metrics["Estimator"] == EST_LABEL["mlp"])]
        if not row_i.empty and not row_m.empty:
            ki, mi = row_i["Kappa"].values[0], row_i["MAE"].values[0]
            km, mm = row_m["Kappa"].values[0], row_m["MAE"].values[0]
            ann = f"Internal  κ={ki:.2f}  MAE={mi:.2f}\nMLP(8)    κ={km:.2f}  MAE={mm:.2f}"
            ax.text(0.03, 0.97, ann, transform=ax.transAxes,
                    va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        n_feats = int(df_metrics[df_metrics["Method"] == method]["N_Features"].iloc[0])
        ax.set_title(f"{method}  (n={n_feats})", fontweight="bold")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    patches = [mpatches.Patch(color=cat_col,        label=EST_LABEL["internal"]),
               mpatches.Patch(color=EST_COLOR["mlp"], label=EST_LABEL["mlp"])]
    fig.legend(handles=patches, title="Estimator",
               loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=10)
    fig.suptitle(f"LOO-CV Predictions: Internal Estimator vs MLP(8)\n"
                 f"Category: {category.capitalize()}  "
                 f"(jitter applied for readability)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(f"figW1_scatter_grid_{category}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig W2 – Residual Lines  (split by category)
# ─────────────────────────────────────────────────────────────────────────────
def figW2_residual_lines(df_preds, category):
    methods = [m for m, (c, _) in FS_ESTIMATORS.items()
               if c == category and m in df_preds["Method"].unique()]
    if not methods:
        return
    n     = len(methods)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    cat_col = CAT_COLOR[category]

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)

    for ax_idx, method in enumerate(methods):
        ax  = axes[ax_idx // ncols][ax_idx % ncols]
        sub = df_preds[df_preds["Method"] == method].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        sub = sub.sort_values("y_true").reset_index(drop=True)
        x   = np.arange(len(sub))

        for est_key, col, color in [
            ("internal", "residual_int_internal", cat_col),
            ("mlp",      "residual_int_mlp",      EST_COLOR["mlp"]),
        ]:
            ax.plot(x, sub[col].values, color=color, lw=1.5,
                    label=EST_LABEL[est_key], alpha=0.85)
            ax.scatter(x, sub[col].values, color=color,
                       marker=EST_MARKER[est_key], s=25, alpha=0.7, zorder=3)

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.fill_between(x, -1, 1, color="gray", alpha=0.08, label="±1 tolerance")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.1f}" for v in sub["y_true"].values],
                           rotation=90, fontsize=7)
        ax.set_xlabel("Samples (sorted by true score →)")
        ax.set_ylabel("Integer residual  (round(ŷ) − round(y_true))")
        ax.set_ylim(-MAX_SCORE - 0.5, MAX_SCORE + 0.5)
        ax.set_yticks(range(-MAX_SCORE, MAX_SCORE + 1))
        ax.set_title(method, fontweight="bold")

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    handles = [mpatches.Patch(color=cat_col,          label=EST_LABEL["internal"]),
               mpatches.Patch(color=EST_COLOR["mlp"],  label=EST_LABEL["mlp"]),
               mpatches.Patch(color="gray", alpha=0.2, label="±1 tolerance")]
    fig.legend(handles=handles, title="Estimator",
               loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=10)
    fig.suptitle(f"LOO-CV Integer Residuals  (round(ŷ) − round(y_true))\n"
                 f"Category: {category.capitalize()}  — sorted by true score",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(f"figW2_residual_{category}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig W3 – Metric Summary Bar Chart  (all methods, grouped by category)
# ─────────────────────────────────────────────────────────────────────────────
def figW3_metric_summary(df_metrics):
    METRICS = ["Kappa", "MAE", "Spearman", "AAcc"]
    METRIC_LABEL = {
        "Kappa":    "Weighted Kappa ↑",
        "MAE":      "MAE ↓",
        "Spearman": "Spearman r ↑",
        "AAcc":     "Adjacent Acc. ↑",
    }
    LOWER_BETTER = {"MAE"}

    # Order: embedded first, then wrapper; within each, by internal Kappa desc
    def sort_key(m):
        cat = FS_ESTIMATORS[m][0]
        row = df_metrics[(df_metrics["Method"] == m) &
                          (df_metrics["Estimator"] == EST_LABEL["internal"])]
        k   = row["Kappa"].values[0] if not row.empty else 0
        return (0 if cat == "embedded" else 1, -k)

    methods = [m for m in FS_ESTIMATORS if m in df_metrics["Method"].unique()]
    methods.sort(key=sort_key)

    ncols = 2
    nrows = (len(METRICS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5 * nrows),
                             squeeze=False)

    for ax_idx, metric in enumerate(METRICS):
        ax    = axes[ax_idx // ncols][ax_idx % ncols]
        x     = np.arange(len(methods))
        width = 0.35

        vals_i, vals_m, bar_colors = [], [], []
        for m in methods:
            cat = FS_ESTIMATORS[m][0]
            row_i = df_metrics[(df_metrics["Method"] == m) &
                                (df_metrics["Estimator"] == EST_LABEL["internal"])]
            row_m = df_metrics[(df_metrics["Method"] == m) &
                                (df_metrics["Estimator"] == EST_LABEL["mlp"])]
            vals_i.append(row_i[metric].values[0] if not row_i.empty else np.nan)
            vals_m.append(row_m[metric].values[0] if not row_m.empty else np.nan)
            bar_colors.append(CAT_COLOR[cat])

        bars_i = ax.bar(x - width / 2, vals_i, width,
                        color=bar_colors, edgecolor="white", lw=0.5,
                        label=EST_LABEL["internal"], alpha=0.9)
        bars_m = ax.bar(x + width / 2, vals_m, width,
                        color=EST_COLOR["mlp"], edgecolor="white", lw=0.5,
                        label=EST_LABEL["mlp"], alpha=0.9)

        for bar in list(bars_i) + list(bars_m):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

        # separator between embedded and wrapper
        n_emb = sum(1 for m in methods if FS_ESTIMATORS[m][0] == "embedded")
        ax.axvline(n_emb - 0.5, color="gray", lw=1, ls=":", alpha=0.7)
        ax.text(n_emb / 2 - 0.5, ax.get_ylim()[1] * 0.02,
                "Embedded", ha="center", fontsize=8, color="gray")
        ax.text(n_emb + (len(methods) - n_emb) / 2 - 0.5, ax.get_ylim()[1] * 0.02,
                "Wrapper", ha="center", fontsize=8, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABEL[metric], fontweight="bold")
        ax.set_ylabel(metric)
        if metric not in LOWER_BETTER:
            finite = [v for v in vals_i + vals_m if not np.isnan(v)]
            ax.set_ylim(bottom=(min(0, min(finite)) - 0.05) if finite else -0.05)

    # legend with category colours + MLP
    handles = [
        mpatches.Patch(color=CAT_COLOR["embedded"], label="Embedded estimator"),
        mpatches.Patch(color=CAT_COLOR["wrapper"],  label="Wrapper estimator"),
        mpatches.Patch(color=EST_COLOR["mlp"],       label="MLP(8)"),
    ]
    for k in range(len(METRICS), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.legend(handles=handles, title="Estimator",
               loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=10)
    fig.suptitle("Internal Estimator vs MLP(8) — Performance Summary\n"
                 "(same selected features per method, LOO-CV, n=53)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("figW3_metric_summary")


# ─────────────────────────────────────────────────────────────────────────────
# Fig W4 – Kappa delta  (MLP Kappa − Internal Kappa vs Internal Kappa)
# ─────────────────────────────────────────────────────────────────────────────
def figW4_kappa_delta(df_metrics):
    OUTLIER = "RFECV (SVR)"

    # collect data
    points = {}
    for method in FS_ESTIMATORS:
        cat   = FS_ESTIMATORS[method][0]
        row_i = df_metrics[(df_metrics["Method"] == method) &
                            (df_metrics["Estimator"] == EST_LABEL["internal"])]
        row_m = df_metrics[(df_metrics["Method"] == method) &
                            (df_metrics["Estimator"] == EST_LABEL["mlp"])]
        if row_i.empty or row_m.empty:
            continue
        points[method] = dict(
            ki=row_i["Kappa"].values[0],
            dk=row_m["Kappa"].values[0] - row_i["Kappa"].values[0],
            nf=int(row_i["N_Features"].values[0]),
            cat=cat,
        )

    main   = {m: v for m, v in points.items() if m != OUTLIER}
    outlier = points[OUTLIER]

    # x limits
    main_xs  = [v["ki"] for v in main.values()]
    x_lo = min(main_xs) - 0.04
    x_hi = max(main_xs) + 0.04
    all_ys   = [v["dk"] for v in points.values()]
    y_lo = min(all_ys) - 0.05
    y_hi = max(all_ys) + 0.07

    # two-panel broken x-axis layout
    fig = plt.figure(figsize=(10, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 5], wspace=0.08)
    ax_l = fig.add_subplot(gs[0])   # outlier panel
    ax_r = fig.add_subplot(gs[1], sharey=ax_l)   # main panel

    def _plot(ax, pts):
        for method, v in pts.items():
            ax.scatter(v["ki"], v["dk"],
                       color=CAT_COLOR[v["cat"]],
                       s=100,
                       edgecolors="black", linewidths=0.6, zorder=3,
                       marker="o" if v["cat"] == "embedded" else "s")
            ax.annotate(method.replace(" (", "\n("),
                        (v["ki"], v["dk"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5)

    _plot(ax_l, {OUTLIER: outlier})
    _plot(ax_r, main)

    for ax in (ax_l, ax_r):
        ax.axhline(0, color="gray", lw=1, ls="--", alpha=0.6)
        ax.set_ylim(y_lo, y_hi)

    ax_l.set_xlim(outlier["ki"] - 0.06, outlier["ki"] + 0.06)
    ax_r.set_xlim(x_lo, x_hi)
    ax_l.set_ylabel("Δκ  =  MLP Kappa − Internal Kappa")
    ax_r.set_ylabel("")
    plt.setp(ax_r.get_yticklabels(), visible=False)

    # shared x label via fig.text
    fig.text(0.5, -0.02, "Internal Estimator — Weighted Kappa (LOO-CV)",
             ha="center", fontsize=11)

    # broken-axis diagonal tick marks
    d = 0.015
    kw = dict(transform=ax_l.transAxes, color="k", clip_on=False, lw=1)
    ax_l.plot((1 - d, 1 + d), (-d, +d), **kw)
    ax_l.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
    kw.update(transform=ax_r.transAxes)
    ax_r.plot((-d, +d), (-d, +d), **kw)
    ax_r.plot((-d, +d), (1 - d, 1 + d), **kw)

    # hide inner spines
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(left=False)

    patches = [
        mpatches.Patch(color=CAT_COLOR["embedded"], label="Embedded (circle)"),
        mpatches.Patch(color=CAT_COLOR["wrapper"],  label="Wrapper (square)"),
    ]
    ax_r.legend(handles=patches, fontsize=10, loc="lower right")

    fig.suptitle("Does a weak internal estimator still find good features for MLP?\n"
                 "Δκ > 0 → MLP improves on internal;  Δκ < 0 → internal is stronger",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig("figW4_kappa_delta")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("Internal Estimator vs MLP(8) — Embedded + Wrapper LOO-CV")
    print("=" * 62)

    print("\n[1/6] Loading data...")
    X, y, feat_cols = load_data()

    print("\n[2/6] Loading feature selections & running LOO-CV...")
    df_metrics, df_preds = run_all(X, y, feat_cols)

    print("\n[3/6] Scatter grid — Embedded...")
    figW1_scatter_grid(df_metrics, df_preds, "embedded")

    print("\n[4/6] Scatter grid — Wrapper...")
    figW1_scatter_grid(df_metrics, df_preds, "wrapper")

    print("\n[5/6] Residual lines (integer) — Embedded + Wrapper...")
    figW2_residual_lines(df_preds, "embedded")
    figW2_residual_lines(df_preds, "wrapper")

    print("\n[6/6] Summary bars + Kappa delta...")
    figW3_metric_summary(df_metrics)
    figW4_kappa_delta(df_metrics)

    print("\n" + "=" * 62)
    print("Done. Results in:", OUT_DIR)
    print("=" * 62)
    print("\nMetrics summary:")
    print(df_metrics.to_string(index=False))
