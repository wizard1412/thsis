"""
Fixed Feature Selection – All Classifiers Comparison
=====================================================
固定 4 種代表性特徵選擇方法（每類最佳 + LLM-Lasso），
對所有回歸器與分類器做 LOOCV，比較預測器效果。

4 種特徵選擇方法:
  Filter   → Spearman     (8 個特徵)
  Embedded → RF Importance (8 個特徵)
  Wrapper  → RFECV(RF)    (42 個特徵)
  LLM      → LLM-Lasso    (6 個特徵)

輸出:
  fixed_fs_results/png/  &  fixed_fs_results/pdf/
  fixed_fs_results/loocv_results.csv
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
from scipy.stats import spearmanr

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingRegressor, GradientBoostingClassifier,
                               ExtraTreesRegressor, ExtraTreesClassifier,
                               AdaBoostRegressor, BaggingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
FEAT_CSV = "../feature_selection/features_dataset.csv"
FS_JSON  = "../feature_selection/experiments_llm_lasso_deepseek70b/results/traditional_selection_results.json"
OUT_DIR  = "fixed_fs_results"
PNG_DIR  = os.path.join(OUT_DIR, "png")
PDF_DIR  = os.path.join(OUT_DIR, "pdf")
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
# 4 種固定特徵選擇方法
# ─────────────────────────────────────────────────────────────────────────────
SELECTED_FS = {
    "LLM-Lasso (LLM)": "LLM-Lasso",
}

FS_COLOR = {
    "LLM-Lasso (LLM)": "#9B59B6",
}

# ─────────────────────────────────────────────────────────────────────────────
# 所有預測器（回歸 + 分類）
# ─────────────────────────────────────────────────────────────────────────────
def make_pipe(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])

REGRESSORS = [
    ("LinearReg",      make_pipe(LinearRegression())),
    ("Ridge",          make_pipe(Ridge(alpha=1.0))),
    ("Lasso",          make_pipe(Lasso(alpha=0.1))),
    ("ElasticNet",     make_pipe(ElasticNet(alpha=0.1, l1_ratio=0.5))),
    ("SVR(linear)",    make_pipe(SVR(kernel="linear", C=1.0))),
    ("SVR(rbf)",       make_pipe(SVR(kernel="rbf",    C=1.0))),
    ("KNN-reg(k=5)",   make_pipe(KNeighborsRegressor(n_neighbors=5))),
    ("RandomForest-R", make_pipe(RandomForestRegressor(n_estimators=100, random_state=42))),
    ("GradBoost-R",    make_pipe(GradientBoostingRegressor(n_estimators=100, random_state=42))),
    ("XGBoost-R",      make_pipe(XGBRegressor(n_estimators=100, random_state=42,
                                              verbosity=0, eval_metric="rmse"))),
    ("AdaBoost-R",     make_pipe(AdaBoostRegressor(n_estimators=50, random_state=42))),
    ("ExtraTrees-R",   make_pipe(ExtraTreesRegressor(n_estimators=100, random_state=42))),
    ("MLP-R",          make_pipe(MLPRegressor(hidden_layer_sizes=(64,32),
                                              max_iter=500, random_state=42))),
]

CLASSIFIERS = [
    ("LogisticReg",    make_pipe(LogisticRegression(multi_class="multinomial",
                                                    max_iter=500, random_state=42))),
    ("LDA",            make_pipe(LinearDiscriminantAnalysis())),
    ("QDA",            make_pipe(QuadraticDiscriminantAnalysis(reg_param=0.5))),
    ("NaiveBayes",     make_pipe(GaussianNB())),
    ("DecisionTree",   make_pipe(DecisionTreeClassifier(max_depth=5, random_state=42))),
    ("RandomForest-C", make_pipe(RandomForestClassifier(n_estimators=100, random_state=42))),
    ("SVM(rbf)",       make_pipe(SVC(kernel="rbf", C=1.0))),
    ("KNN-clf(k=5)",   make_pipe(KNeighborsClassifier(n_neighbors=5))),
    ("GradBoost-C",    make_pipe(GradientBoostingClassifier(n_estimators=100, random_state=42))),
    ("XGBoost-C",      make_pipe(XGBClassifier(n_estimators=100, random_state=42,
                                               verbosity=0, eval_metric="mlogloss",
                                               use_label_encoder=False))),
    ("MLP-C",          make_pipe(MLPClassifier(hidden_layer_sizes=(64,32),
                                               max_iter=500, random_state=42))),
    ("ExtraTrees-C",   make_pipe(ExtraTreesClassifier(n_estimators=100, random_state=42))),
    ("Bagging-C",      make_pipe(BaggingClassifier(n_estimators=20, random_state=42))),
]

PREDICTOR_TYPE = {name: "regression" for name, _ in REGRESSORS}
PREDICTOR_TYPE.update({name: "classifier" for name, _ in CLASSIFIERS})

PRED_COLOR = {
    "regression": "#2980B9",
    "classifier": "#27AE60",
}

# ─────────────────────────────────────────────────────────────────────────────
# Circular combinations: FS method uses the same algorithm as predictor
# These are flagged (marked with *) but not excluded
# ─────────────────────────────────────────────────────────────────────────────
CIRCULAR_COMBOS = {
    ("RF Importance (Embedded)", "RandomForest-R"),
    ("RF Importance (Embedded)", "RandomForest-C"),
    ("RFECV-RF (Wrapper)",       "RandomForest-R"),
    ("RFECV-RF (Wrapper)",       "RandomForest-C"),
    ("LLM-Lasso (LLM)",          "Lasso"),
}

def is_circular(fs_label, pred_name):
    return (fs_label, pred_name) in CIRCULAR_COMBOS

# Predictors without built-in feature selection (fairest comparison group)
CLEAN_REGRESSORS  = {"LinearReg", "Ridge", "SVR(linear)", "SVR(rbf)", "KNN-reg(k=5)", "MLP-R"}
CLEAN_CLASSIFIERS = {"LogisticReg", "LDA", "QDA", "NaiveBayes", "SVM(rbf)", "KNN-clf(k=5)", "MLP-C"}

# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
METRICS      = ["MAE", "Spearman", "Kappa", "AAcc"]
LOWER_BETTER = {"MAE"}
METRIC_LABEL = {
    "MAE":      "MAE ↓",
    "Spearman": "Spearman r ↑",
    "Kappa":    "Weighted Kappa ↑",
    "AAcc":     "Adjacent Acc. ↑",
}

def round_score(arr):
    return np.clip(np.round(arr).astype(int), 0, MAX_SCORE)

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
    records = [compute_metrics(y_true[RNG.integers(0,n,n)], y_pred[RNG.integers(0,n,n)])
               for _ in range(N_BOOT)]
    bdf = pd.DataFrame(records)
    return {f"{c}_CI_lo": bdf[c].quantile(0.025) for c in bdf} | \
           {f"{c}_CI_hi": bdf[c].quantile(0.975) for c in bdf}

def normalise(series, lower_is_better=False):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    norm = (series - lo) / (hi - lo)
    return (1 - norm) if lower_is_better else norm

def add_composite(df):
    df = df.copy()
    for m in METRICS:
        df[f"norm_{m}"] = normalise(df[m], m in LOWER_BETTER)
    df["Composite"] = df[[f"norm_{m}" for m in METRICS]].mean(axis=1)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Save fig
# ─────────────────────────────────────────────────────────────────────────────
def save_fig(name, **kw):
    opts = dict(dpi=300, bbox_inches="tight")
    opts.update(kw)
    plt.savefig(os.path.join(PNG_DIR, f"{name}.png"), **opts)
    plt.savefig(os.path.join(PDF_DIR, f"{name}.pdf"), **opts)
    plt.close()
    print(f"  Saved: {name}")

# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    drop_cols = ["hand", "filename", "Rating1", "Rating2"]
    feat_cols = [c for c in df.columns if c not in drop_cols + ["Rating"]]
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    return df[feat_cols].values.astype(float), df["Rating"].values.astype(float), feat_cols

def load_feature_selections():
    with open(FS_JSON) as f:
        d = json.load(f)
    return d["selections"]

# ─────────────────────────────────────────────────────────────────────────────
# LOOCV
# ─────────────────────────────────────────────────────────────────────────────
def run_loocv(X_full, y, feat_cols, all_selections):
    loo     = LeaveOneOut()
    feat_idx = {f: i for i, f in enumerate(feat_cols)}
    y_int   = round_score(y)
    rows    = []
    # key: (fs_label, pred_name) → np.array of per-sample predictions
    all_preds = {}

    all_predictors = REGRESSORS + CLASSIFIERS

    for fs_label, fs_key in SELECTED_FS.items():
        feats = all_selections[fs_key]
        idx   = [feat_idx[f] for f in feats if f in feat_idx]
        X_sub = X_full[:, idx]
        n_feats = len(idx)

        for pred_name, pipe in all_predictors:
            ptype = PREDICTOR_TYPE[pred_name]
            y_target = y if ptype == "regression" else y_int
            print(f"  LOOCV: {fs_label:30s} + {pred_name:16s} ({n_feats}f) ...")

            preds = np.full(len(y), np.nan)
            for train_idx, test_idx in loo.split(X_sub):
                try:
                    pipe.fit(X_sub[train_idx], y_target[train_idx])
                    preds[test_idx[0]] = pipe.predict(X_sub[test_idx])[0]
                except Exception:
                    pass

            all_preds[(fs_label, pred_name)] = preds

            metrics = compute_metrics(y, preds)
            ci      = bootstrap_ci(y, preds)
            rows.append({
                "FS_Label":   fs_label,
                "FS_Key":     fs_key,
                "Predictor":  pred_name,
                "Type":       ptype,
                "N_Features": n_feats,
                **metrics, **ci,
            })

    return pd.DataFrame(rows), all_preds

# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: build annotated heatmap for one predictor group
# ─────────────────────────────────────────────────────────────────────────────
def _heatmap_for_group(df_group, title, fname):
    """Composite-score heatmap. Circular cells annotated with *."""
    fs_labels = list(SELECTED_FS.keys())
    pivot = df_group.pivot(index="Predictor", columns="FS_Label", values="Composite")
    pivot = pivot[fs_labels]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    # Build annotation matrix (value + * for circular)
    annot = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
    for pred in pivot.index:
        for fs in pivot.columns:
            val = pivot.loc[pred, fs]
            mark = "*" if is_circular(fs, pred) else ""
            annot.loc[pred, fs] = (f"{val:.3f}{mark}" if pd.notna(val) else "–")

    fig, ax = plt.subplots(figsize=(8, max(5, len(pivot) * 0.42)))
    sns.heatmap(pivot, annot=annot, fmt="", cmap="RdYlGn",
                linewidths=0.5, linecolor="lightgray",
                vmin=0, vmax=1,
                cbar_kws={"label": "Composite Score", "shrink": 0.7}, ax=ax)
    ax.set_title(title + "\n(sorted by mean composite; * = circular combination)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Predictor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    save_fig(fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 – Composite heatmap: Regressors only
# ─────────────────────────────────────────────────────────────────────────────
def fig1_heatmap_reg(df):
    sub = df[df["Type"] == "regression"]
    _heatmap_for_group(sub,
                       "Regressors × 4 Feature Selection Methods – Composite Score (LOOCV, n=53)",
                       "fig1_heatmap_reg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 – Composite heatmap: Classifiers only
# ─────────────────────────────────────────────────────────────────────────────
def fig2_heatmap_clf(df):
    sub = df[df["Type"] == "classifier"]
    _heatmap_for_group(sub,
                       "Classifiers × 4 Feature Selection Methods – Composite Score (LOOCV, n=53)",
                       "fig2_heatmap_clf")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: ranked bar per FS method for one predictor group
# ─────────────────────────────────────────────────────────────────────────────
def _ranked_bars(df_group, color, title, fname):
    """Ranked horizontal bar chart per FS method.
    Circular predictors get a red dashed edge and * suffix in label.
    """
    fs_labels = list(SELECTED_FS.keys())
    fig, axes = plt.subplots(1, len(fs_labels),
                             figsize=(6 * len(fs_labels), max(6, len(df_group["Predictor"].unique()) * 0.38)))
    if len(fs_labels) == 1:
        axes = [axes]

    for ax, fs_label in zip(axes, fs_labels):
        sub = df_group[df_group["FS_Label"] == fs_label].sort_values("Composite", ascending=True)
        labels = [
            (p + " *") if is_circular(fs_label, p) else p
            for p in sub["Predictor"]
        ]
        edge_colors = [
            "#E74C3C" if is_circular(fs_label, p) else "white"
            for p in sub["Predictor"]
        ]
        edge_widths = [
            1.5 if is_circular(fs_label, p) else 0.4
            for p in sub["Predictor"]
        ]
        bars = ax.barh(labels, sub["Composite"],
                       color=color, edgecolor=edge_colors,
                       linewidth=edge_widths)
        for bar, val in zip(bars, sub["Composite"]):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=8)
        fs_short = fs_label.split("(")[0].strip()
        n_feat = sub["N_Features"].iloc[0]
        ax.set_title(f"{fs_short}\n({n_feat} features)", fontweight="bold", fontsize=10)
        ax.set_xlabel("Composite Score (1=best)")
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()
        ax.tick_params(labelsize=9)

    circ_p = mpatches.Patch(facecolor=color, edgecolor="#E74C3C",
                            linewidth=1.5, label="* circular combination")
    fig.legend(handles=[circ_p], fontsize=9,
               loc="lower right", bbox_to_anchor=(1.0, 0.0))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 – Ranked bars: Regressors
# ─────────────────────────────────────────────────────────────────────────────
def fig3_ranked_reg(df):
    sub = df[df["Type"] == "regression"]
    _ranked_bars(sub, PRED_COLOR["regression"],
                 "Regressors Ranked by Composite Score per Feature Selection Method\n"
                 "(LOOCV, n=53;  * = circular combination)",
                 "fig3_ranked_reg")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 – Ranked bars: Classifiers
# ─────────────────────────────────────────────────────────────────────────────
def fig4_ranked_clf(df):
    sub = df[df["Type"] == "classifier"]
    _ranked_bars(sub, PRED_COLOR["classifier"],
                 "Classifiers Ranked by Composite Score per Feature Selection Method\n"
                 "(LOOCV, n=53;  * = circular combination)",
                 "fig4_ranked_clf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 – Clean predictors only (no built-in FS): fair side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig5_clean_comparison(df):
    """Only models without internal feature weighting/selection.
    Ranked by Borda Score (sum of per-metric ranks; lower = better).
    Two panels: regressors (left) and classifiers (right).
    """
    BORDA_METRICS = ["MAE", "Kappa", "AAcc"]
    sub = df.copy().dropna(subset=BORDA_METRICS)

    # Compute Borda ranks within each type separately
    for m in BORDA_METRICS:
        ascending = m in LOWER_BETTER
        sub[f"rank_{m}"] = sub.groupby(["FS_Label", "Type"])[m].rank(
            ascending=ascending, method="average"
        )
    sub["Borda"] = sub[[f"rank_{m}" for m in BORDA_METRICS]].sum(axis=1)

    groups = [
        ("Regressors (clean)",  "regression", CLEAN_REGRESSORS,  PRED_COLOR["regression"]),
        ("Classifiers (clean)", "classifier", CLEAN_CLASSIFIERS, PRED_COLOR["classifier"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, len(CLEAN_REGRESSORS) * 0.42 + 1)))

    for ax, (group_title, ptype, clean_set, color) in zip(axes, groups):
        grp = (sub[(sub["Type"] == ptype) & sub["Predictor"].isin(clean_set)]
               .sort_values("Borda", ascending=False))

        bars = ax.barh(grp["Predictor"], grp["Borda"],
                       color=color, edgecolor="white", linewidth=0.4)
        for bar, (_, r) in zip(bars, grp.iterrows()):
            detail = "  " + "  ".join(
                [f"{m[0]}:{r[f'rank_{m}']:.0f}" for m in BORDA_METRICS]
            )
            ax.text(bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{r['Borda']:.0f}{detail}",
                    va="center", fontsize=8, color="#333333")

        ax.set_title(group_title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Borda Score (lower = better)")
        ax.invert_yaxis()
        ax.tick_params(labelsize=9)

    fig.suptitle("Clean Predictor Comparison – Borda Count (no built-in feature selection)\n"
                 "LLM-Lasso features (6 features, LOOCV n=53)\n"
                 "Per-metric rank breakdown: M=MAE  K=Kappa  A=AAcc",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    save_fig("fig5_clean_comparison")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 – Borda Count ranking
# ─────────────────────────────────────────────────────────────────────────────
def fig6_borda(df):
    """Rank each predictor per metric, sum ranks (lower = better).
    Separate panels for regressors and classifiers.
    """
    BORDA_METRICS = ["MAE", "Kappa", "AAcc"]
    sub = df.copy().dropna(subset=BORDA_METRICS)

    # Assign per-metric ranks (1 = best)
    for m in BORDA_METRICS:
        ascending = m in LOWER_BETTER
        sub[f"rank_{m}"] = sub.groupby("FS_Label")[m].rank(
            ascending=ascending, method="average"
        )
    sub["Borda"] = sub[[f"rank_{m}" for m in BORDA_METRICS]].sum(axis=1)

    groups = [("Regressors", "regression"), ("Classifiers", "classifier")]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, sub["Predictor"].nunique() * 0.38)))

    for ax, (title, ptype) in zip(axes, groups):
        grp = sub[sub["Type"] == ptype].sort_values("Borda", ascending=False)
        colors = [
            "#E74C3C" if is_circular(grp["FS_Label"].iloc[0], p) else PRED_COLOR[ptype]
            for p in grp["Predictor"]
        ]
        bars = ax.barh(grp["Predictor"], grp["Borda"],
                       color=colors, edgecolor="white", linewidth=0.4)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax.text(bar.get_width() + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['Borda']:.1f}", va="center", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Borda Score (lower = better rank)")
        ax.invert_yaxis()
        ax.tick_params(labelsize=9)

        # add per-metric rank breakdown as text
        for i, (_, row) in enumerate(grp.iterrows()):
            detail = "  " + "  ".join(
                [f"{m[0]}:{row[f'rank_{m}']:.0f}" for m in BORDA_METRICS]
            )
            ax.text(ax.get_xlim()[1] * 0.02,
                    i, detail, va="center", fontsize=6.5,
                    color="#555555")

    circ_p = mpatches.Patch(color="#E74C3C", label="* circular combination")
    fig.legend(handles=[circ_p], fontsize=9, loc="lower right")
    fig.suptitle(
        "Borda Count Ranking – LLM-Lasso Features (6 features, LOOCV n=53)\n"
        "Per-metric ranks summed: MAE(M) Kappa(K) AAcc(A)  |  lower = better",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    save_fig("fig6_borda")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 – Pareto Dominance
# ─────────────────────────────────────────────────────────────────────────────
def _pareto_dominant(df_sub):
    """Return boolean mask: True if that row is Pareto-optimal.
    A method is dominated if another method is >= on ALL metrics and > on at least one.
    (MAE is inverted so all metrics are higher-is-better.)
    """
    vals = df_sub[METRICS].copy()
    vals["MAE"] = -vals["MAE"]   # invert so all higher = better
    arr = vals.values
    n = len(arr)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i ?
            if np.all(arr[j] >= arr[i]) and np.any(arr[j] > arr[i]):
                dominated[i] = True
                break
    return ~dominated


def fig7_pareto(df):
    """2×2 scatter-plot matrix of key metric pairs.
    Pareto-optimal methods highlighted; others greyed out.
    Separate for regressors and classifiers.
    """
    metric_pairs = [
        ("MAE", "Kappa"),
        ("Spearman", "AAcc"),
        ("MAE", "AAcc"),
        ("Spearman", "Kappa"),
    ]
    pair_labels = {
        "MAE":      "MAE (lower better)",
        "Kappa":    "Weighted Kappa",
        "Spearman": "Spearman r",
        "AAcc":     "Adjacent Acc.",
    }

    groups = [("Regressors", "regression"), ("Classifiers", "classifier")]

    for group_title, ptype in groups:
        sub = df[df["Type"] == ptype].dropna(subset=METRICS).copy()
        pareto_mask = _pareto_dominant(sub)
        sub["Pareto"] = pareto_mask

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes_flat = axes.flatten()

        for ax, (mx, my) in zip(axes_flat, metric_pairs):
            # non-Pareto
            non_p = sub[~sub["Pareto"]]
            ax.scatter(non_p[mx], non_p[my],
                       c="#BDC3C7", s=55, zorder=2, label="Non-Pareto")
            # Pareto
            par = sub[sub["Pareto"]]
            colors_p = [
                "#E74C3C" if is_circular(row["FS_Label"], row["Predictor"])
                else PRED_COLOR[ptype]
                for _, row in par.iterrows()
            ]
            ax.scatter(par[mx], par[my],
                       c=colors_p, s=110, zorder=3,
                       edgecolors="black", linewidths=0.6,
                       label="Pareto-optimal")
            # labels for Pareto methods
            for _, row in par.iterrows():
                ax.annotate(
                    row["Predictor"],
                    (row[mx], row[my]),
                    fontsize=7, xytext=(4, 3),
                    textcoords="offset points"
                )
            ax.set_xlabel(pair_labels[mx], fontsize=9)
            ax.set_ylabel(pair_labels[my], fontsize=9)
            # invert MAE axis (lower is better)
            if mx == "MAE":
                ax.invert_xaxis()
            if my == "MAE":
                ax.invert_yaxis()
            ax.tick_params(labelsize=8)

        # shared legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=PRED_COLOR[ptype],
                   markeredgecolor='black', markersize=9, label="Pareto-optimal"),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#BDC3C7', markersize=9, label="Dominated"),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#E74C3C', markeredgecolor='black',
                   markersize=9, label="Pareto-optimal (circular *)"),
        ]
        fig.legend(handles=handles, fontsize=9, loc="lower right",
                   bbox_to_anchor=(0.99, 0.01))

        pareto_names = sub.loc[sub["Pareto"], "Predictor"].tolist()
        fig.suptitle(
            f"Pareto Dominance Analysis – {group_title} (LLM-Lasso, 6 features, LOOCV n=53)\n"
            f"Pareto-optimal: {', '.join(pareto_names)}",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        fname = f"fig7_pareto_{'reg' if ptype == 'regression' else 'clf'}"
        save_fig(fname)
        print(f"    Pareto-optimal {group_title}: {pareto_names}")


# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────
def print_table(df):
    print("\n" + "="*100)
    print(f"{'FS Method':<32} {'Predictor':<16} {'Type':<12} {'N':>4} "
          f"{'MAE':>6} {'Spear':>6} {'Kappa':>6} {'AAcc':>6} {'Comp':>6}")
    print("="*100)
    for fs_label in SELECTED_FS:
        sub = df[df["FS_Label"] == fs_label].sort_values("Composite", ascending=False)
        print(f"\n--- {fs_label} ---")
        for _, row in sub.iterrows():
            print(f"  {row['Predictor']:<30} {row['Type']:<12} {row['N_Features']:>4}"
                  f" {row['MAE']:>6.3f} {row['Spearman']:>6.3f}"
                  f" {row['Kappa']:>6.3f} {row['AAcc']:>6.3f}"
                  f" {row['Composite']:>6.3f}")
    print("="*100)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Fixed Feature Selection – All Classifiers Comparison")
    print("=" * 60)

    print("\n[1] Loading data ...")
    X_full, y, feat_cols = load_data()
    all_selections = load_feature_selections()
    print(f"    {X_full.shape[0]} samples, {X_full.shape[1]} features")
    print(f"    Fixed FS methods: {list(SELECTED_FS.keys())}")
    print(f"    Predictors: {len(REGRESSORS)} regressors + {len(CLASSIFIERS)} classifiers")

    print("\n[2] Running LOOCV ...")
    results, all_preds = run_loocv(X_full, y, feat_cols, all_selections)
    results = add_composite(results)
    results.to_csv(os.path.join(OUT_DIR, "loocv_results.csv"), index=False)
    print(f"  Saved: {OUT_DIR}/loocv_results.csv")

    # 儲存 per-sample predictions（loocv_results 的逐樣本版本）
    feat_df = pd.read_csv(FEAT_CSV)
    feat_df = feat_df.dropna(subset=["Rating"]).reset_index(drop=True)
    nan_col = pd.Series([np.nan] * len(feat_df))
    pred_df = pd.DataFrame({
        "filename":              feat_df["filename"],
        "label_Dr. Tan":         feat_df["Rating1"] if "Rating1" in feat_df.columns else nan_col,
        "label_Dr. Chien":       feat_df["Rating2"] if "Rating2" in feat_df.columns else nan_col,
        "average_score_Doctors": feat_df["Rating"],
    })
    for (fs_label, pred_name), preds in all_preds.items():
        col = f"{fs_label}__{pred_name}"
        pred_df[col] = np.clip(np.round(preds), 0, MAX_SCORE)
    pred_df.to_csv(os.path.join(OUT_DIR, "per_sample_predictions.csv"), index=False)
    print(f"  Saved: {OUT_DIR}/per_sample_predictions.csv")

    # 輸出與 LLM CSV 相同格式的檔案（欄位 "0"~"9" 填同一個預測值）
    # 放在 result/ 目錄，multi_model_comparison.py 直接讀取
    for (fs_label, pred_name), preds in all_preds.items():
        safe_name = f"{fs_label}__{pred_name}".replace(" ", "_").replace("(", "").replace(")", "")
        out_path  = os.path.join(".", f"scored_by_ml_{safe_name}.csv")
        out_df    = pd.DataFrame({
            "filename":              feat_df["filename"],
            "label_Dr. Tan":         feat_df["Rating1"] if "Rating1" in feat_df.columns else nan_col,
            "label_Dr. Chien":       feat_df["Rating2"] if "Rating2" in feat_df.columns else nan_col,
            "average_score_Doctors": feat_df["Rating"],
        })
        rounded = np.clip(np.round(preds), 0, MAX_SCORE)
        for i in range(10):
            out_df[str(i)] = rounded
        out_df.to_csv(out_path, index=False)
    print(f"  Saved {len(all_preds)} scored_by_ml_*.csv to result/")

    print_table(results)

    print("\n[3] Generating figures ...")
    fig1_heatmap_reg(results)
    fig2_heatmap_clf(results)
    fig3_ranked_reg(results)
    fig4_ranked_clf(results)
    fig5_clean_comparison(results)
    fig6_borda(results)
    fig7_pareto(results)

    print(f"\nDone! → {PNG_DIR}/  and  {PDF_DIR}/")
