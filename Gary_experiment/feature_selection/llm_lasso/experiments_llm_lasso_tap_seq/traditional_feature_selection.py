"""
Traditional Feature Selection Methods — Expanded Comparison (~20 methods)
==========================================================================

Categories
----------
Filter  : Pearson, Spearman, Kendall, Mutual-Info, F-regression,
          F-classif (ANOVA), mRMR
Embedded: LASSO-CV, ElasticNet-CV, Ridge Top-K,
          RF Importance, Extra-Trees Importance, GradBoost Importance,
          Stability Selection, Group LASSO (tap_amp_norm_* as one group)
Wrapper : RFECV (RF), RFECV (SVR), Sequential Forward Selection (SFS)
Baseline: Random (k=8)
External: LLM-Lasso, Standard LASSO  (loaded from prior run results)

Group LASSO requires: pip install group-lasso
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LassoCV, ElasticNetCV, Ridge, RidgeCV,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.feature_selection import (
    RFECV, SequentialFeatureSelector,
    mutual_info_regression, f_regression, f_classif,
)
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score,
)
from sklearn.metrics import mean_absolute_error, make_scorer
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import Counter

from llm_lasso_config import (
    USER_DATASET_PATH, EXCLUDE_COLS,
    OUTPUT_DIR, PLOT_DIR,
    N_CV_FOLDS, RANDOM_SEED,
)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

RESULTS_PATH = OUTPUT_DIR / "traditional_selection_results.json"
K_DEFAULT = 8   # default number of features to select


# ===========================
# Data Loading
# ===========================

def _clean_X(X, feature_names):
    """Drop all-NaN columns, then impute remaining NaN with column mean."""
    all_nan_mask = np.all(np.isnan(X), axis=0)
    if all_nan_mask.any():
        dropped = [feature_names[i] for i in np.where(all_nan_mask)[0]]
        print(f"  [load_data] Dropping {len(dropped)} all-NaN columns: {dropped}")
        keep = ~all_nan_mask
        X = X[:, keep]
        feature_names = [f for f, k in zip(feature_names, keep) if k]
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X, feature_names


def load_data():
    """Load user dataset. Returns two feature matrices:
      - X_scalar / scalar_names : tap_amp_norm_* EXCLUDED  → used by all regular methods
      - X_seq    / seq_names    : tap_amp_norm_* INCLUDED  → used by Group LASSO only
    The amplitude sequence is treated as a single group, not as individual scalar features.
    """
    df = pd.read_csv(USER_DATASET_PATH)
    all_cols = [c for c in df.columns
                if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

    df["Rating1"] = pd.to_numeric(df["Rating1"], errors="coerce")
    df["Rating2"] = pd.to_numeric(df["Rating2"], errors="coerce")
    df["consensus"] = (df["Rating1"] + df["Rating2"]) / 2.0

    valid = df.dropna(subset=["consensus"])
    y = valid["consensus"].values.astype(float)

    # Full set — for Group LASSO (includes tap_amp_norm_*)
    X_seq = valid[all_cols].values.astype(float)
    X_seq, seq_names = _clean_X(X_seq, list(all_cols))

    # Scalar set — for all other methods (tap_amp_norm_* removed)
    scalar_cols = [c for c in all_cols if not c.startswith(TAP_SEQ_PREFIX)]
    X_scalar = valid[scalar_cols].values.astype(float)
    X_scalar, scalar_names = _clean_X(X_scalar, list(scalar_cols))

    n_seq = sum(1 for c in seq_names if c.startswith(TAP_SEQ_PREFIX))
    print(f"  Scalar features : {len(scalar_names)}  (tap_amp_norm_* excluded)")
    print(f"  Sequence features: {n_seq} tap_amp_norm_* columns (Group LASSO only)")

    return (X_scalar, X_seq), y, (scalar_names, seq_names)


def _bin_y(y):
    """Round y to ordinal class label (0–4) for classifier-based methods."""
    return np.clip(np.round(y).astype(int), 0, 4)


# ===========================
# FILTER METHODS
# ===========================

def pearson_select(X, y, k=K_DEFAULT):
    """Top-K features by |Pearson r| with target."""
    corrs = np.array([
        abs(pearsonr(X[:, j], y)[0]) if np.std(X[:, j]) > 1e-10 else 0.0
        for j in range(X.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    return np.argsort(corrs)[-k:][::-1], corrs


def spearman_select(X, y, k=K_DEFAULT):
    """Top-K features by |Spearman ρ| with target."""
    corrs = np.array([
        abs(spearmanr(X[:, j], y).correlation) if np.std(X[:, j]) > 1e-10 else 0.0
        for j in range(X.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    return np.argsort(corrs)[-k:][::-1], corrs


def kendall_select(X, y, k=K_DEFAULT):
    """Top-K features by |Kendall τ| with target."""
    corrs = np.array([
        abs(kendalltau(X[:, j], y).statistic) if np.std(X[:, j]) > 1e-10 else 0.0
        for j in range(X.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    return np.argsort(corrs)[-k:][::-1], corrs


def mutual_info_select(X, y, k=K_DEFAULT):
    """
    Top-K features by Mutual Information (regression).

    Reference: Kraskov et al. (2004), nearest-neighbor MI estimator.
    sklearn: mutual_info_regression.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = mutual_info_regression(X_scaled, y, random_state=RANDOM_SEED)
    return np.argsort(scores)[-k:][::-1], scores


def f_regression_select(X, y, k=K_DEFAULT):
    """
    Top-K features by F-statistic (linear regression F-test).

    Reference: sklearn f_regression — univariate linear regression test.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    f_scores, _ = f_regression(X_scaled, y)
    f_scores = np.nan_to_num(f_scores)
    return np.argsort(f_scores)[-k:][::-1], f_scores


def f_classif_select(X, y, k=K_DEFAULT):
    """
    Top-K features by ANOVA F-statistic (classification formulation).

    Uses binned y as class labels. Tests between-class variance.
    Reference: sklearn f_classif.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_binned = _bin_y(y)
    f_scores, _ = f_classif(X_scaled, y_binned)
    f_scores = np.nan_to_num(f_scores)
    return np.argsort(f_scores)[-k:][::-1], f_scores


def mrmr_select(X, y, k=K_DEFAULT):
    """
    Minimum Redundancy Maximum Relevance (mRMR).

    Score = relevance(f) − (1/|S|) × Σ redundancy(f, s) for s in S.

    Reference: Ding & Peng (2005), JBCB.
    """
    n_features = X.shape[1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    relevance = np.array([
        abs(pearsonr(X_scaled[:, j], y)[0]) if np.std(X_scaled[:, j]) > 1e-10 else 0.0
        for j in range(n_features)
    ])
    relevance = np.nan_to_num(relevance)

    redundancy_matrix = np.abs(np.corrcoef(X_scaled.T))
    np.fill_diagonal(redundancy_matrix, 0)

    selected, remaining = [], list(range(n_features))
    for _ in range(min(k, n_features)):
        scores = {
            f: relevance[f] if not selected
               else relevance[f] - np.mean([redundancy_matrix[f, s] for s in selected])
            for f in remaining
        }
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.remove(best)

    return np.array(selected), relevance


# ===========================
# EMBEDDED METHODS
# ===========================

def lasso_cv_select(X, y, k=K_DEFAULT):
    """
    LASSO with cross-validated α — top-K non-zero coefficients.

    Reference: Tibshirani (1996), JRSS-B.
    sklearn: LassoCV.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=100)
    lasso.fit(X_scaled, y)
    coef_abs = np.abs(lasso.coef_)
    top_k = np.argsort(coef_abs)[-k:][::-1]
    return top_k, coef_abs


def elasticnet_select(X, y, k=K_DEFAULT):
    """
    ElasticNet with cross-validated α and l1_ratio — top-K coefficients.

    Combines L1 (sparsity) and L2 (grouping) penalties.
    Reference: Zou & Hastie (2005), JRSS-B.
    sklearn: ElasticNetCV.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    en = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=50,
    )
    en.fit(X_scaled, y)
    coef_abs = np.abs(en.coef_)
    top_k = np.argsort(coef_abs)[-k:][::-1]
    return top_k, coef_abs


def ridge_select(X, y, k=K_DEFAULT):
    """
    Ridge regression — top-K features by |coefficient|.

    L2 regularization keeps all features but shrinks small ones.
    Reference: Hoerl & Kennard (1970), Technometrics.
    sklearn: RidgeCV.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 50), cv=5)
    ridge.fit(X_scaled, y)
    coef_abs = np.abs(ridge.coef_)
    top_k = np.argsort(coef_abs)[-k:][::-1]
    return top_k, coef_abs


def rf_importance_select(X, y, k=K_DEFAULT):
    """
    Random Forest feature importances (mean decrease in impurity).

    Uses RandomForestRegressor for continuous target.
    Reference: Breiman (2001), Machine Learning.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    top_k = np.argsort(importances)[-k:][::-1]
    return top_k, importances


def extra_trees_select(X, y, k=K_DEFAULT):
    """
    Extremely Randomized Trees feature importances.

    Further randomizes split thresholds vs Random Forest.
    Reference: Geurts et al. (2006), Machine Learning.
    sklearn: ExtraTreesRegressor.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    et = ExtraTreesRegressor(
        n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1,
    )
    et.fit(X_scaled, y)
    importances = et.feature_importances_
    top_k = np.argsort(importances)[-k:][::-1]
    return top_k, importances


def gradient_boosting_select(X, y, k=K_DEFAULT):
    """
    Gradient Boosting feature importances (mean decrease in loss).

    Sequentially builds trees to minimize residual loss.
    Reference: Friedman (2001), Annals of Statistics.
    sklearn: GradientBoostingRegressor.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=RANDOM_SEED,
    )
    gb.fit(X_scaled, y)
    importances = gb.feature_importances_
    top_k = np.argsort(importances)[-k:][::-1]
    return top_k, importances


def stability_selection(X, y, n_bootstrap=200, threshold=0.6):
    """
    Stability Selection with LASSO.

    Repeatedly subsample 50% of data + run LASSO; keep features
    selected in ≥ threshold fraction of bootstraps.

    Reference: Meinshausen & Bühlmann (2010), JRSS-B.
    """
    n_samples, n_features = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    subsample_size = max(n_samples // 2, 10)
    selection_count = np.zeros(n_features)

    for b in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=subsample_size, replace=False)
        try:
            lasso = LassoCV(cv=3, max_iter=50000, random_state=b, n_alphas=50)
            lasso.fit(X_scaled[idx], y[idx])
            selection_count += (np.abs(lasso.coef_) > 1e-8).astype(int)
        except Exception:
            continue

    frequency = selection_count / n_bootstrap
    selected = np.where(frequency >= threshold)[0]
    for t in [0.5, 0.4, 0.3]:
        if len(selected) >= 3:
            break
        selected = np.where(frequency >= t)[0]

    order = np.argsort(frequency[selected])[::-1]
    return selected[order], frequency


# ===========================
# WRAPPER METHODS
# ===========================

def rfecv_rf_select(X, y):
    """
    RFECV with Random Forest classifier (binned target).

    Recursively removes least-important features; CV determines
    optimal subset size.
    Reference: Guyon et al. (2002), Machine Learning.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_binned = _bin_y(y)
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(
        estimator=RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=RANDOM_SEED,
            class_weight="balanced",
        ),
        step=1,
        cv=skf.split(X_scaled, y_binned),
        scoring="neg_mean_absolute_error",
        min_features_to_select=3,
    )
    rfecv.fit(X_scaled, y_binned)
    selected = np.where(rfecv.support_)[0]
    return selected, rfecv.ranking_


def rfecv_svr_select(X, y):
    """
    RFECV with Support Vector Regressor (continuous target).

    Uses linear SVR whose dual coefficients define feature ranking.
    Reference: Weston et al. (2000), Advances in Neural Information
    Processing Systems.
    sklearn: RFECV + SVR(kernel='linear').
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(
        estimator=SVR(kernel="linear", C=1.0),
        step=1,
        cv=kf,
        scoring="neg_mean_absolute_error",
        min_features_to_select=3,
    )
    rfecv.fit(X_scaled, y)
    selected = np.where(rfecv.support_)[0]
    return selected, rfecv.ranking_


def sequential_forward_select(X, y, k=K_DEFAULT):
    """
    Sequential Forward Selection (SFS).

    Greedily adds the feature that most improves CV score at each step.
    Reference: Whitney (1971); sklearn SequentialFeatureSelector.
    Uses Ridge regressor for speed on small datasets.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    sfs = SequentialFeatureSelector(
        estimator=Ridge(alpha=1.0),
        n_features_to_select=k,
        direction="forward",
        scoring="neg_mean_absolute_error",
        cv=kf,
        n_jobs=-1,
    )
    sfs.fit(X_scaled, y)
    selected = np.where(sfs.get_support())[0]
    return selected, sfs.get_support().astype(float)


# ===========================
# GROUP LASSO (sequence-aware)
# ===========================

TAP_SEQ_PREFIX = "tap_amp_"   # matches both tap_amp_norm_* and tap_amp_px_*

def group_lasso_select(X, y, feature_names):
    """
    Group LASSO: tap_amp_norm_* 全部當成一個 group，
    其餘每個特徵各自為一個 group。
    整個振幅序列要選就一起選、要排除就一起排除。

    Requires: pip install group-lasso

    Returns:
        selected_indices: 選到的特徵 index array
        scores: None（無 per-feature scores）
    """
    try:
        from group_lasso import GroupLasso
    except ImportError:
        print("  [Group LASSO] group-lasso not installed. Run: pip install group-lasso")
        return None, None

    # Build group labels: tap_amp_norm_* → group 0; others → unique groups 1, 2, ...
    groups = np.zeros(len(feature_names), dtype=int)
    next_group = 1
    for i, f in enumerate(feature_names):
        if not f.startswith(TAP_SEQ_PREFIX):
            groups[i] = next_group
            next_group += 1

    tap_indices = [i for i, f in enumerate(feature_names) if f.startswith(TAP_SEQ_PREFIX)]
    n_tap = len(tap_indices)
    if n_tap == 0:
        print("  [Group LASSO] No tap_amp_norm_* columns found — skipping")
        return None, None

    print(f"  [Group LASSO] Tap sequence group: {n_tap} features "
          f"(tap_amp_norm_01..{feature_names[tap_indices[-1]][-2:]})")
    print(f"  [Group LASSO] Total groups: {next_group} "
          f"(group 0 = sequence, 1..{next_group-1} = scalar features)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validate group_reg
    reg_candidates = np.logspace(-3, 0, 10)
    best_reg = reg_candidates[0]
    best_mae = np.inf

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    print(f"  [Group LASSO] CV over {len(reg_candidates)} reg values...", flush=True)
    for reg in reg_candidates:
        fold_maes = []
        for train_idx, val_idx in kf.split(X_scaled):
            gl = GroupLasso(
                groups=groups,
                group_reg=reg,
                l1_reg=0.0,
                n_iter=1000,
                tol=1e-5,
                scale_reg="group_size",
                supress_warning=True,
            )
            try:
                gl.fit(X_scaled[train_idx], y[train_idx].reshape(-1, 1))
                pred = gl.predict(X_scaled[val_idx]).flatten()
                fold_maes.append(mean_absolute_error(y[val_idx], pred))
            except Exception:
                fold_maes.append(np.inf)
        mae = np.mean(fold_maes)
        if mae < best_mae:
            best_mae = mae
            best_reg = reg

    print(f"  [Group LASSO] Best reg={best_reg:.4f}, CV-MAE={best_mae:.4f}")

    # Final fit with best reg
    gl = GroupLasso(
        groups=groups,
        group_reg=best_reg,
        l1_reg=0.0,
        n_iter=2000,
        tol=1e-6,
        scale_reg="group_size",
        supress_warning=True,
    )
    gl.fit(X_scaled, y.reshape(-1, 1))
    coef = gl.coef_.flatten()

    selected = np.where(np.abs(coef) > 1e-8)[0]

    # Report whether the tap sequence group was selected
    seq_selected = any(i in selected for i in tap_indices)
    print(f"  [Group LASSO] Sequence group selected: {seq_selected}")
    if seq_selected:
        print(f"  [Group LASSO] All {n_tap} tap_amp_norm_* features included as a group")

    return selected, None


# ===========================
# EVALUATION
# ===========================

def evaluate_feature_subset(X, y, selected_indices, method_name):
    """Evaluate a feature subset using LASSO CV-MAE (5-fold stratified)."""
    selected_indices = np.asarray(selected_indices)
    if len(selected_indices) == 0:
        return {"method": method_name, "cv_mae": np.nan, "n_features": 0,
                "max_inter_corr": np.nan, "mean_inter_corr": np.nan}

    X_sub = X[:, selected_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    y_binned = _bin_y(y)
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    lasso = LassoCV(cv=3, max_iter=50000, random_state=RANDOM_SEED)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    scores = cross_val_score(
        lasso, X_scaled, y,
        cv=skf.split(X_scaled, y_binned),
        scoring=mae_scorer,
    )
    cv_mae = float(-scores.mean())

    if len(selected_indices) > 1:
        corr_mat = np.abs(np.corrcoef(X_sub.T))
        np.fill_diagonal(corr_mat, 0)
        max_corr = float(corr_mat.max())
        mean_corr = float(corr_mat[np.triu_indices_from(corr_mat, k=1)].mean())
    else:
        max_corr = mean_corr = 0.0

    return {
        "method": method_name,
        "cv_mae": cv_mae,
        "n_features": int(len(selected_indices)),
        "max_inter_corr": max_corr,
        "mean_inter_corr": mean_corr,
    }


def evaluate_random_baseline(X, y, k=K_DEFAULT, n_trials=100):
    """Average CV-MAE over many random feature subsets."""
    maes = []
    for _ in range(n_trials):
        idx = np.random.choice(X.shape[1], size=k, replace=False)
        r = evaluate_feature_subset(X, y, idx, "random_trial")
        maes.append(r["cv_mae"])
    return {
        "method": f"Random (k={k})",
        "cv_mae": float(np.nanmean(maes)),
        "cv_mae_std": float(np.nanstd(maes)),
        "n_features": k,
        "max_inter_corr": np.nan,
        "mean_inter_corr": np.nan,
    }


# ===========================
# VISUALIZATION
# ===========================

def plot_comparison(results, output_dir):
    """Grouped bar chart: CV-MAE + #Features per method."""
    methods = [r["method"] for r in results]
    maes = [r["cv_mae"] for r in results]
    n_feats = [r["n_features"] for r in results]
    errs = [r.get("cv_mae_std", 0) or 0 for r in results]

    fig, ax1 = plt.subplots(figsize=(max(14, len(methods) * 0.8), 6))
    x = np.arange(len(methods))
    w = 0.4

    bars = ax1.bar(x - w/2, maes, w, color="#3498db", edgecolor="gray",
                   label="CV-MAE ↓", yerr=errs, capsize=3)
    ax1.set_ylabel("Cross-Validated MAE ↓", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)

    for bar, val in zip(bars, maes):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, n_feats, w, color="#e74c3c", alpha=0.5,
                    edgecolor="gray", label="#Features")
    ax2.set_ylabel("# Selected Features", fontsize=11, color="#e74c3c")
    for bar, val in zip(bars2, n_feats):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 str(val), ha="center", va="bottom", fontsize=7, color="#e74c3c")

    ax1.set_title("Feature Selection Method Comparison (~20 methods) — CV-MAE",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_mae_ranked(results, output_dir):
    """Horizontal bar chart ranked by CV-MAE."""
    sorted_r = sorted(results, key=lambda r: r["cv_mae"])
    methods = [r["method"] for r in sorted_r]
    maes = [r["cv_mae"] for r in sorted_r]
    errs = [r.get("cv_mae_std", 0) or 0 for r in sorted_r]

    colors = []
    for r in sorted_r:
        m = r["method"]
        if "LLM" in m:
            colors.append("#2ecc71")
        elif "Random" in m:
            colors.append("#95a5a6")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(8, max(6, len(methods) * 0.4)))
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, maes, xerr=errs, color=colors, edgecolor="gray",
            capsize=3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Cross-Validated MAE ↓", fontsize=11)
    ax.set_title("Feature Selection Methods Ranked by CV-MAE\n"
                 "(green = LLM-Lasso, gray = random baseline)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    for i, (val, err) in enumerate(zip(maes, errs)):
        ax.text(val + (err or 0) + 0.002, i, f"{val:.3f}",
                va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "mae_ranked.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "mae_ranked.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_redundancy_scatter(results, output_dir):
    """Scatter: CV-MAE vs max inter-feature correlation (bottom-left = best)."""
    valid = [r for r in results if not np.isnan(r.get("max_inter_corr", np.nan))]
    if len(valid) < 3:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for r in valid:
        color = "#2ecc71" if "LLM" in r["method"] else "#3498db"
        ax.scatter(r["max_inter_corr"], r["cv_mae"], s=80, color=color,
                   zorder=5, edgecolors="gray", linewidths=0.5)
        ax.annotate(r["method"], (r["max_inter_corr"], r["cv_mae"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Max Inter-Feature |Correlation| (redundancy →)", fontsize=11)
    ax.set_ylabel("CV-MAE ↓", fontsize=11)
    ax.set_title("Accuracy vs Redundancy Trade-off\n(bottom-left = best)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_redundancy.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "accuracy_vs_redundancy.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_overlap(all_selections, feature_names, output_dir):
    """Heatmap: which features each method selected."""
    methods = list(all_selections.keys())
    n_features = len(feature_names)

    mat = np.zeros((len(methods), n_features))
    for i, (method, indices) in enumerate(all_selections.items()):
        for idx in indices:
            mat[i, idx] = 1

    selected_any = mat.sum(axis=0) > 0
    mat_f = mat[:, selected_any]
    names_f = [feature_names[j] for j in range(n_features) if selected_any[j]]

    order = np.argsort(mat_f.sum(axis=0))[::-1]
    mat_f = mat_f[:, order]
    names_f = [names_f[j] for j in order]

    fig, ax = plt.subplots(
        figsize=(max(12, len(names_f) * 0.45), max(4, len(methods) * 0.45))
    )
    sns.heatmap(mat_f, annot=False, cmap="YlOrRd",
                xticklabels=names_f, yticklabels=methods,
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "Selected (1=yes)"})
    ax.set_title("Feature Selection Overlap Across Methods",
                 fontsize=13, fontweight="bold")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_overlap_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "feature_overlap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_category_comparison(results, output_dir):
    """Box plot of CV-MAE by method category."""
    category_map = {
        "Filter":   ["Pearson", "Spearman", "Kendall", "Mutual Info",
                     "F-regression", "F-classif (ANOVA)", "mRMR"],
        "Embedded": ["LASSO-CV", "ElasticNet-CV", "Ridge",
                     "RF Importance", "Extra Trees", "Gradient Boosting",
                     "Stability Selection", "Group LASSO"],
        "Wrapper":  ["RFECV (RF)", "RFECV (SVR)", "Seq. Forward"],
        "Baseline/External": ["Random", "LLM-Lasso", "Standard LASSO"],
    }

    cat_maes = {cat: [] for cat in category_map}
    for r in results:
        for cat, prefixes in category_map.items():
            if any(r["method"].startswith(p) for p in prefixes):
                if not np.isnan(r["cv_mae"]):
                    cat_maes[cat].append(r["cv_mae"])
                break

    cats = [c for c, vals in cat_maes.items() if vals]
    data = [cat_maes[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=cats, patch_artist=True, notch=False)
    colors = ["#3498db", "#e67e22", "#9b59b6", "#2ecc71"]
    for patch, color in zip(bp["boxes"], colors[:len(cats)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("CV-MAE ↓", fontsize=11)
    ax.set_title("CV-MAE Distribution by Method Category", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "category_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "category_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


# ===========================
# MAIN
# ===========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    (X, X_seq), y, (feature_names, seq_names) = load_data()
    n_feat = X.shape[1]
    print(f"  Samples: {X.shape[0]}, Features (scalar): {n_feat}")

    results = []
    # all_selections stores feature NAMES (not indices) to avoid index ambiguity
    # between the scalar (X) and full-sequence (X_seq) feature sets.
    all_selections = {}  # {method: [feat_name, ...]}

    def _run(label, fn, *args, **kwargs):
        """Helper: call selection fn on X (scalar only), evaluate, print summary."""
        print(f"\n  Running {label}...")
        try:
            idx, _ = fn(*args, **kwargs)
            if idx is None or len(idx) == 0:
                print(f"    [WARNING] No features selected — skipping.")
                return
            ev = evaluate_feature_subset(X, y, idx, label)
            results.append(ev)
            feat_names = [feature_names[i] for i in idx]
            all_selections[label] = feat_names
            print(f"    Selected ({len(idx)}): {feat_names}")
            print(f"    CV-MAE: {ev['cv_mae']:.4f}  |  Max|r|: {ev['max_inter_corr']:.3f}")
        except Exception as e:
            print(f"    [ERROR] {label}: {e}")

    # ------------------------------------------------------------------
    print("\n=== FILTER METHODS ===")
    _run("Pearson",            pearson_select,       X, y)
    _run("Spearman",           spearman_select,      X, y)
    _run("Kendall",            kendall_select,       X, y)
    _run("Mutual Info",        mutual_info_select,   X, y)
    _run("F-regression",       f_regression_select,  X, y)
    _run("F-classif (ANOVA)",  f_classif_select,     X, y)
    _run("mRMR",               mrmr_select,          X, y)

    # ------------------------------------------------------------------
    print("\n=== EMBEDDED METHODS ===")
    _run("LASSO-CV",           lasso_cv_select,          X, y)
    _run("ElasticNet-CV",      elasticnet_select,        X, y)
    _run("Ridge",              ridge_select,             X, y)
    _run("RF Importance",      rf_importance_select,     X, y)
    _run("Extra Trees",        extra_trees_select,       X, y)
    _run("Gradient Boosting",  gradient_boosting_select, X, y)

    print("\n  Running Stability Selection (200 bootstraps)...")
    try:
        stab_idx, stab_freq = stability_selection(X, y)
        ev_stab = evaluate_feature_subset(X, y, stab_idx, "Stability Selection")
        results.append(ev_stab)
        stab_feat_names = [feature_names[i] for i in stab_idx]
        all_selections["Stability Selection"] = stab_feat_names
        print(f"    Selected ({len(stab_idx)}): {stab_feat_names}")
        print(f"    CV-MAE: {ev_stab['cv_mae']:.4f}")
    except Exception as e:
        print(f"    [ERROR] Stability Selection: {e}")
        stab_freq = np.zeros(n_feat)

    # ------------------------------------------------------------------
    print("\n=== WRAPPER METHODS ===")

    print("  Running RFECV (RF)...")
    try:
        rfecv_rf_idx, _ = rfecv_rf_select(X, y)
        ev_rfecv_rf = evaluate_feature_subset(X, y, rfecv_rf_idx, "RFECV (RF)")
        results.append(ev_rfecv_rf)
        all_selections["RFECV (RF)"] = [feature_names[i] for i in rfecv_rf_idx]
        print(f"    Selected ({len(rfecv_rf_idx)}): {all_selections['RFECV (RF)']}")
        print(f"    CV-MAE: {ev_rfecv_rf['cv_mae']:.4f}")
    except Exception as e:
        print(f"    [ERROR] RFECV (RF): {e}")

    print("  Running RFECV (SVR)...")
    try:
        rfecv_svr_idx, _ = rfecv_svr_select(X, y)
        ev_rfecv_svr = evaluate_feature_subset(X, y, rfecv_svr_idx, "RFECV (SVR)")
        results.append(ev_rfecv_svr)
        all_selections["RFECV (SVR)"] = [feature_names[i] for i in rfecv_svr_idx]
        print(f"    Selected ({len(rfecv_svr_idx)}): {all_selections['RFECV (SVR)']}")
        print(f"    CV-MAE: {ev_rfecv_svr['cv_mae']:.4f}")
    except Exception as e:
        print(f"    [ERROR] RFECV (SVR): {e}")

    _run("Seq. Forward",   sequential_forward_select, X, y)

    # ------------------------------------------------------------------
    print("\n=== GROUP LASSO (sequence-aware) ===")
    print("  Running Group LASSO (uses FULL feature set incl. tap_amp_norm_*)...")
    try:
        gl_idx, _ = group_lasso_select(X_seq, y, seq_names)
        if gl_idx is not None and len(gl_idx) > 0:
            ev_gl = evaluate_feature_subset(X_seq, y, gl_idx, "Group LASSO")
            results.append(ev_gl)
            gl_feat_names = [seq_names[i] for i in gl_idx]
            all_selections["Group LASSO"] = gl_feat_names
            print(f"    Selected ({len(gl_idx)}): {gl_feat_names}")
            print(f"    CV-MAE: {ev_gl['cv_mae']:.4f}  |  Max|r|: {ev_gl['max_inter_corr']:.3f}")
        else:
            print("    [WARNING] Group LASSO returned no features — skipping.")
    except Exception as e:
        print(f"    [ERROR] Group LASSO: {e}")

    # ------------------------------------------------------------------
    print("\n=== BASELINE ===")
    print("  Running Random baseline (100 trials)...")
    ev_rand = evaluate_random_baseline(X, y, k=K_DEFAULT, n_trials=100)
    results.append(ev_rand)
    print(f"    CV-MAE: {ev_rand['cv_mae']:.4f} ± {ev_rand['cv_mae_std']:.4f}")

    # ------------------------------------------------------------------
    print("\n=== EXTERNAL RESULTS (LLM-Lasso run) ===")
    llm_results_path = OUTPUT_DIR / "feature_selection_results.json"
    if llm_results_path.exists():
        with open(llm_results_path) as f:
            llm_data = json.load(f)

        llm_feats = llm_data.get("llm_lasso_selected_features", [])
        llm_idx = [feature_names.index(ft) for ft in llm_feats if ft in feature_names]
        if llm_idx:
            ev_llm = evaluate_feature_subset(X, y, np.array(llm_idx), "LLM-Lasso")
            results.append(ev_llm)
            all_selections["LLM-Lasso"] = [feature_names[i] for i in llm_idx]
            print(f"  LLM-Lasso ({len(llm_idx)}): {llm_feats}")
            print(f"    CV-MAE: {ev_llm['cv_mae']:.4f}")

        std_feats = llm_data.get("standard_lasso_selected", [])
        std_idx = [feature_names.index(ft) for ft in std_feats if ft in feature_names]
        if std_idx:
            ev_std = evaluate_feature_subset(X, y, np.array(std_idx), "Standard LASSO")
            results.append(ev_std)
            all_selections["Standard LASSO"] = [feature_names[i] for i in std_idx]
            print(f"  Standard LASSO ({len(std_idx)}): {std_feats}")
            print(f"    CV-MAE: {ev_std['cv_mae']:.4f}")
    else:
        print("  (No LLM-Lasso results found at", llm_results_path, ")")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    results_sorted = sorted(results, key=lambda r: r["cv_mae"])

    print("\n" + "=" * 90)
    print("FEATURE SELECTION COMPARISON SUMMARY")
    print("=" * 90)
    print(f"{'Method':<28} {'#Feat':>5} {'CV-MAE':>9} {'Max|r|':>9} {'Mean|r|':>9}")
    print("-" * 90)
    for r in results_sorted:
        max_c  = f"{r['max_inter_corr']:.3f}"  if not np.isnan(r.get('max_inter_corr',  np.nan)) else "N/A"
        mean_c = f"{r['mean_inter_corr']:.3f}" if not np.isnan(r.get('mean_inter_corr', np.nan)) else "N/A"
        std_s  = f" ±{r['cv_mae_std']:.3f}" if r.get("cv_mae_std") else ""
        print(f"{r['method']:<28} {r['n_features']:>5} {r['cv_mae']:>9.4f}{std_s:<8} {max_c:>9} {mean_c:>9}")

    # Consensus features (all_selections now stores names directly)
    print("\n" + "=" * 90)
    print("CONSENSUS FEATURES")
    print("=" * 90)
    flat = [fname for names in all_selections.values() for fname in names]
    feat_counts = Counter(flat)
    n_methods = len(all_selections)

    for min_count in [max(3, n_methods // 3), 3, 2]:
        consensus = sorted(
            [(feat, cnt) for feat, cnt in feat_counts.items() if cnt >= min_count],
            key=lambda x: x[1], reverse=True,
        )
        if consensus:
            print(f"(Features selected by ≥ {min_count}/{n_methods} methods:)")
            break

    for feat, cnt in consensus:
        using = [m for m, names in all_selections.items() if feat in names]
        print(f"  {feat:40s} {cnt:>2}/{n_methods}  {using}")

    # Save (all_selections already stores names, no conversion needed)
    save_data = {
        "comparison": results_sorted,
        "selections": all_selections,
        "consensus_features": [f for f, _ in consensus],
        "stability_frequencies": {feature_names[i]: float(stab_freq[i])
                                  for i in range(len(feature_names))},
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_PATH}")

    # Plots — build a name→index map for the overlap heatmap (scalar features only)
    all_feat_names = list(dict.fromkeys(
        fname for names in all_selections.values() for fname in names
    ))
    name_to_idx = {f: i for i, f in enumerate(all_feat_names)}
    # Convert back to index-based dict for plot_feature_overlap
    all_sel_idx = {
        method: [name_to_idx[f] for f in names]
        for method, names in all_selections.items()
    }
    print("Generating plots...")
    plot_comparison(results_sorted, PLOT_DIR)
    plot_mae_ranked(results_sorted, PLOT_DIR)
    plot_feature_overlap(all_sel_idx, all_feat_names, PLOT_DIR)
    plot_category_comparison(results_sorted, PLOT_DIR)
    plot_redundancy_scatter(results_sorted, PLOT_DIR)

    print(f"Plots saved → {PLOT_DIR}")

    return results_sorted


if __name__ == "__main__":
    main()
