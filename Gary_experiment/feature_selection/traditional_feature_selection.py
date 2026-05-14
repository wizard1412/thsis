"""
Traditional Feature Selection Methods — Expanded Comparison (28 methods)
==========================================================================

Categories
----------
Filter   (12): Pearson, Spearman, Kendall, Mutual Info, F-regression, mRMR,
               Distance Correlation, HSIC, Fisher Score, ReliefF, FCBF,
               Variance Threshold
Embedded (10): LASSO (Standard), ElasticNet, Adaptive LASSO, LARS,
               RF Importance, Extra Trees, Gradient Boosting, XGBoost,
               Permutation Importance, Stability Selection
Wrapper  (6) : RFECV (RF), RFECV (SVR), RFECV (Lasso),
               Sequential Forward (SFS), Sequential Backward (SBS), Boruta
Baseline (1) : Random (k=8)

Removed (per methodology review)
--------------------------------
- Ridge Top-K       : L2 does not produce sparse solutions, not a true FS method
- F-classif (ANOVA) : bins continuous y, redundant with F-regression
- LLM-Lasso external load : handled by separate experiment scripts
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
    Lasso, LassoCV, LassoLarsCV, ElasticNetCV, Ridge, RidgeCV,
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesRegressor, GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.feature_selection import (
    RFECV, SequentialFeatureSelector,
    mutual_info_regression, f_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score,
)
from sklearn.metrics import mean_absolute_error, make_scorer
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import Counter

# Optional packages
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from boruta import BorutaPy
    _HAS_BORUTA = True
except ImportError:
    _HAS_BORUTA = False

# ===========================
# Standalone config
# ===========================
_HERE = Path(__file__).resolve().parent
USER_DATASET_PATH = _HERE / "features_dataset_v2.csv"
EXCLUDE_COLS      = {"hand", "filename", "Rating1", "Rating2", "Rating"}
N_CV_FOLDS        = 5
RANDOM_SEED       = 42

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

OUTPUT_DIR   = _HERE / "traditional_fs_results_v2"
PLOT_DIR     = OUTPUT_DIR / "plots"
RESULTS_PATH = OUTPUT_DIR / "traditional_selection_results.json"
K_DEFAULT    = 8


# ===========================
# Data Loading
# ===========================

def load_data():
    """Load user dataset, return X (float), y (float), feature_names."""
    df = pd.read_csv(USER_DATASET_PATH)
    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

    df["Rating1"] = pd.to_numeric(df["Rating1"], errors="coerce")
    df["Rating2"] = pd.to_numeric(df["Rating2"], errors="coerce")
    df["consensus"] = (df["Rating1"] + df["Rating2"]) / 2.0

    valid = df.dropna(subset=["consensus"])
    X = valid[feature_cols].values.astype(float)
    y = valid["consensus"].values.astype(float)

    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return X, y, feature_cols


def _bin_y(y):
    """Round y to ordinal class label (0–4) for classifier-based methods.
    Uses standard rounding (half up) instead of NumPy banker's rounding."""
    return np.clip(np.floor(np.asarray(y) + 0.5).astype(int), 0, 4)


def _cv_select_k(X, y, ranked_indices, k_min=2, k_max=None):
    """Select optimal feature count via cross-validation over a pre-ranked order.

    ranked_indices: feature indices sorted from most to least relevant.
    Returns the k in [k_min, k_max] that minimises CV-MAE.
    """
    n_total = X.shape[1]
    if k_max is None:
        k_max = min(n_total, 20)
    k_max = min(k_max, len(ranked_indices))
    k_min = min(k_min, k_max)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    best_k, best_score = k_min, -np.inf
    for k in range(k_min, k_max + 1):
        idx = np.asarray(ranked_indices[:k])
        s = cross_val_score(Ridge(), X_sc[:, idx], y,
                            cv=kf, scoring="neg_mean_absolute_error").mean()
        if s > best_score:
            best_score = s
            best_k = k
    return best_k


# ===========================
# FILTER METHODS
# ===========================

def pearson_select(X, y, k=None):
    """Top-K features by |Pearson r|; k chosen by CV if not specified."""
    corrs = np.array([
        abs(pearsonr(X[:, j], y)[0]) if np.std(X[:, j]) > 1e-10 else 0.0
        for j in range(X.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    ranked = np.argsort(corrs)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], corrs


def spearman_select(X, y, k=None):
    """Top-K features by |Spearman ρ|; k chosen by CV if not specified."""
    corrs = np.array([
        abs(spearmanr(X[:, j], y).correlation) if np.std(X[:, j]) > 1e-10 else 0.0
        for j in range(X.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    ranked = np.argsort(corrs)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], corrs


def kendall_select(X, y, k=None):
    """Top-K features by |Kendall τ|; k chosen by CV if not specified."""
    corrs = np.array([
        abs(kendalltau(X[:, j], y).statistic) if np.std(X[:, j]) > 1e-10 else 0.0
        for j in range(X.shape[1])
    ])
    corrs = np.nan_to_num(corrs)
    ranked = np.argsort(corrs)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], corrs


def mutual_info_select(X, y, k=None):
    """Top-K features by Mutual Information; k chosen by CV if not specified."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = mutual_info_regression(X_scaled, y, random_state=RANDOM_SEED)
    ranked = np.argsort(scores)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], scores


def f_regression_select(X, y, k=None):
    """Top-K features by F-statistic; k chosen by CV if not specified."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    f_scores, _ = f_regression(X_scaled, y)
    f_scores = np.nan_to_num(f_scores)
    ranked = np.argsort(f_scores)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], f_scores


def mrmr_select(X, y, k=None):
    """
    Minimum Redundancy Maximum Relevance (mRMR); k chosen by CV if not specified.
    Score = relevance(f) − (1/|S|) × Σ redundancy(f, s).
    Reference: Ding & Peng (2005), JBCB.
    """
    n_features = X.shape[1]
    k_run = min(n_features, 20) if k is None else k
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
    for _ in range(min(k_run, n_features)):
        scores = {
            f: relevance[f] if not selected
               else relevance[f] - np.mean([redundancy_matrix[f, s] for s in selected])
            for f in remaining
        }
        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.remove(best)

    if k is None:
        k = _cv_select_k(X, y, selected)
    return np.array(selected[:k]), relevance


def _distance_correlation(x, y):
    """Distance correlation (Székely et al. 2007). Captures nonlinear dependence."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, dcov2) / denom))


def distance_correlation_select(X, y, k=None):
    """
    Top-K features by Distance Correlation; k chosen by CV if not specified.
    dCor = 0 iff variables are independent. Captures nonlinear dependence.
    Reference: Székely et al. (2007), Ann. Stat.
    """
    scores = np.array([_distance_correlation(X[:, j], y) for j in range(X.shape[1])])
    ranked = np.argsort(scores)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], scores


def _rbf_kernel_matrix(a):
    """Gaussian RBF kernel with median heuristic for bandwidth."""
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    sq = (a - a.T) ** 2
    med = np.median(sq[sq > 0]) if np.any(sq > 0) else 1.0
    sigma2 = max(med, 1e-10)
    return np.exp(-sq / (2 * sigma2))


def _hsic(x, y):
    """Empirical HSIC with Gaussian kernels (Gretton et al. 2005)."""
    n = len(x)
    K = _rbf_kernel_matrix(x)
    L = _rbf_kernel_matrix(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    return float(np.trace(Kc @ L) / ((n - 1) ** 2))


def hsic_select(X, y, k=None):
    """
    Top-K features by HSIC; k chosen by CV if not specified.
    Reference: Gretton et al. (2005), ALT.
    Kernel-based independence measure; detects any functional dependence.
    """
    scores = np.array([_hsic(X[:, j], y) for j in range(X.shape[1])])
    scores = np.nan_to_num(scores)
    ranked = np.argsort(scores)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], scores


def fisher_score_select(X, y, k=None):
    """
    Fisher Score — between-class variance / within-class variance on binned y; k by CV.
    Reference: Duda, Hart & Stork "Pattern Classification".
    """
    y_binned = _bin_y(y)
    classes = np.unique(y_binned)
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    mu_global = X.mean(axis=0)
    for j in range(n_features):
        num = 0.0
        den = 0.0
        for c in classes:
            mask = y_binned == c
            n_c = mask.sum()
            if n_c == 0:
                continue
            mu_c = X[mask, j].mean()
            var_c = X[mask, j].var()
            num += n_c * (mu_c - mu_global[j]) ** 2
            den += n_c * var_c
        scores[j] = num / (den + 1e-10)
    ranked = np.argsort(scores)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], scores


def relieff_select(X, y, k=None, n_neighbors=5, n_iter=None):
    """
    ReliefF — distance-based FS that considers feature interactions.
    References: Kira & Rendell (1992); Kononenko (1994).
    Classification-style: bins continuous y into ordinal classes.
    """
    n_samples, n_features = X.shape
    X_rng = X.max(axis=0) - X.min(axis=0)
    X_rng[X_rng < 1e-10] = 1.0
    X_norm = (X - X.min(axis=0)) / X_rng

    y_binned = _bin_y(y)
    classes = np.unique(y_binned)
    class_probs = {c: float((y_binned == c).mean()) for c in classes}

    weights = np.zeros(n_features)
    n_iter = n_iter or n_samples
    rng = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_iter):
        i = int(rng.integers(n_samples))
        xi = X_norm[i]
        ci = y_binned[i]

        dist = np.sum(np.abs(X_norm - xi), axis=1)
        dist[i] = np.inf

        # Nearest hits (same class)
        hit_mask = (y_binned == ci).copy()
        hit_mask[i] = False
        if hit_mask.any():
            hit_dist = np.where(hit_mask, dist, np.inf)
            nearest_hits = np.argsort(hit_dist)[:n_neighbors]
            for h in nearest_hits:
                weights -= np.abs(xi - X_norm[h]) / n_iter

        # Nearest misses (each other class, weighted by prior)
        p_ci = class_probs[ci]
        for c in classes:
            if c == ci:
                continue
            miss_mask = y_binned == c
            if not miss_mask.any():
                continue
            miss_dist = np.where(miss_mask, dist, np.inf)
            nearest_misses = np.argsort(miss_dist)[:n_neighbors]
            w_c = class_probs[c] / max(1.0 - p_ci, 1e-10)
            for m in nearest_misses:
                weights += w_c * np.abs(xi - X_norm[m]) / n_iter

    ranked = np.argsort(weights)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], weights


def _symmetric_uncertainty(x, y, bins=10):
    """Symmetric Uncertainty = 2 * I(X;Y) / (H(X) + H(Y))."""
    def _discretize(a):
        edges = np.linspace(a.min(), a.max() + 1e-9, bins + 1)
        return np.digitize(a, edges)

    x_d = _discretize(np.asarray(x, dtype=float))
    y_d = _discretize(np.asarray(y, dtype=float))

    def _entropy(a):
        _, counts = np.unique(a, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    joint = np.stack([x_d, y_d], axis=1)
    _, counts = np.unique(joint, axis=0, return_counts=True)
    p_joint = counts / counts.sum()
    h_joint = -np.sum(p_joint * np.log2(p_joint + 1e-12))

    h_x = _entropy(x_d)
    h_y = _entropy(y_d)
    mi = max(0.0, h_x + h_y - h_joint)
    denom = h_x + h_y
    return 2.0 * mi / (denom + 1e-12)


def fcbf_select(X, y, k=None):
    """
    Fast Correlation-Based Filter (FCBF); k chosen by CV if not specified.
    FCBF first runs to natural termination (SU-based redundancy elimination),
    then CV selects how many of the ranked survivors to keep.
    Reference: Yu & Liu (2003), ICML.
    """
    n_features = X.shape[1]
    k_cap = min(n_features, 20) if k is None else k
    su_fy = np.array([_symmetric_uncertainty(X[:, j], y) for j in range(n_features)])
    order = list(np.argsort(su_fy)[::-1])

    selected = []
    remaining = order[:]
    while remaining and len(selected) < k_cap:
        fp = remaining.pop(0)
        selected.append(fp)
        remaining = [
            fq for fq in remaining
            if _symmetric_uncertainty(X[:, fp], X[:, fq]) < su_fy[fq]
        ]

    if len(selected) < 2:
        selected = order[:max(2, len(selected))]

    if k is None:
        k = _cv_select_k(X, y, selected)
    return np.array(selected[:k]), su_fy


def variance_threshold_select(X, y, k=None):
    """
    Top-K features by variance (unsupervised baseline); k chosen by CV if not specified.
    Does NOT look at y for ranking — purely a measure of feature dispersion.
    """
    variances = np.var(X, axis=0)
    ranked = np.argsort(variances)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], variances


# ===========================
# EMBEDDED METHODS
# ===========================

def _nonzero_selected(coef_abs, fallback_k=K_DEFAULT):
    """Return indices of non-zero coefficients, sorted by magnitude (descending).
    Falls back to top-fallback_k if the model selected nothing."""
    selected = np.where(coef_abs > 1e-8)[0]
    if len(selected) == 0:
        selected = np.argsort(coef_abs)[-fallback_k:]
    return selected[np.argsort(coef_abs[selected])[::-1]]


def lasso_cv_select(X, y, k=None):
    """
    Standard LASSO with cross-validated α — returns features with non-zero coefficients.
    k is ignored when None; pass an integer to force exactly k features (for baselines).
    Reference: Tibshirani (1996), JRSS-B.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=100)
    lasso.fit(X_scaled, y)
    coef_abs = np.abs(lasso.coef_)
    if k is not None:
        return np.argsort(coef_abs)[-k:][::-1], coef_abs
    return _nonzero_selected(coef_abs), coef_abs


def elasticnet_select(X, y, k=None):
    """
    ElasticNet (L1 + L2 mix) with CV α — returns features with non-zero coefficients.
    Reference: Zou & Hastie (2005), JRSS-B.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    en = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=50,
    )
    en.fit(X_scaled, y)
    coef_abs = np.abs(en.coef_)
    if k is not None:
        return np.argsort(coef_abs)[-k:][::-1], coef_abs
    return _nonzero_selected(coef_abs), coef_abs


def adaptive_lasso_select(X, y, k=None, gamma=1.0):
    """
    Adaptive LASSO — two-step reweighted LASSO with oracle property.
    Returns features with non-zero final coefficients.
    Reference: Zou (2006), JASA.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge.fit(X_scaled, y)
    init_beta = np.abs(ridge.coef_)
    weights = 1.0 / (init_beta ** gamma + 1e-6)

    X_weighted = X_scaled / weights
    lasso = LassoCV(cv=5, max_iter=100000, random_state=RANDOM_SEED, n_alphas=100)
    lasso.fit(X_weighted, y)
    coef = lasso.coef_ / weights
    coef_abs = np.abs(coef)
    if k is not None:
        return np.argsort(coef_abs)[-k:][::-1], coef_abs
    return _nonzero_selected(coef_abs), coef_abs


def lars_select(X, y, k=None):
    """
    Least Angle Regression (LARS) — returns features with non-zero coefficients.
    Reference: Efron et al. (2004), Ann. Stat.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lars = LassoLarsCV(cv=5, max_iter=500)
    lars.fit(X_scaled, y)
    coef_abs = np.abs(lars.coef_)
    if k is not None:
        return np.argsort(coef_abs)[-k:][::-1], coef_abs
    return _nonzero_selected(coef_abs), coef_abs


def rf_importance_select(X, y, k=None):
    """Random Forest feature importance (MDI); k chosen by CV. Reference: Breiman (2001)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_scaled, y)
    imp = rf.feature_importances_
    ranked = np.argsort(imp)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], imp


def extra_trees_select(X, y, k=None):
    """Extremely Randomized Trees importance; k chosen by CV. Reference: Geurts et al. (2006)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    et = ExtraTreesRegressor(n_estimators=500, random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X_scaled, y)
    imp = et.feature_importances_
    ranked = np.argsort(imp)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], imp


def gradient_boosting_select(X, y, k=None):
    """Gradient Boosting importance; k chosen by CV. Reference: Friedman (2001), Ann. Stat."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gb = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=RANDOM_SEED,
    )
    gb.fit(X_scaled, y)
    imp = gb.feature_importances_
    ranked = np.argsort(imp)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], imp


def xgboost_select(X, y, k=None):
    """XGBoost gain importance; k chosen by CV. Reference: Chen & Guestrin (2016), KDD."""
    if not _HAS_XGB:
        raise ImportError("xgboost not installed")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    xgb = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_SEED, verbosity=0,
    )
    xgb.fit(X_scaled, y)
    imp = xgb.feature_importances_
    ranked = np.argsort(imp)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], imp


def permutation_importance_select(X, y, k=None):
    """
    Permutation Importance with RF; k chosen by CV.
    Reference: Breiman (2001). sklearn.inspection.permutation_importance.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_scaled, y)
    result = permutation_importance(
        rf, X_scaled, y, n_repeats=30,
        random_state=RANDOM_SEED, n_jobs=-1,
        scoring="neg_mean_absolute_error",
    )
    imp = result.importances_mean
    ranked = np.argsort(imp)[::-1]
    if k is None:
        k = _cv_select_k(X, y, ranked)
    return ranked[:k], imp


def stability_selection(X, y, n_bootstrap=200, threshold=0.6):
    """
    Stability Selection with LASSO.
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
    """RFECV with Random Forest classifier on binned y."""
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
    return np.where(rfecv.support_)[0], rfecv.ranking_


def rfecv_svr_select(X, y):
    """RFECV with linear SVR. Reference: Weston et al. (2000), NeurIPS."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(
        estimator=SVR(kernel="linear", C=1.0),
        step=1, cv=kf,
        scoring="neg_mean_absolute_error",
        min_features_to_select=3,
    )
    rfecv.fit(X_scaled, y)
    return np.where(rfecv.support_)[0], rfecv.ranking_


def rfecv_lasso_select(X, y):
    """RFECV with Lasso — sparse-estimator RFE variant."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rfecv = RFECV(
        estimator=Lasso(alpha=0.01, max_iter=50000, random_state=RANDOM_SEED),
        step=1, cv=kf,
        scoring="neg_mean_absolute_error",
        min_features_to_select=3,
    )
    rfecv.fit(X_scaled, y)
    return np.where(rfecv.support_)[0], rfecv.ranking_


def sequential_forward_select(X, y, k=None):
    """Sequential Forward Selection (Whitney 1971) with Ridge evaluator; k by CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    if k is None:
        n_total = X.shape[1]
        k_max = min(n_total, 20)
        best_k, best_score = 2, -np.inf
        for ki in range(2, k_max + 1):
            sfs = SequentialFeatureSelector(
                estimator=Ridge(alpha=1.0), n_features_to_select=ki,
                direction="forward", scoring="neg_mean_absolute_error",
                cv=kf, n_jobs=-1,
            )
            sfs.fit(X_scaled, y)
            idx = np.where(sfs.get_support())[0]
            s = cross_val_score(Ridge(), X_scaled[:, idx], y,
                                cv=kf, scoring="neg_mean_absolute_error").mean()
            if s > best_score:
                best_score = s
                best_k = ki
        k = best_k

    sfs = SequentialFeatureSelector(
        estimator=Ridge(alpha=1.0), n_features_to_select=k,
        direction="forward", scoring="neg_mean_absolute_error",
        cv=kf, n_jobs=-1,
    )
    sfs.fit(X_scaled, y)
    return np.where(sfs.get_support())[0], sfs.get_support().astype(float)


def sequential_backward_select(X, y, k=None):
    """Sequential Backward Selection (Marill & Green 1963) with Ridge evaluator; k by CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    if k is None:
        n_total = X.shape[1]
        k_max = min(n_total, 20)
        best_k, best_score = 2, -np.inf
        for ki in range(2, k_max + 1):
            sbs = SequentialFeatureSelector(
                estimator=Ridge(alpha=1.0), n_features_to_select=ki,
                direction="backward", scoring="neg_mean_absolute_error",
                cv=kf, n_jobs=-1,
            )
            sbs.fit(X_scaled, y)
            idx = np.where(sbs.get_support())[0]
            s = cross_val_score(Ridge(), X_scaled[:, idx], y,
                                cv=kf, scoring="neg_mean_absolute_error").mean()
            if s > best_score:
                best_score = s
                best_k = ki
        k = best_k

    sbs = SequentialFeatureSelector(
        estimator=Ridge(alpha=1.0), n_features_to_select=k,
        direction="backward", scoring="neg_mean_absolute_error",
        cv=kf, n_jobs=-1,
    )
    sbs.fit(X_scaled, y)
    return np.where(sbs.get_support())[0], sbs.get_support().astype(float)


def boruta_select(X, y, k=None):
    """
    Boruta — all-relevant FS with RF + shadow features.
    Reference: Kursa & Rudnicki (2010), J. Stat. Softw.
    """
    if not _HAS_BORUTA:
        raise ImportError("boruta not installed")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=5, random_state=RANDOM_SEED, n_jobs=-1,
    )
    boruta = BorutaPy(
        rf, n_estimators="auto", max_iter=100,
        random_state=RANDOM_SEED, verbose=0,
    )
    boruta.fit(X_scaled, y)
    selected = np.where(boruta.support_)[0]
    if len(selected) == 0:
        selected = np.where(boruta.support_weak_)[0]
    if len(selected) == 0:
        # Fallback: top by inverse ranking
        selected = np.argsort(boruta.ranking_)[:K_DEFAULT]
    return selected, boruta.ranking_.astype(float)


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
                   label="CV-MAE", yerr=errs, capsize=3)
    ax1.set_ylabel("Cross-Validated MAE", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)

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

    ax1.set_title("Feature Selection Method Comparison (28 methods) — CV-MAE",
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

    colors = ["#95a5a6" if "Random" in r["method"] else "#3498db" for r in sorted_r]

    fig, ax = plt.subplots(figsize=(8, max(6, len(methods) * 0.4)))
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, maes, xerr=errs, color=colors, edgecolor="gray",
            capsize=3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Cross-Validated MAE", fontsize=11)
    ax.set_title("Feature Selection Methods Ranked by CV-MAE",
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
    """Scatter: CV-MAE vs max inter-feature correlation."""
    valid = [r for r in results if not np.isnan(r.get("max_inter_corr", np.nan))]
    if len(valid) < 3:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    for r in valid:
        ax.scatter(r["max_inter_corr"], r["cv_mae"], s=80, color="#3498db",
                   zorder=5, edgecolors="gray", linewidths=0.5)
        ax.annotate(r["method"], (r["max_inter_corr"], r["cv_mae"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Max Inter-Feature |Correlation| (redundancy →)", fontsize=11)
    ax.set_ylabel("CV-MAE", fontsize=11)
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
                     "F-regression", "mRMR", "Distance Correlation",
                     "HSIC", "Fisher Score", "ReliefF", "FCBF",
                     "Variance Threshold"],
        "Embedded": ["LASSO", "ElasticNet", "Adaptive LASSO", "LARS",
                     "RF Importance", "Extra Trees", "Gradient Boosting",
                     "XGBoost", "Permutation Importance",
                     "Stability Selection"],
        "Wrapper":  ["RFECV (RF)", "RFECV (SVR)", "RFECV (Lasso)",
                     "SFS Forward", "SBS Backward", "Boruta"],
        "Baseline": ["Random"],
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
    ax.set_ylabel("CV-MAE", fontsize=11)
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
    X, y, feature_names = load_data()
    n_feat = X.shape[1]
    print(f"  Samples: {X.shape[0]}, Features: {n_feat}")

    results = []
    all_selections = {}
    stab_freq = np.zeros(n_feat)

    def _run(label, fn, *args, **kwargs):
        """Helper: call selection fn, evaluate, print summary."""
        print(f"\n  Running {label}...")
        try:
            idx, _ = fn(*args, **kwargs)
            if idx is None or len(idx) == 0:
                print(f"    [WARNING] No features selected — skipping.")
                return
            ev = evaluate_feature_subset(X, y, idx, label)
            results.append(ev)
            all_selections[label] = list(map(int, idx))
            feat_names = [feature_names[i] for i in idx]
            print(f"    Selected ({len(idx)}): {feat_names}")
            print(f"    CV-MAE: {ev['cv_mae']:.4f}  |  Max|r|: {ev['max_inter_corr']:.3f}")
        except Exception as e:
            print(f"    [ERROR] {label}: {e}")

    # ------------------------------------------------------------------
    print("\n=== FILTER METHODS (12) ===")
    _run("Pearson",              pearson_select,              X, y)
    _run("Spearman",             spearman_select,             X, y)
    _run("Kendall",              kendall_select,              X, y)
    _run("Mutual Info",          mutual_info_select,          X, y)
    _run("F-regression",         f_regression_select,         X, y)
    _run("mRMR",                 mrmr_select,                 X, y)
    _run("Distance Correlation", distance_correlation_select, X, y)
    _run("HSIC",                 hsic_select,                 X, y)
    _run("Fisher Score",         fisher_score_select,         X, y)
    _run("ReliefF",              relieff_select,              X, y)
    _run("FCBF",                 fcbf_select,                 X, y)
    _run("Variance Threshold",   variance_threshold_select,   X, y)

    # ------------------------------------------------------------------
    print("\n=== EMBEDDED METHODS (10) ===")
    _run("LASSO (Standard)",       lasso_cv_select,                X, y)
    _run("ElasticNet",             elasticnet_select,              X, y)
    _run("Adaptive LASSO",         adaptive_lasso_select,          X, y)
    _run("LARS",                   lars_select,                    X, y)
    _run("RF Importance",          rf_importance_select,           X, y)
    _run("Extra Trees",            extra_trees_select,             X, y)
    _run("Gradient Boosting",      gradient_boosting_select,       X, y)
    if _HAS_XGB:
        _run("XGBoost",            xgboost_select,                 X, y)
    else:
        print("  [SKIP] XGBoost — package not installed")
    _run("Permutation Importance", permutation_importance_select,  X, y)

    print("\n  Running Stability Selection (200 bootstraps)...")
    try:
        stab_idx, stab_freq = stability_selection(X, y)
        ev_stab = evaluate_feature_subset(X, y, stab_idx, "Stability Selection")
        results.append(ev_stab)
        all_selections["Stability Selection"] = list(map(int, stab_idx))
        print(f"    Selected ({len(stab_idx)}): {[feature_names[i] for i in stab_idx]}")
        print(f"    CV-MAE: {ev_stab['cv_mae']:.4f}")
    except Exception as e:
        print(f"    [ERROR] Stability Selection: {e}")

    # ------------------------------------------------------------------
    print("\n=== WRAPPER METHODS (6) ===")

    for label, fn in [
        ("RFECV (RF)",    rfecv_rf_select),
        ("RFECV (SVR)",   rfecv_svr_select),
        ("RFECV (Lasso)", rfecv_lasso_select),
    ]:
        print(f"  Running {label}...")
        try:
            idx, _ = fn(X, y)
            ev = evaluate_feature_subset(X, y, idx, label)
            results.append(ev)
            all_selections[label] = list(map(int, idx))
            print(f"    Selected ({len(idx)}): {[feature_names[i] for i in idx]}")
            print(f"    CV-MAE: {ev['cv_mae']:.4f}")
        except Exception as e:
            print(f"    [ERROR] {label}: {e}")

    _run("SFS Forward",  sequential_forward_select,  X, y)
    _run("SBS Backward", sequential_backward_select, X, y)

    if _HAS_BORUTA:
        print("  Running Boruta...")
        try:
            idx, _ = boruta_select(X, y)
            ev = evaluate_feature_subset(X, y, idx, "Boruta")
            results.append(ev)
            all_selections["Boruta"] = list(map(int, idx))
            print(f"    Selected ({len(idx)}): {[feature_names[i] for i in idx]}")
            print(f"    CV-MAE: {ev['cv_mae']:.4f}")
        except Exception as e:
            print(f"    [ERROR] Boruta: {e}")
    else:
        print("  [SKIP] Boruta — package not installed")

    # ------------------------------------------------------------------
    print("\n=== BASELINE (1) ===")
    print("  Running Random baseline (100 trials)...")
    ev_rand = evaluate_random_baseline(X, y, k=K_DEFAULT, n_trials=100)
    results.append(ev_rand)
    print(f"    CV-MAE: {ev_rand['cv_mae']:.4f} ± {ev_rand['cv_mae_std']:.4f}")

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

    # Consensus features
    print("\n" + "=" * 90)
    print("CONSENSUS FEATURES")
    print("=" * 90)
    flat = [idx for indices in all_selections.values() for idx in indices]
    feat_counts = Counter(flat)
    n_methods = len(all_selections)

    consensus = []
    for min_count in [max(3, n_methods // 3), 3, 2]:
        consensus = sorted(
            [(feature_names[idx], cnt) for idx, cnt in feat_counts.items() if cnt >= min_count],
            key=lambda x: x[1], reverse=True,
        )
        if consensus:
            print(f"(Features selected by ≥ {min_count}/{n_methods} methods:)")
            break

    for feat, cnt in consensus:
        using = [m for m, ids in all_selections.items() if feature_names.index(feat) in ids]
        print(f"  {feat:40s} {cnt:>2}/{n_methods}  {using}")

    # Save
    save_data = {
        "comparison": results_sorted,
        "selections": {k: [feature_names[i] for i in v]
                       for k, v in all_selections.items()},
        "consensus_features": [f for f, _ in consensus],
        "stability_frequencies": {feature_names[i]: float(stab_freq[i])
                                  for i in range(len(feature_names))},
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_PATH}")

    # Plots
    print("Generating plots...")
    plot_comparison(results_sorted, PLOT_DIR)
    plot_mae_ranked(results_sorted, PLOT_DIR)
    plot_feature_overlap(all_selections, feature_names, PLOT_DIR)
    plot_category_comparison(results_sorted, PLOT_DIR)
    plot_redundancy_scatter(results_sorted, PLOT_DIR)

    print(f"Plots saved → {PLOT_DIR}")

    return results_sorted


if __name__ == "__main__":
    main()
