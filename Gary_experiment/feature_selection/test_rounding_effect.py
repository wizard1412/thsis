"""
Test: does standard rounding vs banker's rounding change selected features?

Affected methods (those that use _bin_y internally):
  - RFECV (RF)    : trains RandomForestClassifier on binned y
  - Fisher Score  : computes between/within-class variance on binned y
  - ReliefF       : computes near-hit/miss on binned y

Runs each method with both rounding strategies and compares selected features.
"""

import json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE     = Path(__file__).resolve().parent
FEAT_CSV = HERE / "features_dataset_v2.csv"
FS_JSON  = HERE / "traditional_fs_results_v2" / "traditional_selection_results.json"
N_CV_FOLDS  = 5
RANDOM_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(FEAT_CSV)
    drop_cols = {"hand", "filename", "Rating1", "Rating2", "Rating", "tap_amplitudes"}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=["Rating"]).reset_index(drop=True)
    col_means = np.nanmean(df[feat_cols].values, axis=0)
    X = df[feat_cols].values.astype(float)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    y = df["Rating"].values.astype(float)
    return X, y, feat_cols

# ─────────────────────────────────────────────────────────────────────────────
# Two rounding functions
# ─────────────────────────────────────────────────────────────────────────────
def bin_y_banker(y):
    """np.round — banker's (half to even)"""
    return np.clip(np.round(y).astype(int), 0, 4)

def bin_y_standard(y):
    """floor(y+0.5) — standard (half up)"""
    return np.clip(np.floor(y + 0.5).astype(int), 0, 4)

# ─────────────────────────────────────────────────────────────────────────────
# RFECV (RF)
# ─────────────────────────────────────────────────────────────────────────────
def rfecv_rf(X, y, bin_fn):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_binned = bin_fn(y)
    skf      = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
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
    return set(np.where(rfecv.support_)[0])

# ─────────────────────────────────────────────────────────────────────────────
# Fisher Score
# ─────────────────────────────────────────────────────────────────────────────
def _cv_select_k(X, y, ranked, k_min=2, k_max=20):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_k, best_score = k_min, -np.inf
    for k in range(k_min, min(k_max, len(ranked)) + 1):
        idx = np.asarray(ranked[:k])
        s = cross_val_score(Ridge(), X_sc[:, idx], y,
                            cv=kf, scoring="neg_mean_absolute_error").mean()
        if s > best_score:
            best_score = s; best_k = k
    return best_k

def fisher_score(X, y, bin_fn):
    y_binned = bin_fn(y)
    classes  = np.unique(y_binned)
    mu_global = X.mean(axis=0)
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        num = den = 0.0
        for c in classes:
            mask = y_binned == c
            n_c = mask.sum()
            if n_c == 0: continue
            mu_c  = X[mask, j].mean()
            var_c = X[mask, j].var()
            num  += n_c * (mu_c - mu_global[j]) ** 2
            den  += n_c * var_c
        scores[j] = num / (den + 1e-10)
    ranked = np.argsort(scores)[::-1]
    k = _cv_select_k(X, y, ranked)
    return set(ranked[:k])

# ─────────────────────────────────────────────────────────────────────────────
# ReliefF
# ─────────────────────────────────────────────────────────────────────────────
def relieff(X, y, bin_fn, n_neighbors=5):
    n_samples, n_features = X.shape
    X_rng  = X.max(axis=0) - X.min(axis=0)
    X_rng[X_rng < 1e-10] = 1.0
    X_norm = (X - X.min(axis=0)) / X_rng

    y_binned     = bin_fn(y)
    classes      = np.unique(y_binned)
    class_probs  = {c: float((y_binned == c).mean()) for c in classes}
    weights      = np.zeros(n_features)
    rng          = np.random.default_rng(RANDOM_SEED)

    for _ in range(n_samples):
        i  = int(rng.integers(n_samples))
        xi = X_norm[i]
        ci = y_binned[i]
        dist = np.sum(np.abs(X_norm - xi), axis=1)
        dist[i] = np.inf

        hit_mask = (y_binned == ci).copy(); hit_mask[i] = False
        if hit_mask.any():
            for h in np.argsort(np.where(hit_mask, dist, np.inf))[:n_neighbors]:
                weights -= np.abs(xi - X_norm[h]) / n_samples

        p_ci = class_probs[ci]
        for c in classes:
            if c == ci: continue
            miss_mask = y_binned == c
            if not miss_mask.any(): continue
            w_c = class_probs[c] / max(1.0 - p_ci, 1e-10)
            for m in np.argsort(np.where(miss_mask, dist, np.inf))[:n_neighbors]:
                weights += w_c * np.abs(xi - X_norm[m]) / n_samples

    ranked = np.argsort(weights)[::-1]
    k = _cv_select_k(X, y, ranked)
    return set(ranked[:k])

# ─────────────────────────────────────────────────────────────────────────────
# Compare
# ─────────────────────────────────────────────────────────────────────────────
def compare(name, idx_banker, idx_standard, feat_cols):
    feats_b = set(feat_cols[i] for i in idx_banker)
    feats_s = set(feat_cols[i] for i in idx_standard)
    only_b  = sorted(feats_b - feats_s)
    only_s  = sorted(feats_s - feats_b)
    same    = sorted(feats_b & feats_s)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Banker's  : {len(feats_b)} features")
    print(f"  Standard  : {len(feats_s)} features")
    print(f"  共同選出  : {len(same)}")
    print(f"  只有 banker 選: {len(only_b)}")
    if only_b: [print(f"    - {f}") for f in only_b]
    print(f"  只有 standard 選: {len(only_s)}")
    if only_s: [print(f"    + {f}") for f in only_s]

    return {"banker": feats_b, "standard": feats_s,
            "only_banker": only_b, "only_standard": only_s}


if __name__ == "__main__":
    print("Loading data...")
    X, y, feat_cols = load_data()
    feat_cols = np.array(feat_cols)

    print(f"\ny distribution (unique values): {np.unique(y)}")
    b_arr = bin_y_banker(y)
    s_arr = bin_y_standard(y)
    diff  = np.where(b_arr != s_arr)[0]
    print(f"Samples where rounding differs: {len(diff)}/53")
    for i in diff:
        print(f"  sample {i:2d}  y={y[i]}  banker->{b_arr[i]}  standard->{s_arr[i]}")

    # ── RFECV (RF) ────────────────────────────────────────────────────────────
    print("\n[1/3] RFECV (RF) with banker's rounding...")
    idx_rf_b = rfecv_rf(X, y, bin_y_banker)
    print(f"  Selected: {len(idx_rf_b)} features")

    print("[1/3] RFECV (RF) with standard rounding...")
    idx_rf_s = rfecv_rf(X, y, bin_y_standard)
    print(f"  Selected: {len(idx_rf_s)} features")

    r_rf = compare("RFECV (RF)", idx_rf_b, idx_rf_s, feat_cols)

    # ── Fisher Score ──────────────────────────────────────────────────────────
    print("\n[2/3] Fisher Score with banker's rounding...")
    idx_fs_b = fisher_score(X, y, bin_y_banker)
    print(f"  Selected: {len(idx_fs_b)} features")

    print("[2/3] Fisher Score with standard rounding...")
    idx_fs_s = fisher_score(X, y, bin_y_standard)
    print(f"  Selected: {len(idx_fs_s)} features")

    r_fs = compare("Fisher Score", idx_fs_b, idx_fs_s, feat_cols)

    # ── ReliefF ───────────────────────────────────────────────────────────────
    print("\n[3/3] ReliefF with banker's rounding...")
    idx_rf2_b = relieff(X, y, bin_y_banker)
    print(f"  Selected: {len(idx_rf2_b)} features")

    print("[3/3] ReliefF with standard rounding...")
    idx_rf2_s = relieff(X, y, bin_y_standard)
    print(f"  Selected: {len(idx_rf2_s)} features")

    r_rf2 = compare("ReliefF", idx_rf2_b, idx_rf2_s, feat_cols)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    # Also compare against original JSON selections
    with open(FS_JSON) as f:
        orig = json.load(f)["selections"]

    for name, key, res in [
        ("RFECV (RF)",    "RFECV (RF)",    r_rf),
        ("Fisher Score",  "Fisher Score",  r_fs),
        ("ReliefF",       "ReliefF",       r_rf2),
    ]:
        orig_set = set(orig.get(key, []))
        match_b  = len(orig_set & res["banker"])   / max(len(orig_set), 1)
        match_s  = len(orig_set & res["standard"]) / max(len(orig_set), 1)
        changed  = len(res["only_banker"]) + len(res["only_standard"])
        print(f"\n  {name}")
        print(f"    Banker vs Standard: {changed} feature(s) differ")
        print(f"    Overlap with original JSON — banker: {match_b:.0%}  standard: {match_s:.0%}")
