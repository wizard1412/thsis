"""
LLM-Lasso Feature Selection — CVE Comparison Variant
=====================================================
Implements two eta (η) selection criteria and compares them:

  1. CV-MAE (current method):
       Pick η with lowest cross-validated Mean Absolute Error at its best lambda.

  2. CVE-area (original paper method — Zhang et al. 2025):
       For each η, compute the full regularization path as a curve
       (n_features, CV_MSE).  Select η whose curve has the largest
       positive signed area *above* the standard-LASSO (η=0) reference curve.
       Positive area = LLM-LASSO achieves lower error at the same sparsity level.

Reuses LLM scores from experiments_llm_lasso_deepseek70b.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.linear_model import Lasso, LassoCV, lasso_path as sklearn_lasso_path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from scipy.stats import pearsonr

from llm_lasso_config import (
    USER_DATASET_PATH, EXCLUDE_COLS,
    OUTPUT_DIR, SCORES_JSON, SELECTION_RESULTS, PLOT_DIR,
    ETA_CANDIDATES, N_CV_FOLDS, RANDOM_SEED, N_ALPHAS_PATH,
    FEATURE_DESCRIPTIONS,
)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 150})
sns.set_theme(style="whitegrid")


# ===========================
# Data Loading
# ===========================

def load_data():
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
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return X, y, feature_cols


def load_llm_scores(feature_names):
    with open(SCORES_JSON, "r", encoding="utf-8") as f:
        scores_dict = json.load(f)

    importance = np.ones(len(feature_names)) * 5.0
    for i, feat in enumerate(feature_names):
        if feat in scores_dict:
            importance[i] = scores_dict[feat]["median"]
    return importance


# ===========================
# Penalty Weights
# ===========================

def compute_penalty_weights(importance_scores, eta):
    """w_j = (I_j_normalized)^(-η).  η=0 → uniform (standard LASSO)."""
    I_min, I_max = importance_scores.min(), importance_scores.max()
    if I_max > I_min:
        I_norm = 0.1 + 0.9 * (importance_scores - I_min) / (I_max - I_min)
    else:
        I_norm = np.ones_like(importance_scores) * 0.5
    weights = I_norm ** (-eta)
    return weights / weights.mean()


# ===========================
# Selection Method 1: CV-MAE
# ===========================

def weighted_lasso_cv_mae(X, y, penalty_weights, cv_splits):
    """
    Fit weighted LASSO (via feature rescaling) and return CV-MAE at best lambda.
    Uses pre-computed CV splits for consistency.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_weighted = X_scaled / penalty_weights[np.newaxis, :]

    lasso_cv = LassoCV(
        cv=cv_splits,
        max_iter=50000,
        random_state=RANDOM_SEED,
        n_alphas=100,
    )
    lasso_cv.fit(X_weighted, y)

    coefs_original = lasso_cv.coef_ / penalty_weights
    selected = np.where(np.abs(coefs_original) > 1e-8)[0]

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    final_lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    cv_scores = cross_val_score(final_lasso, X_weighted, y, cv=cv_splits, scoring=mae_scorer)
    cv_mae = -cv_scores.mean()

    return cv_mae, selected, coefs_original, lasso_cv.alpha_


# ===========================
# Selection Method 2: CVE Area
# ===========================

def cve_area(cvm, non_zero, ref_cvm, ref_non_zero):
    """
    Signed area between the LLM-LASSO CV curve and the standard-LASSO reference curve,
    measured over the (n_features) axis.  Directly from Zhang et al. (2025).

    Positive = LLM-LASSO has lower CV error at the same feature count → better.
    """
    df1 = pd.DataFrame({'x': ref_non_zero, 'y': ref_cvm})
    df1 = df1.groupby('x', as_index=False)['y'].min()
    df2 = pd.DataFrame({'x': non_zero, 'y': cvm})
    df2 = df2.groupby('x', as_index=False)['y'].min()

    x1 = df1['x'].values.astype(float)
    y1 = df1['y'].values
    x2 = df2['x'].values.astype(float)
    y2 = df2['y'].values

    if len(x1) < 2 or len(x2) < 2:
        return 0.0

    interp_func = interp1d(x1, y1, bounds_error=False, fill_value='extrapolate')
    y1_interp = interp_func(x2)

    area = 0.0
    for i in range(len(x2) - 1):
        width = x2[i + 1] - x2[i]
        height = ((y1_interp[i] - y2[i]) + (y1_interp[i + 1] - y2[i + 1])) / 2
        area += width * height
    return float(area)


def get_cv_path_for_cve(X_weighted_scaled, y, cv_splits):
    """
    Extract the regularization path as (cv_mse, n_features) arrays, needed for CVE.

    Uses LassoCV's mse_path_ (CV MSE for each alpha across folds) and lasso_path
    on the full dataset to count non-zero features at each alpha point.

    Returns:
        cv_mse    : array (n_alphas,) — mean CV MSE at each alpha
        non_zeros : array (n_alphas,) — non-zero feature count at each alpha
    """
    lasso_cv = LassoCV(
        n_alphas=N_ALPHAS_PATH,
        cv=cv_splits,
        max_iter=50000,
        random_state=RANDOM_SEED,
    )
    lasso_cv.fit(X_weighted_scaled, y)

    alphas = lasso_cv.alphas_                         # decreasing, shape (n_alphas,)
    cv_mse = lasso_cv.mse_path_.mean(axis=1)          # mean over folds, shape (n_alphas,)

    # Count non-zeros along the path on the full dataset
    _, coefs_path, _ = sklearn_lasso_path(
        X_weighted_scaled, y, alphas=alphas, max_iter=10000
    )
    non_zeros = (np.abs(coefs_path) > 1e-8).sum(axis=0)  # shape (n_alphas,)

    return cv_mse, non_zeros


# ===========================
# Joint Search
# ===========================

def search_eta(X, y, importance_scores):
    """
    Search for optimal η using both CV-MAE and CVE-area criteria.

    Returns:
        best_mae   : result dict for the η chosen by CV-MAE
        best_cve   : result dict for the η chosen by CVE-area
        all_results: list of dicts with metrics for every η candidate
        cv_paths   : dict mapping eta → (cv_mse_path, non_zeros_path) for plotting
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_binned = np.clip(np.round(y).astype(int), 0, 4)
    cv_splits = list(skf.split(X_scaled, y_binned))   # fixed splits for all etas

    results = []
    cv_paths = {}
    ref_cv_mse = None
    ref_non_zeros = None

    print("\n" + "=" * 70)
    print("SEARCHING OPTIMAL η — CV-MAE  vs  CVE-area (original paper method)")
    print("=" * 70)
    print(f"  {'η':>5}  {'CV-MAE':>8}  {'CVE-area':>12}  {'#features':>9}  note")
    print("  " + "-" * 55)

    for eta in ETA_CANDIDATES:
        weights = compute_penalty_weights(importance_scores, eta)
        X_weighted = X_scaled / weights[np.newaxis, :]

        # --- Method 1: CV-MAE ---
        cv_mae, selected, coefs, alpha = weighted_lasso_cv_mae(X, y, weights, cv_splits)

        # --- Method 2: CVE path ---
        cv_mse_path, non_zeros_path = get_cv_path_for_cve(X_weighted, y, cv_splits)
        cv_paths[eta] = (cv_mse_path, non_zeros_path)

        if ref_cv_mse is None:          # eta==0 sets the reference
            ref_cv_mse = cv_mse_path
            ref_non_zeros = non_zeros_path

        area = (cve_area(cv_mse_path, non_zeros_path, ref_cv_mse, ref_non_zeros)
                if eta > 0 else 0.0)

        note = "reference" if eta == 0 else ""
        print(f"  {eta:>5.2f}  {cv_mae:>8.4f}  {area:>+12.6f}  {len(selected):>9}  {note}")

        results.append({
            "eta": eta,
            "cv_mae": cv_mae,
            "cve_area": area,
            "n_selected": len(selected),
            "selected_indices": selected.tolist(),
            "alpha": float(alpha),
        })

    best_mae = min(results, key=lambda r: r["cv_mae"])
    best_cve = max(results, key=lambda r: r["cve_area"])

    print()
    print(f"  → Best η by CV-MAE  : η={best_mae['eta']:<5}  "
          f"CV-MAE={best_mae['cv_mae']:.4f}  #features={best_mae['n_selected']}")
    print(f"  → Best η by CVE-area: η={best_cve['eta']:<5}  "
          f"CVE-area={best_cve['cve_area']:+.6f}  #features={best_cve['n_selected']}")

    if best_mae["eta"] == best_cve["eta"]:
        print("  ✓ Both criteria agree on the same η.")
    else:
        print("  ✗ Criteria disagree — see plots for details.")

    return best_mae, best_cve, results, cv_paths


# ===========================
# Final Model Fit
# ===========================

def fit_final_model(X, y, importance_scores, eta):
    """Fit the final weighted LASSO with the selected η and return results."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_binned = np.clip(np.round(y).astype(int), 0, 4)
    cv_splits = list(skf.split(X_scaled, y_binned))

    weights = compute_penalty_weights(importance_scores, eta)
    cv_mae, selected, coefs, alpha = weighted_lasso_cv_mae(X, y, weights, cv_splits)
    return cv_mae, selected, coefs, alpha


# ===========================
# Visualization
# ===========================

def plot_eta_comparison(results, cv_paths, best_mae, best_cve, output_dir):
    """
    Four-panel figure:
      Top-left : CV-MAE vs η  (with best marked)
      Top-right: CVE-area vs η  (with best marked)
      Bottom   : regularization path curves for selected etas (CV_MSE vs n_features)
    """
    etas = [r["eta"] for r in results]
    maes = [r["cv_mae"] for r in results]
    areas = [r["cve_area"] for r in results]
    n_feats = [r["n_selected"] for r in results]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # --- Panel 1: CV-MAE ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1b = ax1.twinx()
    ax1.plot(etas, maes, "o-", color="#2c3e50", linewidth=2, markersize=7, label="CV-MAE")
    ax1b.bar(etas, n_feats, alpha=0.15, color="#e74c3c", width=0.18, label="#Features")
    best_idx = next(i for i, r in enumerate(results) if r["eta"] == best_mae["eta"])
    ax1.scatter([etas[best_idx]], [maes[best_idx]], s=200, c="gold",
                edgecolors="black", zorder=5, label=f"Best η={etas[best_idx]}")
    ax1.set_xlabel("η")
    ax1.set_ylabel("CV-MAE ↓", color="#2c3e50")
    ax1b.set_ylabel("# Features", color="#e74c3c")
    ax1.set_title("Method 1: Select by CV-MAE", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)

    # --- Panel 2: CVE-area ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2b = ax2.twinx()
    ax2.plot(etas, areas, "s-", color="#27ae60", linewidth=2, markersize=7, label="CVE-area")
    ax2b.bar(etas, n_feats, alpha=0.15, color="#e74c3c", width=0.18, label="#Features")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    best_idx2 = next(i for i, r in enumerate(results) if r["eta"] == best_cve["eta"])
    ax2.scatter([etas[best_idx2]], [areas[best_idx2]], s=200, c="gold",
                edgecolors="black", zorder=5, label=f"Best η={etas[best_idx2]}")
    ax2.set_xlabel("η")
    ax2.set_ylabel("CVE area (↑ better)", color="#27ae60")
    ax2b.set_ylabel("# Features", color="#e74c3c")
    ax2.set_title("Method 2: Select by CVE-area (original paper)", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8)

    # --- Panel 3: Path curves — highlight the two selected etas ---
    ax3 = fig.add_subplot(gs[1, :])
    highlight = {best_mae["eta"]: ("CV-MAE best η", "#2c3e50", "-"),
                 best_cve["eta"]: ("CVE best η", "#27ae60", "--")}
    # reference (eta=0)
    ref_mse, ref_nz = cv_paths[0.0]
    ax3.plot(ref_nz, ref_mse, color="gray", linewidth=1.5, alpha=0.7,
             linestyle=":", label="η=0 (standard LASSO, reference)")

    for eta, (cv_mse, non_zeros) in cv_paths.items():
        if eta == 0.0:
            continue
        if eta in highlight:
            label, color, ls = highlight[eta]
            ax3.plot(non_zeros, cv_mse, color=color, linewidth=2.5,
                     linestyle=ls, label=f"η={eta} ({label})", zorder=5)
        else:
            ax3.plot(non_zeros, cv_mse, color="#bdc3c7", linewidth=0.8, alpha=0.5)

    ax3.set_xlabel("Number of selected features")
    ax3.set_ylabel("CV MSE (mean over folds)")
    ax3.set_title("Regularization Path: CV MSE vs #Features\n"
                  "(CVE = signed area between colored curve and gray reference)",
                  fontweight="bold")
    ax3.legend(fontsize=9)

    plt.suptitle("LLM-LASSO η Selection: CV-MAE vs CVE-area comparison",
                 fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(output_dir / "eta_selection_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "eta_selection_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'eta_selection_comparison.png'}")


def plot_feature_sets(feature_names, importance, result_mae, result_cve, output_dir):
    """Bar chart comparing the two selected feature sets."""
    set_mae = set(feature_names[i] for i in result_mae["selected_indices"])
    set_cve = set(feature_names[i] for i in result_cve["selected_indices"])
    all_feat = sorted(set_mae | set_cve,
                      key=lambda f: importance[feature_names.index(f)], reverse=True)

    colors = []
    for f in all_feat:
        in_mae = f in set_mae
        in_cve = f in set_cve
        if in_mae and in_cve:
            colors.append("#2ecc71")      # both
        elif in_mae:
            colors.append("#3498db")      # MAE only
        elif in_cve:
            colors.append("#e67e22")      # CVE only

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_feat) * 0.35)))
    y_pos = range(len(all_feat))
    imp_vals = [importance[feature_names.index(f)] for f in all_feat]
    ax.barh(y_pos, imp_vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_feat, fontsize=9)
    ax.set_xlabel("LLM Importance Score")
    ax.set_title("Features selected by each criterion\n"
                 "Green=both, Blue=CV-MAE only, Orange=CVE only",
                 fontweight="bold")
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#2ecc71", label=f"Both (n={len(set_mae & set_cve)})"),
        Patch(color="#3498db", label=f"CV-MAE only (η={result_mae['eta']}, n={len(set_mae - set_cve)})"),
        Patch(color="#e67e22", label=f"CVE only (η={result_cve['eta']}, n={len(set_cve - set_mae)})"),
    ]
    ax.legend(handles=legend_handles, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_set_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "feature_set_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir / 'feature_set_comparison.png'}")


# ===========================
# Main
# ===========================

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X, y, feature_names = load_data()
    feature_names = list(feature_names)
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")

    print(f"\nLoading LLM scores from {SCORES_JSON}...")
    importance = load_llm_scores(feature_names)
    print(f"  Score range: {importance.min():.1f} – {importance.max():.1f}")

    # Search η with both criteria
    best_mae, best_cve, all_results, cv_paths = search_eta(X, y, importance)

    # Final model fits (needed to get coefs for saving)
    print(f"\nFitting final model — CV-MAE best η={best_mae['eta']}...")
    cv_mae_final, sel_mae, coefs_mae, _ = fit_final_model(X, y, importance, best_mae["eta"])
    sel_names_mae = [feature_names[i] for i in sel_mae]

    print(f"Fitting final model — CVE best η={best_cve['eta']}...")
    cv_mae_cve, sel_cve, coefs_cve, _ = fit_final_model(X, y, importance, best_cve["eta"])
    sel_names_cve = [feature_names[i] for i in sel_cve]

    # Standard LASSO (η=0) for reference
    std_result = next(r for r in all_results if r["eta"] == 0.0)
    _, sel_std, _, _ = fit_final_model(X, y, importance, 0.0)
    sel_names_std = [feature_names[i] for i in sel_std]

    # Print summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Method':<35} {'η':>5}  {'#feat':>5}  {'CV-MAE':>8}")
    print("  " + "-" * 58)
    print(f"  {'Standard LASSO (η=0)':<35} {'0.0':>5}  {len(sel_std):>5}  {std_result['cv_mae']:>8.4f}")
    print(f"  {'LLM-LASSO (CV-MAE selection)':<35} {best_mae['eta']:>5}  {len(sel_mae):>5}  {cv_mae_final:>8.4f}")
    print(f"  {'LLM-LASSO (CVE-area selection)':<35} {best_cve['eta']:>5}  {len(sel_cve):>5}  {cv_mae_cve:>8.4f}")

    print(f"\n  Standard LASSO features:  {sel_names_std}")
    print(f"\n  CV-MAE selected features: {sel_names_mae}")
    print(f"\n  CVE-area selected features: {sel_names_cve}")

    overlap = set(sel_names_mae) & set(sel_names_cve)
    only_mae = set(sel_names_mae) - set(sel_names_cve)
    only_cve = set(sel_names_cve) - set(sel_names_mae)
    print(f"\n  Feature set overlap: {len(overlap)}/{max(len(sel_names_mae), len(sel_names_cve))}")
    if overlap:       print(f"    Both:       {sorted(overlap)}")
    if only_mae:      print(f"    MAE only:   {sorted(only_mae)}")
    if only_cve:      print(f"    CVE only:   {sorted(only_cve)}")

    # Save results
    result_data = {
        "cv_mae_selection": {
            "best_eta": best_mae["eta"],
            "cv_mae": cv_mae_final,
            "n_features": len(sel_mae),
            "selected_features": sel_names_mae,
            "feature_coefficients": {
                feature_names[i]: float(coefs_mae[i])
                for i in range(len(feature_names)) if abs(coefs_mae[i]) > 1e-8
            },
        },
        "cve_area_selection": {
            "best_eta": best_cve["eta"],
            "cve_area": best_cve["cve_area"],
            "cv_mae": cv_mae_cve,
            "n_features": len(sel_cve),
            "selected_features": sel_names_cve,
            "feature_coefficients": {
                feature_names[i]: float(coefs_cve[i])
                for i in range(len(feature_names)) if abs(coefs_cve[i]) > 1e-8
            },
        },
        "standard_lasso": {
            "eta": 0.0,
            "cv_mae": std_result["cv_mae"],
            "n_features": len(sel_std),
            "selected_features": sel_names_std,
        },
        "eta_search_results": all_results,
    }

    with open(SELECTION_RESULTS, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {SELECTION_RESULTS}")

    # Plots
    print("\nGenerating plots...")
    best_mae_full = {**best_mae, "selected_indices": [feature_names.index(n) for n in sel_names_mae]}
    best_cve_full = {**best_cve, "selected_indices": [feature_names.index(n) for n in sel_names_cve]}
    plot_eta_comparison(all_results, cv_paths, best_mae, best_cve, PLOT_DIR)
    plot_feature_sets(feature_names, importance, best_mae_full, best_cve_full, PLOT_DIR)
    print(f"Plots saved to: {PLOT_DIR}")

    return result_data


if __name__ == "__main__":
    main()
