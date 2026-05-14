"""
Step 2: LLM-Lasso Feature Selection
=====================================
Uses LLM importance scores as penalty weights in weighted LASSO.
Cross-validates η (trust level) to robustly determine how much
to rely on LLM knowledge vs. pure data-driven selection.

Based on: Zhang et al. (2025) "LLM-Lasso"
  - Penalty weight: w_j = (I_j)^(-η)
  - η = 0 → standard LASSO (ignore LLM)
  - η > 0 → trust LLM (important features penalized less)
  - η selected by CV to protect against hallucination

Output: results/feature_selection_results.json + comparison plots
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from scipy.stats import pearsonr

from llm_lasso_config import (
    USER_DATASET_PATH, EXCLUDE_COLS,
    OUTPUT_DIR, SCORES_JSON, SELECTION_RESULTS, PLOT_DIR,
    ETA_CANDIDATES, N_FEATURES_TO_SELECT, N_CV_FOLDS, RANDOM_SEED,
    FEATURE_DESCRIPTIONS,
)

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 150})
sns.set_theme(style="whitegrid")


# ===========================
# Data Loading
# ===========================

def load_data():
    """Load user dataset and return X, y, feature names."""
    df = pd.read_csv(USER_DATASET_PATH)
    feature_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

    # Target: consensus of two raters
    df["Rating1"] = pd.to_numeric(df["Rating1"], errors="coerce")
    df["Rating2"] = pd.to_numeric(df["Rating2"], errors="coerce")
    df["consensus"] = (df["Rating1"] + df["Rating2"]) / 2.0

    # Drop rows with missing target
    valid = df.dropna(subset=["consensus"])

    X = valid[feature_cols].values.astype(float)
    y = valid["consensus"].values.astype(float)
    feature_names = feature_cols

    # Handle NaN features
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    return X, y, feature_names


def load_llm_scores(feature_names):
    """Load LLM importance scores from JSON. Return array aligned with feature_names."""
    with open(SCORES_JSON, "r", encoding="utf-8") as f:
        scores_dict = json.load(f)

    importance = np.ones(len(feature_names)) * 5.0  # default neutral
    for i, feat in enumerate(feature_names):
        if feat in scores_dict:
            importance[i] = scores_dict[feat]["median"]

    return importance


# ===========================
# LLM-Lasso Core
# ===========================

def compute_penalty_weights(importance_scores, eta):
    """
    Compute penalty weights: w_j = (I_j)^(-η)

    Args:
        importance_scores: array of LLM scores (1-10) per feature
        eta: trust hyperparameter (0 = standard LASSO)

    Returns:
        weights: array of penalty weights (lower = feature more likely retained)
    """
    # Normalize importance to [0.1, 1.0] to avoid division issues
    I_min, I_max = importance_scores.min(), importance_scores.max()
    if I_max > I_min:
        I_norm = 0.1 + 0.9 * (importance_scores - I_min) / (I_max - I_min)
    else:
        I_norm = np.ones_like(importance_scores) * 0.5

    weights = I_norm ** (-eta)

    # Normalize so mean weight = 1 (keeps lambda scale comparable)
    weights = weights / weights.mean()

    return weights


def weighted_lasso_cv(X, y, penalty_weights, eta, n_folds=N_CV_FOLDS):
    """
    Run weighted LASSO with given penalty weights.
    Uses LassoCV to find optimal lambda, then evaluates with CV.

    The trick: scale features by 1/w_j before standard LASSO,
    which is equivalent to per-feature penalties w_j * |β_j|.

    Returns:
        cv_mae: cross-validated MAE
        selected_features: indices of non-zero coefficients
        coefs: coefficient values
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply penalty weights by rescaling features: X_tilde = X / w
    # Standard LASSO on X_tilde → equivalent to weighted LASSO on X
    X_weighted = X_scaled / penalty_weights[np.newaxis, :]

    # Use LassoCV to find best alpha
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    y_binned = np.clip(np.round(y).astype(int), 0, 4)

    lasso_cv = LassoCV(
        cv=skf.split(X_weighted, y_binned),
        max_iter=50000,
        random_state=RANDOM_SEED,
        n_alphas=100,
    )
    lasso_cv.fit(X_weighted, y)

    # Transform coefficients back to original scale
    # β_original = β_tilde / w
    coefs_original = lasso_cv.coef_ / penalty_weights

    # Cross-validated MAE
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    final_lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
    cv_scores = cross_val_score(
        final_lasso, X_weighted, y,
        cv=skf.split(X_weighted, y_binned),
        scoring=mae_scorer
    )
    cv_mae = -cv_scores.mean()

    selected = np.where(np.abs(coefs_original) > 1e-8)[0]

    return cv_mae, selected, coefs_original, lasso_cv.alpha_


def search_eta(X, y, importance_scores):
    """
    Search for optimal η via cross-validation.
    This is the key robustness mechanism of LLM-Lasso.

    Returns:
        best_eta, all_results
    """
    results = []
    print("\n" + "=" * 60)
    print("SEARCHING OPTIMAL η (trust level for LLM)")
    print("=" * 60)

    for eta in ETA_CANDIDATES:
        weights = compute_penalty_weights(importance_scores, eta)
        cv_mae, selected, coefs, alpha = weighted_lasso_cv(X, y, weights, eta)
        n_selected = len(selected)

        results.append({
            "eta": eta,
            "cv_mae": cv_mae,
            "n_selected": n_selected,
            "selected_indices": selected.tolist(),
            "alpha": float(alpha),
        })

        trust_label = "standard LASSO" if eta == 0 else f"trust LLM (η={eta})"
        print(f"  η={eta:5.2f} | CV-MAE={cv_mae:.4f} | "
              f"#features={n_selected:3d} | {trust_label}")

    # Select best η by lowest CV-MAE
    best = min(results, key=lambda r: r["cv_mae"])
    print(f"\n  → Best η = {best['eta']} (CV-MAE = {best['cv_mae']:.4f}, "
          f"{best['n_selected']} features)")

    return best["eta"], results


# ===========================
# Comparison with Pearson-only
# ===========================

def pearson_baseline(X, y, feature_names, k=8):
    """Current method: select top-k features by |Pearson r| with target."""
    correlations = []
    for j in range(X.shape[1]):
        if np.std(X[:, j]) < 1e-10:  # constant column
            correlations.append(0.0)
        else:
            r, _ = pearsonr(X[:, j], y)
            correlations.append(abs(r) if not np.isnan(r) else 0.0)
    correlations = np.array(correlations)
    top_k = np.argsort(correlations)[-k:][::-1]
    return top_k, correlations


# ===========================
# Visualization
# ===========================

def plot_eta_search(results, output_dir):
    """Plot CV-MAE vs η."""
    etas = [r["eta"] for r in results]
    maes = [r["cv_mae"] for r in results]
    n_feats = [r["n_selected"] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#2c3e50"
    color2 = "#e74c3c"

    ax1.plot(etas, maes, "o-", color=color1, linewidth=2, markersize=8, label="CV-MAE")
    ax1.set_xlabel("η (LLM trust level)", fontsize=12)
    ax1.set_ylabel("Cross-Validated MAE ↓", fontsize=12, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    # Mark best
    best_idx = np.argmin(maes)
    ax1.scatter([etas[best_idx]], [maes[best_idx]], s=200, c="gold",
                edgecolors="black", zorder=5, label=f"Best η={etas[best_idx]}")

    ax2 = ax1.twinx()
    ax2.bar(etas, n_feats, alpha=0.2, color=color2, width=0.15, label="#Features")
    ax2.set_ylabel("# Selected Features", fontsize=12, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotations
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="η=0 (standard LASSO)")
    ax1.set_title("LLM-Lasso: Cross-Validation for η\n(η=0 ignores LLM, η>0 trusts LLM)",
                   fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "eta_search.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "eta_search.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_comparison(feature_names, importance_scores, pearson_corrs,
                            llm_lasso_selected, pearson_selected, output_dir):
    """Compare LLM-Lasso vs Pearson-only feature selection."""
    n = len(feature_names)
    df = pd.DataFrame({
        "Feature": feature_names,
        "LLM Score": importance_scores,
        "|Pearson r|": pearson_corrs,
        "LLM-Lasso": ["Selected" if i in llm_lasso_selected else "" for i in range(n)],
        "Pearson Top-8": ["Selected" if i in pearson_selected else "" for i in range(n)],
    })
    df = df.sort_values("LLM Score", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n * 0.25)))

    # Left: LLM importance scores
    ax = axes[0]
    colors = ["#27ae60" if f in [feature_names[i] for i in llm_lasso_selected]
              else "#bdc3c7" for f in df["Feature"]]
    ax.barh(range(n), df["LLM Score"].values, color=colors)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["Feature"].values, fontsize=7)
    ax.set_xlabel("LLM Importance Score (1-10)")
    ax.set_title("LLM Feature Importance\n(green = LLM-Lasso selected)", fontsize=11)
    ax.set_xlim(0, 11)

    # Right: Pearson correlation
    ax = axes[1]
    colors2 = ["#2980b9" if f in [feature_names[i] for i in pearson_selected]
               else "#bdc3c7" for f in df["Feature"]]
    ax.barh(range(n), df["|Pearson r|"].values, color=colors2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df["Feature"].values, fontsize=7)
    ax.set_xlabel("|Pearson Correlation| with Consensus")
    ax.set_title("Pearson Correlation\n(blue = Pearson top-8 selected)", fontsize=11)
    ax.set_xlim(0, 1)

    plt.suptitle("Feature Selection Comparison: LLM-Lasso vs Pearson-only",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_comparison.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_penalty_weights(feature_names, importance_scores, best_eta, output_dir):
    """Show how LLM scores translate to penalty weights."""
    weights = compute_penalty_weights(importance_scores, best_eta)
    order = np.argsort(weights)

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.25)))
    colors = ["#e74c3c" if w > 1.0 else "#27ae60" for w in weights[order]]
    ax.barh(range(len(order)), weights[order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=7)
    ax.axvline(x=1.0, color="black", linestyle="--", alpha=0.5, label="w=1 (neutral)")
    ax.set_xlabel(f"Penalty Weight w_j = (I_j)^(-{best_eta})")
    ax.set_title(f"LLM-Lasso Penalty Weights (η={best_eta})\n"
                 f"Green = lower penalty (LLM says important), Red = higher penalty",
                 fontsize=11, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "penalty_weights.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "penalty_weights.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_correlation_matrix(X, feature_names, selected_indices, output_dir):
    """Show inter-feature correlation matrix, highlighting selected features."""
    sel_names = [feature_names[i] for i in selected_indices]
    sel_X = X[:, selected_indices]
    corr = np.corrcoef(sel_X.T)

    fig, ax = plt.subplots(figsize=(max(6, len(sel_names) * 0.6),
                                    max(5, len(sel_names) * 0.5)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                xticklabels=sel_names, yticklabels=sel_names,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Inter-Feature Correlation Matrix (LLM-Lasso Selected)",
                 fontsize=12, fontweight="bold")
    plt.xticks(fontsize=7, rotation=45, ha="right")
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "selected_correlation_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "selected_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


# ===========================
# Main
# ===========================

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    X, y, feature_names = load_data()
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Target range: {y.min():.1f} - {y.max():.1f}")

    # 2. Load LLM scores
    print(f"\nLoading LLM scores from {SCORES_JSON}...")
    importance = load_llm_scores(feature_names)
    print(f"  Score range: {importance.min():.1f} - {importance.max():.1f}")

    # 3. Pearson baseline (current method)
    print("\nComputing Pearson baseline (current method)...")
    pearson_top8, pearson_corrs = pearson_baseline(X, y, feature_names, k=8)
    print(f"  Pearson top-8: {[feature_names[i] for i in pearson_top8]}")

    # 4. Search optimal η
    best_eta, eta_results = search_eta(X, y, importance)

    # 5. Run final LLM-Lasso with best η
    print(f"\nRunning final LLM-Lasso with η={best_eta}...")
    weights = compute_penalty_weights(importance, best_eta)
    cv_mae, selected, coefs, alpha = weighted_lasso_cv(X, y, weights, best_eta)

    selected_names = [feature_names[i] for i in selected]
    print(f"  Selected {len(selected)} features: {selected_names}")
    print(f"  CV-MAE: {cv_mae:.4f}")

    # 6. Compare with standard LASSO (η=0)
    weights_std = compute_penalty_weights(importance, 0.0)
    cv_mae_std, selected_std, _, _ = weighted_lasso_cv(X, y, weights_std, 0.0)
    print(f"\n  Standard LASSO (η=0): {len(selected_std)} features, CV-MAE={cv_mae_std:.4f}")
    print(f"  LLM-Lasso (η={best_eta}): {len(selected)} features, CV-MAE={cv_mae:.4f}")
    improvement = cv_mae_std - cv_mae
    print(f"  Improvement: {improvement:+.4f} MAE")

    # 7. Save results
    result_data = {
        "best_eta": best_eta,
        "eta_search_results": eta_results,
        "llm_lasso_selected_features": selected_names,
        "llm_lasso_cv_mae": cv_mae,
        "llm_lasso_n_features": len(selected),
        "standard_lasso_cv_mae": cv_mae_std,
        "standard_lasso_n_features": len(selected_std),
        "standard_lasso_selected": [feature_names[i] for i in selected_std],
        "pearson_top8": [feature_names[i] for i in pearson_top8],
        "pearson_correlations": {feature_names[i]: float(pearson_corrs[i])
                                 for i in range(len(feature_names))},
        "llm_scores": {feature_names[i]: float(importance[i])
                       for i in range(len(feature_names))},
        "improvement_over_standard_lasso": improvement,
        "feature_coefficients": {feature_names[i]: float(coefs[i])
                                 for i in range(len(feature_names)) if abs(coefs[i]) > 1e-8},
    }

    with open(SELECTION_RESULTS, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {SELECTION_RESULTS}")

    # 8. Generate plots
    print("\nGenerating comparison plots...")
    plot_eta_search(eta_results, PLOT_DIR)
    plot_feature_comparison(feature_names, importance, pearson_corrs,
                            selected, pearson_top8, PLOT_DIR)
    plot_penalty_weights(feature_names, importance, best_eta, PLOT_DIR)
    if len(selected) > 1:
        plot_feature_correlation_matrix(X, feature_names, selected, PLOT_DIR)
    print(f"Plots saved to: {PLOT_DIR}")

    # 9. Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: LLM-Lasso vs Pearson-only vs Standard LASSO")
    print("=" * 70)
    print(f"{'Method':<25} {'#Features':>10} {'CV-MAE':>10}")
    print("-" * 50)
    print(f"{'Pearson top-8 (current)':<25} {'8':>10} {'N/A':>10}")
    print(f"{'Standard LASSO (η=0)':<25} {len(selected_std):>10} {cv_mae_std:>10.4f}")
    print(f"{'LLM-Lasso (η=' + str(best_eta) + ')':<25} {len(selected):>10} {cv_mae:>10.4f}")

    # Show overlap between methods
    pearson_set = set(feature_names[i] for i in pearson_top8)
    llm_lasso_set = set(selected_names)
    overlap = pearson_set & llm_lasso_set
    only_pearson = pearson_set - llm_lasso_set
    only_llm = llm_lasso_set - pearson_set

    print(f"\nOverlap with Pearson top-8: {len(overlap)}")
    if overlap:
        print(f"  Both selected: {sorted(overlap)}")
    if only_pearson:
        print(f"  Only Pearson:   {sorted(only_pearson)}")
    if only_llm:
        print(f"  Only LLM-Lasso: {sorted(only_llm)}")

    return result_data


if __name__ == "__main__":
    main()
