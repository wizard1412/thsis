"""
RAG vs No-RAG Comparison
=========================
Reads results from results/rag/ and results/no_rag/ and generates
comparison plots:
  - Feature-method ranking with both LLM-Lasso variants highlighted
  - CV-MAE vs η curves for both variants
  - Per-feature LLM penalty score difference

Run after both RAG and No-RAG scoring + selection have been completed:
    python llm_lasso_score_features.py   # USE_RAG=True
    python llm_lasso_select.py           # USE_RAG=True
    # (switch USE_RAG=False in llm_lasso_score_features.py, repeat)
    python compare_rag_norag.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from llm_lasso_config import OUTPUT_DIR as _BASE_OUTPUT_DIR

RAG_DIR   = _BASE_OUTPUT_DIR / "rag"
NORAG_DIR = _BASE_OUTPUT_DIR / "no_rag"
PLOT_DIR  = _BASE_OUTPUT_DIR / "plots"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 150})


# ===========================
# Comparison plots
# ===========================

def plot_rag_vs_norag_ranking(results_rag, results_norag, output_dir):
    """
    Horizontal bar chart showing both RAG and No-RAG LLM-Lasso
    placed in the full ranking of all methods.
    All non-LLM methods use the shared RAG results (same dataset).
    """
    unified = []
    for r in results_rag:
        if r["method"] == "LLM-Lasso":
            unified.append({**r, "method": "LLM-Lasso (RAG)"})
        else:
            unified.append(r)

    for r in results_norag:
        if r["method"] == "LLM-Lasso":
            unified.append({**r, "method": "LLM-Lasso (No-RAG)"})
            break

    unified_sorted = sorted(unified, key=lambda r: r["cv_mae"])
    methods = [r["method"] for r in unified_sorted]
    maes    = [r["cv_mae"]  for r in unified_sorted]
    errs    = [r.get("cv_mae_std", 0) or 0 for r in unified_sorted]

    colors = []
    for r in unified_sorted:
        m = r["method"]
        if "RAG)" in m:
            colors.append("#27ae60")
        elif "No-RAG" in m:
            colors.append("#f39c12")
        elif "Random" in m:
            colors.append("#95a5a6")
        else:
            colors.append("#3498db")

    fig, ax = plt.subplots(figsize=(9, max(7, len(methods) * 0.42)))
    y_pos = np.arange(len(methods))
    ax.barh(y_pos, maes, xerr=errs, color=colors, edgecolor="gray",
            capsize=3, height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Cross-Validated MAE ↓", fontsize=11)
    ax.set_title(
        "Feature Selection Methods Ranked by CV-MAE\n"
        "(green = LLM-Lasso + RAG,  orange = LLM-Lasso no RAG,  gray = random)",
        fontsize=11, fontweight="bold",
    )
    ax.invert_yaxis()

    for i, (val, err) in enumerate(zip(maes, errs)):
        ax.text(val + (err or 0) + 0.002, i, f"{val:.3f}",
                va="center", fontsize=8, fontweight="bold")

    rag_rank   = next(i for i, r in enumerate(unified_sorted) if "RAG)" in r["method"])
    norag_rank = next(i for i, r in enumerate(unified_sorted) if "No-RAG" in r["method"])
    delta      = unified_sorted[norag_rank]["cv_mae"] - unified_sorted[rag_rank]["cv_mae"]
    x_ann      = max(maes) * 0.88
    mid_y      = (rag_rank + norag_rank) / 2
    ax.annotate(
        f"Δ={delta:.4f}", xy=(x_ann, mid_y),
        fontsize=9, color="#c0392b", fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c0392b", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "rag_vs_norag_ranking.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "rag_vs_norag_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: rag_vs_norag_ranking")


def plot_eta_curves(rag_eta_results, norag_eta_results, output_dir):
    """Line chart: CV-MAE vs η for RAG and No-RAG LLM-Lasso."""
    def _parse(eta_results):
        return [r["eta"] for r in eta_results], [r["cv_mae"] for r in eta_results]

    etas_rag,   maes_rag   = _parse(rag_eta_results)
    etas_norag, maes_norag = _parse(norag_eta_results)

    best_rag   = min(rag_eta_results,   key=lambda r: r["cv_mae"])
    best_norag = min(norag_eta_results, key=lambda r: r["cv_mae"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(etas_rag,   maes_rag,   "o-", color="#27ae60", linewidth=2,
            markersize=6, label="LLM-Lasso + RAG")
    ax.plot(etas_norag, maes_norag, "s--", color="#f39c12", linewidth=2,
            markersize=6, label="LLM-Lasso (No-RAG)")

    ax.scatter([best_rag["eta"]],   [best_rag["cv_mae"]],
               color="#27ae60", s=120, zorder=10,
               marker="*", edgecolors="black", linewidths=0.5)
    ax.scatter([best_norag["eta"]], [best_norag["cv_mae"]],
               color="#f39c12", s=120, zorder=10,
               marker="*", edgecolors="black", linewidths=0.5)

    ax.annotate(f"best η={best_rag['eta']}\nMAE={best_rag['cv_mae']:.4f}",
                xy=(best_rag["eta"], best_rag["cv_mae"]),
                xytext=(best_rag["eta"] + 0.3, best_rag["cv_mae"] + 0.005),
                fontsize=8, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax.annotate(f"best η={best_norag['eta']}\nMAE={best_norag['cv_mae']:.4f}",
                xy=(best_norag["eta"], best_norag["cv_mae"]),
                xytext=(best_norag["eta"] + 0.3, best_norag["cv_mae"] + 0.005),
                fontsize=8, color="#f39c12", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#f39c12"))

    ax.set_xlabel("η (LLM influence strength)", fontsize=11)
    ax.set_ylabel("CV-MAE ↓", fontsize=11)
    ax.set_title("LLM-Lasso: CV-MAE vs η\nRAG vs No-RAG", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "eta_curves_rag_vs_norag.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "eta_curves_rag_vs_norag.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: eta_curves_rag_vs_norag")


def plot_llm_score_diff(rag_scores, norag_scores, output_dir):
    """
    Horizontal bar chart of (No-RAG score − RAG score) per feature.
    Positive = RAG gave lower penalty (higher importance).
    """
    features = [f for f in rag_scores if f in norag_scores]
    diffs = {f: norag_scores[f] - rag_scores[f] for f in features}
    changed = {f: d for f, d in diffs.items() if abs(d) > 0.01}
    if not changed:
        print("  No score differences between RAG and No-RAG — skipping score diff plot.")
        return

    sorted_feats = sorted(changed.items(), key=lambda x: x[1], reverse=True)
    feat_labels = [f[0] for f in sorted_feats]
    diff_vals   = [f[1] for f in sorted_feats]
    colors      = ["#27ae60" if d > 0 else "#e74c3c" for d in diff_vals]

    fig, ax = plt.subplots(figsize=(9, max(4, len(feat_labels) * 0.38)))
    y_pos = np.arange(len(feat_labels))
    ax.barh(y_pos, diff_vals, color=colors, edgecolor="gray", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Score change: (No-RAG) − (RAG)\n"
                  "Positive (green) = RAG lowered penalty = higher importance assigned by RAG",
                  fontsize=9)
    ax.set_title("LLM Penalty Score Change Due to RAG\n"
                 "(only features where score changed)",
                 fontsize=11, fontweight="bold")
    for i, v in enumerate(diff_vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), i,
                f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "llm_score_diff_rag_vs_norag.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "llm_score_diff_rag_vs_norag.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: llm_score_diff_rag_vs_norag")


# ===========================
# Main
# ===========================

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load RAG results ---
    rag_trad_path = RAG_DIR / "traditional_selection_results.json"
    rag_fs_path   = RAG_DIR / "feature_selection_results.json"
    rag_scores_path = RAG_DIR / "llm_feature_scores.json"

    norag_trad_path   = NORAG_DIR / "traditional_selection_results.json"
    norag_fs_path     = NORAG_DIR / "feature_selection_results.json"
    norag_scores_path = NORAG_DIR / "llm_feature_scores.json"

    missing = []
    for p in [rag_trad_path, norag_trad_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("ERROR: Missing result files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun the following first:")
        print("  1. Set USE_RAG=True  in llm_lasso_score_features.py → python llm_lasso_score_features.py → python llm_lasso_select.py → python traditional_feature_selection.py")
        print("  2. Set USE_RAG=False in llm_lasso_score_features.py → repeat step 1")
        return

    with open(rag_trad_path) as f:
        rag_trad = json.load(f)
    with open(norag_trad_path) as f:
        norag_trad = json.load(f)

    print("Generating comparison plots...")

    # 1. Ranking comparison
    plot_rag_vs_norag_ranking(rag_trad["comparison"], norag_trad["comparison"], PLOT_DIR)

    # 2. η curves
    rag_eta, norag_eta = [], []
    if rag_fs_path.exists():
        with open(rag_fs_path) as f:
            rag_fs = json.load(f)
        rag_eta = rag_fs.get("eta_search_results", [])
    if norag_fs_path.exists():
        with open(norag_fs_path) as f:
            norag_fs = json.load(f)
        norag_eta = norag_fs.get("eta_search_results", [])

    if rag_eta and norag_eta:
        plot_eta_curves(rag_eta, norag_eta, PLOT_DIR)
    else:
        print("  (Skipping η curves — feature_selection_results.json not found or missing eta_search_results)")

    # 3. Score diff
    rag_scores, norag_scores = {}, {}
    if rag_scores_path.exists():
        with open(rag_scores_path) as f:
            raw = json.load(f)
        rag_scores = {k: v["median"] for k, v in raw.items()}
    if norag_scores_path.exists():
        with open(norag_scores_path) as f:
            raw = json.load(f)
        norag_scores = {k: v["median"] for k, v in raw.items()}

    if rag_scores and norag_scores:
        plot_llm_score_diff(rag_scores, norag_scores, PLOT_DIR)
    else:
        print("  (Skipping score diff — llm_feature_scores.json not found in one or both dirs)")

    print(f"\nPlots saved → {PLOT_DIR}")


if __name__ == "__main__":
    main()
