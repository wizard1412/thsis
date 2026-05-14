"""
FTDualNet + RCE Result Visualization

Load cross-validation results from rce_cv_results.json and generate:
1. Confusion matrix (confusion_matrix.png)
2. Per-fold metric comparison (fold_comparison.png)
3. True vs predicted score distribution (score_distribution.png)
4. Consensus-level comparison (consensus_comparison.png)
5. Learned RCE rater confusion matrices heatmap (rce_matrices.png)
6. Metrics summary table (metrics_table.png)
"""

import json
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

from rce_config import OUTPUT_DIR, PLOT_DIR, CHECKPOINT_DIR, SCORE_NUM_CLASSES, MAX_NUM_RATERS
from rce_model import RaterConfusionEstimation


def load_results():
    results_file = OUTPUT_DIR / "rce_cv_results.json"
    with open(results_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================
# 1. Confusion Matrix
# ===========================

def plot_confusion_matrix(results, save_path=None):
    """Plot confusion matrix heatmap (aggregated across all folds)"""
    all_preds = []
    all_labels = []
    for r in results["fold_results"]:
        all_preds.extend(r["predictions"])
        all_labels.extend(r["labels"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(SCORE_NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = [str(i) if i < SCORE_NUM_CLASSES - 1 else f"{i}+"
              for i in range(SCORE_NUM_CLASSES)]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Score",
        ylabel="True Score",
        title="FTDualNet + RCE Confusion Matrix",
    )

    thresh = cm.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


# ===========================
# 2. Per-Fold Metric Comparison
# ===========================

def plot_per_fold_comparison(results, save_path=None):
    """Plot per-fold metric comparison"""
    folds = [r["fold"] for r in results["fold_results"]]
    accs = [r["accuracy"] for r in results["fold_results"]]
    maes = [r["mae"] for r in results["fold_results"]]
    obos = [r["off_by_one_accuracy"] for r in results["fold_results"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy
    axes[0].bar(folds, accs, color="steelblue", alpha=0.8)
    axes[0].axhline(y=np.mean(accs), color="red", linestyle="--", label=f"Mean: {np.mean(accs):.3f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy per Fold")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # MAE
    axes[1].bar(folds, maes, color="coral", alpha=0.8)
    axes[1].axhline(y=np.mean(maes), color="red", linestyle="--", label=f"Mean: {np.mean(maes):.3f}")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("MAE per Fold")
    axes[1].legend()

    # Off-by-one Accuracy
    axes[2].bar(folds, obos, color="mediumseagreen", alpha=0.8)
    axes[2].axhline(y=np.mean(obos), color="red", linestyle="--", label=f"Mean: {np.mean(obos):.3f}")
    axes[2].set_xlabel("Fold")
    axes[2].set_ylabel("Off-by-One Accuracy")
    axes[2].set_title("Off-by-One Accuracy per Fold")
    axes[2].legend()
    axes[2].set_ylim(0, 1)

    fig.suptitle("FTDualNet + RCE 5-Fold Cross-Validation Results", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Fold comparison saved to {save_path}")
    plt.close(fig)


# ===========================
# 3. Score Distribution
# ===========================

def plot_score_distribution(results, save_path=None):
    """Plot true vs predicted score distribution"""
    all_preds = []
    all_labels = []
    for r in results["fold_results"]:
        all_preds.extend(r["predictions"])
        all_labels.extend(r["labels"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    fig, ax = plt.subplots(figsize=(8, 5))

    scores = list(range(SCORE_NUM_CLASSES))
    x = np.arange(len(scores))
    width = 0.35

    true_counts = [np.sum(all_labels == s) for s in scores]
    pred_counts = [np.sum(all_preds == s) for s in scores]

    ax.bar(x - width / 2, true_counts, width, label="True Score", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, pred_counts, width, label="Predicted Score", color="coral", alpha=0.8)

    ax.set_xlabel("MDS-UPDRS 3.4 Score")
    ax.set_ylabel("Count")
    ax.set_title("True vs Predicted Score Distribution")
    ax.set_xticks(x)
    tick_labels = [str(s) if s < SCORE_NUM_CLASSES - 1 else f"{s}+" for s in scores]
    ax.set_xticklabels(tick_labels)
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Score distribution saved to {save_path}")
    plt.close(fig)


# ===========================
# 4. Consensus-Level Comparison
# ===========================

def plot_consensus_comparison(results, save_path=None):
    """Plot performance comparison across consensus levels"""
    levels = ["full_consensus", "majority_consensus", "low_consensus"]
    level_labels = ["Full Consensus", "Majority Consensus", "Low Consensus"]

    acc_data = {lvl: [] for lvl in levels}
    kappa_data = {lvl: [] for lvl in levels}

    for r in results["fold_results"]:
        for lvl in levels:
            cr = r["consensus_results"][lvl]
            if cr["n"] > 0:
                acc_data[lvl].append(cr["accuracy"])
                kappa_data[lvl].append(cr["kappa"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    x = np.arange(len(levels))
    means_acc = [np.mean(acc_data[lvl]) for lvl in levels]
    stds_acc = [np.std(acc_data[lvl]) for lvl in levels]

    axes[0].bar(x, means_acc, yerr=stds_acc, color=["#4CAF50", "steelblue", "#FF7043"],
                alpha=0.8, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(level_labels)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by Consensus Level")
    axes[0].set_ylim(0, 1)
    for i, (m, s) in enumerate(zip(means_acc, stds_acc)):
        axes[0].text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=10)

    # QWK
    means_kappa = [np.mean(kappa_data[lvl]) for lvl in levels]
    stds_kappa = [np.std(kappa_data[lvl]) for lvl in levels]

    axes[1].bar(x, means_kappa, yerr=stds_kappa, color=["#4CAF50", "steelblue", "#FF7043"],
                alpha=0.8, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(level_labels)
    axes[1].set_ylabel("QWK")
    axes[1].set_title("Quadratic Weighted Kappa by Consensus Level")
    axes[1].set_ylim(0, 1)
    for i, (m, s) in enumerate(zip(means_kappa, stds_kappa)):
        axes[1].text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=10)

    fig.suptitle("FTDualNet + RCE: Consensus-Level Comparison", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Consensus comparison saved to {save_path}")
    plt.close(fig)


# ===========================
# 5. RCE Rater Confusion Matrices
# ===========================

def plot_rce_matrices(fold=1, save_path=None):
    """Plot learned RCE rater confusion matrices heatmap"""
    ckpt_path = CHECKPOINT_DIR / f"rce_fold{fold}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    rce_module = RaterConfusionEstimation()
    rce_module.load_state_dict(checkpoint["rce"])
    rce_module.eval()

    fig, axes = plt.subplots(1, MAX_NUM_RATERS, figsize=(4 * MAX_NUM_RATERS, 4))

    labels = [str(i) if i < SCORE_NUM_CLASSES - 1 else f"{i}+"
              for i in range(SCORE_NUM_CLASSES)]

    for r in range(MAX_NUM_RATERS):
        ax = axes[r]
        A = rce_module.get_matrix(r).detach().cpu().numpy()

        im = ax.imshow(A, interpolation="nearest", cmap=plt.cm.YlOrRd, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Observed Score")
        ax.set_ylabel("True Score")
        ax.set_title(f"Rater {r + 1}")

        for i in range(SCORE_NUM_CLASSES):
            for j in range(SCORE_NUM_CLASSES):
                ax.text(
                    j, i, f"{A[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if A[i, j] > 0.5 else "black",
                    fontsize=9,
                )

    fig.colorbar(im, ax=axes, shrink=0.8, label="P(Observed | True)")
    fig.suptitle(f"Learned RCE Rater Confusion Matrices (Fold {fold})", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"RCE matrices saved to {save_path}")
    plt.close(fig)


# ===========================
# 6. Metrics Summary Table
# ===========================

def plot_metrics_table(results, save_path=None):
    """Plot 5-fold cross-validation metrics summary table"""
    fold_results = results["fold_results"]

    # --- Table 1: Per-Fold Metrics ---
    col_labels = ["Fold", "Accuracy", "MAE", "Off-by-One Acc", "QWK", "F1 (weighted)"]
    table_data = []
    for r in fold_results:
        table_data.append([
            f"Fold {r['fold']}",
            f"{r['accuracy']:.3f}",
            f"{r['mae']:.3f}",
            f"{r['off_by_one_accuracy']:.3f}",
            f"{r['qwk']:.3f}",
            f"{r['f1_weighted']:.3f}",
        ])
    table_data.append([
        "Mean +/- Std",
        f"{results['mean_accuracy']:.3f} +/- {results['std_accuracy']:.3f}",
        f"{results['mean_mae']:.3f} +/- {results['std_mae']:.3f}",
        f"{results['mean_off_by_one']:.3f} +/- {results['std_off_by_one']:.3f}",
        f"{results['mean_qwk']:.3f} +/- {results['std_qwk']:.3f}",
        f"{results['mean_f1']:.3f} +/- {results['std_f1']:.3f}",
    ])

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                              gridspec_kw={"height_ratios": [1.2, 1]})

    # Table 1
    ax1 = axes[0]
    ax1.axis("off")
    t1 = ax1.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    t1.auto_set_font_size(False)
    t1.set_fontsize(11)
    t1.scale(1, 1.6)

    for j in range(len(col_labels)):
        t1[0, j].set_facecolor("#4472C4")
        t1[0, j].set_text_props(color="white", fontweight="bold")

    last_row = len(table_data)
    for j in range(len(col_labels)):
        t1[last_row, j].set_facecolor("#D6E4F0")
        t1[last_row, j].set_text_props(fontweight="bold")

    ax1.set_title("Table 1: 5-Fold Cross-Validation Metrics", fontsize=13, fontweight="bold", pad=12)

    # --- Table 2: Consensus-Level Metrics ---
    levels = ["full_consensus", "majority_consensus", "low_consensus", "overall"]
    level_names = ["Full Consensus", "Majority Consensus", "Low Consensus", "Overall"]

    col_labels2 = ["Consensus Level", "Avg N", "Accuracy", "MAE", "Off-by-One Acc", "QWK"]
    table_data2 = []

    for lvl, name in zip(levels, level_names):
        ns = [r["consensus_results"][lvl]["n"] for r in fold_results]
        accs = [r["consensus_results"][lvl]["accuracy"]
                for r in fold_results if r["consensus_results"][lvl]["n"] > 0]
        maes = [r["consensus_results"][lvl]["mae"]
                for r in fold_results if r["consensus_results"][lvl]["n"] > 0]
        obos = [r["consensus_results"][lvl]["off_by_one"]
                for r in fold_results if r["consensus_results"][lvl]["n"] > 0]
        kappas = [r["consensus_results"][lvl]["kappa"]
                  for r in fold_results if r["consensus_results"][lvl]["n"] > 0]

        table_data2.append([
            name,
            f"{np.mean(ns):.1f}",
            f"{np.mean(accs):.3f} +/- {np.std(accs):.3f}" if accs else "N/A",
            f"{np.mean(maes):.3f} +/- {np.std(maes):.3f}" if maes else "N/A",
            f"{np.mean(obos):.3f} +/- {np.std(obos):.3f}" if obos else "N/A",
            f"{np.mean(kappas):.3f} +/- {np.std(kappas):.3f}" if kappas else "N/A",
        ])

    ax2 = axes[1]
    ax2.axis("off")
    t2 = ax2.table(
        cellText=table_data2,
        colLabels=col_labels2,
        loc="center",
        cellLoc="center",
    )
    t2.auto_set_font_size(False)
    t2.set_fontsize(11)
    t2.scale(1, 1.6)

    for j in range(len(col_labels2)):
        t2[0, j].set_facecolor("#4472C4")
        t2[0, j].set_text_props(color="white", fontweight="bold")

    for j in range(len(col_labels2)):
        t2[len(table_data2), j].set_facecolor("#D6E4F0")
        t2[len(table_data2), j].set_text_props(fontweight="bold")

    ax2.set_title("Table 2: Metrics by Consensus Level", fontsize=13, fontweight="bold", pad=12)

    fig.suptitle("FTDualNet + RCE Experiment Results Summary", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Metrics table saved to {save_path}")
    plt.close(fig)


# ===========================
# Main
# ===========================

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    print(f"Loaded: {results['method']}")
    print(f"Mean Accuracy: {results['mean_accuracy']:.3f} +/- {results['std_accuracy']:.3f}")
    print(f"Mean QWK:      {results['mean_qwk']:.3f} +/- {results['std_qwk']:.3f}")

    plot_confusion_matrix(results, save_path=PLOT_DIR / "confusion_matrix.png")
    plot_per_fold_comparison(results, save_path=PLOT_DIR / "fold_comparison.png")
    plot_score_distribution(results, save_path=PLOT_DIR / "score_distribution.png")
    plot_consensus_comparison(results, save_path=PLOT_DIR / "consensus_comparison.png")
    plot_rce_matrices(fold=1, save_path=PLOT_DIR / "rce_matrices.png")
    plot_metrics_table(results, save_path=PLOT_DIR / "metrics_table.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()