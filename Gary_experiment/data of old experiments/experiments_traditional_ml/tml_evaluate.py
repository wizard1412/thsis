"""
Traditional ML + Dawid-Skene Result Visualization

Generates plots from tml_cv_results.json:
1. Model comparison bar chart (model_comparison.png)
2. Confusion matrices grid (confusion_matrix.png)
3. Dawid-Skene learned rater confusion matrices (ds_confusion_matrices.png)
4. Metrics summary table (metrics_table.png)
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

from tml_config import OUTPUT_DIR, PLOT_DIR, SCORE_NUM_CLASSES, MAX_NUM_RATERS, LABEL_STRATEGIES, MODEL_NAMES


def load_results():
    with open(OUTPUT_DIR / "tml_cv_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================
# 1. Model Comparison
# ===========================

def plot_model_comparison(results, save_path=None):
    """Bar chart comparing all strategy x model combinations."""
    strategies = LABEL_STRATEGIES
    models = MODEL_NAMES
    metrics = ["mean_accuracy", "mean_mae", "mean_off_by_one", "mean_qwk", "mean_f1"]
    metric_labels = ["Accuracy", "MAE", "Off-by-One Acc", "QWK", "F1 (weighted)"]
    std_keys = ["std_accuracy", "std_mae", "std_off_by_one", "std_qwk", "std_f1"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    x = np.arange(len(models))
    width = 0.25
    colors = {"majority_vote": "steelblue", "dawid_skene": "coral", "mean": "seagreen"}

    for ax_idx, (metric, std_key, label) in enumerate(zip(metrics, std_keys, metric_labels)):
        ax = axes[ax_idx]
        for s_idx, strategy in enumerate(strategies):
            means = [results["strategies"][strategy][m][metric] for m in models]
            stds = [results["strategies"][strategy][m][std_key] for m in models]
            offset = (s_idx - (len(strategies) - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, yerr=stds,
                          label=strategy.replace("_", " ").title(),
                          color=colors[strategy], alpha=0.8, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        if metric != "mean_mae":
            ax.set_ylim(0, 1)

    fig.suptitle("Traditional ML + Dawid-Skene: Model Comparison", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison saved to {save_path}")
    plt.close(fig)


# ===========================
# 2. Confusion Matrices
# ===========================

def plot_confusion_matrices(results, save_path=None):
    """2x3 grid: rows=strategies, cols=models."""
    strategies = LABEL_STRATEGIES
    models = MODEL_NAMES
    labels = [str(i) if i < SCORE_NUM_CLASSES - 1 else f"{i}+" for i in range(SCORE_NUM_CLASSES)]

    fig, axes = plt.subplots(len(strategies), len(models),
                              figsize=(5 * len(models), 5 * len(strategies)))

    for s_idx, strategy in enumerate(strategies):
        for m_idx, model_name in enumerate(models):
            ax = axes[s_idx, m_idx]
            folds = results["strategies"][strategy][model_name]["fold_results"]

            all_preds, all_labels_list = [], []
            for f in folds:
                all_preds.extend(f["predictions"])
                all_labels_list.extend(f["labels"])

            cm = confusion_matrix(all_labels_list, all_preds, labels=list(range(SCORE_NUM_CLASSES)))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            thresh = cm.max() / 2.0
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black", fontsize=12)

            strategy_name = strategy.replace("_", " ").title()
            ax.set_title(f"{model_name}\n({strategy_name})", fontsize=11)
            if m_idx == 0:
                ax.set_ylabel("True Score")
            if s_idx == len(strategies) - 1:
                ax.set_xlabel("Predicted Score")

    fig.suptitle("Confusion Matrices: Strategy x Model", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved to {save_path}")
    plt.close(fig)


# ===========================
# 3. Dawid-Skene Confusion Matrices
# ===========================

def plot_ds_matrices(results, save_path=None):
    """Plot Dawid-Skene learned rater confusion matrices (from fold 1)."""
    ds_folds = results["strategies"]["dawid_skene"]
    # Find fold 1 with DS matrices from any model (they're the same per fold)
    first_model = MODEL_NAMES[0]
    fold1 = ds_folds[first_model]["fold_results"][0]

    if "ds_confusion_matrices" not in fold1:
        print("No Dawid-Skene confusion matrices found in results.")
        return

    A = np.array(fold1["ds_confusion_matrices"])  # (R, C_obs, C_true)
    labels = [str(i) if i < SCORE_NUM_CLASSES - 1 else f"{i}+" for i in range(SCORE_NUM_CLASSES)]

    fig, axes = plt.subplots(1, MAX_NUM_RATERS, figsize=(4 * MAX_NUM_RATERS, 4))

    for r in range(MAX_NUM_RATERS):
        ax = axes[r]
        mat = A[r]  # (C_obs, C_true) -> transpose to (C_true, C_obs) for display
        mat_display = mat.T  # rows=true, cols=observed

        im = ax.imshow(mat_display, interpolation="nearest", cmap=plt.cm.YlOrRd, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Observed Score")
        ax.set_ylabel("True Score")
        ax.set_title(f"Rater {r + 1}")

        for i in range(SCORE_NUM_CLASSES):
            for j in range(SCORE_NUM_CLASSES):
                ax.text(j, i, f"{mat_display[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat_display[i, j] > 0.5 else "black", fontsize=9)

    fig.colorbar(im, ax=axes, shrink=0.8, label="P(Observed | True)")
    fig.suptitle("Dawid-Skene Learned Rater Confusion Matrices (Fold 1)", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"DS matrices saved to {save_path}")
    plt.close(fig)


# ===========================
# 4. Metrics Summary Table
# ===========================

def plot_metrics_table(results, save_path=None):
    """Summary table image."""
    strategies = LABEL_STRATEGIES
    models = MODEL_NAMES

    col_labels = ["Strategy", "Model", "Accuracy", "MAE", "Off-by-One Acc", "QWK", "F1 (weighted)"]
    table_data = []

    for strategy in strategies:
        for model_name in models:
            r = results["strategies"][strategy][model_name]
            strategy_name = strategy.replace("_", " ").title()
            table_data.append([
                strategy_name,
                model_name,
                f"{r['mean_accuracy']:.3f} +/- {r['std_accuracy']:.3f}",
                f"{r['mean_mae']:.3f} +/- {r['std_mae']:.3f}",
                f"{r['mean_off_by_one']:.3f} +/- {r['std_off_by_one']:.3f}",
                f"{r['mean_qwk']:.3f} +/- {r['std_qwk']:.3f}",
                f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}",
            ])

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors for strategies
    for i in range(len(table_data)):
        color = "#F2F2F2" if i < len(models) else "#FFFFFF"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Traditional ML + Dawid-Skene: 5-Fold CV Results Summary",
                 fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
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

    plot_model_comparison(results, save_path=PLOT_DIR / "model_comparison.png")
    plot_confusion_matrices(results, save_path=PLOT_DIR / "confusion_matrix.png")
    plot_ds_matrices(results, save_path=PLOT_DIR / "ds_confusion_matrices.png")
    plot_metrics_table(results, save_path=PLOT_DIR / "metrics_table.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()