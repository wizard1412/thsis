"""
Binary Classification ML Result Visualization

Generates plots from bml_cv_results.json:
1. Model comparison bar chart (model_comparison.png)
2. Confusion matrices grid (confusion_matrix.png)
3. Metrics summary table (metrics_table.png)
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

from bml_config import OUTPUT_DIR, PLOT_DIR, BINARY_SPLITS, SPLIT_NAMES, MODEL_NAMES


def load_results():
    with open(OUTPUT_DIR / "bml_cv_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================
# 1. Model Comparison
# ===========================

def plot_model_comparison(results, save_path=None):
    """Bar chart comparing all split x model combinations."""
    metrics = ["mean_accuracy", "mean_f1", "mean_precision", "mean_recall", "mean_auc", "mean_mcc"]
    metric_labels = ["Accuracy", "F1", "Precision", "Recall", "AUC", "MCC"]
    std_keys = ["std_accuracy", "std_f1", "std_precision", "std_recall", "std_auc", "std_mcc"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    x = np.arange(len(MODEL_NAMES))
    width = 0.35
    colors = {"split_A": "steelblue", "split_B": "coral"}

    for ax_idx, (metric, std_key, label) in enumerate(zip(metrics, std_keys, metric_labels)):
        ax = axes[ax_idx]
        for s_idx, split_name in enumerate(SPLIT_NAMES):
            split_data = results["splits"][split_name]
            means = [split_data["models"][m][metric] for m in MODEL_NAMES]
            stds = [split_data["models"][m][std_key] for m in MODEL_NAMES]
            offset = (s_idx - (len(SPLIT_NAMES) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds,
                   label=split_data["name"],
                   color=colors[split_name], alpha=0.8, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_NAMES, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)

    fig.suptitle("Binary Classification ML: Model Comparison", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison saved to {save_path}")
    plt.close(fig)


# ===========================
# 2. Confusion Matrices
# ===========================

def plot_confusion_matrices(results, save_path=None):
    """Grid: rows=splits, cols=models."""
    fig, axes = plt.subplots(len(SPLIT_NAMES), len(MODEL_NAMES),
                              figsize=(4 * len(MODEL_NAMES), 4 * len(SPLIT_NAMES)))

    for s_idx, split_name in enumerate(SPLIT_NAMES):
        split_data = results["splits"][split_name]
        class_names = BINARY_SPLITS[split_name]["class_names"]

        for m_idx, model_name in enumerate(MODEL_NAMES):
            ax = axes[s_idx, m_idx]
            folds = split_data["models"][model_name]["fold_results"]

            all_preds, all_labels = [], []
            for f in folds:
                all_preds.extend(f["predictions"])
                all_labels.extend(f["labels"])

            cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(class_names, fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)

            thresh = cm.max() / 2.0
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black", fontsize=14)

            ax.set_title(f"{model_name}\n({split_data['name']})", fontsize=10)
            if m_idx == 0:
                ax.set_ylabel("True")
            if s_idx == len(SPLIT_NAMES) - 1:
                ax.set_xlabel("Predicted")

    fig.suptitle("Binary Classification: Confusion Matrices", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved to {save_path}")
    plt.close(fig)


# ===========================
# 3. Metrics Summary Table
# ===========================

def plot_metrics_table(results, save_path=None):
    """Summary table image."""
    col_labels = ["Split", "Model", "Accuracy", "F1", "Precision", "Recall", "AUC", "MCC"]
    table_data = []

    for split_name in SPLIT_NAMES:
        split_data = results["splits"][split_name]
        for model_name in MODEL_NAMES:
            r = split_data["models"][model_name]
            table_data.append([
                split_data["name"],
                model_name,
                f"{r['mean_accuracy']:.3f} +/- {r['std_accuracy']:.3f}",
                f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}",
                f"{r['mean_precision']:.3f} +/- {r['std_precision']:.3f}",
                f"{r['mean_recall']:.3f} +/- {r['std_recall']:.3f}",
                f"{r['mean_auc']:.3f} +/- {r['std_auc']:.3f}",
                f"{r['mean_mcc']:.3f} +/- {r['std_mcc']:.3f}",
            ])

    fig, ax = plt.subplots(figsize=(18, 4))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(len(table_data)):
        color = "#F2F2F2" if i < len(MODEL_NAMES) else "#FFFFFF"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Binary Classification ML: 5-Fold CV Results",
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
    plot_metrics_table(results, save_path=PLOT_DIR / "metrics_table.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
