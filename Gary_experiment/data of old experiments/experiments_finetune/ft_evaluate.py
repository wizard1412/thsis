"""
QLoRA Fine-Tuning Result Visualization

Generates plots from ft_cv_results.json:
1. Strategy comparison bar chart (strategy_comparison.png)
2. Confusion matrices (confusion_matrix.png)
3. Per-fold comparison (fold_comparison.png)
4. Metrics summary table (metrics_table.png)
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
matplotlib.rcParams["axes.unicode_minus"] = False

from ft_config import OUTPUT_DIR, PLOT_DIR, SCORE_NUM_CLASSES, LABEL_STRATEGIES


def load_results():
    with open(OUTPUT_DIR / "ft_cv_results.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================
# 1. Strategy Comparison
# ===========================

def plot_strategy_comparison(results, save_path=None):
    """Bar chart comparing majority_vote vs dawid_skene strategies."""
    strategies = LABEL_STRATEGIES
    metrics = ["mean_accuracy", "mean_mae", "mean_off_by_one", "mean_qwk", "mean_f1"]
    metric_labels = ["Accuracy", "MAE", "Off-by-One Acc", "QWK", "F1 (weighted)"]
    std_keys = ["std_accuracy", "std_mae", "std_off_by_one", "std_qwk", "std_f1"]
    colors = {"majority_vote": "steelblue", "dawid_skene": "coral"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    x = np.arange(len(strategies))
    width = 0.5

    for ax_idx, (metric, std_key, label) in enumerate(zip(metrics, std_keys, metric_labels)):
        ax = axes[ax_idx]
        means = [results["strategies"][s][metric] for s in strategies]
        stds = [results["strategies"][s][std_key] for s in strategies]
        bar_colors = [colors[s] for s in strategies]
        strategy_labels = [s.replace("_", " ").title() for s in strategies]

        bars = ax.bar(x, means, width, yerr=stds, color=bar_colors, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_labels, rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label)
        if metric != "mean_mae":
            ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    model_name = results.get("model", "LLM").split("/")[-1]
    fig.suptitle(f"QLoRA Fine-tuned {model_name}: Strategy Comparison", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Strategy comparison saved to {save_path}")
    plt.close(fig)


# ===========================
# 2. Confusion Matrices
# ===========================

def plot_confusion_matrices(results, save_path=None):
    """1x2 grid: one confusion matrix per strategy."""
    strategies = LABEL_STRATEGIES
    labels = [str(i) if i < SCORE_NUM_CLASSES - 1 else f"{i}+" for i in range(SCORE_NUM_CLASSES)]

    fig, axes = plt.subplots(1, len(strategies), figsize=(6 * len(strategies), 5))

    for s_idx, strategy in enumerate(strategies):
        ax = axes[s_idx]
        folds = results["strategies"][strategy]["fold_results"]

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
                        color="white" if cm[i, j] > thresh else "black", fontsize=14)

        strategy_name = strategy.replace("_", " ").title()
        ax.set_title(f"{strategy_name}", fontsize=12)
        ax.set_ylabel("True Score")
        ax.set_xlabel("Predicted Score")

    model_name = results.get("model", "LLM").split("/")[-1]
    fig.suptitle(f"QLoRA Fine-tuned {model_name}: Confusion Matrices", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrices saved to {save_path}")
    plt.close(fig)


# ===========================
# 3. Per-Fold Comparison
# ===========================

def plot_fold_comparison(results, save_path=None):
    """Per-fold metrics for each strategy."""
    strategies = LABEL_STRATEGIES
    metrics = ["accuracy", "mae", "off_by_one_accuracy", "qwk", "f1_weighted"]
    metric_labels = ["Accuracy", "MAE", "Off-by-One Acc", "QWK", "F1 (weighted)"]
    colors = {"majority_vote": "steelblue", "dawid_skene": "coral"}

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))

    for m_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[m_idx]
        for strategy in strategies:
            folds = results["strategies"][strategy]["fold_results"]
            values = [f[metric] for f in folds]
            fold_nums = [f["fold"] for f in folds]
            strategy_label = strategy.replace("_", " ").title()
            ax.plot(fold_nums, values, "o-", label=strategy_label,
                    color=colors[strategy], linewidth=2, markersize=8)
        ax.set_xlabel("Fold")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(range(1, 6))
        ax.legend()
        ax.grid(True, alpha=0.3)

    model_name = results.get("model", "LLM").split("/")[-1]
    fig.suptitle(f"QLoRA Fine-tuned {model_name}: Per-Fold Results", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Fold comparison saved to {save_path}")
    plt.close(fig)


# ===========================
# 4. Metrics Summary Table
# ===========================

def plot_metrics_table(results, save_path=None):
    """Summary table image."""
    strategies = LABEL_STRATEGIES
    col_labels = ["Strategy", "Accuracy", "MAE", "Off-by-One Acc", "QWK", "F1 (weighted)"]
    table_data = []

    for strategy in strategies:
        r = results["strategies"][strategy]
        strategy_name = strategy.replace("_", " ").title()
        table_data.append([
            strategy_name,
            f"{r['mean_accuracy']:.3f} +/- {r['std_accuracy']:.3f}",
            f"{r['mean_mae']:.3f} +/- {r['std_mae']:.3f}",
            f"{r['mean_off_by_one']:.3f} +/- {r['std_off_by_one']:.3f}",
            f"{r['mean_qwk']:.3f} +/- {r['std_qwk']:.3f}",
            f"{r['mean_f1']:.3f} +/- {r['std_f1']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row colors
    for i in range(len(table_data)):
        color = "#F2F2F2" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    model_name = results.get("model", "LLM").split("/")[-1]
    ax.set_title(f"QLoRA Fine-tuned {model_name}: 5-Fold CV Results Summary",
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
    model_name = results.get("model", "Unknown")
    print(f"Loaded: {results['method']}")
    print(f"Model: {model_name}")

    plot_strategy_comparison(results, save_path=PLOT_DIR / "strategy_comparison.png")
    plot_confusion_matrices(results, save_path=PLOT_DIR / "confusion_matrix.png")
    plot_fold_comparison(results, save_path=PLOT_DIR / "fold_comparison.png")
    plot_metrics_table(results, save_path=PLOT_DIR / "metrics_table.png")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
