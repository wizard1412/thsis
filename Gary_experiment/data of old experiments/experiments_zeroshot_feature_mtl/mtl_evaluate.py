"""
FTDualNet 評估模組

評估指標：
- MAE (平均絕對誤差)
- ±1 容許準確率
- 整體準確率
- 加權 F1 分數
- QWK (Quadratic Weighted Kappa)
- Correlation (Pearson r)
- AAUC (Average AUC, one-vs-rest)
- WD (Wasserstein Distance)
- 混淆矩陣熱力圖
- 各類別精確率/召回率
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    cohen_kappa_score,
    roc_auc_score,
)
from scipy.stats import wasserstein_distance

from mtl_config import OUTPUT_DIR, PLOT_DIR, SCORE_NUM_CLASSES

# 支援中文字型
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_cv_results(results_path=None):
    """載入交叉驗證結果"""
    if results_path is None:
        results_path = OUTPUT_DIR / "cv_results.json"

    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_aauc(y_true, y_pred, max_score=None):
    """
    計算 Average AUC (AAUC)，使用 one-vs-rest 方法
    與 comparison_heatmaps.py 一致
    """
    if max_score is None:
        max_score = SCORE_NUM_CLASSES - 1

    try:
        aucs = []
        for threshold in range(max_score + 1):
            y_true_bin = (y_true >= threshold).astype(int)
            y_pred_bin = (y_pred >= threshold).astype(int)
            if len(set(y_true_bin)) > 1:
                auc = roc_auc_score(y_true_bin, y_pred_bin)
                aucs.append(auc)
        if aucs:
            return float(np.mean(aucs))
        return float("nan")
    except Exception:
        return float("nan")


def compute_all_metrics(results):
    """
    從 CV 結果計算所有評估指標

    回傳：
        metrics: dict, 包含所有指標
    """
    # 合併所有 fold 的預測和標籤
    all_preds = []
    all_labels = []
    for fold in results["fold_results"]:
        all_preds.extend(fold["predictions"])
        all_labels.extend(fold["labels"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # --- 全域指標（合併所有 fold）---

    # QWK (Quadratic Weighted Kappa) - 與 comparison_heatmaps.py 一致
    try:
        qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    except Exception:
        qwk = float("nan")

    # Pearson Correlation
    corr = float(np.corrcoef(all_labels, all_preds)[0, 1]) if len(all_labels) > 1 else float("nan")

    # AAUC (Average AUC, one-vs-rest)
    aauc = calculate_aauc(all_labels, all_preds)

    # Wasserstein Distance
    wd = float(wasserstein_distance(all_labels, all_preds))

    # --- Per-fold 指標 ---
    fold_qwks = []
    fold_corrs = []
    fold_aaucs = []
    fold_wds = []
    for fold in results["fold_results"]:
        fp = np.array(fold["predictions"])
        fl = np.array(fold["labels"])
        try:
            fold_qwks.append(cohen_kappa_score(fl, fp, weights="quadratic"))
        except Exception:
            fold_qwks.append(float("nan"))
        fold_corrs.append(float(np.corrcoef(fl, fp)[0, 1]) if len(fl) > 1 else float("nan"))
        fold_aaucs.append(calculate_aauc(fl, fp))
        fold_wds.append(float(wasserstein_distance(fl, fp)))

    metrics = {
        "mae": mean_absolute_error(all_labels, all_preds),
        "accuracy": accuracy_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "off_by_one_accuracy": np.mean(np.abs(all_preds - all_labels) <= 1),
        "qwk": qwk,
        "correlation": corr,
        "aauc": aauc,
        "wasserstein_distance": wd,
        "confusion_matrix": confusion_matrix(
            all_labels, all_preds,
            labels=list(range(SCORE_NUM_CLASSES)),
        ).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds,
            labels=list(range(SCORE_NUM_CLASSES)),
            target_names=[f"Score {i}" if i < SCORE_NUM_CLASSES - 1 else f"Score {i}+"
                          for i in range(SCORE_NUM_CLASSES)],
            zero_division=0,
        ),
        # 從 CV 結果取 per-fold 指標
        "cv_mean_accuracy": results["mean_accuracy"],
        "cv_std_accuracy": results["std_accuracy"],
        "cv_mean_mae": results["mean_mae"],
        "cv_std_mae": results["std_mae"],
        "cv_mean_off_by_one": results["mean_off_by_one"],
        "cv_std_off_by_one": results["std_off_by_one"],
        # Per-fold 新指標
        "cv_mean_qwk": float(np.nanmean(fold_qwks)),
        "cv_std_qwk": float(np.nanstd(fold_qwks)),
        "cv_mean_correlation": float(np.nanmean(fold_corrs)),
        "cv_std_correlation": float(np.nanstd(fold_corrs)),
        "cv_mean_aauc": float(np.nanmean(fold_aaucs)),
        "cv_std_aauc": float(np.nanstd(fold_aaucs)),
        "cv_mean_wd": float(np.nanmean(fold_wds)),
        "cv_std_wd": float(np.nanstd(fold_wds)),
    }

    return metrics, all_preds, all_labels


# ===========================
# 視覺化
# ===========================

def plot_confusion_matrix(cm, save_path=None):
    """繪製混淆矩陣熱力圖"""
    fig, ax = plt.subplots(figsize=(8, 6))

    cm_array = np.array(cm)
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = [str(i) if i < SCORE_NUM_CLASSES - 1 else f"{i}+"
              for i in range(SCORE_NUM_CLASSES)]
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="預測分數",
        ylabel="真實分數",
        title="FTDualNet 混淆矩陣",
    )

    # 在每個格子中顯示數字
    thresh = cm_array.max() / 2.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j, i, format(cm_array[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_array[i, j] > thresh else "black",
                fontsize=14,
            )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"混淆矩陣已儲存至 {save_path}")

    plt.close(fig)


def plot_per_fold_comparison(results, save_path=None):
    """繪製各 Fold 指標比較圖"""
    folds = [r["fold"] for r in results["fold_results"]]
    accs = [r["accuracy"] for r in results["fold_results"]]
    maes = [r["mae"] for r in results["fold_results"]]
    obos = [r["off_by_one_accuracy"] for r in results["fold_results"]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 準確率
    axes[0].bar(folds, accs, color="steelblue", alpha=0.8)
    axes[0].axhline(y=np.mean(accs), color="red", linestyle="--", label=f"平均: {np.mean(accs):.3f}")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("準確率")
    axes[0].set_title("各 Fold 準確率")
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    # MAE
    axes[1].bar(folds, maes, color="coral", alpha=0.8)
    axes[1].axhline(y=np.mean(maes), color="red", linestyle="--", label=f"平均: {np.mean(maes):.3f}")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("各 Fold MAE")
    axes[1].legend()

    # ±1 準確率
    axes[2].bar(folds, obos, color="mediumseagreen", alpha=0.8)
    axes[2].axhline(y=np.mean(obos), color="red", linestyle="--", label=f"平均: {np.mean(obos):.3f}")
    axes[2].set_xlabel("Fold")
    axes[2].set_ylabel("±1 準確率")
    axes[2].set_title("各 Fold ±1 容許準確率")
    axes[2].legend()
    axes[2].set_ylim(0, 1)

    fig.suptitle("FTDualNet 5-Fold 交叉驗證結果", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Fold 比較圖已儲存至 {save_path}")

    plt.close(fig)


def plot_score_distribution(all_preds, all_labels, save_path=None):
    """繪製真實 vs 預測的分數分布比較"""
    fig, ax = plt.subplots(figsize=(8, 5))

    scores = list(range(SCORE_NUM_CLASSES))
    x = np.arange(len(scores))
    width = 0.35

    true_counts = [np.sum(all_labels == s) for s in scores]
    pred_counts = [np.sum(all_preds == s) for s in scores]

    ax.bar(x - width / 2, true_counts, width, label="真實分數", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, pred_counts, width, label="預測分數", color="coral", alpha=0.8)

    ax.set_xlabel("MDS-UPDRS 3.4 分數")
    ax.set_ylabel("次數")
    ax.set_title("真實 vs 預測分數分布")
    ax.set_xticks(x)
    tick_labels = [str(s) if s < SCORE_NUM_CLASSES - 1 else f"{s}+" for s in scores]
    ax.set_xticklabels(tick_labels)
    ax.legend()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"分數分布圖已儲存至 {save_path}")

    plt.close(fig)


# ===========================
# 主要評估流程
# ===========================

def run_evaluation(results_path=None):
    """
    執行完整評估，產生所有指標和圖表
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 載入結果
    results = load_cv_results(results_path)

    # 計算指標
    metrics, all_preds, all_labels = compute_all_metrics(results)

    # 印出結果
    print("=" * 60)
    print("FTDualNet 評估結果")
    print("=" * 60)

    print(f"\n交叉驗證 (5-Fold) Mean ± Std:")
    print(f"  MAE:          {metrics['cv_mean_mae']:.3f} ± {metrics['cv_std_mae']:.3f}")
    print(f"  Exact Acc:    {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
    print(f"  ±1 Acc:       {metrics['cv_mean_off_by_one']:.3f} ± {metrics['cv_std_off_by_one']:.3f}")
    print(f"  QWK (κ):      {metrics['cv_mean_qwk']:.3f} ± {metrics['cv_std_qwk']:.3f}")
    print(f"  Correlation:  {metrics['cv_mean_correlation']:.3f} ± {metrics['cv_std_correlation']:.3f}")
    print(f"  AAUC:         {metrics['cv_mean_aauc']:.3f} ± {metrics['cv_std_aauc']:.3f}")
    print(f"  WD:           {metrics['cv_mean_wd']:.3f} ± {metrics['cv_std_wd']:.3f}")

    print(f"\n合併所有 Fold:")
    print(f"  MAE:          {metrics['mae']:.3f}")
    print(f"  Exact Acc:    {metrics['accuracy']:.3f}")
    print(f"  ±1 Acc:       {metrics['off_by_one_accuracy']:.3f}")
    print(f"  加權 F1:      {metrics['weighted_f1']:.3f}")
    print(f"  Macro F1:     {metrics['macro_f1']:.3f}")
    print(f"  QWK (κ):      {metrics['qwk']:.3f}")
    print(f"  Correlation:  {metrics['correlation']:.3f}")
    print(f"  AAUC:         {metrics['aauc']:.3f}")
    print(f"  WD:           {metrics['wasserstein_distance']:.3f}")

    print(f"\n分類報告:")
    print(metrics["classification_report"])

    # 繪製圖表
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=PLOT_DIR / "confusion_matrix.png",
    )
    plot_per_fold_comparison(
        results,
        save_path=PLOT_DIR / "fold_comparison.png",
    )
    plot_score_distribution(
        all_preds, all_labels,
        save_path=PLOT_DIR / "score_distribution.png",
    )

    # 儲存完整指標
    metrics_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    metrics_save["classification_report_text"] = metrics["classification_report"]

    metrics_file = OUTPUT_DIR / "evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_save, f, ensure_ascii=False, indent=2)
    print(f"\n指標已儲存至 {metrics_file}")
    print(f"圖表已儲存至 {PLOT_DIR}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
