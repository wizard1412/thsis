"""
暫時比較圖：FTDualNet vs LLM Models vs Human Baseline
比較 6 個指標：MAE, ±1 Acc, r (Correlation), κ (QWK), AAUC, WD
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 支援中文字型
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ===========================
# 數據
# ===========================

# FTDualNet (MTL) - from evaluation_metrics.json (CV results, vs rater-expanded labels)
ftdualnet = {
    "MAE": 0.840,
    "±1 Acc": 0.825,
    "r": 0.301,
    "κ": 0.291,
    "AAUC": 0.587,
    "WD": 0.235,
}

# deepseek-r1 vs Consensus
deepseek = {
    "MAE": 0.523,
    "±1 Acc": 0.981,
    "r": 0.528,
    "κ": 0.365,
    "AAUC": 0.772,
    "WD": 0.285,
}

# Gemini3 vs Consensus
gemini3 = {
    "MAE": 0.706,
    "±1 Acc": 0.868,
    "r": 0.520,
    "κ": 0.369,
    "AAUC": 0.724,
    "WD": 0.600,
}

# Human Baseline (Tan vs Chien inter-rater)
human = {
    "MAE": 0.755,
    "±1 Acc": 0.830,
    "r": 0.414,
    "κ": 0.364,
    "AAUC": 0.773,
    "WD": 0.415,
}

# ===========================
# 繪圖
# ===========================

metrics = ["MAE", "±1 Acc", "r", "κ", "AAUC", "WD"]
models = {
    "FTDualNet (MTL)": ftdualnet,
    "DeepSeek-R1": deepseek,
    "Gemini 3 Flash": gemini3,
    "Human (Inter-rater)": human,
}

colors = ["#E74C3C", "#3498DB", "#2ECC71", "#95A5A6"]

fig, axes = plt.subplots(1, 6, figsize=(24, 5))

x = np.arange(len(models))
width = 0.6
model_names = list(models.keys())

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = [models[m][metric] for m in model_names]
    bars = ax.bar(x, values, width, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

    # 在柱狀圖上方顯示數值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title(metric, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, max(values) * 1.25)

    # 標註「越低越好」或「越高越好」
    if metric in ["MAE", "WD"]:
        ax.set_ylabel("↓ Lower is better", fontsize=9, color="gray")
    else:
        ax.set_ylabel("↑ Higher is better", fontsize=9, color="gray")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle(
    "FTDualNet (MTL) vs LLM Zero-Shot vs Human Baseline\n"
    "Note: FTDualNet = 5-fold CV; LLMs = vs Consensus; Human = Inter-rater",
    fontsize=13, fontweight="bold", y=1.02,
)
fig.tight_layout()

save_path = "results/temp_comparison.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"比較圖已儲存至 {save_path}")
plt.close(fig)

# ===========================
# 也印出表格
# ===========================

print("\n" + "=" * 85)
print(f"{'Model':<22} {'MAE':>7} {'±1Acc':>7} {'r':>7} {'κ':>7} {'AAUC':>7} {'WD':>7}")
print("-" * 85)
for name, data in models.items():
    print(f"{name:<22} {data['MAE']:>7.3f} {data['±1 Acc']:>7.3f} {data['r']:>7.3f} {data['κ']:>7.3f} {data['AAUC']:>7.3f} {data['WD']:>7.3f}")
print("=" * 85)
print("\nNote:")
print("  - FTDualNet: 5-fold CV, pred vs rater-expanded labels")
print("  - LLMs: vs Consensus score (mean of 10 evaluations)")
print("  - Human: Inter-rater agreement (Tan vs Chien)")
