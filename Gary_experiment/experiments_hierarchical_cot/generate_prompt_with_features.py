#!/usr/bin/env python3
"""
Hierarchical CoT Prompt Generation
====================================
將特徵按 MDS-UPDRS 3.4 的三個評分維度分組輸出：
  Dimension A — Rhythm / Interruptions (periodEntropy, period_quartile_range)
  Dimension B — Speed                  (period_min)
  Dimension C — Amplitude              (finger_mvmnt_x_mean, finger_mvmnt_x_max)

維度之間提供臨床語言解釋，幫助 LLM 對應到評分標準。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy

from run_feature_extraction_with_ratings import extract_full_features

# ===========================
# 常數
# ===========================
FPS              = 240
BASE_PROMPT_FILE = "Prompt_hierarchical_cot.txt"

# LLM-Lasso 選出的特徵，依維度分組
DIM_A_FEATURES = ["periodEntropy", "period_quartile_range"]   # Rhythm / Interruptions
DIM_B_FEATURES = ["period_min"]                               # Speed
DIM_C_FEATURES = ["finger_mvmnt_x_mean", "finger_mvmnt_x_max"]  # Amplitude

ALL_SELECTED_FEATURES = DIM_A_FEATURES + DIM_B_FEATURES + DIM_C_FEATURES

# ===========================
# 臨床語言描述（根據資料集分布定義閾值）
# ===========================

def describe_period_entropy(val):
    if np.isnan(val): return "N/A"
    if val < 1.0:  return f"{val:.4f}  (low → regular rhythm)"
    if val < 1.6:  return f"{val:.4f}  (moderate → some rhythm irregularity)"
    return         f"{val:.4f}  (high → highly irregular rhythm)"

def describe_period_qr(val):
    if np.isnan(val): return "N/A"
    if val < 0.05: return f"{val:.4f}  (low → consistent timing)"
    if val < 0.09: return f"{val:.4f}  (moderate → some timing variability)"
    return         f"{val:.4f}  (high → large timing variability)"

def describe_period_min(val):
    if np.isnan(val): return "N/A"
    if val < 0.30: return f"{val:.4f} s  (short → capable of fast tapping)"
    if val < 0.42: return f"{val:.4f} s  (moderate → normal tapping speed)"
    return         f"{val:.4f} s  (long → slow even at fastest)"

def describe_mvmnt_x_mean(val):
    if np.isnan(val): return "N/A"
    if val > 1.0:  return f"{val:.4f}  (high → large average amplitude)"
    if val > 0.6:  return f"{val:.4f}  (moderate → reduced amplitude)"
    return         f"{val:.4f}  (low → small average amplitude)"

def describe_mvmnt_x_max(val):
    if np.isnan(val): return "N/A"
    if val > 5:    return f"{val}  (high → large peak amplitude)"
    if val > 2:    return f"{val}  (moderate → moderate peak amplitude)"
    return         f"{val}  (low → small peak amplitude)"

DESCRIBERS = {
    "periodEntropy":        describe_period_entropy,
    "period_quartile_range": describe_period_qr,
    "period_min":           describe_period_min,
    "finger_mvmnt_x_mean":  describe_mvmnt_x_mean,
    "finger_mvmnt_x_max":   describe_mvmnt_x_max,
}

# ===========================
# Prompt 生成
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path=None):
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    features = extract_full_features(csv_path)
    features['num_taps'] = features.get('num_peaks', 0)

    # 按維度組織特徵輸出
    feature_text  = "\n" + "=" * 40 + "\n"
    feature_text += "KINEMATIC FEATURES:\n"
    feature_text += "=" * 40 + "\n"

    # Tap count
    feature_text += f"\n  [Tap Count]\n"
    feature_text += f"  Number of taps: {features['num_taps']}\n"

    # Dimension A
    feature_text += f"\n  [Dimension A — Rhythm / Interruptions]\n"
    for feat in DIM_A_FEATURES:
        val = features.get(feat, np.nan)
        desc = DESCRIBERS[feat](val)
        feature_text += f"  {feat}: {desc}\n"

    # Dimension B
    feature_text += f"\n  [Dimension B — Speed]\n"
    for feat in DIM_B_FEATURES:
        val = features.get(feat, np.nan)
        desc = DESCRIBERS[feat](val)
        feature_text += f"  {feat}: {desc}\n"

    # Dimension C
    feature_text += f"\n  [Dimension C — Amplitude]\n"
    for feat in DIM_C_FEATURES:
        val = features.get(feat, np.nan)
        if isinstance(val, float) and not np.isnan(val):
            val = round(val, 4)
        desc = DESCRIBERS[feat](val)
        feature_text += f"  {feat}: {desc}\n"

    return base_prompt + feature_text, features


# ===========================
# 批次處理
# ===========================

def batch_generate_prompts(csv_dir="./csv_files", output_dir="./prompts",
                           base_prompt_path=None):
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE
    Path(output_dir).mkdir(exist_ok=True)

    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] {csv_name}")
        try:
            prompt, features = generate_prompt_with_features(csv_path, base_prompt_path)
            out = Path(output_dir) / f"{csv_name}_prompt.txt"
            with open(out, 'w', encoding='utf-8') as f:
                f.write(prompt)
            print(f"  ✓ {features['num_taps']} taps → {out}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


# ===========================
# 執行
# ===========================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        prompt, features = generate_prompt_with_features(csv_file)
        out = Path(csv_file).stem + "_hcot_prompt.txt"
        with open(out, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"✓ Prompt saved to: {out}")
        print(f"\n--- Features ---")
        print(f"Taps: {features['num_taps']}")
        for feat in ALL_SELECTED_FEATURES:
            print(f"  {feat}: {features.get(feat)}")
    else:
        batch_generate_prompts()
