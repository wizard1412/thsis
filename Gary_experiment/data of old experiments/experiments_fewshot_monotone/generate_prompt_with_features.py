#!/usr/bin/env python3
"""
Feature Extraction and Prompt Generation Script
Function: Extract features and append to Prompt
Output: Original Prompt.txt + Raw Data + Statistical Features

Features included:
- Raw data: num_taps, tap_periods, tap_amplitudes
- Statistical features (selected by correlation analysis):
  period_stdev, periodEntropy, finger_mvmnt_x_mean, finger_mvmnt_dist_mean,
  speed_mean, period_max, maxFreezeDuration, aperiodicity
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy
from pathlib import Path

# 引入您現有的全量特徵提取函數
from run_feature_extraction_with_ratings import extract_full_features

# ===========================
# Constants
# ===========================

FPS = 240  # Frame rate
SPEED_THRESHOLD = 50  # Interruption threshold (degrees/sec)
FREEZE_DURATION = 0.30  # Freeze duration threshold (seconds)

# Prompt file configuration
BASE_PROMPT_FILE = "Prompt_fewshot_monotone.txt"

# ===========================
# LLMLasso 選出的特徵配置
# ===========================
# 請將 LLMLasso 模型選出的特徵名稱放入此陣列中
# 此處為範例：您可以隨時根據最新的 Lasso 實驗結果修改此清單
LLMLASSO_SELECTED_FEATURES = [
    "finger_mvmnt_x_mean",
    "finger_mvmnt_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks"
]

# ===========================
# Prompt Generation
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path=None, selected_features=None):
    """
    Generate Prompt with dynamically selected LLMLasso features
    Structure: Base Prompt -> LLMLasso Selected Features

    Args:
        csv_path: Path to CSV file
        base_prompt_path: Path to base Prompt file (default: use BASE_PROMPT_FILE constant)
        selected_features: List of feature names to include (default: use LLMLASSO_SELECTED_FEATURES constant)

    Returns:
        prompt: Complete prompt with features
        features: Feature dictionary (for program use)
    """
    if selected_features is None:
        selected_features = LLMLASSO_SELECTED_FEATURES

    # 1. Read base Prompt
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    # 2. Extract ALL features using the comprehensive script
    full_features = extract_full_features(csv_path)

    # 為了向下相容 batch_score.py 日誌顯示，補上 'num_taps'
    full_features['num_taps'] = full_features.get('num_peaks', 0)

    # 3. Format dynamic features section based on selected_features list
    feature_text = "\n" + "="*35 + "\n"
    feature_text += "TARGET KINEMATIC FEATURES TO SCORE\n"
    feature_text += "="*35 + "\n"
    feature_text += f"Number of taps: {full_features['num_taps']}\n"

    for feat in selected_features:
        if feat in full_features:
            val = full_features[feat]
            if isinstance(val, float) and not np.isnan(val):
                val = round(val, 4)
            feature_text += f"{feat}: {val}\n"

    # 5. Combine: Base Prompt -> Features
    enhanced_prompt = base_prompt + feature_text

    return enhanced_prompt, full_features


# ===========================
# Batch Processing
# ===========================

def batch_generate_prompts(csv_dir="./csv_files", output_dir="./prompts",
                          base_prompt_path=None):
    """
    Batch process all CSV files

    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory for output prompt files
        base_prompt_path: Path to base prompt file (default: use BASE_PROMPT_FILE constant)
    """
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE
    Path(output_dir).mkdir(exist_ok=True)

    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    results = {}

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")

        try:
            # Generate Prompt with features
            prompt, features = generate_prompt_with_features(
                csv_path, base_prompt_path
            )

            # Save Prompt
            output_path = Path(output_dir) / f"{csv_name}_prompt.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt)

            results[csv_name] = {
                'success': True,
                'features': {k: features.get(k) for k in LLMLASSO_SELECTED_FEATURES + ['num_taps']},
                'prompt_path': str(output_path)
            }

            print(f"  ✓ Features: {features['num_taps']} taps")
            print(f"  ✓ Prompt saved: {output_path}\n")

        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results[csv_name] = {
                'success': False,
                'error': str(e)
            }

    print(f"\n✓ Complete! All prompts generated")
    return results


# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single file mode
        csv_file = sys.argv[1]
        print(f"Processing single file: {csv_file}")
        print(f"Using prompt file: {BASE_PROMPT_FILE}\n")

        prompt, features = generate_prompt_with_features(csv_file)

        # Save prompt
        output_file = Path(csv_file).stem + "_prompt.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Display features
        print(f"✓ Prompt saved to: {output_file}")
        print(f"\n--- LLMLASSO SELECTED FEATURES ---")
        print(f"Number of taps: {features['num_taps']}")
        for feat in LLMLASSO_SELECTED_FEATURES:
            if feat in features:
                print(f"{feat}: {features[feat]}")

    else:
        # Batch mode
        print("="*60)
        print("Feature Extraction System (Dynamic LLMLasso Features)")
        print("="*60 + "\n")
        print(f"Using prompt file: {BASE_PROMPT_FILE}\n")
        batch_generate_prompts()
