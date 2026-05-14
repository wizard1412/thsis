#!/usr/bin/env python3
"""
kNN Few-Shot Prompt Generation
================================
對每個受試者，從資料集中為每個分數等級（0-3）各選出
特徵最相似（歐氏距離最近）的樣本作為 few-shot 範例。
使用 LLM-Lasso 選出的 6 個特徵做距離計算（StandardScaler 正規化）。

策略：Stratified kNN — 每個分數等級選一個最近鄰
優點：
  1. 保證 LLM 看到所有分數等級的範例
  2. 每個範例對當前受試者最具參考價值
  3. 避免固定範例造成的系統性偏差
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from run_feature_extraction_with_ratings import extract_full_features

# ===========================
# 常數
# ===========================
FPS = 240
BASE_PROMPT_FILE = "Prompt_knn_header.txt"
CSV_DIR = "../csv_files"
TEMPLATE_CSV = "scored_by_ChatGPT_promptC_template.csv"

# 用於 kNN 距離計算的特徵（LLM-Lasso 選出的 6 個）
LLMLASSO_FEATURES = [
    "finger_mvmnt_x_mean",
    "finger_mvmnt_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks",
]

# prompt 中顯示的特徵順序（同 fewshot_llmlasso）
DISPLAY_FEATURES = LLMLASSO_FEATURES


# ===========================
# 資料集建立
# ===========================

def build_dataset(csv_dir=CSV_DIR, template_csv=TEMPLATE_CSV):
    """
    提取所有受試者的特徵並整合標籤，回傳 DataFrame。
    label 欄位 = round(average_score_Doctors)，只保留有整數標籤的樣本。
    """
    # 載入標籤
    tpl = pd.read_csv(template_csv, encoding='utf-8-sig')

    # 建立 filename → label 映射
    label_map = {}
    for _, row in tpl.iterrows():
        fname = str(row.get('filename', '')).strip().replace('.csv', '')
        if not fname:
            continue
        for col in ['average_score_Doctors', 'Rating']:
            val = row.get(col, None)
            if val is not None and pd.notna(val) and val != '':
                try:
                    label_map[fname] = min(int(np.round(float(val))), 3)
                except (ValueError, TypeError):
                    pass
                break

    # 提取特徵
    rows = []
    for csv_path in sorted(Path(csv_dir).glob("*.csv")):
        stem = csv_path.stem
        if stem not in label_map:
            continue
        try:
            feat = extract_full_features(csv_path)
            feat['filename'] = stem
            feat['num_taps'] = feat.get('num_peaks', 0)
            feat['label'] = label_map[stem]
            rows.append(feat)
        except Exception as e:
            print(f"  [build_dataset] Warning: {stem} — {e}")

    df = pd.DataFrame(rows)
    # 只保留 DISPLAY_FEATURES 無 NaN 且 label 合法的樣本
    df = df.dropna(subset=LLMLASSO_FEATURES + ['label'])
    df['label'] = df['label'].astype(int)
    return df


def fit_scaler(df):
    """在全資料集上 fit StandardScaler（用 LLMLASSO_FEATURES）"""
    scaler = StandardScaler()
    scaler.fit(df[LLMLASSO_FEATURES].values)
    return scaler


# ===========================
# Stratified kNN 選例
# ===========================

def select_stratified_knn(test_features, all_df, scaler, exclude_filename=None):
    """
    對每個分數等級，從資料集（排除當前受試者）找最近鄰。

    Args:
        test_features : dict，當前受試者特徵
        all_df        : DataFrame，全資料集（含 label 欄位）
        scaler        : 已 fit 的 StandardScaler
        exclude_filename : 要排除的受試者 filename（LOOCV）

    Returns:
        list of (score, row_Series)，每個分數等級各一個
    """
    df = all_df.copy()
    if exclude_filename:
        df = df[df['filename'] != exclude_filename]
    df = df.dropna(subset=LLMLASSO_FEATURES + ['label'])

    # 正規化
    X = scaler.transform(df[LLMLASSO_FEATURES].values)
    test_vec_raw = np.array([
        test_features.get(f, np.nan) for f in LLMLASSO_FEATURES
    ], dtype=float)
    test_vec_raw = np.nan_to_num(test_vec_raw)
    test_vec = scaler.transform([test_vec_raw])[0]

    examples = []
    for score in sorted(df['label'].unique()):
        mask = (df['label'] == score).values
        subset_X = X[mask]
        subset_rows = df[mask].reset_index(drop=True)

        dists = np.linalg.norm(subset_X - test_vec, axis=1)
        best_i = int(np.argmin(dists))
        examples.append((score, subset_rows.iloc[best_i]))

    return examples


# ===========================
# 格式化 Few-Shot 範例
# ===========================

def format_example(score, row):
    """將一個 kNN 選出的樣本格式化為 few-shot 文字"""
    text = "───────────────\n"
    text += f"EXAMPLE FOR SCORE {score}\n"
    text += "───────────────\n"
    text += "KINEMATIC FEATURES :\n"
    text += f"  Number of taps: {int(row.get('num_taps', row.get('num_peaks', 'N/A')))}\n"
    for feat in DISPLAY_FEATURES:
        val = row.get(feat, np.nan)
        if isinstance(val, float):
            if np.isnan(val):
                text += f"  {feat}: N/A\n"
            else:
                text += f"  {feat}: {val:.4f}\n"
        else:
            text += f"  {feat}: {val}\n"
    text += f"\nThis example was rated as SCORE {score}.\n\n"
    return text


# ===========================
# Prompt 生成
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path=None,
                                   all_df=None, scaler=None):
    """
    為單一受試者生成包含 kNN few-shot 範例的 prompt。

    Args:
        csv_path        : 受試者 CSV 路徑
        base_prompt_path: header prompt 檔案路徑
        all_df          : 全資料集 DataFrame（由 build_dataset() 產生）
        scaler          : 已 fit 的 StandardScaler

    Returns:
        (prompt_str, features_dict)
    """
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        header = f.read()

    # 提取當前受試者特徵
    features = extract_full_features(csv_path)
    features['num_taps'] = features.get('num_peaks', 0)
    exclude_name = Path(csv_path).stem

    # 選出 kNN 範例
    if all_df is not None and scaler is not None:
        examples = select_stratified_knn(
            features, all_df, scaler, exclude_filename=exclude_name
        )
    else:
        examples = []

    # 組合 few-shot 範例文字
    fewshot_text = ""
    for score, row in examples:
        fewshot_text += format_example(score, row)

    # 目標受試者特徵區塊
    target_text  = "===================================\n"
    target_text += "TARGET SUBJECT — EVALUATE BELOW\n"
    target_text += "===================================\n"
    target_text += "KINEMATIC FEATURES :\n"
    target_text += f"  Number of taps: {features['num_taps']}\n"
    for feat in DISPLAY_FEATURES:
        val = features.get(feat, np.nan)
        if isinstance(val, float) and not np.isnan(val):
            val = round(val, 4)
        target_text += f"  {feat}: {val}\n"
    target_text += "\n"

    prompt = header + fewshot_text + target_text
    return prompt, features
