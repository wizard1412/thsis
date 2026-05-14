#!/usr/bin/env python3
"""
特徵提取系統 - 含 tap amplitude 序列版本

在原有特徵的基礎上，額外新增兩組固定長度的逐拍振幅特徵：

  tap_amp_norm_01..20  — 相對正規化（除以該受試者最大振幅）：保留序列形狀（遞減趨勢）
  tap_amp_px_01..20    — 原始 pixel 值：保留絕對幅度（MDS-UPDRS 重要指標）

- 拍次不足 N_TAP_POSITIONS 的部分填 NaN（LOOCV 中由 SimpleImputer 插補）
- 拍次超過 N_TAP_POSITIONS 的僅取前 N_TAP_POSITIONS 拍

輸出: ./features_dataset_tap_seq.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 引入原有的特徵提取函數（已包含振幅提取邏輯）
from run_feature_extraction_with_ratings import (
    extract_full_features,
    FPS,
)

# ===========================
# 配置
# ===========================

CSV_DIR      = "./csv_files"
RATINGS_FILE = Path(__file__).parent / "scored_by_ChatGPT_promptC_template.csv"
OUTPUT_PATH  = "./features_dataset_tap_seq.csv"

# 固定序列長度（超過部分截斷，不足部分 NaN）
N_TAP_POSITIONS = 20


# ===========================
# 序列特徵欄位名稱
# ===========================

TAP_SEQ_NORM_COLS = [f"tap_amp_norm_{k:02d}" for k in range(1, N_TAP_POSITIONS + 1)]
TAP_SEQ_PX_COLS  = [f"tap_amp_px_{k:02d}"   for k in range(1, N_TAP_POSITIONS + 1)]
TAP_SEQ_COLS = TAP_SEQ_NORM_COLS + TAP_SEQ_PX_COLS


def extract_tap_seq_features(tap_amplitudes: list, n: int = N_TAP_POSITIONS) -> dict:
    """
    將逐拍振幅序列展開成兩組 n 個固定長度的 scalar 欄位：
      tap_amp_norm_* : 相對正規化（除以最大振幅），保留形狀/遞減趨勢
      tap_amp_px_*   : 原始 pixel 值，保留絕對幅度

    Args:
        tap_amplitudes: list of float (來自 extract_full_features 的 'tap_amplitudes')
        n: 固定序列長度

    Returns:
        dict with tap_amp_norm_01..n and tap_amp_px_01..n
    """
    result = {}
    for k in range(1, n + 1):
        result[f"tap_amp_norm_{k:02d}"] = np.nan
        result[f"tap_amp_px_{k:02d}"]   = np.nan

    if not tap_amplitudes or len(tap_amplitudes) == 0:
        return result

    arr = np.array(tap_amplitudes, dtype=float)
    max_amp = arr.max()

    for k in range(min(len(arr), n)):
        result[f"tap_amp_px_{k+1:02d}"] = round(float(arr[k]), 2)
        if max_amp > 0:
            result[f"tap_amp_norm_{k+1:02d}"] = round(float(arr[k] / max_amp), 4)

    return result


# ===========================
# 批次提取
# ===========================

def batch_extract(csv_dir=CSV_DIR, output_path=OUTPUT_PATH, ratings_file=RATINGS_FILE):
    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    # 載入評分
    ratings_df = None
    rating_column_map = {}
    if ratings_file and Path(ratings_file).exists():
        ratings_df = pd.read_csv(ratings_file)
        print(f"Loaded ratings from: {ratings_file}")
        if 'label_Dr. Tan' in ratings_df.columns:
            rating_column_map['Rating1'] = 'label_Dr. Tan'
        if 'label_Dr. Chien' in ratings_df.columns:
            rating_column_map['Rating2'] = 'label_Dr. Chien'
        if 'average_score_Doctors' in ratings_df.columns:
            rating_column_map['Rating'] = 'average_score_Doctors'
        for col in ['Rating1', 'Rating2', 'Rating']:
            if col in ratings_df.columns:
                rating_column_map[col] = col

    all_features = []

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] {csv_name}...", end=" ")

        try:
            # 完整特徵提取（含 tap_amplitudes list）
            features = extract_full_features(csv_path)
            features['filename'] = csv_name

            # 手的資訊
            if '_Hands_L_' in csv_name:
                features['hand'] = 'L'
            elif '_Hands_R_' in csv_name:
                features['hand'] = 'R'
            else:
                features['hand'] = ''

            # 評分
            if ratings_df is not None:
                matching_row = ratings_df[
                    (ratings_df['filename'] == csv_name) |
                    (ratings_df['filename'] == csv_name + '.csv')
                ]
                if not matching_row.empty:
                    row = matching_row.iloc[0]
                    for target_col, source_col in rating_column_map.items():
                        if source_col in row:
                            value = row[source_col]
                            import pandas as pd_inner
                            if pd_inner.notna(value) and value != '':
                                features[target_col] = value

            # 新增：展開逐拍振幅序列
            tap_amplitudes = features.get('tap_amplitudes', [])
            seq_features = extract_tap_seq_features(tap_amplitudes)
            features.update(seq_features)

            # tap_amplitudes 是 list，不適合直接存 CSV → 移除
            features.pop('tap_amplitudes', None)

            all_features.append(features)
            n_taps = features.get('num_peaks', 0)
            print(f"OK ({n_taps} taps)")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Tap sequence columns ({N_TAP_POSITIONS}): {TAP_SEQ_COLS[:3]} ... {TAP_SEQ_COLS[-1]}")

    return df


# ===========================
# 執行
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("特徵提取系統 (含 tap amplitude 序列)")
    print("=" * 60 + "\n")

    df = batch_extract()

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n總共處理檔案數: {len(df)}")
    print(f"特徵數量: {len(df.columns)}")
    print(f"其中 tap_amp_norm_* 欄位數: {N_TAP_POSITIONS}")
    print(f"其中 tap_amp_px_*   欄位數: {N_TAP_POSITIONS}")

    # 簡單統計：各位置的覆蓋率（用 norm 代表，px 相同）
    print(f"\n各拍次覆蓋率（non-NaN 比例）:")
    for col in TAP_SEQ_NORM_COLS:
        if col in df.columns:
            coverage = df[col].notna().mean()
            print(f"  {col}: {coverage:.1%}")
        else:
            print(f"  {col}: missing")
