"""
FTDualNet 資料載入與對齊模組

功能：
1. 載入外部資料集 (488 samples, 5 raters) 和使用者資料集 (53 samples, 2 raters)
2. 對齊兩個資料集的共同特徵（wrist → finger → 統一名稱）
3. 將每位評分者的分數展開為獨立訓練對
4. 各資料集分別做 Z-score 標準化
5. 提供 PyTorch Dataset 類別
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

from mtl_config import (
    EXTERNAL_DATASET_PATH,
    USER_DATASET_PATH,
    EXTERNAL_FEATURE_MAPPING,
    USER_FEATURE_MAPPING,
    UNIFIED_FEATURE_NAMES,
    EXTERNAL_RATER_COLUMNS,
    USER_RATER_COLUMNS,
    EXTERNAL_DIAGNOSIS_COLUMN,
    NUM_FOLDS,
    NUM_FEATURES,
    SCORE_NUM_CLASSES,
    RANDOM_SEED,
)


# ===========================
# 資料載入
# ===========================

def load_external_dataset(path=None):
    """
    載入外部資料集（488 samples, 5 raters, ~120 features）

    回傳：
        features_df: 僅包含對應特徵的 DataFrame（已重新命名為統一名稱）
        ratings_df:  包含 Rating1-5 的 DataFrame
        diagnosed:   Series, "yes"/"no" → 1/0
        sample_ids:  Series, 每筆資料的唯一識別
    """
    if path is None:
        path = EXTERNAL_DATASET_PATH

    df = pd.read_csv(path)

    # 提取並重新命名特徵
    ext_cols = list(EXTERNAL_FEATURE_MAPPING.keys())
    missing_cols = [c for c in ext_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"外部資料集缺少以下欄位: {missing_cols}")

    features_df = df[ext_cols].rename(columns=EXTERNAL_FEATURE_MAPPING)
    # 確保欄位順序一致
    features_df = features_df[UNIFIED_FEATURE_NAMES]

    # 提取評分
    ratings_df = df[EXTERNAL_RATER_COLUMNS].copy()

    # 提取診斷資訊
    diagnosed = df[EXTERNAL_DIAGNOSIS_COLUMN].map({"yes": 1, "no": 0}).astype(float)

    # 樣本 ID（使用 filename 或 index）
    if "filename" in df.columns:
        sample_ids = df["filename"].astype(str)
    else:
        sample_ids = pd.Series([f"ext_{i}" for i in range(len(df))], name="sample_id")

    return features_df, ratings_df, diagnosed, sample_ids


def load_user_dataset(path=None):
    """
    載入使用者資料集（53 samples, 2 raters, 56 features）

    回傳：
        features_df: 僅包含對應特徵的 DataFrame（已重新命名為統一名稱）
        ratings_df:  包含 Rating1-2 的 DataFrame
        sample_ids:  Series, 每筆資料的唯一識別
    """
    if path is None:
        path = USER_DATASET_PATH

    df = pd.read_csv(path)

    # 提取並重新命名特徵
    user_cols = list(USER_FEATURE_MAPPING.keys())
    missing_cols = [c for c in user_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"使用者資料集缺少以下欄位: {missing_cols}")

    features_df = df[user_cols].rename(columns=USER_FEATURE_MAPPING)
    features_df = features_df[UNIFIED_FEATURE_NAMES]

    # 提取評分
    ratings_df = df[USER_RATER_COLUMNS].copy()

    # 樣本 ID
    if "filename" in df.columns:
        sample_ids = df["filename"].astype(str)
    else:
        sample_ids = pd.Series([f"user_{i}" for i in range(len(df))], name="sample_id")

    return features_df, ratings_df, sample_ids


# ===========================
# 特徵標準化
# ===========================

def standardize_features(ext_features, user_features):
    """
    各資料集分別做 Z-score 標準化
    （因為 wrist vs finger 的量值尺度不同，需各自標準化）

    回傳：
        ext_scaled:    標準化後的外部特徵 (np.ndarray)
        user_scaled:   標準化後的使用者特徵 (np.ndarray)
        ext_scaler:    外部資料集的 scaler（推論時需要）
        user_scaler:   使用者資料集的 scaler（推論時需要）
    """
    ext_scaler = StandardScaler()
    user_scaler = StandardScaler()

    ext_scaled = ext_scaler.fit_transform(ext_features.values)
    user_scaled = user_scaler.fit_transform(user_features.values)

    return ext_scaled, user_scaled, ext_scaler, user_scaler


# ===========================
# 評分者展開
# ===========================

def expand_by_raters(features, ratings_df, sample_ids, diagnosed=None):
    """
    將每位評分者的分數展開為獨立的訓練對

    Input:  1 row  × [features, R1=3, R2=2, R3=2]
    Output: 3 rows × [features, score=3], [features, score=2], [features, score=2]

    參數：
        features:    np.ndarray, shape (N, num_features)
        ratings_df:  DataFrame, 包含各 rater 的分數
        sample_ids:  Series, 每筆資料的唯一識別
        diagnosed:   Series or None, 診斷標籤 (1/0)

    回傳：
        expanded_features: np.ndarray, shape (M, num_features)
        expanded_scores:   np.ndarray, shape (M,), 整數 0-4
        expanded_diag:     np.ndarray, shape (M,), 1/0 或 None
        expanded_groups:   np.ndarray, shape (M,), 樣本群組 ID（用於 CV 分組）
    """
    expanded_features = []
    expanded_scores = []
    expanded_diag = []
    expanded_groups = []

    rater_columns = ratings_df.columns.tolist()

    for i in range(len(features)):
        for rater_col in rater_columns:
            score = ratings_df.iloc[i][rater_col]

            # 跳過缺失值或非數字
            if pd.isna(score):
                continue

            score_int = int(round(float(score)))
            # 確保分數在 0-4 範圍內
            if score_int < 0 or score_int > 4:
                continue

            expanded_features.append(features[i])
            expanded_scores.append(score_int)
            expanded_groups.append(i)  # 同一 sample 的所有展開列共用 group ID

            if diagnosed is not None:
                expanded_diag.append(diagnosed.iloc[i])

    expanded_features = np.array(expanded_features)
    expanded_scores = np.array(expanded_scores, dtype=np.int64)
    expanded_groups = np.array(expanded_groups, dtype=np.int64)

    if diagnosed is not None:
        expanded_diag = np.array(expanded_diag, dtype=np.float32)
    else:
        expanded_diag = None

    return expanded_features, expanded_scores, expanded_diag, expanded_groups


# ===========================
# PyTorch Dataset
# ===========================

class FingerTappingDataset(Dataset):
    """
    手指敲擊特徵資料集

    每筆資料包含：
        - features: 標準化後的特徵向量
        - score:    MDS-UPDRS 3.4 分數 (0-4)
        - diagnosed: 是否確診 PD (1/0)，可選
    """

    def __init__(self, features, scores, diagnosed=None):
        self.features = torch.FloatTensor(features)
        self.scores = torch.LongTensor(scores)

        if diagnosed is not None:
            self.diagnosed = torch.FloatTensor(diagnosed)
        else:
            self.diagnosed = None

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "score": self.scores[idx],
        }
        if self.diagnosed is not None:
            item["diagnosed"] = self.diagnosed[idx]
        return item


class AutoencoderDataset(Dataset):
    """
    自編碼器預訓練用資料集（不需要標籤）
    僅包含特徵，用於重建任務
    """

    def __init__(self, features):
        self.features = torch.FloatTensor(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# ===========================
# 交叉驗證分割
# ===========================

def create_cv_splits(scores, groups, n_folds=None):
    """
    建立分層群組交叉驗證分割

    確保：
    1. 按分數分布做分層抽樣
    2. 同一樣本的所有評分者展開列在同一 fold

    參數：
        scores: np.ndarray, 分數標籤
        groups: np.ndarray, 群組 ID（同一 sample 共用）
        n_folds: int, 折數

    回傳：
        splits: list of (train_idx, val_idx)
    """
    if n_folds is None:
        n_folds = NUM_FOLDS

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    splits = []
    for train_idx, val_idx in sgkf.split(np.zeros(len(scores)), scores, groups):
        splits.append((train_idx, val_idx))

    return splits


# ===========================
# 主要資料準備流程
# ===========================

def prepare_all_data():
    """
    完整的資料準備流程

    回傳：
        data: dict 包含所有準備好的資料
            - "ae_features":     用於自編碼器預訓練的特徵 (所有唯一樣本)
            - "train_features":  用於分類訓練的特徵 (評分者展開後)
            - "train_scores":    對應的分數標籤
            - "train_diagnosed": 對應的診斷標籤
            - "train_groups":    對應的群組 ID
            - "cv_splits":       交叉驗證分割
            - "ext_scaler":      外部資料集的 scaler
            - "user_scaler":     使用者資料集的 scaler
            - "feature_names":   統一特徵名列表
    """
    print("=" * 60)
    print("載入資料...")
    print("=" * 60)

    # 1. 載入兩個資料集
    ext_features, ext_ratings, ext_diagnosed, ext_ids = load_external_dataset()
    user_features, user_ratings, user_ids = load_user_dataset()

    print(f"外部資料集: {len(ext_features)} 筆樣本, {len(ext_features.columns)} 個特徵")
    print(f"使用者資料集: {len(user_features)} 筆樣本, {len(user_features.columns)} 個特徵")
    print(f"統一特徵數: {len(UNIFIED_FEATURE_NAMES)}")

    # 2. 各自標準化
    print("\n標準化特徵...")
    ext_scaled, user_scaled, ext_scaler, user_scaler = standardize_features(
        ext_features, user_features
    )

    # 3. 合併為自編碼器預訓練資料（所有唯一樣本，不展開）
    ae_features = np.vstack([ext_scaled, user_scaled])
    print(f"自編碼器預訓練資料: {ae_features.shape[0]} 筆")

    # 4. 評分者展開
    print("\n展開評分者分數...")

    # 外部資料集：展開 5 位評分者
    ext_exp_feat, ext_exp_scores, ext_exp_diag, ext_exp_groups = expand_by_raters(
        ext_scaled, ext_ratings, ext_ids, ext_diagnosed
    )
    print(f"  外部: {len(ext_scaled)} 筆 × 5 raters → {len(ext_exp_scores)} 筆訓練對")

    # 使用者資料集：展開 2 位評分者（無 diagnosed 資訊）
    user_exp_feat, user_exp_scores, _, user_exp_groups = expand_by_raters(
        user_scaled, user_ratings, user_ids
    )
    # 使用者資料集的 group ID 需要偏移，避免與外部重疊
    user_exp_groups = user_exp_groups + len(ext_scaled)
    print(f"  使用者: {len(user_scaled)} 筆 × 2 raters → {len(user_exp_scores)} 筆訓練對")

    # 合併
    all_features = np.vstack([ext_exp_feat, user_exp_feat])
    all_scores = np.concatenate([ext_exp_scores, user_exp_scores])
    all_groups = np.concatenate([ext_exp_groups, user_exp_groups])

    # diagnosed 標籤：使用者資料集沒有，用 -1 表示缺失
    user_exp_diag = np.full(len(user_exp_scores), -1.0, dtype=np.float32)
    all_diagnosed = np.concatenate([ext_exp_diag, user_exp_diag])

    # 合併分數類別：將 score 4 → 3（合併為 3+ 類別）
    max_class = SCORE_NUM_CLASSES - 1  # 3
    all_scores = np.minimum(all_scores, max_class)
    print(f"\n分數合併: 原始 0-4 → 0-{max_class}（分數 {max_class} 以上合併為 {max_class}+）")

    print(f"合併後總訓練對: {len(all_scores)} 筆")
    print(f"分數分布:")
    for s in range(SCORE_NUM_CLASSES):
        count = np.sum(all_scores == s)
        label = f"{s}+" if s == max_class else str(s)
        print(f"  分數 {label}: {count} 筆 ({count/len(all_scores)*100:.1f}%)")

    # 5. 建立交叉驗證分割
    print(f"\n建立 {NUM_FOLDS}-Fold 交叉驗證分割...")
    cv_splits = create_cv_splits(all_scores, all_groups)
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"  Fold {i+1}: 訓練 {len(train_idx)} 筆, 驗證 {len(val_idx)} 筆")

    print("=" * 60)
    print("資料準備完成！")
    print("=" * 60)

    return {
        "ae_features": ae_features,
        "train_features": all_features,
        "train_scores": all_scores,
        "train_diagnosed": all_diagnosed,
        "train_groups": all_groups,
        "cv_splits": cv_splits,
        "ext_scaler": ext_scaler,
        "user_scaler": user_scaler,
        "feature_names": UNIFIED_FEATURE_NAMES,
    }


# ===========================
# 獨立執行：測試資料載入
# ===========================

if __name__ == "__main__":
    data = prepare_all_data()

    print(f"\n自編碼器資料形狀: {data['ae_features'].shape}")
    print(f"訓練資料形狀: {data['train_features'].shape}")
    print(f"分數標籤形狀: {data['train_scores'].shape}")
    print(f"診斷標籤形狀: {data['train_diagnosed'].shape}")
    print(f"群組 ID 形狀: {data['train_groups'].shape}")
    print(f"特徵名列表: {data['feature_names']}")
