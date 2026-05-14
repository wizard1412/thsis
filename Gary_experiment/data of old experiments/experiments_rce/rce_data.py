"""
RCE 資料載入模組

與 mtl_data.py 的關鍵差異：
- 不展開 rater！每個 sample 保持一筆，所有 rater 分數存為向量
- Dataset 回傳 ratings: (MAX_NUM_RATERS,)，缺失 rater 填 -1
- 交叉驗證以 sample 為單位（不需要 group）
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from rce_config import (
    EXTERNAL_DATASET_PATH,
    USER_DATASET_PATH,
    EXTERNAL_FEATURE_MAPPING,
    USER_FEATURE_MAPPING,
    UNIFIED_FEATURE_NAMES,
    EXTERNAL_RATER_COLUMNS,
    USER_RATER_COLUMNS,
    EXTERNAL_DIAGNOSIS_COLUMN,
    MAX_NUM_RATERS,
    NUM_FOLDS,
    SCORE_NUM_CLASSES,
    RANDOM_SEED,
)


# ===========================
# 資料載入
# ===========================

def load_external_dataset(path=None):
    """載入外部資料集（488 samples, 5 raters）"""
    if path is None:
        path = EXTERNAL_DATASET_PATH

    df = pd.read_csv(path)

    ext_cols = list(EXTERNAL_FEATURE_MAPPING.keys())
    missing_cols = [c for c in ext_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"外部資料集缺少以下欄位: {missing_cols}")

    features_df = df[ext_cols].rename(columns=EXTERNAL_FEATURE_MAPPING)
    features_df = features_df[UNIFIED_FEATURE_NAMES]

    # 提取所有 rater 的分數為矩陣 (N, 5)
    ratings = df[EXTERNAL_RATER_COLUMNS].values.copy()

    # 診斷資訊
    diagnosed = df[EXTERNAL_DIAGNOSIS_COLUMN].map({"yes": 1, "no": 0}).values.astype(np.float32)

    return features_df, ratings, diagnosed


def load_user_dataset(path=None):
    """載入使用者資料集（53 samples, 2 raters）"""
    if path is None:
        path = USER_DATASET_PATH

    df = pd.read_csv(path)

    user_cols = list(USER_FEATURE_MAPPING.keys())
    missing_cols = [c for c in user_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"使用者資料集缺少以下欄位: {missing_cols}")

    features_df = df[user_cols].rename(columns=USER_FEATURE_MAPPING)
    features_df = features_df[UNIFIED_FEATURE_NAMES]

    # 只有 2 位 rater，需要 padding 到 MAX_NUM_RATERS
    raw_ratings = df[USER_RATER_COLUMNS].values.copy()
    # (53, 2) → (53, 5)，填 -1 表示缺失
    ratings = np.full((len(raw_ratings), MAX_NUM_RATERS), -1, dtype=np.float64)
    ratings[:, :raw_ratings.shape[1]] = raw_ratings

    return features_df, ratings


# ===========================
# 特徵標準化
# ===========================

def standardize_features(ext_features, user_features):
    """各資料集分別做 Z-score 標準化"""
    ext_scaler = StandardScaler()
    user_scaler = StandardScaler()

    ext_scaled = ext_scaler.fit_transform(ext_features.values)
    user_scaled = user_scaler.fit_transform(user_features.values)

    return ext_scaled, user_scaled, ext_scaler, user_scaler


# ===========================
# PyTorch Dataset
# ===========================

class RCEDataset(Dataset):
    """
    RCE 訓練用資料集

    與 FingerTappingDataset 的關鍵差異：
    - 不展開 rater，每個 sample 一筆
    - ratings: (MAX_NUM_RATERS,) 所有 rater 的分數，-1 表示缺失
    - majority_score: 多數決分數（用於驗證評估）
    """

    def __init__(self, features, ratings, diagnosed=None):
        """
        features: (N, D) 特徵
        ratings:  (N, R) 每位 rater 的分數，-1 表示缺失
        diagnosed: (N,) 或 None
        """
        self.features = torch.FloatTensor(features)
        self.ratings = torch.LongTensor(ratings)  # (N, R)

        # 計算多數決分數（用於驗證和分層抽樣）
        self.majority_scores = self._compute_majority_scores(ratings)

        if diagnosed is not None:
            self.diagnosed = torch.FloatTensor(diagnosed)
        else:
            self.diagnosed = None

    def _compute_majority_scores(self, ratings):
        """計算每個 sample 的多數決分數"""
        majority = []
        for row in ratings:
            valid = row[row >= 0]
            if len(valid) == 0:
                majority.append(0)
            else:
                valid_int = valid.astype(int)
                # 合併 4 → 3
                valid_int = np.minimum(valid_int, SCORE_NUM_CLASSES - 1)
                counts = np.bincount(valid_int, minlength=SCORE_NUM_CLASSES)
                majority.append(counts.argmax())
        return torch.LongTensor(majority)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "ratings": self.ratings[idx],            # (R,) 所有 rater 的分數
            "majority_score": self.majority_scores[idx],  # 多數決分數
        }
        if self.diagnosed is not None:
            item["diagnosed"] = self.diagnosed[idx]
        return item


class AutoencoderDataset(Dataset):
    """自編碼器預訓練用資料集"""

    def __init__(self, features):
        self.features = torch.FloatTensor(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


# ===========================
# 交叉驗證分割
# ===========================

def create_cv_splits(majority_scores, n_folds=None):
    """
    建立分層交叉驗證分割

    注意：RCE 不展開 rater，因此不需要 GroupKFold，
    直接用 StratifiedKFold 以 sample 為單位分割。
    """
    if n_folds is None:
        n_folds = NUM_FOLDS

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    splits = []
    for train_idx, val_idx in skf.split(np.zeros(len(majority_scores)), majority_scores):
        splits.append((train_idx, val_idx))

    return splits


# ===========================
# 主要資料準備流程
# ===========================

def prepare_all_data():
    """
    完整的資料準備流程

    回傳：
        data: dict
            - "ae_features":       自編碼器預訓練特徵 (541, 33)
            - "all_features":      所有樣本特徵 (541, 33)
            - "all_ratings":       所有 rater 分數 (541, 5)，-1=缺失
            - "all_diagnosed":     診斷標籤 (541,)，-1=無標籤
            - "majority_scores":   多數決分數 (541,)，已合併 3+
            - "cv_splits":         交叉驗證分割
            - "ext_scaler":        外部 scaler
            - "user_scaler":       使用者 scaler
            - "feature_names":     特徵名列表
            - "dataset_source":    各 sample 來源 ("ext" / "user")
    """
    print("=" * 60)
    print("Loading data (RCE mode: no rater expansion)...")
    print("=" * 60)

    # 1. 載入
    ext_features, ext_ratings, ext_diagnosed = load_external_dataset()
    user_features, user_ratings = load_user_dataset()

    print(f"External dataset: {len(ext_features)} samples, {ext_ratings.shape[1]} raters")
    print(f"User dataset: {len(user_features)} samples, {user_ratings.shape[1]} raters (with padding)")

    # 2. 標準化
    print("\nStandardizing features...")
    ext_scaled, user_scaled, ext_scaler, user_scaler = standardize_features(
        ext_features, user_features
    )

    # 3. 合併
    ae_features = np.vstack([ext_scaled, user_scaled])
    all_features = ae_features.copy()

    # ratings: 處理 NaN，合併 score 4 → 3
    ext_ratings_clean = ext_ratings.copy()
    ext_ratings_clean[np.isnan(ext_ratings_clean)] = -1
    user_ratings_clean = user_ratings.copy()
    user_ratings_clean[np.isnan(user_ratings_clean)] = -1

    all_ratings = np.vstack([ext_ratings_clean, user_ratings_clean]).astype(np.int64)

    # 合併 4 → 3（只對有效分數）
    max_class = SCORE_NUM_CLASSES - 1
    valid_mask = all_ratings >= 0
    all_ratings[valid_mask] = np.minimum(all_ratings[valid_mask], max_class)

    # diagnosed
    user_diagnosed = np.full(len(user_features), -1.0, dtype=np.float32)
    all_diagnosed = np.concatenate([ext_diagnosed, user_diagnosed])

    # dataset source
    dataset_source = (
        ["ext"] * len(ext_features) + ["user"] * len(user_features)
    )

    print(f"\nMerged: {len(all_features)} samples x {MAX_NUM_RATERS} raters")

    # 統計分數分布（基於所有有效 rating）
    valid_ratings = all_ratings[all_ratings >= 0]
    print(f"Total valid ratings: {len(valid_ratings)}")
    print("Score distribution:")
    for s in range(SCORE_NUM_CLASSES):
        count = np.sum(valid_ratings == s)
        label = f"{s}+" if s == max_class else str(s)
        print(f"  Score {label}: {count} ({count/len(valid_ratings)*100:.1f}%)")

    # 計算多數決分數
    majority_scores = []
    for row in all_ratings:
        valid = row[row >= 0]
        if len(valid) == 0:
            majority_scores.append(0)
        else:
            counts = np.bincount(valid.astype(int), minlength=SCORE_NUM_CLASSES)
            majority_scores.append(counts.argmax())
    majority_scores = np.array(majority_scores)

    # Inter-rater agreement 統計
    print("\nInter-rater agreement statistics:")
    for i in range(len(all_ratings)):
        valid = all_ratings[i][all_ratings[i] >= 0]
        if len(valid) >= 2:
            unique_scores = len(set(valid))
            if unique_scores == 1:
                pass  # 完全一致
    full_agree = sum(
        1 for row in all_ratings
        if len(set(row[row >= 0])) == 1 and len(row[row >= 0]) >= 2
    )
    total_multi = sum(1 for row in all_ratings if sum(row >= 0) >= 2)
    print(f"  Full agreement: {full_agree}/{total_multi} ({full_agree/total_multi*100:.1f}%)")

    # 4. 交叉驗證分割
    print(f"\nCreating {NUM_FOLDS}-Fold cross-validation splits...")
    cv_splits = create_cv_splits(majority_scores)
    for i, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"  Fold {i+1}: Train {len(train_idx)}, Val {len(val_idx)}")

    print("=" * 60)
    print("Data preparation complete!")
    print("=" * 60)

    return {
        "ae_features": ae_features,
        "all_features": all_features,
        "all_ratings": all_ratings,
        "all_diagnosed": all_diagnosed,
        "majority_scores": majority_scores,
        "cv_splits": cv_splits,
        "ext_scaler": ext_scaler,
        "user_scaler": user_scaler,
        "feature_names": UNIFIED_FEATURE_NAMES,
        "dataset_source": dataset_source,
    }


if __name__ == "__main__":
    data = prepare_all_data()

    print(f"\n自編碼器資料: {data['ae_features'].shape}")
    print(f"特徵: {data['all_features'].shape}")
    print(f"Ratings: {data['all_ratings'].shape}")
    print(f"Diagnosed: {data['all_diagnosed'].shape}")
    print(f"Majority scores: {data['majority_scores'].shape}")

    # 展示幾筆 ratings
    print("\n前 5 筆 ratings (外部資料集):")
    for i in range(5):
        print(f"  Sample {i}: {data['all_ratings'][i]} → majority={data['majority_scores'][i]}")

    print("\n前 3 筆 ratings (使用者資料集):")
    start = 488
    for i in range(start, start + 3):
        print(f"  Sample {i}: {data['all_ratings'][i]} → majority={data['majority_scores'][i]}")
