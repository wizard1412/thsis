"""
FTDualNet 推論與 LLM Prompt 整合模組

功能：
1. 載入訓練好的 FTDualNet 模型
2. 從 CSV 提取特徵 → 預測分數、信心度、診斷機率
3. 產生結構化文字注入 LLM Prompt
4. 與現有的 generate_prompt_with_features.py 整合
"""

import sys
import numpy as np
import torch
import joblib
from pathlib import Path

from mtl_config import (
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    NUM_FEATURES,
    UNIFIED_FEATURE_NAMES,
    USER_FEATURE_MAPPING,
    SCORE_NUM_CLASSES,
)
from mtl_model import FTDualNet

# 加入原始特徵擷取模組的路徑
ZEROSHOT_FEATURE_DIR = Path(__file__).parent.parent / "experiments_zeroshot_feature"
sys.path.insert(0, str(ZEROSHOT_FEATURE_DIR))


# ===========================
# 模型載入
# ===========================

def load_trained_model(checkpoint_path=None, device=None):
    """
    載入訓練好的 FTDualNet 模型

    參數：
        checkpoint_path: 模型權重路徑，預設使用最終模型
        device: torch device

    回傳：
        model: FTDualNet
        device: torch device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "ftdualnet_final.pt"

    model = FTDualNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device


def load_scaler(scaler_path=None):
    """載入標準化的 scaler"""
    if scaler_path is None:
        scaler_path = CHECKPOINT_DIR / "user_scaler.pkl"

    return joblib.load(scaler_path)


# ===========================
# 特徵提取（從 CSV）
# ===========================

def extract_features_for_mtl(csv_path):
    """
    從 CSV 提取用於 MTL 模型的特徵

    使用原始 generate_prompt_with_features.py 的特徵擷取功能，
    然後對齊到統一特徵名。

    參數：
        csv_path: CSV 檔案路徑

    回傳：
        feature_vector: dict, 統一特徵名 → 數值
    """
    from generate_prompt_with_features import generate_prompt_with_features

    _, raw_features = generate_prompt_with_features(csv_path, str(ZEROSHOT_FEATURE_DIR / "Prompt.txt"))

    # 從原始特徵中提取並對齊到統一名稱
    feature_vector = {}
    for user_col, unified_name in USER_FEATURE_MAPPING.items():
        if user_col in raw_features:
            feature_vector[unified_name] = raw_features[user_col]
        else:
            feature_vector[unified_name] = 0.0

    return feature_vector


# ===========================
# 單筆預測
# ===========================

def predict_single(model, csv_path, scaler, device=None):
    """
    對單個 CSV 檔案進行預測

    參數：
        model: FTDualNet
        csv_path: CSV 檔案路徑
        scaler: StandardScaler（使用者資料集的 scaler）
        device: torch device

    回傳：
        prediction: dict
            - predicted_score: int, 0-3+
            - score_probs: list, 各分數的機率 (0, 1, 2, 3+)
            - confidence: float, 預測信心度
            - diag_prob: float, PD 診斷機率
    """
    if device is None:
        device = next(model.parameters()).device

    # 提取特徵
    feature_dict = extract_features_for_mtl(csv_path)

    # 轉為陣列（按統一特徵名排序）
    feature_values = [feature_dict.get(name, 0.0) for name in UNIFIED_FEATURE_NAMES]
    feature_array = np.array(feature_values, dtype=np.float32).reshape(1, -1)

    # 標準化
    feature_scaled = scaler.transform(feature_array)

    # 預測
    feature_tensor = torch.FloatTensor(feature_scaled).to(device)
    prediction = model.predict(feature_tensor)

    return prediction


# ===========================
# Prompt 注入格式
# ===========================

def format_mtl_output(prediction):
    """
    將 MTL 預測結果格式化為可注入 LLM Prompt 的文字

    參數：
        prediction: dict，來自 predict_single 或 model.predict

    回傳：
        text: str, 結構化的分析文字
    """
    score = prediction["predicted_score"]
    confidence = prediction["confidence"]
    diag_prob = prediction["diag_prob"]
    score_probs = prediction["score_probs"]

    # 信心度描述
    if confidence >= 0.6:
        conf_desc = "High"
    elif confidence >= 0.4:
        conf_desc = "Moderate"
    else:
        conf_desc = "Low"

    # 診斷可能性描述
    if diag_prob >= 0.8:
        diag_desc = "High"
    elif diag_prob >= 0.5:
        diag_desc = "Moderate"
    else:
        diag_desc = "Low"

    # 分數機率分布
    prob_labels = [f"Score {i}" if i < SCORE_NUM_CLASSES - 1 else f"Score {i}+"
                   for i in range(SCORE_NUM_CLASSES)]
    prob_dist = ", ".join([f"{prob_labels[i]}: {p*100:.1f}%" for i, p in enumerate(score_probs)])

    text = (
        f"\n# Auxiliary ML Model Analysis (trained on 488+ clinical samples)\n"
        f"An auxiliary multi-task learning model (FTDualNet) has analyzed the kinematic features.\n"
        f"Predicted MDS-UPDRS 3.4 score: {score} (confidence: {conf_desc}, {confidence*100:.0f}%)\n"
        f"Score probability distribution: {prob_dist}\n"
        f"Parkinson's disease likelihood: {diag_desc} ({diag_prob*100:.0f}%)\n"
        f"Note: This is a reference prediction from a machine learning model. "
        f"Please make your own clinical judgment based on the kinematic features above.\n"
    )

    return text


# ===========================
# 完整推論流程
# ===========================

def predict_and_format(csv_path, model=None, scaler=None, device=None):
    """
    完整推論流程：載入模型 → 預測 → 格式化

    參數：
        csv_path: CSV 檔案路徑
        model: FTDualNet（如果 None 則自動載入）
        scaler: StandardScaler（如果 None 則自動載入）
        device: torch device

    回傳：
        mtl_text: str, 可直接附加到 Prompt 的文字
        prediction: dict, 原始預測結果
    """
    if model is None:
        model, device = load_trained_model()

    if scaler is None:
        scaler = load_scaler()

    prediction = predict_single(model, csv_path, scaler, device)
    mtl_text = format_mtl_output(prediction)

    return mtl_text, prediction


# ===========================
# 獨立執行：測試推論
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTDualNet 推論")
    parser.add_argument("csv_path", help="要預測的 CSV 檔案路徑")
    parser.add_argument("--checkpoint", default=None, help="模型權重路徑")
    args = parser.parse_args()

    model, device = load_trained_model(args.checkpoint)
    scaler = load_scaler()

    mtl_text, prediction = predict_and_format(args.csv_path, model, scaler, device)

    print("=" * 60)
    print("FTDualNet 推論結果")
    print("=" * 60)
    print(f"CSV: {args.csv_path}")
    print(f"預測分數: {prediction['predicted_score']}")
    print(f"信心度: {prediction['confidence']:.3f}")
    print(f"各分數機率: {[f'{p:.3f}' for p in prediction['score_probs']]}")
    print(f"診斷機率: {prediction['diag_prob']:.3f}")
    print(f"\n注入 Prompt 的文字:")
    print(mtl_text)
