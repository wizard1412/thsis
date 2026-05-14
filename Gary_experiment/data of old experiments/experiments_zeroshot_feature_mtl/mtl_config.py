"""
FTDualNet 設定檔
基於 PDualNet (Scientific Reports, 2025) 改造的多任務學習手指敲擊評分系統

所有超參數、檔案路徑、特徵對應皆集中於此檔案管理。
"""

from pathlib import Path

# ===========================
# 路徑設定
# ===========================

# 專案根目錄
PROJECT_ROOT = Path(__file__).parent

# 資料集路徑
EXTERNAL_DATASET_PATH = PROJECT_ROOT.parent / "fingertapping_features_severity_diagnosis_June13_2023.csv"
USER_DATASET_PATH = PROJECT_ROOT.parent / "experiments_zeroshot_feature" / "features_dataset.csv"

# 輸出路徑
OUTPUT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "mtl_plots"
LOG_DIR = OUTPUT_DIR / "logs"

# 原始 CSV 目錄（用於推論時提取特徵）
CSV_DIR = PROJECT_ROOT.parent / "experiments_zeroshot_feature" / "csv_files"

# 基礎 Prompt 路徑
BASE_PROMPT_PATH = PROJECT_ROOT.parent / "experiments_zeroshot_feature" / "Prompt.txt"

# ===========================
# 特徵對應（外部資料集 → 統一名稱 ← 使用者資料集）
# ===========================

# 外部資料集欄位名 → 統一特徵名
EXTERNAL_FEATURE_MAPPING = {
    # 運動特徵 (wrist → 統一名)
    "wrist_mvmnt_x_median":           "mvmnt_x_median",
    "wrist_mvmnt_x_quartile_range":   "mvmnt_x_quartile_range",
    "wrist_mvmnt_x_mean":             "mvmnt_x_mean",
    "wrist_mvmnt_x_stdev":            "mvmnt_x_stdev",
    "wrist_mvmnt_y_median":           "mvmnt_y_median",
    "wrist_mvmnt_y_quartile_range":   "mvmnt_y_quartile_range",
    "wrist_mvmnt_y_mean":             "mvmnt_y_mean",
    "wrist_mvmnt_y_stdev":            "mvmnt_y_stdev",
    "wrist_mvmnt_dist_median":        "mvmnt_dist_median",
    "wrist_mvmnt_dist_quartile_range":"mvmnt_dist_quartile_range",
    "wrist_mvmnt_dist_mean":          "mvmnt_dist_mean",
    "wrist_mvmnt_dist_stdev":         "mvmnt_dist_stdev",
    # 節律特徵 (使用 _denoised 版本)
    "aperiodicity_denoised":          "aperiodicity",
    "periodEntropy_denoised":         "periodEntropy",
    "periodVarianceNorm_denoised":    "periodVarianceNorm",
    "maxFreezeDuration_denoised":     "maxFreezeDuration",
    # 週期特徵
    "period_median_denoised":         "period_median",
    "period_mean_denoised":           "period_mean",
    "period_stdev_denoised":          "period_stdev",
    "period_max_denoised":            "period_max",
    # 頻率特徵
    "frequency_mean_denoised":        "frequency_mean",
    "frequency_stdev_denoised":       "frequency_stdev",
    "frequency_median_denoised":      "frequency_median",
    "frequency_lr_fitness_r2_denoised": "frequency_lr_fitness_r2",
    "frequency_lr_slope_denoised":    "frequency_lr_slope",
    # 速度特徵
    "speed_mean_denoised":            "speed_mean",
    "speed_stdev_denoised":           "speed_stdev",
    "speed_max_denoised":             "speed_max",
    # 加速度特徵
    "acceleration_mean_denoised":     "acceleration_mean",
    "acceleration_stdev_denoised":    "acceleration_stdev",
    # 計數特徵
    "num_peaks_denoised":             "num_peaks",
    "num_interruptions_norm_denoised":"num_interruptions_norm",
    "num_freeze_norm_denoised":       "num_freeze_norm",
}

# 使用者資料集欄位名 → 統一特徵名
USER_FEATURE_MAPPING = {
    # 運動特徵 (finger → 統一名)
    "finger_mvmnt_x_median":           "mvmnt_x_median",
    "finger_mvmnt_x_quartile_range":   "mvmnt_x_quartile_range",
    "finger_mvmnt_x_mean":             "mvmnt_x_mean",
    "finger_mvmnt_x_stdev":            "mvmnt_x_stdev",
    "finger_mvmnt_y_median":           "mvmnt_y_median",
    "finger_mvmnt_y_quartile_range":   "mvmnt_y_quartile_range",
    "finger_mvmnt_y_mean":             "mvmnt_y_mean",
    "finger_mvmnt_y_stdev":            "mvmnt_y_stdev",
    "finger_mvmnt_dist_median":        "mvmnt_dist_median",
    "finger_mvmnt_dist_quartile_range":"mvmnt_dist_quartile_range",
    "finger_mvmnt_dist_mean":          "mvmnt_dist_mean",
    "finger_mvmnt_dist_stdev":         "mvmnt_dist_stdev",
    # 節律特徵 (直接對應)
    "aperiodicity":                    "aperiodicity",
    "periodEntropy":                   "periodEntropy",
    "periodVarianceNorm":              "periodVarianceNorm",
    "maxFreezeDuration":               "maxFreezeDuration",
    # 週期特徵
    "period_median":                   "period_median",
    "period_mean":                     "period_mean",
    "period_stdev":                    "period_stdev",
    "period_max":                      "period_max",
    # 頻率特徵
    "frequency_mean":                  "frequency_mean",
    "frequency_stdev":                 "frequency_stdev",
    "frequency_median":                "frequency_median",
    "frequency_lr_fitness_r2":         "frequency_lr_fitness_r2",
    "frequency_lr_slope":              "frequency_lr_slope",
    # 速度特徵
    "speed_mean":                      "speed_mean",
    "speed_stdev":                     "speed_stdev",
    "speed_max":                       "speed_max",
    # 加速度特徵
    "acceleration_mean":               "acceleration_mean",
    "acceleration_stdev":              "acceleration_stdev",
    # 計數特徵
    "num_peaks":                       "num_peaks",
    "num_interruptions_norm":          "num_interruptions_norm",
    "num_freeze_norm":                 "num_freeze_norm",
}

# 統一特徵名列表（排序後，確保順序一致）
UNIFIED_FEATURE_NAMES = sorted(set(EXTERNAL_FEATURE_MAPPING.values()))
NUM_FEATURES = len(UNIFIED_FEATURE_NAMES)  # 應為 33

# ===========================
# 評分者欄位名稱
# ===========================

EXTERNAL_RATER_COLUMNS = ["Rating1", "Rating2", "Rating3", "Rating4", "Rating5"]
USER_RATER_COLUMNS = ["Rating1", "Rating2"]

# 輔助任務欄位
EXTERNAL_DIAGNOSIS_COLUMN = "diagnosed"  # "yes" / "no"

# ===========================
# 模型超參數
# ===========================

# 自編碼器 (SiVE-inspired)
LATENT_DIM = 16          # 潛在空間維度
ENCODER_HIDDEN_DIMS = [64, 32]  # 編碼器隱藏層
DECODER_HIDDEN_DIMS = [32, 64]  # 解碼器隱藏層 (對稱)

# Self-Attention (DiSE-inspired)
ATTENTION_HEADS = 4      # 多頭注意力的頭數
ATTENTION_DROPOUT = 0.1  # 注意力層的 dropout

# 任務 Head
SCORE_HEAD_HIDDEN = 32   # 評分 Head 隱藏層維度
SCORE_NUM_CLASSES = 4    # 0, 1, 2, 3+（原始 3 和 4 合併）
DIAG_HEAD_HIDDEN = 16    # 診斷 Head 隱藏層維度

# 正則化
DROPOUT = 0.3            # 主要 dropout 比率

# ===========================
# 訓練超參數
# ===========================

# 第一階段：自編碼器預訓練
PHASE1_EPOCHS = 200
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_PATIENCE = 20     # 早停耐心值

# 第二階段：雙 Head 訓練（編碼器凍結）
PHASE2_EPOCHS = 100
PHASE2_LR = 1e-3
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_PATIENCE = 15
PHASE2_DIAG_LOSS_WEIGHT = 0.3  # 診斷任務的損失權重

# 第三階段：聯合微調（全部解凍）
PHASE3_EPOCHS = 50
PHASE3_ENCODER_LR = 1e-5       # 編碼器使用較低學習率
PHASE3_HEAD_LR = 1e-4          # Head 使用較高學習率
PHASE3_WEIGHT_DECAY = 1e-4
PHASE3_PATIENCE = 10

# 通用
BATCH_SIZE = 32
NUM_FOLDS = 5            # 交叉驗證折數
RANDOM_SEED = 42

# ===========================
# Ollama / LLM 設定（推論整合用）
# ===========================

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:30b"
EVALUATION_TIMES = 10
DELAY_BETWEEN_CALLS = 25
REQUEST_TIMEOUT = 300
MAX_RETRIES = 5
NUM_PREDICT = 16384
