"""
FTDualNet + RCE (Rater Confusion Estimation) 設定檔

基於 PARQ (2025) 和 Li et al. (2021) 的方法：
- 為每位評分者學習一個 confusion matrix，建模其評分偏差
- 使用 Ordinal Focal Loss 強化序數分類的有序性
- 聯合訓練嚴重度預測與 rater 偏差估計
"""

from pathlib import Path

# ===========================
# 路徑設定
# ===========================

PROJECT_ROOT = Path(__file__).parent

# 資料集路徑
EXTERNAL_DATASET_PATH = PROJECT_ROOT.parent / "fingertapping_features_severity_diagnosis_June13_2023.csv"
USER_DATASET_PATH = PROJECT_ROOT.parent / "experiments_fewshot" / "features_dataset.csv"

# 輸出路徑
OUTPUT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

# ===========================
# 特徵對應（與 mtl_config 完全一致）
# ===========================

EXTERNAL_FEATURE_MAPPING = {
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
    "aperiodicity_denoised":          "aperiodicity",
    "periodEntropy_denoised":         "periodEntropy",
    "periodVarianceNorm_denoised":    "periodVarianceNorm",
    "maxFreezeDuration_denoised":     "maxFreezeDuration",
    "period_median_denoised":         "period_median",
    "period_mean_denoised":           "period_mean",
    "period_stdev_denoised":          "period_stdev",
    "period_max_denoised":            "period_max",
    "frequency_mean_denoised":        "frequency_mean",
    "frequency_stdev_denoised":       "frequency_stdev",
    "frequency_median_denoised":      "frequency_median",
    "frequency_lr_fitness_r2_denoised": "frequency_lr_fitness_r2",
    "frequency_lr_slope_denoised":    "frequency_lr_slope",
    "speed_mean_denoised":            "speed_mean",
    "speed_stdev_denoised":           "speed_stdev",
    "speed_max_denoised":             "speed_max",
    "acceleration_mean_denoised":     "acceleration_mean",
    "acceleration_stdev_denoised":    "acceleration_stdev",
    "num_peaks_denoised":             "num_peaks",
    "num_interruptions_norm_denoised":"num_interruptions_norm",
    "num_freeze_norm_denoised":       "num_freeze_norm",
}

USER_FEATURE_MAPPING = {
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
    "aperiodicity":                    "aperiodicity",
    "periodEntropy":                   "periodEntropy",
    "periodVarianceNorm":              "periodVarianceNorm",
    "maxFreezeDuration":               "maxFreezeDuration",
    "period_median":                   "period_median",
    "period_mean":                     "period_mean",
    "period_stdev":                    "period_stdev",
    "period_max":                      "period_max",
    "frequency_mean":                  "frequency_mean",
    "frequency_stdev":                 "frequency_stdev",
    "frequency_median":                "frequency_median",
    "frequency_lr_fitness_r2":         "frequency_lr_fitness_r2",
    "frequency_lr_slope":              "frequency_lr_slope",
    "speed_mean":                      "speed_mean",
    "speed_stdev":                     "speed_stdev",
    "speed_max":                       "speed_max",
    "acceleration_mean":               "acceleration_mean",
    "acceleration_stdev":              "acceleration_stdev",
    "num_peaks":                       "num_peaks",
    "num_interruptions_norm":          "num_interruptions_norm",
    "num_freeze_norm":                 "num_freeze_norm",
}

UNIFIED_FEATURE_NAMES = sorted(set(EXTERNAL_FEATURE_MAPPING.values()))
NUM_FEATURES = len(UNIFIED_FEATURE_NAMES)  # 33

# ===========================
# 評分者設定
# ===========================

EXTERNAL_RATER_COLUMNS = ["Rating1", "Rating2", "Rating3", "Rating4", "Rating5"]
USER_RATER_COLUMNS = ["Rating1", "Rating2"]
EXTERNAL_DIAGNOSIS_COLUMN = "diagnosed"

# RCE 中統一使用的最大 rater 數量
MAX_NUM_RATERS = 5  # 外部 5 位，使用者 2 位（其餘填 -1）

# ===========================
# 模型超參數（與原始 FTDualNet 一致）
# ===========================

LATENT_DIM = 16
ENCODER_HIDDEN_DIMS = [64, 32]
DECODER_HIDDEN_DIMS = [32, 64]
ATTENTION_HEADS = 4
ATTENTION_DROPOUT = 0.1
SCORE_HEAD_HIDDEN = 32
SCORE_NUM_CLASSES = 4  # 0, 1, 2, 3+
DIAG_HEAD_HIDDEN = 16
DROPOUT = 0.3

# ===========================
# RCE 超參數
# ===========================

# Ordinal Focal Loss
OF_ALPHA = 1.0       # focal loss alpha
OF_GAMMA = 2.0       # focal loss gamma（越大越專注於難分類樣本）
OF_BETA = 1.0        # ordinal distance penalty 權重

# RCE 正則化
RCE_TRACE_LAMBDA = 0.1   # trace regularization 權重
RCE_INIT_NOISE = 0.01    # confusion matrix 初始化時加的隨機噪音

# ===========================
# 訓練超參數
# ===========================

# 第一階段：自編碼器預訓練（與原版一致）
PHASE1_EPOCHS = 200
PHASE1_LR = 1e-3
PHASE1_WEIGHT_DECAY = 1e-4
PHASE1_PATIENCE = 20

# 第二階段：RCE + Dual Head 訓練（編碼器凍結）
PHASE2_EPOCHS = 150       # 比原版多，因為 RCE 需要更多 epoch 收斂
PHASE2_LR = 1e-3
PHASE2_RCE_LR = 5e-3     # confusion matrix 的學習率（較高，使其快速收斂）
PHASE2_WEIGHT_DECAY = 1e-4
PHASE2_PATIENCE = 20
PHASE2_DIAG_LOSS_WEIGHT = 0.3

# 第三階段：聯合微調
PHASE3_EPOCHS = 50
PHASE3_ENCODER_LR = 1e-5
PHASE3_HEAD_LR = 1e-4
PHASE3_RCE_LR = 1e-4     # 微調階段 RCE 學習率
PHASE3_WEIGHT_DECAY = 1e-4
PHASE3_PATIENCE = 10

# 通用
BATCH_SIZE = 32
NUM_FOLDS = 5
RANDOM_SEED = 42
