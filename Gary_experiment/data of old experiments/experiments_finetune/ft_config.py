"""
LLM LoRA/QLoRA Fine-Tuning Configuration

Fine-tune Qwen2.5-7B-Instruct with LoRA (Mac) or QLoRA (CUDA) on serialized
tabular features for Parkinson's disease severity classification (MDS-UPDRS Item 3.4).

Platform auto-detection:
  - CUDA available  → 4-bit QLoRA + paged_adamw_32bit
  - MPS available   → float16 LoRA + adamw_torch
  - CPU only        → float32 LoRA + adamw_torch
"""

import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Import shared constants from rce_config
sys.path.insert(0, str(PROJECT_ROOT.parent / "experiments_rce"))
from rce_config import (
    EXTERNAL_DATASET_PATH,
    USER_DATASET_PATH,
    EXTERNAL_FEATURE_MAPPING,
    USER_FEATURE_MAPPING,
    UNIFIED_FEATURE_NAMES,
    NUM_FEATURES,
    EXTERNAL_RATER_COLUMNS,
    USER_RATER_COLUMNS,
    EXTERNAL_DIAGNOSIS_COLUMN,
    MAX_NUM_RATERS,
    SCORE_NUM_CLASSES,
    NUM_FOLDS,
    RANDOM_SEED,
)

# ===========================
# Platform auto-detection
# ===========================

if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
else:
    DEVICE_TYPE = "cpu"

USE_QUANTIZATION = (DEVICE_TYPE == "cuda")

# ===========================
# Output paths
# ===========================

OUTPUT_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

# ===========================
# Model (choose one)
# ===========================

AVAILABLE_MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen3-30b": "Qwen/Qwen3-30B-A3B",
    "deepseek-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

MODEL_ID = AVAILABLE_MODELS["qwen2.5-7b"]  # <-- change this to switch model

# ===========================
# QLoRA / LoRA
# ===========================

# Quantization (only used when USE_QUANTIZATION=True, i.e. CUDA)
QLORA_BITS = 4
QLORA_QUANT_TYPE = "nf4"
QLORA_DOUBLE_QUANT = True
QLORA_COMPUTE_DTYPE = "bfloat16"

# LoRA (shared across platforms)
QLORA_RANK = 16
QLORA_ALPHA = 32
QLORA_DROPOUT = 0.05
QLORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ===========================
# Training
# ===========================

EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4  # effective batch = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
MAX_SEQ_LENGTH = 1024
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
MAX_GRAD_NORM = 0.3

# Platform-specific optimizer and precision
if DEVICE_TYPE == "cuda":
    OPTIM = "paged_adamw_32bit"
    USE_FP16 = False
    USE_BF16 = True
elif DEVICE_TYPE == "mps":
    OPTIM = "adamw_torch"
    USE_FP16 = False
    USE_BF16 = False
else:
    OPTIM = "adamw_torch"
    USE_FP16 = False
    USE_BF16 = False

# ===========================
# Inference
# ===========================

INFERENCE_MAX_NEW_TOKENS = 4
INFERENCE_DO_SAMPLE = False  # greedy decoding

# ===========================
# Experiment
# ===========================

LABEL_STRATEGIES = ["majority_vote", "dawid_skene"]

# ===========================
# Dawid-Skene EM settings (reuse from tml)
# ===========================

DS_MAX_ITER = 100
DS_TOL = 1e-4
DS_SMOOTHING = 1e-6

# ===========================
# Prompt templates
# ===========================

SYSTEM_PROMPT = (
    "You are a clinical assessment tool for Parkinson's disease severity scoring. "
    "Your task is to classify finger-tapping performance according to the MDS-UPDRS Item 3.4 scale.\n\n"
    "Scoring criteria:\n"
    "0: Normal - No problems.\n"
    "1: Slight - Regular rhythm broken with one or two interruptions or hesitations; "
    "slight slowing; amplitude decrements near the end of the 10 taps.\n"
    "2: Mild - 3 to 5 interruptions during tapping; mild slowing; "
    "amplitude decrements midway in the sequence.\n"
    "3: Moderate - More than 5 interruptions or at least one longer arrest (freeze); "
    "moderate slowing; amplitude decrements starting after the 1st tap.\n\n"
    "Based on the provided finger-tapping motion features, "
    "output ONLY the severity score as a single digit (0, 1, 2, or 3)."
)

USER_TEMPLATE = (
    "Finger-tapping motion features for this subject:\n\n"
    "{serialized_features}\n\n"
    "Severity score:"
)

# ===========================
# Feature descriptions for serialization
# ===========================

FEATURE_GROUPS = {
    "Movement Amplitude": [
        "mvmnt_x_median", "mvmnt_x_quartile_range", "mvmnt_x_mean", "mvmnt_x_stdev",
        "mvmnt_y_median", "mvmnt_y_quartile_range", "mvmnt_y_mean", "mvmnt_y_stdev",
        "mvmnt_dist_median", "mvmnt_dist_quartile_range", "mvmnt_dist_mean", "mvmnt_dist_stdev",
    ],
    "Rhythm and Periodicity": [
        "aperiodicity", "periodEntropy", "periodVarianceNorm",
        "period_median", "period_mean", "period_stdev", "period_max",
    ],
    "Frequency": [
        "frequency_mean", "frequency_stdev", "frequency_median",
        "frequency_lr_fitness_r2", "frequency_lr_slope",
    ],
    "Speed and Acceleration": [
        "speed_mean", "speed_stdev", "speed_max",
        "acceleration_mean", "acceleration_stdev",
    ],
    "Tap Count and Freezes": [
        "num_peaks", "num_interruptions_norm", "num_freeze_norm", "maxFreezeDuration",
    ],
}

FEATURE_DESCRIPTIONS = {
    "mvmnt_x_median":           "Median horizontal movement amplitude",
    "mvmnt_x_quartile_range":   "Horizontal movement interquartile range",
    "mvmnt_x_mean":             "Mean horizontal movement amplitude",
    "mvmnt_x_stdev":            "Horizontal movement variability",
    "mvmnt_y_median":           "Median vertical movement amplitude",
    "mvmnt_y_quartile_range":   "Vertical movement interquartile range",
    "mvmnt_y_mean":             "Mean vertical movement amplitude",
    "mvmnt_y_stdev":            "Vertical movement variability",
    "mvmnt_dist_median":        "Median finger-tapping distance",
    "mvmnt_dist_quartile_range": "Finger-tapping distance interquartile range",
    "mvmnt_dist_mean":          "Mean finger-tapping distance",
    "mvmnt_dist_stdev":         "Finger-tapping distance variability",
    "aperiodicity":             "Rhythm irregularity (aperiodicity)",
    "periodEntropy":            "Rhythm complexity (period entropy)",
    "periodVarianceNorm":       "Normalized period variance",
    "period_median":            "Median tapping period",
    "period_mean":              "Mean tapping period",
    "period_stdev":             "Tapping period variability",
    "period_max":               "Maximum tapping period",
    "frequency_mean":           "Mean tapping frequency",
    "frequency_stdev":          "Frequency variability",
    "frequency_median":         "Median tapping frequency",
    "frequency_lr_fitness_r2":  "Frequency trend R-squared",
    "frequency_lr_slope":       "Frequency trend slope",
    "speed_mean":               "Mean finger speed",
    "speed_stdev":              "Speed variability",
    "speed_max":                "Maximum finger speed",
    "acceleration_mean":        "Mean finger acceleration",
    "acceleration_stdev":       "Acceleration variability",
    "num_peaks":                "Number of detected taps",
    "num_interruptions_norm":   "Normalized interruption count",
    "num_freeze_norm":          "Normalized freeze count",
    "maxFreezeDuration":        "Longest freeze duration",
}
