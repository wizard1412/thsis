"""
Traditional ML + Dawid-Skene EM Configuration

Replaces FTDualNet with traditional ML classifiers (XGBoost, RF, SVM),
using Dawid-Skene EM as the label aggregation analog of neural RCE.
"""

import sys
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
# Output paths
# ===========================

OUTPUT_DIR = PROJECT_ROOT / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

# ===========================
# Dawid-Skene EM settings
# ===========================

DS_MAX_ITER = 100
DS_TOL = 1e-4
DS_SMOOTHING = 1e-6

# ===========================
# Model hyperparameters
# ===========================

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "multi:softprob",
    "num_class": SCORE_NUM_CLASSES,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

SVM_PARAMS = {
    "C": 10.0,
    "kernel": "rbf",
    "gamma": "scale",
    "decision_function_shape": "ovr",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-3,
    "learning_rate": "adaptive",
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "random_state": RANDOM_SEED,
}

# ===========================
# Experiment settings
# ===========================

LABEL_STRATEGIES = ["majority_vote", "dawid_skene", "mean"]
MODEL_NAMES = ["XGBoost", "RandomForest", "SVM", "MLP"]