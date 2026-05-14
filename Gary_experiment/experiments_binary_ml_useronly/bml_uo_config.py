"""
Binary Classification ML (User-Only, LOO CV) Configuration

Same binary splits as experiments_binary_ml but only uses
the 53-sample user dataset with Leave-One-Out cross-validation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

sys.path.insert(0, str(PROJECT_ROOT.parent / "experiments_rce"))
from rce_config import (
    USER_DATASET_PATH,
    USER_FEATURE_MAPPING,
    UNIFIED_FEATURE_NAMES,
    NUM_FEATURES,
    USER_RATER_COLUMNS,
    MAX_NUM_RATERS,
    SCORE_NUM_CLASSES,
    RANDOM_SEED,
)

# ===========================
# Output paths
# ===========================

OUTPUT_DIR = PROJECT_ROOT / "results"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

# ===========================
# Binary split definitions
# ===========================

BINARY_SPLITS = {
    "split_A": {
        "name": "Mild(0-1) vs Severe(2-3)",
        "threshold": 2,
        "class_names": ["Mild (0-1)", "Severe (2-3)"],
    },
    "split_B": {
        "name": "Normal(0) vs Symptomatic(1-3)",
        "threshold": 1,
        "class_names": ["Normal (0)", "Symptomatic (1-3)"],
    },
}

# ===========================
# Model hyperparameters (binary, smaller data -> lighter models)
# ===========================

XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

SVM_PARAMS = {
    "C": 10.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-2,
    "learning_rate": "adaptive",
    "learning_rate_init": 1e-3,
    "max_iter": 500,
    "early_stopping": False,
    "random_state": RANDOM_SEED,
}

# ===========================
# Experiment settings
# ===========================

SPLIT_NAMES = list(BINARY_SPLITS.keys())
MODEL_NAMES = ["XGBoost", "RandomForest", "SVM", "MLP"]
