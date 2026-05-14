"""
Binary Classification ML Configuration

Two binary split strategies:
  - split_A: 0-1 vs 2-3 (mild vs severe)
  - split_B: 0 vs 1-3 (normal vs symptomatic)

Uses majority_vote labels only (best from 4-class experiments).
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
# Binary split definitions
# ===========================

BINARY_SPLITS = {
    "split_A": {
        "name": "Mild(0-1) vs Severe(2-3)",
        "threshold": 2,  # scores >= 2 -> positive class
        "class_names": ["Mild (0-1)", "Severe (2-3)"],
    },
    "split_B": {
        "name": "Normal(0) vs Symptomatic(1-3)",
        "threshold": 1,  # scores >= 1 -> positive class
        "class_names": ["Normal (0)", "Symptomatic (1-3)"],
    },
}

# ===========================
# Model hyperparameters (binary)
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
    "objective": "binary:logistic",
    "eval_metric": "logloss",
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

SPLIT_NAMES = list(BINARY_SPLITS.keys())
MODEL_NAMES = ["XGBoost", "RandomForest", "SVM", "MLP"]
