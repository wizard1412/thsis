"""
Data loading for Binary Classification ML experiments.
Reuses rce_data and adds binary label conversion.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments_rce"))
from rce_data import prepare_all_data


def binarize_labels(labels, threshold):
    """Convert multi-class labels to binary: scores >= threshold -> 1, else -> 0."""
    return (labels >= threshold).astype(int)


def load_data():
    """
    Load all data as numpy arrays.

    Returns dict with:
        all_features:    (541, 33) standardized features
        all_ratings:     (541, 5) rater scores, -1 = missing
        majority_scores: (541,) majority vote labels {0,1,2,3}
        cv_splits:       list of (train_idx, val_idx) for 5 folds
        feature_names:   list of 33 feature names
        dataset_source:  list of "ext"/"user" per sample
    """
    data = prepare_all_data()
    return {
        "all_features": data["all_features"],
        "all_ratings": data["all_ratings"],
        "majority_scores": data["majority_scores"],
        "cv_splits": data["cv_splits"],
        "feature_names": data["feature_names"],
        "dataset_source": data["dataset_source"],
    }
