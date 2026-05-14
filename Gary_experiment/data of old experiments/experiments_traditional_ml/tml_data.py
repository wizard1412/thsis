"""
Data loading for Traditional ML experiments.
Thin wrapper around rce_data.prepare_all_data().
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments_rce"))
from rce_data import prepare_all_data


def compute_mean_scores(ratings):
    """Compute rounded mean of valid rater scores per sample."""
    N = ratings.shape[0]
    mean_scores = np.zeros(N, dtype=int)
    for i in range(N):
        valid = ratings[i][ratings[i] >= 0]
        mean_scores[i] = int(np.round(valid.mean()))
    return mean_scores


def load_data():
    """
    Load all data as numpy arrays.

    Returns dict with:
        all_features:    (541, 33) standardized features
        all_ratings:     (541, 5) rater scores, -1 = missing
        majority_scores: (541,) majority vote labels {0,1,2,3}
        mean_scores:     (541,) rounded mean labels {0,1,2,3}
        cv_splits:       list of (train_idx, val_idx) for 5 folds
        feature_names:   list of 33 feature names
        dataset_source:  list of "ext"/"user" per sample
    """
    data = prepare_all_data()
    ratings = data["all_ratings"]
    return {
        "all_features": data["all_features"],
        "all_ratings": ratings,
        "majority_scores": data["majority_scores"],
        "mean_scores": compute_mean_scores(ratings),
        "cv_splits": data["cv_splits"],
        "feature_names": data["feature_names"],
        "dataset_source": data["dataset_source"],
    }