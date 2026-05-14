"""
Data loading for User-Only Binary ML experiments.
Reuses rce_data.prepare_all_data() and filters to user-only samples.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments_rce"))
from rce_data import prepare_all_data
from rce_config import SCORE_NUM_CLASSES


def binarize_labels(labels, threshold):
    """Convert multi-class labels to binary: scores >= threshold -> 1, else -> 0."""
    return (labels >= threshold).astype(int)


def load_user_data():
    """
    Load user-only data (53 samples) by filtering from the full dataset.

    Returns dict with:
        features:        (53, 33) standardized features
        majority_scores: (53,) majority vote labels {0,1,2,3}
        ratings:         (53, 5) rater scores, -1 = missing
        feature_names:   list of 33 feature names
    """
    data = prepare_all_data()

    # Filter to user samples only
    sources = data["dataset_source"]
    user_mask = np.array([s == "user" for s in sources])

    features = data["all_features"][user_mask]
    majority_scores = data["majority_scores"][user_mask]
    ratings = data["all_ratings"][user_mask]
    feature_names = data["feature_names"]

    print(f"User dataset: {len(features)} samples, {len(feature_names)} features")
    print(f"Score distribution: {np.bincount(majority_scores.astype(int), minlength=SCORE_NUM_CLASSES)}")

    return {
        "features": features,
        "majority_scores": majority_scores,
        "ratings": ratings,
        "feature_names": feature_names,
    }
