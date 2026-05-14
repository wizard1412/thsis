"""
Traditional ML model factories for PD severity classification.
Models: XGBoost, Random Forest, SVM, MLP (4-class ordinal classification).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from tml_config import (
    XGBOOST_PARAMS,
    RANDOM_FOREST_PARAMS,
    SVM_PARAMS,
    MLP_PARAMS,
    SCORE_NUM_CLASSES,
)


def create_model(model_name):
    """Create an ML model by name."""
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(**XGBOOST_PARAMS)
    elif model_name == "RandomForest":
        return RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    elif model_name == "SVM":
        return SVC(**SVM_PARAMS, probability=True)
    elif model_name == "MLP":
        return MLPClassifier(**MLP_PARAMS)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model, X_train, y_train):
    """Train a model with given features and labels."""
    model.fit(X_train, y_train)
    return model


def predict_model(model, X):
    """
    Get predictions and probabilities from a trained model.

    Returns:
        preds: (N,) predicted class labels
        probs: (N, C) class probabilities or None
    """
    preds = model.predict(X).astype(int)

    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # Ensure shape is (N, SCORE_NUM_CLASSES) even if some classes missing
        if probs.shape[1] < SCORE_NUM_CLASSES:
            full_probs = np.zeros((len(X), SCORE_NUM_CLASSES))
            for i, c in enumerate(model.classes_):
                full_probs[:, int(c)] = probs[:, i]
            probs = full_probs

    return preds, probs