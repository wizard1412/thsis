"""
Binary Classification ML training pipeline.

For each binary split (A: 0-1 vs 2-3, B: 0 vs 1-3):
  For each fold:
    1. Binarize majority_vote labels by threshold
    2. Train ML models
    3. Evaluate on validation set

Output: bml_cv_results.json
"""

import json
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef,
)

from bml_config import (
    OUTPUT_DIR, LOG_DIR, PLOT_DIR,
    NUM_FOLDS, RANDOM_SEED,
    BINARY_SPLITS, SPLIT_NAMES, MODEL_NAMES,
)
from bml_data import load_data, binarize_labels
from bml_models import create_model, train_model, predict_model


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"bml_train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_probs is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_probs[:, 1]))
    else:
        metrics["auc"] = 0.0
    return metrics


def run_cross_validation():
    np.random.seed(RANDOM_SEED)
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data...")
    data = load_data()
    all_features = data["all_features"]
    majority_scores = data["majority_scores"]
    cv_splits = data["cv_splits"]

    logging.info(f"Total samples: {len(all_features)}, Features: {all_features.shape[1]}")
    logging.info(f"Original score distribution: {np.bincount(majority_scores.astype(int), minlength=4)}")

    # Log binary distributions
    for split_name, split_cfg in BINARY_SPLITS.items():
        binary_labels = binarize_labels(majority_scores, split_cfg["threshold"])
        counts = np.bincount(binary_labels, minlength=2)
        logging.info(f"{split_name} ({split_cfg['name']}): class 0={counts[0]}, class 1={counts[1]}")

    # results[split][model] = list of fold dicts
    all_results = {s: {m: [] for m in MODEL_NAMES} for s in SPLIT_NAMES}

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        logging.info(f"\n{'#' * 60}")
        logging.info(f"Fold {fold_idx + 1}/{NUM_FOLDS}")
        logging.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        logging.info(f"{'#' * 60}")

        X_train = all_features[train_idx]
        X_val = all_features[val_idx]

        for split_name in SPLIT_NAMES:
            split_cfg = BINARY_SPLITS[split_name]
            threshold = split_cfg["threshold"]
            logging.info(f"\n--- Binary Split: {split_cfg['name']} ---")

            y_train = binarize_labels(majority_scores[train_idx], threshold)
            y_val = binarize_labels(majority_scores[val_idx], threshold)

            logging.info(f"  Train: {np.bincount(y_train, minlength=2)}, "
                         f"Val: {np.bincount(y_val, minlength=2)}")

            for model_name in MODEL_NAMES:
                logging.info(f"  Training {model_name}...")

                model = create_model(model_name)
                model = train_model(model, X_train, y_train)
                y_pred, y_probs = predict_model(model, X_val)

                metrics = compute_metrics(y_val, y_pred, y_probs)

                fold_result = {
                    "fold": fold_idx + 1,
                    **metrics,
                    "predictions": y_pred.tolist(),
                    "labels": y_val.tolist(),
                }
                all_results[split_name][model_name].append(fold_result)

                logging.info(
                    f"    Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
                    f"Prec={metrics['precision']:.3f}, Rec={metrics['recall']:.3f}, "
                    f"AUC={metrics['auc']:.3f}, MCC={metrics['mcc']:.3f}"
                )

    # === Build summary ===
    summary = {"method": "Binary Classification ML", "splits": {}}

    for split_name in SPLIT_NAMES:
        split_cfg = BINARY_SPLITS[split_name]
        summary["splits"][split_name] = {"name": split_cfg["name"], "models": {}}

        for model_name in MODEL_NAMES:
            folds = all_results[split_name][model_name]
            metric_keys = ["accuracy", "f1", "precision", "recall", "auc", "mcc"]

            model_summary = {"fold_results": folds}
            for key in metric_keys:
                vals = [r[key] for r in folds]
                model_summary[f"mean_{key}"] = float(np.mean(vals))
                model_summary[f"std_{key}"] = float(np.std(vals))

            summary["splits"][split_name]["models"][model_name] = model_summary

    # Print summary
    logging.info(f"\n{'=' * 90}")
    logging.info("BINARY CLASSIFICATION RESULTS SUMMARY")
    logging.info(f"{'=' * 90}")
    logging.info(f"{'Split':<30} {'Model':<14} {'Acc':>10} {'F1':>10} "
                 f"{'Prec':>10} {'Rec':>10} {'AUC':>10} {'MCC':>10}")
    logging.info("-" * 100)

    for split_name in SPLIT_NAMES:
        split_cfg = BINARY_SPLITS[split_name]
        for model_name in MODEL_NAMES:
            r = summary["splits"][split_name]["models"][model_name]
            logging.info(
                f"{split_cfg['name']:<30} {model_name:<14} "
                f"{r['mean_accuracy']:.3f}+/-{r['std_accuracy']:.3f} "
                f"{r['mean_f1']:.3f}+/-{r['std_f1']:.3f} "
                f"{r['mean_precision']:.3f}+/-{r['std_precision']:.3f} "
                f"{r['mean_recall']:.3f}+/-{r['std_recall']:.3f} "
                f"{r['mean_auc']:.3f}+/-{r['std_auc']:.3f} "
                f"{r['mean_mcc']:.3f}+/-{r['std_mcc']:.3f}"
            )

    # Save
    results_file = OUTPUT_DIR / "bml_cv_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved to {results_file}")

    return summary


if __name__ == "__main__":
    print("Running 5-Fold CV (Binary Classification ML)...")
    run_cross_validation()
