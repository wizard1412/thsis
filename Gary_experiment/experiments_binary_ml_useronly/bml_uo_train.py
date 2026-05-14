"""
Binary Classification ML (User-Only) with Leave-One-Out CV.

For each binary split (A: 0-1 vs 2-3, B: 0 vs 1-3):
  LOO CV over 53 user-only samples.

Output: bml_uo_results.json
"""

import json
import logging
import numpy as np
from datetime import datetime
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
)

from bml_uo_config import (
    OUTPUT_DIR, LOG_DIR, PLOT_DIR, RANDOM_SEED,
    BINARY_SPLITS, SPLIT_NAMES, MODEL_NAMES,
)
from bml_uo_data import load_user_data, binarize_labels

# Reuse model factories from binary_ml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments_binary_ml"))
from bml_models import create_model, train_model, predict_model


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"bml_uo_train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def run_loo_cv():
    np.random.seed(RANDOM_SEED)
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading user-only data...")
    data = load_user_data()
    features = data["features"]
    majority_scores = data["majority_scores"]
    n_samples = len(features)

    logging.info(f"Samples: {n_samples}, Features: {features.shape[1]}")

    loo = LeaveOneOut()
    summary = {"method": "Binary ML (User-Only, LOO CV)", "n_samples": n_samples, "splits": {}}

    for split_name in SPLIT_NAMES:
        split_cfg = BINARY_SPLITS[split_name]
        threshold = split_cfg["threshold"]
        y_all = binarize_labels(majority_scores, threshold)
        counts = np.bincount(y_all, minlength=2)

        logging.info(f"\n{'='*60}")
        logging.info(f"Split: {split_cfg['name']}")
        logging.info(f"  Class 0: {counts[0]}, Class 1: {counts[1]}")
        logging.info(f"{'='*60}")

        split_results = {"name": split_cfg["name"], "models": {}}

        for model_name in MODEL_NAMES:
            logging.info(f"\n  --- {model_name} (LOO, {n_samples} iterations) ---")

            all_preds = np.zeros(n_samples, dtype=int)
            all_probs = np.zeros(n_samples, dtype=float)

            for i, (train_idx, test_idx) in enumerate(loo.split(features)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train = y_all[train_idx]

                model = create_model(model_name)
                model = train_model(model, X_train, y_train)
                pred, prob = predict_model(model, X_test)

                all_preds[test_idx[0]] = pred[0]
                if prob is not None and prob.shape[1] == 2:
                    all_probs[test_idx[0]] = prob[0, 1]

            # Compute metrics
            acc = float(accuracy_score(y_all, all_preds))
            f1 = float(f1_score(y_all, all_preds, zero_division=0))
            prec = float(precision_score(y_all, all_preds, zero_division=0))
            rec = float(recall_score(y_all, all_preds, zero_division=0))
            mcc = float(matthews_corrcoef(y_all, all_preds))

            if len(np.unique(y_all)) == 2:
                auc = float(roc_auc_score(y_all, all_probs))
            else:
                auc = 0.0

            cm = confusion_matrix(y_all, all_preds, labels=[0, 1]).tolist()

            split_results["models"][model_name] = {
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "auc": auc,
                "mcc": mcc,
                "confusion_matrix": cm,
                "predictions": all_preds.tolist(),
                "labels": y_all.tolist(),
                "probabilities": all_probs.tolist(),
            }

            logging.info(
                f"    Acc={acc:.3f}, F1={f1:.3f}, Prec={prec:.3f}, "
                f"Rec={rec:.3f}, AUC={auc:.3f}, MCC={mcc:.3f}"
            )

        summary["splits"][split_name] = split_results

    # Print final summary
    logging.info(f"\n{'='*90}")
    logging.info("LOO CV RESULTS SUMMARY (User-Only, N=53)")
    logging.info(f"{'='*90}")
    logging.info(f"{'Split':<30} {'Model':<14} {'Acc':>8} {'F1':>8} "
                 f"{'Prec':>8} {'Rec':>8} {'AUC':>8} {'MCC':>8}")
    logging.info("-" * 96)

    for split_name in SPLIT_NAMES:
        split_data = summary["splits"][split_name]
        for model_name in MODEL_NAMES:
            r = split_data["models"][model_name]
            logging.info(
                f"{split_data['name']:<30} {model_name:<14} "
                f"{r['accuracy']:>8.3f} {r['f1']:>8.3f} {r['precision']:>8.3f} "
                f"{r['recall']:>8.3f} {r['auc']:>8.3f} {r['mcc']:>8.3f}"
            )

    # Save
    results_file = OUTPUT_DIR / "bml_uo_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved to {results_file}")

    return summary


if __name__ == "__main__":
    print("Running LOO CV (Binary ML, User-Only, N=53)...")
    run_loo_cv()
