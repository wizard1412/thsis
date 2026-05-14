"""
Traditional ML + Dawid-Skene training pipeline.

For each fold:
1. Run Dawid-Skene EM on training set -> inferred labels
2. Train ML models with majority_vote labels (baseline)
   and dawid_skene labels (RCE analog)
3. Evaluate on validation set using majority_score as ground truth

Output: tml_cv_results.json
"""

import json
import logging
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score

from tml_config import (
    OUTPUT_DIR, LOG_DIR, PLOT_DIR,
    NUM_FOLDS, RANDOM_SEED, SCORE_NUM_CLASSES,
    LABEL_STRATEGIES, MODEL_NAMES,
)
from tml_data import load_data
from tml_dawid_skene import dawid_skene_em
from tml_models import create_model, train_model, predict_model


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"tml_train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "off_by_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def evaluate_consensus_levels(y_pred, val_ratings, val_majority):
    """Evaluate at different inter-rater consensus levels."""
    consensus_full = []
    consensus_major = []
    consensus_low = []

    for i in range(len(val_ratings)):
        valid = val_ratings[i][val_ratings[i] >= 0]
        if len(valid) < 2:
            consensus_major.append(i)
            continue
        unique = len(set(valid))
        max_count = max(np.bincount(valid.astype(int), minlength=SCORE_NUM_CLASSES))
        agreement_ratio = max_count / len(valid)
        if unique == 1:
            consensus_full.append(i)
        elif agreement_ratio >= 0.6:
            consensus_major.append(i)
        else:
            consensus_low.append(i)

    results = {}
    for name, indices in [
        ("full_consensus", consensus_full),
        ("majority_consensus", consensus_major),
        ("low_consensus", consensus_low),
        ("overall", list(range(len(y_pred)))),
    ]:
        if len(indices) == 0:
            results[name] = {"n": 0, "accuracy": 0, "mae": 0, "off_by_one": 0, "kappa": 0}
            continue
        idx = np.array(indices)
        p, m = y_pred[idx], val_majority[idx]
        kappa = float(cohen_kappa_score(m, p, weights="quadratic")) if len(set(m)) > 1 else 0.0
        results[name] = {
            "n": len(indices),
            "accuracy": float(np.mean(p == m)),
            "mae": float(np.mean(np.abs(p - m))),
            "off_by_one": float(np.mean(np.abs(p - m) <= 1)),
            "kappa": kappa,
        }
    return results


def get_train_labels(strategy, train_ratings, train_majority, train_mean=None):
    """Get training labels based on label strategy."""
    if strategy == "majority_vote":
        return train_majority.copy(), None
    elif strategy == "dawid_skene":
        ds_result = dawid_skene_em(ratings=train_ratings, init_labels=train_majority)
        return ds_result["estimated_labels"], ds_result
    elif strategy == "mean":
        return train_mean.copy(), None
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_cross_validation():
    np.random.seed(RANDOM_SEED)
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data...")
    data = load_data()
    all_features = data["all_features"]
    all_ratings = data["all_ratings"]
    majority_scores = data["majority_scores"]
    mean_scores = data["mean_scores"]
    cv_splits = data["cv_splits"]

    logging.info(f"Total samples: {len(all_features)}, Features: {all_features.shape[1]}")
    logging.info(f"Score distribution: {np.bincount(majority_scores.astype(int), minlength=SCORE_NUM_CLASSES)}")

    # results[strategy][model_name] = list of fold dicts
    all_results = {s: {m: [] for m in MODEL_NAMES} for s in LABEL_STRATEGIES}

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        logging.info(f"\n{'#' * 60}")
        logging.info(f"Fold {fold_idx + 1}/{NUM_FOLDS}")
        logging.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        logging.info(f"{'#' * 60}")

        X_train = all_features[train_idx]
        X_val = all_features[val_idx]
        train_ratings = all_ratings[train_idx]
        val_ratings = all_ratings[val_idx]
        train_majority = majority_scores[train_idx]
        val_majority = majority_scores[val_idx]
        train_mean = mean_scores[train_idx]

        for strategy in LABEL_STRATEGIES:
            logging.info(f"\n--- Label Strategy: {strategy} ---")

            y_train, ds_result = get_train_labels(strategy, train_ratings, train_majority, train_mean)

            if ds_result is not None:
                changed = np.sum(y_train != train_majority)
                logging.info(
                    f"  Dawid-Skene: {ds_result['n_iterations']} iters, "
                    f"converged={ds_result['converged']}, "
                    f"labels changed: {changed}/{len(y_train)} ({100*changed/len(y_train):.1f}%)"
                )

            for model_name in MODEL_NAMES:
                logging.info(f"  Training {model_name}...")

                model = create_model(model_name)
                model = train_model(model, X_train, y_train)
                y_pred, y_probs = predict_model(model, X_val)

                metrics = compute_metrics(val_majority, y_pred)
                consensus = evaluate_consensus_levels(y_pred, val_ratings, val_majority)

                fold_result = {
                    "fold": fold_idx + 1,
                    **metrics,
                    "consensus_results": consensus,
                    "predictions": y_pred.tolist(),
                    "labels": val_majority.tolist(),
                }

                # Store DS confusion matrices for fold 1
                if ds_result is not None and fold_idx == 0:
                    fold_result["ds_confusion_matrices"] = ds_result["confusion_matrices"].tolist()
                    fold_result["ds_class_priors"] = ds_result["class_priors"].tolist()

                all_results[strategy][model_name].append(fold_result)

                logging.info(
                    f"    Acc={metrics['accuracy']:.3f}, MAE={metrics['mae']:.3f}, "
                    f"+-1={metrics['off_by_one_accuracy']:.3f}, "
                    f"QWK={metrics['qwk']:.3f}, F1={metrics['f1_weighted']:.3f}"
                )

    # === Build summary ===
    summary = {"method": "Traditional ML + Dawid-Skene EM", "strategies": {}}

    for strategy in LABEL_STRATEGIES:
        summary["strategies"][strategy] = {}
        for model_name in MODEL_NAMES:
            folds = all_results[strategy][model_name]
            accs = [r["accuracy"] for r in folds]
            maes = [r["mae"] for r in folds]
            obos = [r["off_by_one_accuracy"] for r in folds]
            qwks = [r["qwk"] for r in folds]
            f1s = [r["f1_weighted"] for r in folds]

            summary["strategies"][strategy][model_name] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "mean_mae": float(np.mean(maes)),
                "std_mae": float(np.std(maes)),
                "mean_off_by_one": float(np.mean(obos)),
                "std_off_by_one": float(np.std(obos)),
                "mean_qwk": float(np.mean(qwks)),
                "std_qwk": float(np.std(qwks)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "fold_results": folds,
            }

    # Print summary
    logging.info(f"\n{'=' * 80}")
    logging.info("CROSS-VALIDATION RESULTS SUMMARY")
    logging.info(f"{'=' * 80}")
    logging.info(f"{'Strategy':<16} {'Model':<14} {'Acc':>14} {'MAE':>14} "
                 f"{'+-1 Acc':>14} {'QWK':>14} {'F1':>14}")
    logging.info("-" * 90)

    for strategy in LABEL_STRATEGIES:
        for model_name in MODEL_NAMES:
            r = summary["strategies"][strategy][model_name]
            logging.info(
                f"{strategy:<16} {model_name:<14} "
                f"{r['mean_accuracy']:.3f}+/-{r['std_accuracy']:.3f}  "
                f"{r['mean_mae']:.3f}+/-{r['std_mae']:.3f}  "
                f"{r['mean_off_by_one']:.3f}+/-{r['std_off_by_one']:.3f}  "
                f"{r['mean_qwk']:.3f}+/-{r['std_qwk']:.3f}  "
                f"{r['mean_f1']:.3f}+/-{r['std_f1']:.3f}"
            )

    # Save
    results_file = OUTPUT_DIR / "tml_cv_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved to {results_file}")

    return summary


if __name__ == "__main__":
    print("Running 5-Fold Cross-Validation (Traditional ML + Dawid-Skene)...")
    run_cross_validation()