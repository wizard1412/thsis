"""
QLoRA Fine-Tuning Pipeline for Parkinson's Disease Severity Classification.

5-fold cross-validation with QLoRA fine-tuning of Qwen2.5-7B-Instruct.
For each fold:
  1. Load fresh base model + QLoRA adapter
  2. Fine-tune on training data (serialized features -> severity score)
  3. Run constrained inference on validation data
  4. Compute metrics and save results

Output: results/ft_cv_results.json
"""

import gc
import json
import logging
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from ft_config import (
    MODEL_ID,
    DEVICE_TYPE, USE_QUANTIZATION,
    QLORA_BITS, QLORA_QUANT_TYPE, QLORA_DOUBLE_QUANT, QLORA_COMPUTE_DTYPE,
    QLORA_RANK, QLORA_ALPHA, QLORA_DROPOUT, QLORA_TARGET_MODULES,
    EPOCHS, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_RATIO, LR_SCHEDULER,
    MAX_SEQ_LENGTH, LOGGING_STEPS, SAVE_STRATEGY,
    MAX_GRAD_NORM, OPTIM, USE_FP16, USE_BF16,
    INFERENCE_MAX_NEW_TOKENS, INFERENCE_DO_SAMPLE,
    LABEL_STRATEGIES, NUM_FOLDS, RANDOM_SEED, SCORE_NUM_CLASSES,
    OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
)
from ft_data import (
    load_data,
    build_train_dataset,
    build_inference_prompts,
    get_train_labels,
)


# ===========================
# Logging
# ===========================

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"ft_train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_file


# ===========================
# Constrained Generation
# ===========================

class ScoreConstraintProcessor(LogitsProcessor):
    """Force model to only output valid score tokens (0, 1, 2, 3) or EOS."""

    def __init__(self, valid_token_ids, eos_token_id):
        self.valid_ids = set(valid_token_ids)
        self.valid_ids.add(eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for tid in self.valid_ids:
            mask[:, tid] = 0.0
        return scores + mask


def get_score_token_ids(tokenizer):
    """Get token IDs for digits 0-3."""
    token_ids = []
    for digit in ["0", "1", "2", "3"]:
        ids = tokenizer.encode(digit, add_special_tokens=False)
        token_ids.append(ids[-1])
    return token_ids


def parse_score(response):
    """Parse severity score from model output."""
    import re
    response = response.strip()
    # Remove <think>...</think> blocks from thinking models (e.g. Qwen3)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Also handle unclosed <think> tags (truncated output)
    response = re.sub(r"<think>.*", "", response, flags=re.DOTALL).strip()
    for char in response:
        if char in "0123":
            return int(char)
    # Fallback: return -1 to indicate parse failure
    return -1


# ===========================
# Metrics (identical to tml_train.py)
# ===========================

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


# ===========================
# Model Loading
# ===========================

def load_model_and_tokenizer():
    """Load base model with QLoRA (CUDA) or LoRA (MPS/CPU) and apply adapter."""
    logging.info(f"Loading model: {MODEL_ID} (device: {DEVICE_TYPE}, quantization: {USE_QUANTIZATION})")

    load_kwargs = {}

    if USE_QUANTIZATION:
        # CUDA: 4-bit QLoRA
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=QLORA_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, QLORA_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=QLORA_DOUBLE_QUANT,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["torch_dtype"] = getattr(torch, QLORA_COMPUTE_DTYPE)
        load_kwargs["device_map"] = "auto"
    elif DEVICE_TYPE == "mps":
        # Mac: float16 LoRA
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "mps"
    else:
        # CPU fallback
        load_kwargs["torch_dtype"] = torch.float32

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Note: prepare_model_for_kbit_training skipped — it causes OOM on
    # multi-GPU setups by converting params to float32. get_peft_model
    # and SFTTrainer handle freezing and dtype setup automatically.

    # Apply LoRA
    lora_config = LoraConfig(
        r=QLORA_RANK,
        lora_alpha=QLORA_ALPHA,
        lora_dropout=QLORA_DROPOUT,
        target_modules=QLORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer


# ===========================
# Inference
# ===========================

def run_inference(model, tokenizer, features, feature_names):
    """Run constrained inference on all samples."""
    model.eval()

    # Setup constrained generation
    score_token_ids = get_score_token_ids(tokenizer)
    processor = ScoreConstraintProcessor(score_token_ids, tokenizer.eos_token_id)

    # Build prompts
    prompts = build_inference_prompts(features, feature_names, tokenizer)

    predictions = []
    parse_failures = 0

    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=INFERENCE_MAX_NEW_TOKENS,
                do_sample=INFERENCE_DO_SAMPLE,
                logits_processor=[processor],
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        score = parse_score(response)
        if score == -1:
            parse_failures += 1
            score = 1  # fallback to most common class
        predictions.append(score)

    if parse_failures > 0:
        logging.warning(f"Parse failures: {parse_failures}/{len(prompts)}")

    return np.array(predictions)


# ===========================
# Training
# ===========================

def train_fold(fold_idx, train_features, train_labels, val_features, val_ratings,
               val_majority, feature_names, strategy):
    """Train one fold: load model, fine-tune, inference, compute metrics."""

    logging.info(f"  Loading fresh model for fold {fold_idx + 1}...")
    model, tokenizer = load_model_and_tokenizer()

    # Build training dataset
    train_dataset = build_train_dataset(train_features, train_labels, feature_names)
    logging.info(f"  Training samples: {len(train_dataset)}")

    # SFT Config (replaces TrainingArguments in trl >= 0.29)
    fold_output_dir = str(CHECKPOINT_DIR / f"{strategy}_fold{fold_idx + 1}")
    sft_config = SFTConfig(
        output_dir=fold_output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        fp16=USE_FP16,
        bf16=USE_BF16,
        max_grad_norm=MAX_GRAD_NORM,
        optim=OPTIM,
        seed=RANDOM_SEED,
        report_to="tensorboard",
        logging_dir=str(LOG_DIR / f"{strategy}_fold{fold_idx + 1}"),
        remove_unused_columns=False,
        max_length=MAX_SEQ_LENGTH,
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=sft_config,
    )

    logging.info(f"  Training fold {fold_idx + 1}...")
    trainer.train()

    # Inference
    logging.info(f"  Running inference on {len(val_features)} validation samples...")
    predictions = run_inference(model, tokenizer, val_features, feature_names)

    # Compute metrics
    metrics = compute_metrics(val_majority, predictions)
    consensus = evaluate_consensus_levels(predictions, val_ratings, val_majority)

    fold_result = {
        "fold": fold_idx + 1,
        **metrics,
        "consensus_results": consensus,
        "predictions": predictions.tolist(),
        "labels": val_majority.tolist(),
    }

    logging.info(
        f"  Fold {fold_idx + 1} Results: "
        f"Acc={metrics['accuracy']:.3f}, MAE={metrics['mae']:.3f}, "
        f"+-1={metrics['off_by_one_accuracy']:.3f}, "
        f"QWK={metrics['qwk']:.3f}, F1={metrics['f1_weighted']:.3f}"
    )

    # Cleanup GPU memory
    del model, trainer, tokenizer
    if DEVICE_TYPE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE_TYPE == "mps":
        torch.mps.empty_cache()
    gc.collect()

    return fold_result


def run_cross_validation():
    """Full 5-fold CV with QLoRA fine-tuning."""
    np.random.seed(RANDOM_SEED)
    setup_logging()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 60)
    logging.info("QLoRA Fine-Tuning: Parkinson's Disease Severity Classification")
    logging.info(f"Model: {MODEL_ID}")
    logging.info(f"QLoRA: rank={QLORA_RANK}, alpha={QLORA_ALPHA}, bits={QLORA_BITS}")
    logging.info(f"Training: epochs={EPOCHS}, lr={LEARNING_RATE}, batch={BATCH_SIZE}x{GRADIENT_ACCUMULATION_STEPS}")
    logging.info("=" * 60)

    logging.info("Loading data...")
    data = load_data()
    all_features = data["all_features"]
    all_ratings = data["all_ratings"]
    majority_scores = data["majority_scores"]
    cv_splits = data["cv_splits"]
    feature_names = data["feature_names"]

    logging.info(f"Total samples: {len(all_features)}, Features: {all_features.shape[1]}")
    logging.info(f"Score distribution: {np.bincount(majority_scores.astype(int), minlength=SCORE_NUM_CLASSES)}")

    # Results storage
    all_results = {s: [] for s in LABEL_STRATEGIES}

    for strategy in LABEL_STRATEGIES:
        logging.info(f"\n{'#' * 60}")
        logging.info(f"Label Strategy: {strategy}")
        logging.info(f"{'#' * 60}")

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            logging.info(f"\n--- Fold {fold_idx + 1}/{NUM_FOLDS} (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

            train_ratings = all_ratings[train_idx]
            train_majority = majority_scores[train_idx]
            val_ratings = all_ratings[val_idx]
            val_majority = majority_scores[val_idx]

            # Get training labels
            y_train, ds_result = get_train_labels(strategy, train_ratings, train_majority)

            if ds_result is not None:
                changed = np.sum(y_train != train_majority)
                logging.info(
                    f"  Dawid-Skene: {ds_result['n_iterations']} iters, "
                    f"converged={ds_result['converged']}, "
                    f"labels changed: {changed}/{len(y_train)} ({100*changed/len(y_train):.1f}%)"
                )

            # Train and evaluate this fold
            fold_result = train_fold(
                fold_idx=fold_idx,
                train_features=all_features[train_idx],
                train_labels=y_train,
                val_features=all_features[val_idx],
                val_ratings=val_ratings,
                val_majority=val_majority,
                feature_names=feature_names,
                strategy=strategy,
            )

            # Store DS info for fold 1
            if ds_result is not None and fold_idx == 0:
                fold_result["ds_confusion_matrices"] = ds_result["confusion_matrices"].tolist()
                fold_result["ds_class_priors"] = ds_result["class_priors"].tolist()

            all_results[strategy].append(fold_result)

    # === Build summary ===
    summary = {
        "method": f"QLoRA Fine-tuned {MODEL_ID}",
        "model": MODEL_ID,
        "qlora_config": {
            "rank": QLORA_RANK,
            "alpha": QLORA_ALPHA,
            "bits": QLORA_BITS,
            "target_modules": QLORA_TARGET_MODULES,
        },
        "training_config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "scheduler": LR_SCHEDULER,
        },
        "strategies": {},
    }

    logging.info(f"\n{'=' * 80}")
    logging.info("CROSS-VALIDATION RESULTS SUMMARY")
    logging.info(f"{'=' * 80}")
    logging.info(f"{'Strategy':<16} {'Acc':>14} {'MAE':>14} {'+-1 Acc':>14} {'QWK':>14} {'F1':>14}")
    logging.info("-" * 80)

    for strategy in LABEL_STRATEGIES:
        folds = all_results[strategy]
        accs = [r["accuracy"] for r in folds]
        maes = [r["mae"] for r in folds]
        obos = [r["off_by_one_accuracy"] for r in folds]
        qwks = [r["qwk"] for r in folds]
        f1s = [r["f1_weighted"] for r in folds]

        summary["strategies"][strategy] = {
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

        logging.info(
            f"{strategy:<16} "
            f"{np.mean(accs):.3f}+/-{np.std(accs):.3f}  "
            f"{np.mean(maes):.3f}+/-{np.std(maes):.3f}  "
            f"{np.mean(obos):.3f}+/-{np.std(obos):.3f}  "
            f"{np.mean(qwks):.3f}+/-{np.std(qwks):.3f}  "
            f"{np.mean(f1s):.3f}+/-{np.std(f1s):.3f}"
        )

    # Save results
    results_file = OUTPUT_DIR / "ft_cv_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved to {results_file}")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("QLoRA Fine-Tuning: 5-Fold Cross-Validation")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)
    run_cross_validation()
