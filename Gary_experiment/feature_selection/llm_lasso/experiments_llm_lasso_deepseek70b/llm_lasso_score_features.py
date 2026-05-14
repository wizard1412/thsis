"""
Step 1: LLM-Lasso Penalty Factor Generation
=============================================
Follows the exact methodology from Zhang et al. (2025) LLM-Lasso paper.

Uses the same prompt format and score extraction regex as the original repo:
  - Prompt template with {genes} and {category} placeholders
  - Output format: **FeatureName**: score (integer 2-5)
  - Penalty factor 2 = strongly relevant (penalized LEAST)
  - Penalty factor 5 = minimal relevance (penalized MOST)

Adapted for local Ollama inference instead of OpenAI API.

Output: results/llm_feature_scores.json + results/final_scores_plain.pkl
"""

import json
import time
import re
import pickle
import requests
import numpy as np
import pandas as pd
from pathlib import Path

from llm_lasso_config import (
    OLLAMA_URL, LLM_MODEL, REQUEST_TIMEOUT, NUM_PREDICT,
    DELAY_BETWEEN_CALLS, LLM_SCORING_REPEATS,
    USER_DATASET_PATH, EXCLUDE_COLS,
    FEATURE_DESCRIPTIONS, OUTPUT_DIR, SCORES_JSON,
)

PROMPT_PATH = Path(__file__).parent / "prompts" / "finger_tapping_prompt.txt"
CATEGORY = "MDS-UPDRS Item 3.4 Finger Tapping severity"


# ===========================
# Prompt construction (follows original repo's create_general_prompt)
# ===========================

def create_prompt(prompt_path, category, features):
    """
    Load prompt template and fill placeholders.
    Matches: llm_lasso.utils.score_collection.create_general_prompt()
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    filled = template.replace("{category}", category)
    filled = filled.replace("{genes}", ", ".join(features))
    return filled


# ===========================
# Score extraction (follows original repo's extract_scores_from_responses)
# ===========================

def extract_scores_from_response(response_text, feature_names):
    """
    Extract scores using the same regex as the original LLM-Lasso repo.
    Pattern: **FeatureName**: score
    """
    scores = {feat: None for feat in feature_names}

    # Original repo regex (from score_collection.py line 132)
    matches = re.findall(
        r"\*\*(.*?)\*\*:\s*(-?\d+(?:\.\d+)?|inf(?:inity)?|∞)",
        response_text, re.IGNORECASE
    )

    for name, score_str in matches:
        # Clean up feature name
        cleaned = name.strip().replace(" ", "_")

        # Try exact match first
        if cleaned in scores:
            scores[cleaned] = float(score_str)
            continue

        # Try case-insensitive match
        for feat in feature_names:
            if feat.lower() == cleaned.lower():
                scores[feat] = float(score_str)
                break

    return scores


# ===========================
# Ollama query
# ===========================

def call_ollama(prompt, model=LLM_MODEL):
    """Call local Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": 0,  # Deterministic (same as original paper)
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["response"]


# ===========================
# Batch scoring (follows original repo's collect_penalties)
# ===========================

def score_all_features():
    """
    Score all features following the LLM-Lasso paper methodology.
    Matches: llm_lasso.llm_penalty.penalty_collection.collect_penalties()
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature names
    df = pd.read_csv(USER_DATASET_PATH)
    feature_names = [c for c in df.columns
                     if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[c])]

    print(f"Dataset: {USER_DATASET_PATH}")
    print(f"Features to score: {len(feature_names)}")
    print(f"LLM model: {LLM_MODEL}")
    print(f"Prompt: {PROMPT_PATH}")
    print(f"Trials: {LLM_SCORING_REPEATS}")
    print("=" * 60)

    # --- Batch processing (original repo uses batch_size=30) ---
    BATCH_SIZE = 30
    all_trial_scores = []

    for trial in range(LLM_SCORING_REPEATS):
        print(f"\n--- Trial {trial + 1}/{LLM_SCORING_REPEATS} ---")
        trial_scores = {feat: None for feat in feature_names}

        for batch_start in range(0, len(feature_names), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(feature_names))
            batch_features = feature_names[batch_start:batch_end]

            print(f"  Batch [{batch_start+1}-{batch_end}] / {len(feature_names)}...", end=" ")

            # Build prompt for this batch
            prompt = create_prompt(PROMPT_PATH, CATEGORY, batch_features)

            # Query LLM with retry
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = call_ollama(prompt)

                    # Save raw response
                    raw_path = OUTPUT_DIR / f"raw_response_trial{trial}_batch{batch_start}.txt"
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(response)

                    # Extract scores
                    batch_scores = extract_scores_from_response(response, batch_features)

                    # Check completeness
                    n_found = sum(1 for v in batch_scores.values() if v is not None)
                    n_missing = len(batch_features) - n_found

                    if n_missing == 0:
                        print(f"OK ({n_found}/{len(batch_features)} extracted)")
                        break
                    elif n_missing <= 3 and attempt < max_retries - 1:
                        missing = [f for f, v in batch_scores.items() if v is None]
                        print(f"RETRY (missing {n_missing}: {missing})")
                        time.sleep(DELAY_BETWEEN_CALLS)
                        continue
                    else:
                        print(f"PARTIAL ({n_found}/{len(batch_features)}, "
                              f"filling missing with default=3)")
                        break

                except Exception as e:
                    print(f"ERROR: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(DELAY_BETWEEN_CALLS)
                    continue

            # Merge batch scores into trial
            for feat in batch_features:
                if batch_scores.get(feat) is not None:
                    trial_scores[feat] = batch_scores[feat]
                else:
                    trial_scores[feat] = 3.5  # Neutral default

            time.sleep(DELAY_BETWEEN_CALLS)

        all_trial_scores.append(trial_scores)

    # --- Average across trials ---
    final_scores = {}
    detailed_results = {}

    for feat in feature_names:
        trial_vals = [ts[feat] for ts in all_trial_scores if ts[feat] is not None]
        if trial_vals:
            median_score = float(np.median(trial_vals))
            mean_score = float(np.mean(trial_vals))
        else:
            median_score = 3.5
            mean_score = 3.5
            trial_vals = [3.5]

        final_scores[feat] = median_score
        detailed_results[feat] = {
            "scores": trial_vals,
            "median": median_score,
            "mean": mean_score,
            "std": float(np.std(trial_vals)),
            "description": FEATURE_DESCRIPTIONS.get(feat, feat),
        }

    # --- Save results in multiple formats ---

    # 1. JSON (detailed, our format)
    with open(SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    # 2. Pickle (compatible with original repo's final_scores_plain.pkl)
    scores_array = np.array([final_scores[f] for f in feature_names])
    pkl_path = OUTPUT_DIR / "final_scores_plain.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(scores_array, f)

    # 3. Text (human readable, compatible with original repo)
    txt_path = OUTPUT_DIR / "final_scores_plain.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for feat in feature_names:
            f.write(f"{feat}: {final_scores[feat]}\n")

    # 4. Feature names pickle (for original repo compatibility)
    names_pkl = OUTPUT_DIR / "feature_names.pkl"
    with open(names_pkl, "wb") as f:
        pickle.dump(feature_names, f)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PENALTY FACTOR RANKING (lower = more important, penalized less)")
    print("=" * 60)
    ranked = sorted(final_scores.items(), key=lambda x: x[1])
    for rank, (feat, score) in enumerate(ranked, 1):
        print(f"  #{rank:2d}  PF={score:4.1f}  {feat}")

    print(f"\nResults saved to:")
    print(f"  JSON:   {SCORES_JSON}")
    print(f"  Pickle: {pkl_path}")
    print(f"  Text:   {txt_path}")

    return final_scores


if __name__ == "__main__":
    score_all_features()
