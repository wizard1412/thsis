#!/usr/bin/env python3
"""
Complete Batch Scoring System
Features:
1. Extract CSV features (raw data + statistical features)
2. Generate prompts containing features
3. Call Ollama for scoring
4. Save results

Features include:
- Raw data: num_taps, tap_periods, tap_amplitudes
- Statistical features: period_stdev, periodEntropy, finger_mvmnt_x_mean,
  finger_mvmnt_dist_mean, speed_mean, period_max, maxFreezeDuration, aperiodicity
"""

import numpy as np
from pathlib import Path
import requests
import json
import csv
import re
import time
import os
import logging
from datetime import datetime

from generate_prompt_with_features import generate_prompt_with_features

# ===========================
# Configuration
# ===========================

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:32b"

# Directory settings
CSV_DIR = "./csv_files"
OUTPUT_DIR = "./results"
BASE_PROMPT_PATH = "Prompt.txt"  # Base prompt

# CSV output settings (for filling scores into template CSV)
TEMPLATE_CSV = "scored_by_ChatGPT_promptC_template.csv"
SCORED_CSV = f"scored_by_{MODEL.replace(':', '_')}.csv"
FILENAME_COL = 0         # Column index for filenames (0 = Column A)
SCORE_START_COL = 5      # Start column for scores (5 = Column F)
SCORE_END_COL = 14       # End column for scores (14 = Column O)

# Number of evaluations per CSV
EVALUATION_TIMES = 10

# Delay between evaluations (seconds)
DELAY_BETWEEN_CALLS = 25

# Request timeout (seconds)
# Good responses arrive in 15-60s; if no response after 180s, it's likely stuck - retry faster
REQUEST_TIMEOUT = 300  # 5 minutes (thinking mode needs more time)

# Retry settings - more retries with shorter delay since we fail-fast now
MAX_RETRIES = 5  # Maximum retry attempts
RETRY_DELAY = 10  # Retry delay (seconds)

# Max tokens to generate (prevents runaway generation / thinking loops)
NUM_PREDICT = 16384

# ===========================
# Logging Configuration
# ===========================

def setup_logging():
    """Setup logging system"""
    log_dir = Path(OUTPUT_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_score_{timestamp}.log"

    # Setup log format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Output to file only (no terminal output)
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )

    logging.info(f"Log file: {log_file}")
    return log_file

# ===========================
# LLM Call
# ===========================

def check_ollama_connection():
    """Check if Ollama service is available"""
    try:
        logging.info("Checking Ollama connection...")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json()
        logging.info(f"Ollama connection OK. Available models: {len(models.get('models', []))}")

        # Check if our target model is available
        model_names = [m['name'] for m in models.get('models', [])]
        if MODEL in model_names:
            logging.info(f"Target model '{MODEL}' is available")
        else:
            logging.warning(f"Target model '{MODEL}' not found in available models: {model_names}")

        return True
    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        logging.error(f"Please make sure Ollama is running at {OLLAMA_URL}")
        return False

def should_disable_thinking(model):
    """Check if thinking mode should be disabled for this model.
    Model name format: qwen3:30b, qwen3:70b, qwen3:235b, etc."""
    return model == "qwen3:30b"

def call_ollama(prompt, model=MODEL, retry_count=0):
    """
    Call Ollama API with detailed logging and retry mechanism
    """
    # Disable thinking mode via API parameter for qwen3 models < 70b
    disable_thinking = should_disable_thinking(model)
    if disable_thinking:
        logging.info(f"Thinking mode disabled for {model} (< 70b) via API 'think' parameter")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": NUM_PREDICT
        }
    }

    # Add think: false at the API level to properly disable thinking mode
    if disable_thinking:
        payload["think"] = False

    # Record request start time
    start_time = time.time()
    attempt = retry_count + 1

    logging.info(f"Starting Ollama API call (attempt {attempt}/{MAX_RETRIES})")
    logging.info(f"Model: {model}")
    logging.info(f"Prompt length: {len(prompt)} characters")
    logging.info(f"Timeout setting: {REQUEST_TIMEOUT} seconds")

    try:
        # Send request
        logging.info(f"Sending request to {OLLAMA_URL}...")
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logging.info(f"Response received, elapsed time: {elapsed_time:.2f} seconds")

        # Check HTTP status code
        response.raise_for_status()

        # Parse JSON
        result = response.json()

        # Check response content
        if 'response' not in result:
            logging.error(f"Missing 'response' field in result: {result}")
            raise ValueError("Invalid response format")

        response_text = result['response']
        logging.info(f"Successfully got response, length: {len(response_text)} characters")
        logging.debug(f"Response preview: {response_text[:200]}...")

        # Debug: if response is empty, log the full raw JSON to understand why
        if len(response_text) == 0:
            logging.warning(f"Empty response detected! Raw JSON keys: {list(result.keys())}")
            # Log all fields except 'response' to see what Ollama returned
            for key, value in result.items():
                if key != 'response':
                    val_str = str(value)
                    if len(val_str) > 500:
                        val_str = val_str[:500] + "..."
                    logging.warning(f"  raw['{key}']: {val_str}")

            # Retry on empty response (model likely consumed all tokens on thinking)
            if retry_count < MAX_RETRIES - 1:
                logging.warning(f"Retrying due to empty response in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
                return call_ollama(prompt, model, retry_count + 1)
            else:
                logging.error(f"Max retries ({MAX_RETRIES}) reached, all returned empty")

        return response_text

    except requests.exceptions.Timeout as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Request timeout! Waited {elapsed_time:.2f} seconds")
        logging.error(f"Timeout error: {str(e)}")

        # Retry if attempts remaining
        if retry_count < MAX_RETRIES - 1:
            logging.warning(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return call_ollama(prompt, model, retry_count + 1)
        else:
            logging.error(f"Max retries ({MAX_RETRIES}) reached, giving up")
            return None

    except requests.exceptions.ConnectionError as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Connection error! Elapsed time: {elapsed_time:.2f} seconds")
        logging.error(f"Connection error details: {str(e)}")
        logging.error(f"Please check if Ollama service is running: {OLLAMA_URL}")

        # Retry
        if retry_count < MAX_RETRIES - 1:
            logging.warning(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return call_ollama(prompt, model, retry_count + 1)
        else:
            logging.error(f"Max retries ({MAX_RETRIES}) reached, giving up")
            return None

    except requests.exceptions.HTTPError as e:
        elapsed_time = time.time() - start_time
        logging.error(f"HTTP error! Elapsed time: {elapsed_time:.2f} seconds")
        logging.error(f"Status code: {response.status_code}")
        logging.error(f"Error message: {str(e)}")
        logging.error(f"Response content: {response.text[:500]}")
        return None

    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Unexpected error! Elapsed time: {elapsed_time:.2f} seconds")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")

        # Retry
        if retry_count < MAX_RETRIES - 1:
            logging.warning(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return call_ollama(prompt, model, retry_count + 1)
        else:
            logging.error(f"Max retries ({MAX_RETRIES}) reached, giving up")
            return None

def parse_score(response):
    """
    Extract score from LLM response
    Find 0-4 score in the last line
    """
    if response is None:
        return None

    # Find number in the last line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        # Find 0-4 number
        match = re.search(r'\b([0-4])\b', line)
        if match:
            return int(match.group(1))

    return None

# ===========================
# Save Results
# ===========================

def save_results(all_results, output_dir=OUTPUT_DIR):
    """
    Save results to file
    """
    # Prepare serializable results
    serializable_results = {}
    for csv_name, result in all_results.items():
        serializable_results[csv_name] = {
            'scores': result['scores'],
            'valid_scores': result['valid_scores'],
            'mean_score': float(result['mean_score']) if result['mean_score'] is not None else None,
            'std_score': float(result['std_score']) if result['std_score'] is not None else None,
            'mode_score': result['mode_score'],
            'features': result['features']
        }

    # Save JSON
    output_file = Path(output_dir) / "all_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    # Save detailed responses
    responses_file = Path(output_dir) / "detailed_responses.txt"
    with open(responses_file, 'w', encoding='utf-8') as f:
        for csv_name, result in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{csv_name}\n")
            f.write(f"{'='*60}\n\n")

            for i, (score, response) in enumerate(zip(result['scores'], result['responses']), 1):
                f.write(f"--- Evaluation {i} (Score: {score}) ---\n")
                f.write(response if response else "No response")
                f.write("\n\n")

    return output_file, responses_file

# ===========================
# Fill Scores into Template CSV
# ===========================

def normalize_filename(filename):
    """Normalize filename by removing .csv extension."""
    if not filename:
        return ""
    return str(filename).strip().replace('.csv', '')

def fill_scores_to_csv(json_data, csv_path, output_path):
    """
    Map scores from JSON results into a template CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"Error: Template CSV not found: {csv_path}")
        logging.error(f"Template CSV not found: {csv_path}")
        return 0, []

    print(f"Loading template CSV: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        print("Error: CSV file is empty")
        return 0, []

    # Build filename -> scores mapping
    json_scores = {}
    for key, value in json_data.items():
        normalized_key = normalize_filename(key)
        scores = value.get('scores', [])
        json_scores[normalized_key] = scores[:10] if len(scores) >= 10 else scores

    print(f"Total records in JSON: {len(json_scores)}")
    print(f"Total rows in CSV: {len(rows)}")

    matched_count = 0
    unmatched_files = []

    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        if len(row) <= FILENAME_COL:
            continue

        filename = row[FILENAME_COL]
        if not filename:
            continue

        normalized_filename = normalize_filename(filename)

        if normalized_filename in json_scores:
            scores = json_scores[normalized_filename]

            while len(row) < SCORE_END_COL + 1:
                row.append('')

            for i, score in enumerate(scores):
                col_idx = SCORE_START_COL + i
                if col_idx <= SCORE_END_COL:
                    row[col_idx] = score

            rows[row_idx] = row
            matched_count += 1
            print(f"  Filled: {filename} -> {scores}")
        else:
            unmatched_files.append(filename)

    print(f"Saving scored CSV to: {output_path}")
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Successfully matched: {matched_count} records")
    if unmatched_files:
        print(f"Unmatched: {len(unmatched_files)} records")
        for fname in unmatched_files[:10]:
            print(f"  - {fname}")
        if len(unmatched_files) > 10:
            print(f"  ... and {len(unmatched_files) - 10} more")

    logging.info(f"CSV fill complete: {matched_count} matched, {len(unmatched_files)} unmatched -> {output_path}")
    return matched_count, unmatched_files

# ===========================
# Single CSV Processing
# ===========================

def process_single_csv(csv_path):
    """
    Process a single CSV evaluation
    Use new feature extraction function (raw data + statistical features)
    """
    logging.info(f"Processing CSV: {csv_path.name}")

    # 1. Generate prompt (including raw data and statistical features)
    logging.info("Generating prompt...")
    try:
        prompt, features = generate_prompt_with_features(csv_path, BASE_PROMPT_PATH)
        logging.info(f"Prompt generated successfully, feature count: {len(features) if features else 0}")
    except Exception as e:
        logging.error(f"Failed to generate prompt: {e}")
        raise

    # 2. Call Ollama
    logging.info("Calling Ollama API...")
    response = call_ollama(prompt)

    if response is None:
        logging.warning("No valid response received")
    else:
        logging.info(f"Response received, length: {len(response)} characters")

    # 3. Parse score
    logging.info("Parsing score...")
    score = parse_score(response)

    if score is not None:
        logging.info(f"Score parsed successfully: {score}")
    else:
        logging.warning("Score parsing failed")
        if response:
            logging.warning(f"Response content preview: {response[:300]}...")

    return {
        'score': score,
        'response': response,
        'features': features
    }

# ===========================
# Batch Processing Main Flow
# ===========================

def batch_process_all():
    """
    Batch process all CSV files
    Save immediately after each evaluation to avoid data loss due to interruption
    """
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    # Find all CSVs
    csv_files = list(Path(CSV_DIR).glob("*.csv"))

    if len(csv_files) == 0:
        error_msg = f"No CSV files found in {CSV_DIR}"
        logging.error(error_msg)
        print(f"Error: {error_msg}")
        return

    logging.info(f"Found {len(csv_files)} CSV files")
    print(f"Found {len(csv_files)} CSV files")
    print(f"Each file will be evaluated {EVALUATION_TIMES} times\n")

    all_results = {}

    # Process each CSV
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"\n{'='*60}")
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")
        print(f"{'='*60}")

        logging.info(f"\n{'='*60}")
        logging.info(f"[{i}/{len(csv_files)}] Processing: {csv_name}")
        logging.info(f"CSV path: {csv_path}")
        logging.info(f"{'='*60}")

        scores = []
        responses = []
        features_list = []

        # Evaluate each CSV multiple times
        for eval_num in range(1, EVALUATION_TIMES + 1):
            logging.info(f"\n--- Evaluation {eval_num}/{EVALUATION_TIMES} ---")
            print(f"  Evaluation {eval_num}/{EVALUATION_TIMES}...", end=' ', flush=True)

            try:
                result = process_single_csv(csv_path)

                scores.append(result['score'])
                responses.append(result['response'])
                features_list.append(result['features'])

                if result['score'] is not None:
                    print(f"Score: {result['score']}", end=' ')
                    logging.info(f"Evaluation {eval_num} completed - Score: {result['score']}")
                else:
                    print("Score parsing failed", end=' ')
                    logging.warning(f"Evaluation {eval_num} - Score parsing failed")

            except Exception as e:
                logging.error(f"Evaluation {eval_num} failed with exception: {e}", exc_info=True)
                print(f"Error: {e}", end=' ')
                scores.append(None)
                responses.append(str(e))
                features_list.append(None)

            # Save immediately after each evaluation
            # Calculate current statistics
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                mode_score = max(set(valid_scores), key=valid_scores.count)
            else:
                mean_score = None
                std_score = None
                mode_score = None

            # Update results dictionary
            all_results[csv_name] = {
                'scores': scores,
                'valid_scores': valid_scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'mode_score': mode_score,
                'responses': responses,
                'features': features_list[0] if features_list else None
            }

            # Save to file immediately
            try:
                save_results(all_results)
                print(f"✓", flush=True)
                logging.info(f"Results saved successfully")
            except Exception as e:
                print(f"⚠ Save failed: {e}", flush=True)
                logging.error(f"Failed to save results: {e}")

            # Avoid calling too fast
            if eval_num < EVALUATION_TIMES:
                logging.info(f"Waiting {DELAY_BETWEEN_CALLS} seconds before next evaluation...")
                time.sleep(DELAY_BETWEEN_CALLS)

        # Display summary
        result = all_results[csv_name]
        print(f"\n  Completed! Statistics:")
        print(f"    Valid scores: {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        if result['valid_scores']:
            print(f"    Average score: {result['mean_score']:.2f} ± {result['std_score']:.2f}")
            print(f"    Mode score: {result['mode_score']}")
            print(f"    All scores: {result['valid_scores']}")

    # Final save
    output_file, responses_file = save_results(all_results)

    print(f"\n{'='*60}")
    print(f"✓ Completed! Results saved to: {output_file}")
    print(f"✓ Detailed responses saved to: {responses_file}")
    print(f"{'='*60}")

    # Display summary table
    print(f"\n{'='*60}")
    print("Scoring Summary")
    print(f"{'='*60}")
    print(f"{'File Name':<40} {'Average Score':<12} {'Mode':<8} {'Valid/Total'}")
    print(f"{'-'*60}")

    for csv_name, result in all_results.items():
        name_short = csv_name[:37] + "..." if len(csv_name) > 40 else csv_name
        if result['mean_score'] is not None:
            print(f"{name_short:<40} {result['mean_score']:>6.2f}±{result['std_score']:.2f}   "
                  f"{result['mode_score']:>4}     {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        else:
            print(f"{name_short:<40} {'N/A':<12} {'N/A':<8} {len(result['valid_scores'])}/{EVALUATION_TIMES}")

    # Fill scores into template CSV
    print(f"\n{'='*60}")
    print("Filling scores into template CSV...")
    print(f"{'='*60}")
    try:
        # Load the saved JSON results
        results_json_path = Path(OUTPUT_DIR) / "all_results.json"
        with open(results_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        matched, unmatched = fill_scores_to_csv(json_data, TEMPLATE_CSV, SCORED_CSV)
        print(f"Scored CSV saved to: {SCORED_CSV}")
    except Exception as e:
        print(f"Warning: Failed to fill scores into CSV: {e}")
        logging.error(f"Failed to fill scores into CSV: {e}")

    return all_results

# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    # Setup logging first
    log_file = setup_logging()

    print("="*60)
    print("Finger Tapping Automatic Scoring System (with Statistical Features)")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Evaluations per file: {EVALUATION_TIMES}")
    print(f"Request Timeout: {REQUEST_TIMEOUT} seconds")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"Max Tokens (num_predict): {NUM_PREDICT}")
    print(f"Log file: {log_file}")
    print()
    print("Features include:")
    print("  - Raw data: num_taps, tap_periods, tap_amplitudes")
    print("  - Statistical features: period_stdev, periodEntropy, finger_mvmnt_x_mean,")
    print("              finger_mvmnt_dist_mean, speed_mean, period_max,")
    print("              maxFreezeDuration, aperiodicity")
    print("="*60 + "\n")

    logging.info("="*60)
    logging.info("Batch scoring process started")
    logging.info(f"Model: {MODEL}")
    logging.info(f"Request timeout: {REQUEST_TIMEOUT} seconds")
    logging.info(f"Max retries: {MAX_RETRIES}")
    logging.info(f"Delay between calls: {DELAY_BETWEEN_CALLS} seconds")
    logging.info(f"Max tokens (num_predict): {NUM_PREDICT}")
    logging.info("="*60)

    # Check if base prompt exists
    if not Path(BASE_PROMPT_PATH).exists():
        logging.error(f"Cannot find {BASE_PROMPT_PATH}")
        print(f"Error: Cannot find {BASE_PROMPT_PATH}")
        print("Please ensure Prompt.txt is in the current directory")
        exit(1)

    # Check Ollama connection
    if not check_ollama_connection():
        print(f"\nError: Cannot connect to Ollama service at {OLLAMA_URL}")
        print("Please make sure Ollama is running before starting the batch process")
        exit(1)

    print("\nOllama connection verified. Starting batch processing...\n")

    # Execute batch processing
    try:
        batch_process_all()
        logging.info("Batch processing completed successfully")
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        raise
