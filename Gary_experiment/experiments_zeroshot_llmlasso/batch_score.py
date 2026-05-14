#!/usr/bin/env python3
"""
Complete Batch Scoring System (Zero-Shot + LLMLasso)
Features:
1. Extract CSV features (LLMLasso selected features)
2. Generate zero-shot prompts containing features
3. Call Ollama for scoring
4. Save results

Features include LLMLasso selected features (dynamically configurable)
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

from generate_prompt_with_features import generate_prompt_with_features, LLMLASSO_SELECTED_FEATURES

# ===========================
# Configuration (defaults)
# ===========================

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:70b"

# Directory settings
CSV_DIR = "./csv_files"
OUTPUT_DIR = "./results"
BASE_PROMPT_PATH = "Prompt.txt"  # Zero-shot prompt

# CSV output settings (for filling scores into template CSV)
TEMPLATE_CSV = "scored_by_ChatGPT_promptC_template.csv"
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
# Experiment Configurations
# ===========================
# 在這裡定義所有實驗。每個實驗有獨立的輸出目錄，不會互相蓋掉。
# 執行方式:
#   單一實驗: python batch_score.py --experiment lasso_v1
#   多個實驗: python batch_score.py --experiment lasso_v1 lasso_v2
#   全部實驗: python batch_score.py --all
#   平行執行: python run_experiments.py --experiments lasso_v1 lasso_v2

EXPERIMENTS = {
    "qwen72b_no_amplitude": {
        # LLM-LASSO (deepseek70b版, η=3.0) 選出的 6 個特徵 (cv_mae=0.4555)，使用 qwen2.5:72b
        "features": [
            "finger_mvmnt_x_mean",
            "finger_mvmnt_x_max",
            "periodEntropy",
            "period_quartile_range",
            "period_min",
            "num_peaks",
        ],
        "output_dir": "./results/qwen72b_no_amplitude",
        "prompt": "Prompt.txt",
        "model": "qwen2.5:72b",
    },
    "qwen72b_amplitude": {
        # LLM-LASSO (amplitude版, η=2.0) 選出的 8 個特徵 (cv_mae=0.4477)，使用 qwen2.5:72b
        "features": [
            "finger_mvmnt_x_max",
            "periodEntropy",
            "period_quartile_range",
            "period_min",
            "num_peaks",
            "amplitude_quartile_range",
            "amplitude_min",
            "amplitude_entropy",
        ],
        "output_dir": "./results/qwen72b_amplitude",
        "prompt": "Prompt.txt",
        "model": "qwen2.5:72b",
    },
    "qwen32b_no_amplitude": {
        # LLM-LASSO (deepseek70b版, η=3.0) 選出的 6 個特徵 (cv_mae=0.4555)，使用 qwen2.5:32b
        "features": [
            "finger_mvmnt_x_mean",
            "finger_mvmnt_x_max",
            "periodEntropy",
            "period_quartile_range",
            "period_min",
            "num_peaks",
        ],
        "output_dir": "./results/qwen32b_no_amplitude",
        "prompt": "Prompt.txt",
        "model": "qwen2.5:32b",
    },
    "ds32b_no_amplitude": {
        # LLM-LASSO (deepseek70b版, η=3.0) 選出的 6 個特徵 (cv_mae=0.4555)，使用 deepseek-r1:32b
        "features": [
            "finger_mvmnt_x_mean",
            "finger_mvmnt_x_max",
            "periodEntropy",
            "period_quartile_range",
            "period_min",
            "num_peaks",
        ],
        "output_dir": "./results/ds32b_no_amplitude",
        "prompt": "Prompt.txt",
        "model": "deepseek-r1:32b",
    },
    "ds32b_amplitude": {
        # LLM-LASSO (amplitude版, η=2.0) 選出的 8 個特徵 (cv_mae=0.4477)，使用 deepseek-r1:32b
        "features": [
            "finger_mvmnt_x_max",
            "periodEntropy",
            "period_quartile_range",
            "period_min",
            "num_peaks",
            "amplitude_quartile_range",
            "amplitude_min",
            "amplitude_entropy",
        ],
        "output_dir": "./results/ds32b_amplitude",
        "prompt": "Prompt.txt",
        "model": "deepseek-r1:32b",
    },
    "qwen32b_amplitude": {
        # LLM-LASSO (amplitude版, η=2.0) 選出的 8 個特徵 (cv_mae=0.4477)，使用 qwen2.5:32b
        "features": [
            "finger_mvmnt_x_max",
            "periodEntropy",
            "period_quartile_range",
            "period_min",
            "num_peaks",
            "amplitude_quartile_range",
            "amplitude_min",
            "amplitude_entropy",
        ],
        "output_dir": "./results/qwen32b_amplitude",
        "prompt": "Prompt.txt",
        "model": "qwen2.5:32b",
    },
}

# ===========================
# Logging Configuration
# ===========================

def setup_logging(output_dir=None):
    """Setup logging system"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    log_dir = Path(output_dir) / "logs"
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

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

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
        json.dump(serializable_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

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

def process_single_csv(csv_path, config):
    """
    Process a single CSV evaluation.

    Args:
        csv_path: Path to the CSV file
        config: Experiment config dict with keys: model, prompt, features, output_dir
    """
    logging.info(f"Processing CSV: {csv_path.name}")

    # 1. Generate prompt (including LLMLasso selected features)
    logging.info("Generating prompt...")
    try:
        prompt, features = generate_prompt_with_features(
            csv_path,
            base_prompt_path=config.get('prompt', BASE_PROMPT_PATH),
            selected_features=config.get('features'),
        )
        logging.info(f"Prompt generated successfully, feature count: {len(features) if features else 0}")
    except Exception as e:
        logging.error(f"Failed to generate prompt: {e}")
        raise

    # 2. Call Ollama
    model = config.get('model', MODEL)
    logging.info("Calling Ollama API...")
    response = call_ollama(prompt, model=model)

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

def batch_process_all(config):
    """
    Batch process all CSV files.
    Save immediately after each evaluation to avoid data loss due to interruption.

    Args:
        config: Experiment config dict with keys:
                  output_dir, model, prompt, features (list of feature names)
    """
    output_dir = config.get('output_dir', OUTPUT_DIR)
    model = config.get('model', MODEL)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

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
                result = process_single_csv(csv_path, config)

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

            # ★★★ Save immediately after each evaluation ★★★
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
                save_results(all_results, output_dir=output_dir)
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
    output_file, responses_file = save_results(all_results, output_dir=output_dir)

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
    scored_csv = Path(output_dir) / f"scored_by_{model.replace(':', '_')}.csv"
    print(f"\n{'='*60}")
    print("Filling scores into template CSV...")
    print(f"{'='*60}")
    try:
        results_json_path = Path(output_dir) / "all_results.json"
        with open(results_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        fill_scores_to_csv(json_data, TEMPLATE_CSV, str(scored_csv))
        print(f"Scored CSV saved to: {scored_csv}")
    except Exception as e:
        print(f"Warning: Failed to fill scores into CSV: {e}")
        logging.error(f"Failed to fill scores into CSV: {e}")

    return all_results


def run_experiment(experiment_name, experiments_dict=None):
    """
    執行單一實驗（供 run_experiments.py 平行呼叫用）。

    Args:
        experiment_name: EXPERIMENTS dict 中的 key
        experiments_dict: 實驗設定 dict（None 時使用模組內的 EXPERIMENTS）
    """
    if experiments_dict is None:
        experiments_dict = EXPERIMENTS

    if experiment_name not in experiments_dict:
        raise ValueError(f"Unknown experiment: '{experiment_name}'. Available: {list(experiments_dict.keys())}")

    config = experiments_dict[experiment_name]
    output_dir = config.get('output_dir', OUTPUT_DIR)
    model = config.get('model', MODEL)

    # 每個實驗有獨立的 log 檔
    log_file = setup_logging(output_dir=output_dir)
    logging.info(f"Starting experiment: {experiment_name}")
    logging.info(f"Config: {config}")

    prefix = f"[{experiment_name}] "
    print(f"\n{'='*60}")
    print(f"{prefix}Starting experiment (Zero-Shot + LLMLasso)")
    print(f"{prefix}Model   : {model}")
    print(f"{prefix}Features: {config.get('features', LLMLASSO_SELECTED_FEATURES)}")
    print(f"{prefix}Output  : {output_dir}")
    print(f"{prefix}Log     : {log_file}")
    print(f"{'='*60}\n")

    if not Path(config.get('prompt', BASE_PROMPT_PATH)).exists():
        raise FileNotFoundError(f"Prompt file not found: {config.get('prompt', BASE_PROMPT_PATH)}")

    if not check_ollama_connection():
        raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_URL}")

    try:
        results = batch_process_all(config)
        logging.info(f"Experiment '{experiment_name}' completed successfully")
        print(f"\n{prefix}✓ Experiment completed")
        return results
    except Exception as e:
        logging.error(f"Experiment '{experiment_name}' failed: {e}")
        print(f"\n{prefix}✗ Experiment failed: {e}")
        raise

# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Finger Tapping Automatic Scoring System (Zero-Shot + LLMLasso)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--experiment", "-e",
        nargs="+",
        metavar="NAME",
        help="要執行的實驗名稱（可指定多個，依序執行）\n例: --experiment qwen72b_no_amplitude qwen32b_amplitude",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="執行 EXPERIMENTS dict 中的所有實驗（依序）",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有可用的實驗名稱",
    )
    args = parser.parse_args()

    # --list: 只列出實驗名稱
    if args.list:
        print("Available experiments:")
        for name, cfg in EXPERIMENTS.items():
            feats = cfg.get('features', LLMLASSO_SELECTED_FEATURES)
            print(f"  {name}")
            print(f"    model   : {cfg.get('model', MODEL)}")
            print(f"    output  : {cfg.get('output_dir', OUTPUT_DIR)}")
            print(f"    features: {feats}")
        exit(0)

    # 決定要跑哪些實驗
    if args.all:
        to_run = list(EXPERIMENTS.keys())
    elif args.experiment:
        to_run = args.experiment
    else:
        # 預設行為：跑第一個實驗（向下相容）
        to_run = [list(EXPERIMENTS.keys())[0]]
        print(f"No experiment specified. Running default: {to_run[0]}")
        print(f"Tip: use --list to see available experiments\n")

    # 驗證實驗名稱
    for name in to_run:
        if name not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{name}'. Use --list to see available experiments.")
            exit(1)

    print("="*60)
    print("Finger Tapping Automatic Scoring System (Zero-Shot + LLMLasso)")
    print("="*60)
    print(f"Experiments to run ({len(to_run)}): {', '.join(to_run)}")
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Evaluations per file: {EVALUATION_TIMES}")
    print(f"Request Timeout: {REQUEST_TIMEOUT} seconds")
    print("="*60 + "\n")

    # 依序執行各實驗（平行執行請用 run_experiments.py）
    for exp_name in to_run:
        try:
            run_experiment(exp_name)
        except Exception as e:
            print(f"\n[{exp_name}] Failed: {e}")
            logging.error(f"Experiment '{exp_name}' failed: {e}")
            # 一個失敗不影響後續實驗
            continue

    print("\n" + "="*60)
    print(f"All done. Ran {len(to_run)} experiment(s): {', '.join(to_run)}")
    print("="*60)
