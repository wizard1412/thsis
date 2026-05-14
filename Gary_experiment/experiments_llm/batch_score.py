#!/usr/bin/env python3
"""
Unified Batch Scoring System
Consolidates: experiments_fewshot_llmlasso, experiments_zeroshot_llmlasso,
              experiments_fewshot_llmlasso_v2, experiments_zeroshot_llmlasso_v2

Experiment naming convention:
  fewshot_*      : v1 format, few-shot prompt
  zeroshot_*     : v1 format, zero-shot prompt
  fewshot_v2_*   : v2 format (AMPLITUDE PATTERN), few-shot prompt
  zeroshot_v2_*  : v2 format (AMPLITUDE PATTERN), zero-shot prompt

Usage:
  python batch_score.py --list
  python batch_score.py --experiment fewshot_v2_ds32b_amplitude
  python batch_score.py --experiment fewshot_v2_ds32b_amplitude zeroshot_v2_ds32b_amplitude
  python batch_score.py --all
  python run_experiments.py --experiments fewshot_v2_ds32b_amplitude zeroshot_v2_ds32b_amplitude
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

from generate_prompt_with_features import (
    generate_prompt_with_features,
    LLMLASSO_SELECTED_FEATURES,
    LLMLASSO_AMPLITUDE_FEATURES,
)

# ===========================
# Configuration (defaults)
# ===========================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:70b"

CSV_DIR = "../csv_files"
OUTPUT_DIR = "./results"
TEMPLATE_CSV = "scored_by_ChatGPT_promptC_template.csv"

FILENAME_COL = 0
SCORE_START_COL = 5
SCORE_END_COL = 14

EVALUATION_TIMES = 10
DELAY_BETWEEN_CALLS = 3
REQUEST_TIMEOUT = 300
MAX_RETRIES = 5
RETRY_DELAY = 10
NUM_PREDICT = 16384

# ===========================
# Experiment Configurations
# ===========================

_NO_AMP = LLMLASSO_SELECTED_FEATURES   # 6 features
_AMP    = LLMLASSO_AMPLITUDE_FEATURES  # 8 features

# ── 特徵頻率排名（依 feature_selection_comparison.py 的 figE 結果，v2 dataset） ──
# 從各 FS 方法中出現頻率由高到低，前 13 個特徵（含 amplitude 類）
_FREQ_RANKED = [
    "V_open_norm_std",                 # #1  count=11
    "A_late",                          # #2  count=10
    "period_quartile_range",           # #3  count=6
    "finger_mvmnt_thumb_dist_stdev",   # #4  count=6
    "finger_mvmnt_thumb_x_max",        # #5  count=5
    "amplitude_norm_median",           # #6  count=5
    "finger_mvmnt_thumb_x_stdev",      # #7  count=5
    "amplitude_norm_mean",             # #8  count=4
    "V_close_norm_std",                # #9  count=3
    "periodEntropy",                   # #10 count=3
    "A_early",                         # #11 count=3
    "amplitude_min",                   # #12 count=3
    "amplitude_median",                # #13 count=3
]

# 逐步加入 1→13 個特徵的子集
_FREQ_SUBSETS = {n: _FREQ_RANKED[:n] for n in range(1, 14)}

# ── 無 amplitude 特徵的頻率排名（依 figE_feature_frequency，v2 dataset） ──
# 從各 FS 方法中出現頻率由高到低，前 10 個（排除 amplitude/A_early/A_late/D_A_* 等）
_FREQ_RANKED_NO_AMP = [
    "V_open_norm_std",                 # #1  count=13
    "finger_mvmnt_thumb_x_max",        # #2  count=8
    "period_quartile_range",           # #3  count=7
    "finger_mvmnt_thumb_x_stdev",      # #4  count=6
    "finger_mvmnt_thumb_dist_stdev",   # #5  count=6
    "V_close_norm_std",                # #6  count=4
    "periodEntropy",                   # #7  count=3
    "frequency_lr_slope",              # #8  count=2
    "R_V_close",                       # #9  count=2
    "finger_mvmnt_thumb_x_mean",       # #10 count=2
]

_FREQ_SUBSETS_NO_AMP = {n: _FREQ_RANKED_NO_AMP[:n] for n in range(1, 11)}

# ── LLM-based FS 特徵頻率排名（依 fs_results_llm/fig9，kappa>=0.5 且 N<10 的方法） ──
# 9 個方法通過篩選：Spearman, Kendall, mRMR, HSIC, Fisher Score,
#                  Gradient Boosting, Permutation Importance, Stability Selection, RFECV (Lasso)
_LLM_FREQ_RANKED = [
    "A_late",                          # #1  count=7
    "amplitude_min",                   # #2  count=5
    "amplitude_norm_median",           # #3  count=5
    "V_open_norm_std",                 # #4  count=5
    "amplitude_mean",                  # #5  count=4
    "finger_mvmnt_thumb_x_max",        # #6  count=4
    "period_quartile_range",           # #7  count=4
    "periodEntropy",                   # #8  count=2
    "amplitude_median",                # #9  count=2
    "periodVarianceNorm",              # #10 count=2
]

_LLM_FREQ_SUBSETS = {n: _LLM_FREQ_RANKED[:n] for n in range(1, 11)}

# ── Few-shot 範例 sessions（取自 scored_by_ChatGPT_promptC_template.csv 的 example 欄）──
# 每個 score 一個代表 session，特徵值會依 selected_features 即時計算
_FEWSHOT_EXAMPLES = [
    (0, "553661824_20200629_Hands_L_7-_R.mp4FT.csv"),
    (1, "445817794_20200720_Hands_R_12-_L.mp4FT.csv"),
    (2, "879885247_20200608_Hands_L_11_R.mp4FT.csv"),
    (3, "546503158_20200310_Hands_L_9_R.mp4FT.csv"),
]

EXPERIMENTS = {
    # ── v2 fewshot ds7b：頻率排名 1→10 個特徵 ──────────────────────
    **{
        f"fewshot_v2_ds7b_freq{n}": {
            "features": _FREQ_SUBSETS[n],
            "output_dir": f"./results/fewshot_v2_ds7b_freq{n}",
            "prompt": "./prompts/Prompt_fewshot_v2_dynamic.txt",
            "prompt_format": "v2",
            "model": "deepseek-r1:7b",
            "fewshot_examples": _FEWSHOT_EXAMPLES,
        }
        for n in range(1, 11)
    },
    # ── v1 fewshot ds32b no_amplitude：頻率排名 1→10 個特徵 ──────────
    **{
        f"fewshot_v1_ds32b_freq{n}": {
            "features": _FREQ_SUBSETS_NO_AMP[n],
            "output_dir": f"./results/fewshot_v1_ds32b_freq{n}",
            "prompt": "./prompts/Prompt_fewshot_v1_dynamic.txt",
            "prompt_format": "v1",
            "model": "deepseek-r1:32b",
            "fewshot_examples": _FEWSHOT_EXAMPLES,
        }
        for n in range(1, 11)
    },
    # ── v1 fewshot ds32b amplitude：頻率排名 1→13 個特徵 ──────────
    **{
        f"fewshot_v1_ds32b_amp_freq{n}": {
            "features": _FREQ_SUBSETS[n],
            "output_dir": f"./results/fewshot_v1_ds32b_amp_freq{n}",
            "prompt": "./prompts/Prompt_fewshot_v1_dynamic.txt",
            "prompt_format": "v1",
            "model": "deepseek-r1:32b",
            "fewshot_examples": _FEWSHOT_EXAMPLES,
        }
        for n in range(1, 14)
    },
    # ── v2 fewshot ds32b amplitude：頻率排名 1→13 個特徵 ──────────
    **{
        f"fewshot_v2_ds32b_amp_freq{n}": {
            "features": _FREQ_SUBSETS[n],
            "output_dir": f"./results/fewshot_v2_ds32b_amp_freq{n}",
            "prompt": "./prompts/Prompt_fewshot_v2_dynamic.txt",
            "prompt_format": "v2",
            "model": "deepseek-r1:32b",
            "fewshot_examples": _FEWSHOT_EXAMPLES,
        }
        for n in range(1, 14)
    },
    # ── v1 fewshot ds32b：LLM-based FS 頻率排名 1→10 個特徵 ──────────
    **{
        f"fewshot_v1_ds32b_llmfreq{n}": {
            "features": _LLM_FREQ_SUBSETS[n],
            "output_dir": f"./results/fewshot_v1_ds32b_llmfreq{n}",
            "prompt": "./prompts/Prompt_fewshot_v1_dynamic.txt",
            "prompt_format": "v1",
            "model": "deepseek-r1:32b",
            "fewshot_examples": _FEWSHOT_EXAMPLES,
        }
        for n in range(1, 11)
    },
}

# ===========================
# Logging Configuration
# ===========================

def setup_logging(output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_score_{timestamp}.log"
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_file, encoding='utf-8')]
    )
    logging.info(f"Log file: {log_file}")
    return log_file

# ===========================
# LLM Call
# ===========================

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json()
        logging.info(f"Ollama connection OK. Available models: {len(models.get('models', []))}")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Ollama: {e}")
        return False

def should_disable_thinking(model):
    return model == "qwen3:30b"

def call_ollama(prompt, model=MODEL, retry_count=0):
    disable_thinking = should_disable_thinking(model)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": NUM_PREDICT}
    }
    if disable_thinking:
        payload["think"] = False

    start_time = time.time()
    attempt = retry_count + 1
    logging.info(f"Starting Ollama API call (attempt {attempt}/{MAX_RETRIES}), model: {model}")

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        elapsed_time = time.time() - start_time
        logging.info(f"Response received in {elapsed_time:.2f}s")
        response.raise_for_status()
        result = response.json()

        if 'response' not in result:
            raise ValueError("Invalid response format")

        response_text = result['response']
        if len(response_text) == 0:
            logging.warning("Empty response detected!")
            if retry_count < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                return call_ollama(prompt, model, retry_count + 1)
        return response_text

    except requests.exceptions.Timeout:
        logging.error(f"Request timeout after {time.time() - start_time:.2f}s")
        if retry_count < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
            return call_ollama(prompt, model, retry_count + 1)
        return None
    except (requests.exceptions.ConnectionError, requests.exceptions.HTTPError, Exception) as e:
        logging.error(f"{type(e).__name__}: {e}")
        if retry_count < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
            return call_ollama(prompt, model, retry_count + 1)
        return None

def parse_score(response):
    if response is None:
        return None
    lines = response.strip().split('\n')
    for line in reversed(lines):
        match = re.search(r'\b([0-4])\b', line)
        if match:
            return int(match.group(1))
    return None

# ===========================
# Save Results
# ===========================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_results(all_results, output_dir=OUTPUT_DIR):
    serializable_results = {}
    for csv_name, result in all_results.items():
        serializable_results[csv_name] = {
            'scores': result['scores'],
            'valid_scores': result['valid_scores'],
            'mean_score': float(result['mean_score']) if result['mean_score'] is not None else None,
            'std_score': float(result['std_score']) if result['std_score'] is not None else None,
            'mode_score': result['mode_score'],
            'features': result['features'],
        }
    output_file = Path(output_dir) / "all_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    responses_file = Path(output_dir) / "detailed_responses.txt"
    with open(responses_file, 'w', encoding='utf-8') as f:
        for csv_name, result in all_results.items():
            f.write(f"\n{'='*60}\n{csv_name}\n{'='*60}\n\n")
            for i, (score, response) in enumerate(zip(result['scores'], result['responses']), 1):
                f.write(f"--- Evaluation {i} (Score: {score}) ---\n")
                f.write(response if response else "No response")
                f.write("\n\n")
    return output_file, responses_file

# ===========================
# Fill Scores into Template CSV
# ===========================

def normalize_filename(filename):
    return str(filename).strip().replace('.csv', '') if filename else ""

def fill_scores_to_csv(json_data, csv_path, output_path):
    if not os.path.exists(csv_path):
        logging.error(f"Template CSV not found: {csv_path}")
        return 0, []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0, []

    json_scores = {
        normalize_filename(k): v.get('scores', [])[:10]
        for k, v in json_data.items()
    }

    matched_count = 0
    unmatched_files = []
    for row_idx in range(1, len(rows)):
        row = rows[row_idx]
        if len(row) <= FILENAME_COL or not row[FILENAME_COL]:
            continue
        key = normalize_filename(row[FILENAME_COL])
        if key in json_scores:
            while len(row) < SCORE_END_COL + 1:
                row.append('')
            for i, score in enumerate(json_scores[key]):
                col_idx = SCORE_START_COL + i
                if col_idx <= SCORE_END_COL:
                    row[col_idx] = score
            rows[row_idx] = row
            matched_count += 1
        else:
            unmatched_files.append(row[FILENAME_COL])

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(rows)
    logging.info(f"CSV fill: {matched_count} matched, {len(unmatched_files)} unmatched -> {output_path}")
    return matched_count, unmatched_files

# ===========================
# Single CSV Processing
# ===========================

def process_single_csv(csv_path, config):
    logging.info(f"Processing CSV: {csv_path.name}")
    try:
        prompt, features = generate_prompt_with_features(
            csv_path,
            base_prompt_path=config['prompt'],
            selected_features=config.get('features'),
            prompt_format=config.get('prompt_format', 'v2'),
            fewshot_examples=config.get('fewshot_examples'),
            example_csv_dir=config.get('example_csv_dir', CSV_DIR),
        )
    except Exception as e:
        logging.error(f"Failed to generate prompt: {e}")
        raise

    model = config.get('model', MODEL)
    response = call_ollama(prompt, model=model)
    score = parse_score(response)

    if score is not None:
        logging.info(f"Score: {score}")
    else:
        logging.warning("Score parsing failed")

    return {'score': score, 'response': response, 'features': features}

# ===========================
# Batch Processing Main Flow
# ===========================

def batch_process_all(config):
    output_dir = config.get('output_dir', OUTPUT_DIR)
    model = config.get('model', MODEL)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_files = list(Path(CSV_DIR).glob("*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in {CSV_DIR}")
        print(f"Error: No CSV files found in {CSV_DIR}")
        return

    print(f"Found {len(csv_files)} CSV files")
    print(f"Each file will be evaluated {EVALUATION_TIMES} times\n")

    all_results = {}

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"\n{'='*60}\n[{i}/{len(csv_files)}] Processing: {csv_name}\n{'='*60}")

        scores, responses, features_list = [], [], []

        for eval_num in range(1, EVALUATION_TIMES + 1):
            print(f"  Evaluation {eval_num}/{EVALUATION_TIMES}...", end=' ', flush=True)
            try:
                result = process_single_csv(csv_path, config)
                scores.append(result['score'])
                responses.append(result['response'])
                features_list.append(result['features'])
                print(f"Score: {result['score']}" if result['score'] is not None else "Parse failed", end=' ')
            except Exception as e:
                logging.error(f"Evaluation {eval_num} failed: {e}", exc_info=True)
                print(f"Error: {e}", end=' ')
                scores.append(None)
                responses.append(str(e))
                features_list.append(None)

            valid_scores = [s for s in scores if s is not None]
            all_results[csv_name] = {
                'scores': scores,
                'valid_scores': valid_scores,
                'mean_score': float(np.mean(valid_scores)) if valid_scores else None,
                'std_score': float(np.std(valid_scores)) if valid_scores else None,
                'mode_score': max(set(valid_scores), key=valid_scores.count) if valid_scores else None,
                'responses': responses,
                'features': features_list[0] if features_list else None,
            }
            try:
                save_results(all_results, output_dir=output_dir)
                print("✓", flush=True)
            except Exception as e:
                print(f"⚠ Save failed: {e}", flush=True)

            if eval_num < EVALUATION_TIMES:
                time.sleep(DELAY_BETWEEN_CALLS)

        result = all_results[csv_name]
        print(f"\n  Valid scores: {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        if result['valid_scores']:
            print(f"  Average: {result['mean_score']:.2f} ± {result['std_score']:.2f}  Mode: {result['mode_score']}")

    output_file, responses_file = save_results(all_results, output_dir=output_dir)
    print(f"\n{'='*60}\n✓ Results saved to: {output_file}\n✓ Responses saved to: {responses_file}\n{'='*60}")

    scored_csv = Path(output_dir) / f"scored_by_{model.replace(':', '_')}.csv"
    try:
        with open(Path(output_dir) / "all_results.json", 'r', encoding='utf-8') as f:
            fill_scores_to_csv(json.load(f), TEMPLATE_CSV, str(scored_csv))
        print(f"Scored CSV: {scored_csv}")
    except Exception as e:
        print(f"Warning: Failed to fill scores into CSV: {e}")

    return all_results

# ===========================
# Run Single Experiment
# ===========================

def run_experiment(experiment_name, experiments_dict=None):
    if experiments_dict is None:
        experiments_dict = EXPERIMENTS
    if experiment_name not in experiments_dict:
        raise ValueError(f"Unknown experiment: '{experiment_name}'. Use --list to see available.")

    config = experiments_dict[experiment_name]
    output_dir = config.get('output_dir', OUTPUT_DIR)
    log_file = setup_logging(output_dir=output_dir)

    logging.info(f"Starting experiment: {experiment_name}")
    print(f"\n{'='*60}")
    print(f"[{experiment_name}]")
    print(f"  Model  : {config.get('model', MODEL)}")
    print(f"  Format : {config.get('prompt_format', 'v2')}")
    print(f"  Prompt : {config.get('prompt')}")
    print(f"  Features ({len(config.get('features', []))}): {config.get('features')}")
    print(f"  Output : {output_dir}")
    print(f"  Log    : {log_file}")
    print(f"{'='*60}\n")

    if not Path(config['prompt']).exists():
        raise FileNotFoundError(f"Prompt file not found: {config['prompt']}")
    if not check_ollama_connection():
        raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_URL}")

    try:
        results = batch_process_all(config)
        print(f"\n[{experiment_name}] ✓ Completed")
        return results
    except Exception as e:
        logging.error(f"Experiment '{experiment_name}' failed: {e}")
        print(f"\n[{experiment_name}] ✗ Failed: {e}")
        raise

# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified LLM-LASSO Scoring System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--experiment", "-e", nargs="+", metavar="NAME",
                        help="Experiment name(s) to run (sequential)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run all experiments sequentially")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available experiments")
    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for name, cfg in EXPERIMENTS.items():
            print(f"  {name}")
            print(f"    model  : {cfg.get('model')}")
            print(f"    format : {cfg.get('prompt_format')}")
            print(f"    output : {cfg.get('output_dir')}")
        exit(0)

    if args.all:
        to_run = list(EXPERIMENTS.keys())
    elif args.experiment:
        to_run = args.experiment
    else:
        to_run = [list(EXPERIMENTS.keys())[0]]
        print(f"No experiment specified. Running default: {to_run[0]}")
        print("Tip: use --list to see available experiments\n")

    for name in to_run:
        if name not in EXPERIMENTS:
            print(f"Error: Unknown experiment '{name}'. Use --list to see available.")
            exit(1)

    print("="*60)
    print("Unified LLM-LASSO Scoring System")
    print("="*60)
    print(f"Experiments to run ({len(to_run)}): {', '.join(to_run)}")
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Evaluations per file: {EVALUATION_TIMES}")
    print("="*60 + "\n")

    for exp_name in to_run:
        try:
            run_experiment(exp_name)
        except Exception as e:
            print(f"\n[{exp_name}] Failed: {e}")
            continue

    print(f"\n{'='*60}\nAll done. Ran {len(to_run)} experiment(s).\n{'='*60}")
