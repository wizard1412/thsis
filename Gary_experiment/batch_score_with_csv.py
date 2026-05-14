#!/usr/bin/env python3
"""
Complete Batch Scoring System with CSV Data
Functions: 
1. Extract CSV features
2. Generate Prompt with CSV data and features
3. Call Ollama for scoring
4. Save results
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import requests
import json
import re
import time

# ===========================
# Configuration
# ===========================

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:70b"  # Your model

# Directory settings
CSV_DIR = "./csv_files"      # Place your CSV files here
OUTPUT_DIR = "./results"     # Output results directory
BASE_PROMPT_PATH = "Prompt3.txt"  # Base Prompt

# Evaluation settings
EVALUATION_TIMES = 10  # Number of evaluations per CSV

# Delay between calls (seconds)
DELAY_BETWEEN_CALLS = 20

# CSV data settings
MAX_CSV_ROWS = None  # Maximum CSV rows to include in prompt (None for all rows)

# ===========================
# Feature Extraction
# ===========================

def extract_features_from_csv(csv_path):
    """
    Extract features using data_feature.py logic
    """
    data = pd.read_csv(csv_path)
    
    # Calculate distance
    data['Distance'] = np.sqrt(
        (data['X_thumb'] - data['X_index'])**2 + 
        (data['Y_thumb'] - data['Y_index'])**2
    )
    data['Distance_Norm'] = (data['Distance'] - np.min(data['Distance'])) / \
                            (np.max(data['Distance']) - np.min(data['Distance']))
    
    # Detect peaks
    peaks, _ = find_peaks(
        data['Distance_Norm'], 
        height=0.2,
        distance=50,
        prominence=0.15,
        width=1,
        plateau_size=[1, 40]
    )
    
    # Calculate three basic features
    num_taps = len(peaks)
    tap_periods = np.diff(data['Frame'].iloc[peaks])
    tap_amplitudes = data['Distance'].iloc[peaks]
    
    return num_taps, tap_periods, tap_amplitudes

# ===========================
# CSV Data Formatting
# ===========================

def format_csv_data(csv_path, max_rows=None):
    """
    Format CSV data for inclusion in prompt
    Args:
        csv_path: Path to CSV file
        max_rows: Maximum number of rows to include (None for all rows)
    Returns:
        Formatted string with CSV data
    """
    data = pd.read_csv(csv_path)
    
    # Limit rows if specified
    if max_rows is not None and len(data) > max_rows:
        data_to_show = data.head(max_rows)
        truncated = True
    else:
        data_to_show = data
        truncated = False
    
    # Format as string
    csv_text = "\n\n## RAW CSV DATA\n\n"
    csv_text += data_to_show.to_string(index=False)
    
    if truncated:
        csv_text += f"\n\n(Showing first {max_rows} rows out of {len(data)} total rows)"
    
    return csv_text

# ===========================
# Prompt Generation
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path=BASE_PROMPT_PATH, max_csv_rows=MAX_CSV_ROWS):
    """
    Generate Prompt with CSV data and extracted features
    Structure: Base Prompt -> CSV Data -> Extracted Features
    """
    # 1. Read base Prompt
    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    
    # 2. Format CSV data
    csv_data_text = format_csv_data(csv_path, max_rows=max_csv_rows)
    
    # 3. Extract features
    num_taps, tap_periods, tap_amplitudes = extract_features_from_csv(csv_path)
    
    # 4. Prepare feature dictionary
    features = {
        'num_taps': int(num_taps),
        'tap_periods': tap_periods.tolist(),
        'tap_amplitudes': tap_amplitudes.tolist()
    }
    
    # 5. Format extracted features
    feature_text = "\n\n## EXTRACTED FEATURES\n\n"
    feature_text += f"Number of taps: {features['num_taps']}\n\n"
    feature_text += f"Tap periods (frames): {features['tap_periods']}\n\n"
    feature_text += f"Tap amplitudes (pixels): {features['tap_amplitudes']}\n"
    
    # 6. Combine: Base Prompt -> CSV Data -> Extracted Features
    enhanced_prompt = base_prompt + csv_data_text + feature_text
    
    return enhanced_prompt, features

# ===========================
# LLM Call
# ===========================

def call_ollama(prompt, model=MODEL):
    """
    Call Ollama API
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result['response']
    except Exception as e:
        print(f"    Ollama call failed: {e}")
        return None

def parse_score(response):
    """
    Extract score from LLM response
    Look for 0-4 score in the last line
    """
    if response is None:
        return None
    
    # Find number in last line
    lines = response.strip().split('\n')
    for line in reversed(lines):
        # Search for 0-4 number
        match = re.search(r'\b([0-4])\b', line)
        if match:
            return int(match.group(1))
    
    return None

# ===========================
# Save Results
# ===========================

def save_results(all_results, output_dir=OUTPUT_DIR):
    """
    Save results to files
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
# Single CSV Processing
# ===========================

def process_single_csv(csv_path, evaluation_num):
    """
    Process single CSV for one evaluation
    """
    # 1. Generate prompt with CSV data and features
    prompt, features = generate_prompt_with_features(csv_path)
    
    # 2. Call Ollama
    response = call_ollama(prompt)
    
    # 3. Parse score
    score = parse_score(response)
    
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
    Save immediately after each evaluation to prevent data loss on interruption
    """
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = list(Path(CSV_DIR).glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"Error: No CSV files found in {CSV_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Each file will be evaluated {EVALUATION_TIMES} times\n")
    
    all_results = {}
    
    # Process each CSV
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"\n{'='*60}")
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")
        print(f"{'='*60}")
        
        scores = []
        responses = []
        features_list = []
        
        # Evaluate each CSV multiple times
        for eval_num in range(1, EVALUATION_TIMES + 1):
            print(f"  Evaluation {eval_num}/{EVALUATION_TIMES}...", end=' ', flush=True)
            
            try:
                result = process_single_csv(csv_path, eval_num)
                
                scores.append(result['score'])
                responses.append(result['response'])
                features_list.append(result['features'])
                
                if result['score'] is not None:
                    print(f"Score: {result['score']}", end=' ')
                else:
                    print("Score parsing failed", end=' ')
                
            except Exception as e:
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
            
            # Save immediately to file
            try:
                save_results(all_results)
                print(f"✓", flush=True)
            except Exception as e:
                print(f"⚠Save failed: {e}", flush=True)
            
            # Delay to avoid overload
            if eval_num < EVALUATION_TIMES:
                time.sleep(DELAY_BETWEEN_CALLS)
        
        # Display summary
        result = all_results[csv_name]
        print(f"\n  Complete! Statistics:")
        print(f"    Valid scores: {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        if result['valid_scores']:
            print(f"    Mean score: {result['mean_score']:.2f} ± {result['std_score']:.2f}")
            print(f"    Mode score: {result['mode_score']}")
            print(f"    All scores: {result['valid_scores']}")
    
    # Final save
    output_file, responses_file = save_results(all_results)
    
    print(f"\n{'='*60}")
    print(f"✓ Complete! Results saved to: {output_file}")
    print(f"✓ Detailed responses saved to: {responses_file}")
    print(f"{'='*60}")
    
    # Display summary table
    print(f"\n{'='*60}")
    print("Scoring Summary")
    print(f"{'='*60}")
    print(f"{'File Name':<40} {'Mean Score':<12} {'Mode':<8} {'Valid/Total'}")
    print(f"{'-'*60}")
    
    for csv_name, result in all_results.items():
        name_short = csv_name[:37] + "..." if len(csv_name) > 40 else csv_name
        if result['mean_score'] is not None:
            print(f"{name_short:<40} {result['mean_score']:>6.2f}±{result['std_score']:.2f}   "
                  f"{result['mode_score']:>4}     {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        else:
            print(f"{name_short:<40} {'N/A':<12} {'N/A':<8} {len(result['valid_scores'])}/{EVALUATION_TIMES}")
    
    return all_results

# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    print("="*60)
    print("Finger Tapping Automated Scoring System with CSV Data")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Evaluations per file: {EVALUATION_TIMES}")
    print(f"Max CSV rows in prompt: {MAX_CSV_ROWS if MAX_CSV_ROWS else 'All'}")
    print("="*60 + "\n")
    
    # Check if base Prompt exists
    if not Path(BASE_PROMPT_PATH).exists():
        print(f"Error: Cannot find {BASE_PROMPT_PATH}")
        print("Please ensure Prompt3.txt is in the current directory")
        exit(1)
    
    # Execute batch processing
    batch_process_all()
