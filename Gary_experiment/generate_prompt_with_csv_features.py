#!/usr/bin/env python3
"""
Feature Extraction and Prompt Generation Script with CSV Data
Function: Extract features and append to Prompt with CSV data in between
Output: Original Prompt3.txt + CSV Data + Extracted Features
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

# ===========================
# Feature Extraction
# ===========================

def extract_features_from_csv(csv_path):
    """
    Extract features using data_feature.py logic
    Returns: num_taps, tap_periods, tap_amplitudes
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

def generate_prompt_with_features(csv_path, base_prompt_path="Prompt3.txt", max_csv_rows=None):
    """
    Generate Prompt with CSV data and extracted features
    Structure: Base Prompt -> CSV Data -> Extracted Features
    
    Args:
        csv_path: Path to CSV file
        base_prompt_path: Path to base Prompt file
        max_csv_rows: Maximum number of CSV rows to include (None for all)
    
    Returns:
        prompt: Complete prompt with CSV data and features
        features: Feature dictionary (for program use)
    """
    
    # 1. Read base Prompt (keep original)
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
    
    # 7. Return complete prompt and feature dictionary
    return enhanced_prompt, features

# ===========================
# Batch Processing
# ===========================

def batch_generate_prompts(csv_dir="./csv_files", output_dir="./prompts", 
                          base_prompt_path="Prompt3.txt", max_csv_rows=None):
    """
    Batch process all CSV files
    - Prompt file: Original Prompt3.txt + CSV Data + Extracted Features
    - No separate features_summary.txt is saved
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory for output prompt files
        base_prompt_path: Path to base prompt file
        max_csv_rows: Maximum CSV rows to include (None for all)
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")
    
    results = {}
    
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")
        
        try:
            # Generate Prompt with CSV data and extracted features
            prompt, features = generate_prompt_with_features(
                csv_path, base_prompt_path, max_csv_rows
            )
            
            # Save Prompt (with CSV data and features)
            output_path = Path(output_dir) / f"{csv_name}_prompt.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            
            results[csv_name] = {
                'success': True,
                'features': features,
                'prompt_path': str(output_path)
            }
            
            print(f"  ✓ Features: {features['num_taps']} taps")
            print(f"  ✓ Prompt saved: {output_path}\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results[csv_name] = {
                'success': False,
                'error': str(e)
            }
    
    print(f"\n✓ Complete! All prompts generated")
    return results

# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Single file mode
        csv_file = sys.argv[1]
        print(f"Processing single file: {csv_file}\n")
        
        prompt, features = generate_prompt_with_features(csv_file)
        
        # Save prompt (with CSV data and features)
        output_file = Path(csv_file).stem + "_prompt.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # Display features
        print(f"✓ Prompt saved to: {output_file}")
        print(f"✓ Detected {features['num_taps']} taps")
        print(f"✓ Tap periods: {features['tap_periods']}")
        print(f"✓ Tap amplitudes: {features['tap_amplitudes']}")
        
    else:
        # Batch mode
        print("="*60)
        print("Feature Extraction System with CSV Data")
        print("="*60 + "\n")
        batch_generate_prompts()