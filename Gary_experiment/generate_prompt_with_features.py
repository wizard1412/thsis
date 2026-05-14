#!/usr/bin/env python3
"""
Feature Extraction and Prompt Generation Script (No Raw Data)
Function: Extract features and append ONLY features to Prompt (skips raw CSV data)
Output: Original Prompt.txt + Extracted Features
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path

# ===========================
# Feature Extractio
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
# Prompt Generation
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path="Prompt.txt"):
    """
    Generate Prompt with ONLY extracted features (No raw CSV data)
    Structure: Base Prompt -> Extracted Features
    
    Args:
        csv_path: Path to CSV file
        base_prompt_path: Path to base Prompt file
    
    Returns:
        prompt: Complete prompt with features
        features: Feature dictionary (for program use)
    """
    
    # 1. Read base Prompt (keep original)
    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()
    
    # 2. Extract features
    num_taps, tap_periods, tap_amplitudes = extract_features_from_csv(csv_path)
    
    # 3. Prepare feature dictionary
    features = {
        'num_taps': int(num_taps),
        'tap_periods': tap_periods.tolist(),
        'tap_amplitudes': tap_amplitudes.tolist()
    }
    
    # 4. Format extracted features
    feature_text = f"Number of taps: {features['num_taps']}\n"
    feature_text += f"Tap periods (frames): {features['tap_periods']}\n"
    feature_text += f"Tap amplitudes (pixels): {features['tap_amplitudes']}\n"
    
    # 5. Combine: Base Prompt -> Extracted Features
    # Note: csv_data_text has been removed
    enhanced_prompt = base_prompt.rstrip() + "\n" + feature_text
    
    # 6. Return complete prompt and feature dictionary
    return enhanced_prompt, features

# ===========================
# Batch Processing
# ===========================

def batch_generate_prompts(csv_dir="./csv_files", output_dir="./prompts", 
                          base_prompt_path="Prompt.txt"):
    """
    Batch process all CSV files
    - Prompt file: Original Prompt3.txt + Extracted Features
    
    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory for output prompt files
        base_prompt_path: Path to base prompt file
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")
    
    results = {}
    
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")
        
        try:
            # Generate Prompt with ONLY extracted features
            prompt, features = generate_prompt_with_features(
                csv_path, base_prompt_path
            )
            
            # Save Prompt
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
    
    print(f"\n✓ Complete! All prompts generated (without raw CSV data)")
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
        
        # Save prompt
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
        print("Feature Extraction System (Features Only, No Raw Data)")
        print("="*60 + "\n")
        batch_generate_prompts()