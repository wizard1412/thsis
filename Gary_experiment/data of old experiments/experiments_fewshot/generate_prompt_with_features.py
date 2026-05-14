#!/usr/bin/env python3
"""
Feature Extraction and Prompt Generation Script
Function: Extract features and append to Prompt
Output: Original Prompt.txt + Raw Data + Statistical Features

Features included:
- Raw data: num_taps, tap_periods, tap_amplitudes
- Statistical features (selected by correlation analysis):
  period_stdev, periodEntropy, finger_mvmnt_x_mean, finger_mvmnt_dist_mean,
  speed_mean, period_max, maxFreezeDuration, aperiodicity
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy
from pathlib import Path

# ===========================
# Constants
# ===========================

FPS = 240  # Frame rate
SPEED_THRESHOLD = 50  # Interruption threshold (degrees/sec)
FREEZE_DURATION = 0.30  # Freeze duration threshold (seconds)

# Prompt file configuration
BASE_PROMPT_FILE = "Prompt_fewshot.txt"  # Change this to "Prompt_fewshot.txt" for few-shot mode

# ===========================
# Helper Functions
# ===========================

def safe_stats(arr):
    """Safely calculate statistics, handle empty arrays"""
    if len(arr) == 0:
        return {
            'median': np.nan,
            'quartile_range': np.nan,
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan,
            'stdev': np.nan
        }
    return {
        'median': np.median(arr),
        'quartile_range': np.percentile(arr, 75) - np.percentile(arr, 25),
        'mean': np.mean(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'stdev': np.std(arr)
    }


def calculate_entropy(arr, bins=10, range_vals=None):
    """Calculate distribution entropy"""
    if len(arr) == 0:
        return np.nan
    hist, _ = np.histogram(arr, bins=bins, range=range_vals, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0
    return entropy(hist)


def denoise_signal(signal, window=11, polyorder=3):
    """Denoise signal using Savitzky-Golay filter"""
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window, polyorder)


# ===========================
# Feature Extraction
# ===========================

def extract_features_from_csv(csv_path, fps=FPS):
    """
    Extract all features from CSV file

    Returns:
        features dict containing:
        - Raw data: num_taps, tap_periods, tap_amplitudes
        - Statistical features: period_stdev, periodEntropy, finger_mvmnt_x_mean,
          finger_mvmnt_dist_mean, speed_mean, period_max, maxFreezeDuration, aperiodicity
    """
    data = pd.read_csv(csv_path)

    # ===========================
    # 1. Basic Distance Calculation
    # ===========================

    data['Distance'] = np.sqrt(
        (data['X_thumb'] - data['X_index'])**2 +
        (data['Y_thumb'] - data['Y_index'])**2
    )

    # Normalize distance
    dist_min = data['Distance'].min()
    dist_max = data['Distance'].max()
    dist_range = dist_max - dist_min if dist_max > dist_min else 1
    data['Distance_Norm'] = (data['Distance'] - dist_min) / dist_range

    # Calculate angle (convert pixel distance to approximate angle)
    max_distance = data['Distance'].max()
    data['Angle'] = (data['Distance'] / max_distance) * 90

    # ===========================
    # 2. Denoise Signal
    # ===========================

    distance_denoised = denoise_signal(data['Distance'].values)
    angle_denoised = denoise_signal(data['Angle'].values)

    # ===========================
    # 3. Speed Calculation (degrees/second)
    # ===========================

    dt = 1.0 / fps
    speed_raw = np.gradient(angle_denoised, dt)
    speed = speed_raw

    # ===========================
    # 4. Peak Detection (Taps)
    # ===========================

    distance_norm = (distance_denoised - distance_denoised.min()) / \
                    (distance_denoised.max() - distance_denoised.min() + 1e-10)

    peaks, _ = find_peaks(
        distance_norm,
        height=0.2,
        distance=int(fps * 0.15),
        prominence=0.15,
        width=1
    )

    num_taps = len(peaks)

    # ===========================
    # 5. Raw Data: Tap Periods and Amplitudes
    # ===========================

    if num_taps > 1:
        tap_periods_frames = np.diff(peaks)
        periods_seconds = tap_periods_frames / fps
    else:
        tap_periods_frames = np.array([])
        periods_seconds = np.array([])

    tap_amplitudes = data['Distance'].iloc[peaks].values if num_taps > 0 else np.array([])

    # ===========================
    # 6. Statistical Features
    # ===========================

    # Period statistics
    period_stats = safe_stats(periods_seconds)
    period_stdev = period_stats['stdev']
    period_max = period_stats['max']

    # Period entropy
    periodEntropy = calculate_entropy(periods_seconds, bins=50, range_vals=(0, 2))

    # Speed statistics
    speed_stats = safe_stats(np.abs(speed))
    speed_mean = speed_stats['mean']

    # Finger movement features
    thumb_mvmnt_x = np.abs(np.diff(data['X_thumb'].values))
    thumb_mvmnt_dist = np.sqrt(
        np.diff(data['X_thumb'].values)**2 +
        np.diff(data['Y_thumb'].values)**2
    )

    finger_mvmnt_x_stats = safe_stats(thumb_mvmnt_x)
    finger_mvmnt_dist_stats = safe_stats(thumb_mvmnt_dist)

    finger_mvmnt_x_mean = finger_mvmnt_x_stats['mean']
    finger_mvmnt_dist_mean = finger_mvmnt_dist_stats['mean']

    # Aperiodicity (FFT power spectrum entropy)
    if len(distance_denoised) > 10:
        fft_result = np.fft.fft(distance_denoised)
        power_spectrum = np.abs(fft_result) ** 2
        power_spectrum_norm = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else power_spectrum
        aperiodicity = entropy(power_spectrum_norm[power_spectrum_norm > 0])
    else:
        aperiodicity = np.nan

    # Max freeze duration
    abs_speed = np.abs(speed)
    maxFreezeDuration = 0
    current_freeze_len = 0

    for i in range(len(abs_speed)):
        if abs_speed[i] <= SPEED_THRESHOLD:
            current_freeze_len += 1
            maxFreezeDuration = max(maxFreezeDuration, current_freeze_len)
        else:
            current_freeze_len = 0

    maxFreezeDuration = maxFreezeDuration / fps  # Convert to seconds

    # ===========================
    # 7. Compile Features
    # ===========================

    features = {
        # Raw data
        'num_taps': int(num_taps),
        'tap_periods': tap_periods_frames.tolist(),
        'tap_amplitudes': [round(x, 2) for x in tap_amplitudes.tolist()],

        # Statistical features (selected by correlation analysis)
        'period_stdev': round(period_stdev, 4) if not np.isnan(period_stdev) else None,
        'periodEntropy': round(periodEntropy, 4) if not np.isnan(periodEntropy) else None,
        'finger_mvmnt_x_mean': round(finger_mvmnt_x_mean, 4) if not np.isnan(finger_mvmnt_x_mean) else None,
        'finger_mvmnt_dist_mean': round(finger_mvmnt_dist_mean, 4) if not np.isnan(finger_mvmnt_dist_mean) else None,
        'speed_mean': round(speed_mean, 4) if not np.isnan(speed_mean) else None,
        'period_max': round(period_max, 4) if not np.isnan(period_max) else None,
        'maxFreezeDuration': round(maxFreezeDuration, 4),
        'aperiodicity': round(aperiodicity, 4) if not np.isnan(aperiodicity) else None,
    }

    return features


# ===========================
# Prompt Generation
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path=None):
    """
    Generate Prompt with extracted features
    Structure: Base Prompt -> Raw Data -> Statistical Features

    Args:
        csv_path: Path to CSV file
        base_prompt_path: Path to base Prompt file (default: use BASE_PROMPT_FILE constant)

    Returns:
        prompt: Complete prompt with features
        features: Feature dictionary (for program use)
    """

    # 1. Read base Prompt
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    # 2. Extract features
    features = extract_features_from_csv(csv_path)

    # 3. Format raw data section
    feature_text = "# RAW DATA\n"
    feature_text += f"Number of taps: {features['num_taps']}\n"
    feature_text += f"Tap periods (frames): {features['tap_periods']}\n"
    feature_text += f"Tap amplitudes (pixels): {features['tap_amplitudes']}\n"

    # 4. Format statistical features section
    feature_text += "\n# STATISTICAL FEATURES\n"
    feature_text += f"period_stdev (rhythm stability): {features['period_stdev']}\n"
    feature_text += f"periodEntropy (rhythm complexity): {features['periodEntropy']}\n"
    feature_text += f"finger_mvmnt_x_mean (movement amplitude): {features['finger_mvmnt_x_mean']}\n"
    feature_text += f"finger_mvmnt_dist_mean (finger distance): {features['finger_mvmnt_dist_mean']}\n"
    feature_text += f"speed_mean (average speed): {features['speed_mean']}\n"
    feature_text += f"period_max (longest interval): {features['period_max']}\n"
    feature_text += f"maxFreezeDuration (longest freeze in seconds): {features['maxFreezeDuration']}\n"
    feature_text += f"aperiodicity (rhythm irregularity): {features['aperiodicity']}\n"

    # 5. Combine: Base Prompt -> Features
    enhanced_prompt = base_prompt + feature_text

    return enhanced_prompt, features


# ===========================
# Batch Processing
# ===========================

def batch_generate_prompts(csv_dir="./csv_files", output_dir="./prompts",
                          base_prompt_path=None):
    """
    Batch process all CSV files

    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory for output prompt files
        base_prompt_path: Path to base prompt file (default: use BASE_PROMPT_FILE constant)
    """
    if base_prompt_path is None:
        base_prompt_path = BASE_PROMPT_FILE
    Path(output_dir).mkdir(exist_ok=True)

    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    results = {}

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")

        try:
            # Generate Prompt with features
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
        print(f"Processing single file: {csv_file}")
        print(f"Using prompt file: {BASE_PROMPT_FILE}\n")

        prompt, features = generate_prompt_with_features(csv_file)

        # Save prompt
        output_file = Path(csv_file).stem + "_prompt.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt)

        # Display features
        print(f"✓ Prompt saved to: {output_file}")
        print(f"\n--- RAW DATA ---")
        print(f"Number of taps: {features['num_taps']}")
        print(f"Tap periods: {features['tap_periods']}")
        print(f"Tap amplitudes: {features['tap_amplitudes']}")
        print(f"\n--- STATISTICAL FEATURES ---")
        print(f"period_stdev: {features['period_stdev']}")
        print(f"periodEntropy: {features['periodEntropy']}")
        print(f"finger_mvmnt_x_mean: {features['finger_mvmnt_x_mean']}")
        print(f"finger_mvmnt_dist_mean: {features['finger_mvmnt_dist_mean']}")
        print(f"speed_mean: {features['speed_mean']}")
        print(f"period_max: {features['period_max']}")
        print(f"maxFreezeDuration: {features['maxFreezeDuration']}")
        print(f"aperiodicity: {features['aperiodicity']}")

    else:
        # Batch mode
        print("="*60)
        print("Feature Extraction System (Raw Data + Statistical Features)")
        print("="*60 + "\n")
        print(f"Using prompt file: {BASE_PROMPT_FILE}\n")
        batch_generate_prompts()
