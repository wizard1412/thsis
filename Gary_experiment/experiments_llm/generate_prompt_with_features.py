#!/usr/bin/env python3
"""
Unified Feature Extraction and Prompt Generation Script
Supports two prompt formats:
  v1: Flat feature list under "TARGET KINEMATIC FEATURES TO SCORE"
  v2: "AMPLITUDE PATTERN" (tap_amplitudes_normalized) + "STATISTICAL FEATURES"
"""

import numpy as np
from pathlib import Path

from run_feature_extraction_with_ratings_v2 import extract_full_features_v2 as extract_full_features

# ===========================
# Constants
# ===========================

# LLM-LASSO (deepseek70b, η=3.0) 選出的 6 個特徵
LLMLASSO_SELECTED_FEATURES = [
    "finger_mvmnt_thumb_x_mean",
    "finger_mvmnt_thumb_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks",
]

# LLM-LASSO (amplitude, η=2.0) 選出的 8 個特徵
LLMLASSO_AMPLITUDE_FEATURES = [
    "finger_mvmnt_thumb_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks",
    "amplitude_quartile_range",
    "amplitude_min",
    "amplitude_entropy",
]

# ===========================
# Feature descriptions (only for non-self-explanatory names)
# ===========================

FEATURE_DESCRIPTIONS = {
    # Amplitude decrement (teacher's formulas, opaque names)
    "A_early": "mean amplitude of the first 3 taps",
    "A_late": "mean amplitude of the last 3 taps",
    "R_A_3_3": "ratio A_late / A_early (1=no decrement, <1=decrement)",
    "D_A_pct": "relative drop from A_early to A_late (0=none, 1=full collapse)",
    "D_A_max": "max relative amplitude drop across the sequence",
    "k_collapse": "tap index where amplitude first drops below 0.5*A_early (-1 if none)",
    "tilde_A_median": "median of amplitudes normalized by A_early",
    "beta_tilde_A": "linear slope of normalized amplitudes over tap index",
    "CV_tilde_A": "coefficient of variation of normalized amplitudes",
    "amplitude_MAD": "median absolute deviation of tap amplitudes",
    "amplitude_decrement_fitness_r2": "R^2 of linear fit of amplitude vs tap index",
    "amplitude_decrement_slope": "slope of amplitude decrement fit",
    "amplitude_decrement_end_to_mean": "ratio last-tap amplitude / mean amplitude",
    "amplitude_decrement_last_to_first_half": "mean(last half) / mean(first half) of amplitudes",
    "amplitude_decrement_fit_min_degree": "min polynomial degree for amplitude decrement fit",

    # ITI / rhythm
    "I_ITI": "inter-tap-interval irregularity index",
    "I_ITI_norm": "ITI irregularity normalized by mean ITI",
    "ITI_MAD": "median absolute deviation of inter-tap intervals",

    # Halt / freeze / completion
    "S_halt": "total halt duration during the tapping (seconds)",
    "F_halt": "halt frequency (events per second)",
    "N_valid": "number of valid taps",
    "C_10": "1 if completed 10 taps, else 0",
    "T_10": "time to complete 10 taps (seconds)",
    "F_incomplete": "fraction of expected taps missing",
    "num_interruptions_norm": "interruption events per second",
    "num_freeze_norm": "freeze events per second",

    # Opening / closing phase
    "V_open_median": "median opening-phase peak velocity in pixels/s (finger lifting)",
    "V_open_mean": "mean opening-phase peak velocity in pixels/s",
    "V_open_std": "std of opening-phase peak velocity across taps",
    "V_close_median": "median closing-phase peak velocity in pixels/s (finger tapping down)",
    "V_close_mean": "mean closing-phase peak velocity in pixels/s",
    "V_close_std": "std of closing-phase peak velocity across taps",
    "V_open_norm_median": "median opening-phase velocity normalized by distance range",
    "V_open_norm_mean": "mean opening-phase velocity normalized by distance range",
    "V_open_norm_std": "std of normalized opening-phase velocity",
    "V_close_norm_median": "median closing-phase velocity normalized by distance range",
    "V_close_norm_mean": "mean closing-phase velocity normalized by distance range",
    "V_close_norm_std": "std of normalized closing-phase velocity",
    "T_open_median": "median opening-phase duration (seconds)",
    "T_open_mean": "mean opening-phase duration",
    "T_open_std": "std of opening-phase duration",
    "T_close_median": "median closing-phase duration",
    "T_close_mean": "mean closing-phase duration",
    "T_close_std": "std of closing-phase duration",
    "R_OC_median": "median ratio opening/closing duration",
    "R_OC_mean": "mean ratio opening/closing duration",
    "R_OC_std": "std of ratio opening/closing duration",
    "R_V_open": "opening-velocity decrement ratio (late/early taps)",
    "R_V_close": "closing-velocity decrement ratio (late/early taps)",

    # Period / frequency (shape metrics)
    "aperiodicity": "aperiodicity score of tapping rhythm",
    "periodEntropy": "entropy of inter-tap period distribution",
    "periodVarianceNorm": "variance of periods normalized by mean period",
    "frequency_lr_fitness_r2": "R^2 of linear fit of frequency vs tap index",
    "frequency_lr_slope": "slope of tap frequency over time (negative = slowing)",
    "frequency_fit_min_degree": "min polynomial degree for frequency fit",

    # Finger movement (thumb / index)
    "finger_mvmnt_thumb_x_median": "median frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_x_mean": "mean frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_x_max": "max frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_x_stdev": "std of frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_dist_stdev": "std of frame-to-frame thumb trajectory distance",
    "finger_mvmnt_thumb_dist_quartile_range": "IQR of frame-to-frame thumb trajectory distance",
    "finger_mvmnt_index_x_median": "median frame-to-frame index finger x displacement",
    "finger_mvmnt_index_x_mean": "mean frame-to-frame index finger x displacement",
    "finger_mvmnt_index_x_max": "max frame-to-frame index finger x displacement",
    "finger_mvmnt_index_x_stdev": "std of frame-to-frame index finger x displacement",
    "finger_mvmnt_index_dist_stdev": "std of frame-to-frame index finger trajectory distance",
    "finger_mvmnt_index_dist_quartile_range": "IQR of frame-to-frame index finger trajectory distance",

    # Normalized amplitude
    "amplitude_norm_median": "median tap amplitude normalized by subject's full distance range",
    "amplitude_norm_mean": "mean normalized tap amplitude",
    "amplitude_norm_min": "min normalized tap amplitude",
    "amplitude_norm_max": "max normalized tap amplitude",
    "amplitude_norm_stdev": "std of normalized tap amplitudes",
    "amplitude_norm_quartile_range": "IQR of normalized tap amplitudes",

    # Speed / acceleration (normalized)
    "speed_norm_median": "median tapping speed in Norm_Distance/s",
    "speed_norm_mean": "mean tapping speed in Norm_Distance/s",
    "speed_norm_max": "max tapping speed in Norm_Distance/s",
    "speed_norm_stdev": "std of tapping speed in Norm_Distance/s",
    "acceleration_norm_median": "median tapping acceleration in Norm_Distance/s²",
    "acceleration_norm_mean": "mean tapping acceleration in Norm_Distance/s²",
}


def _fmt_line(feat, val, indent="  "):
    """Format one feature line, appending inline description if the name is non-obvious."""
    desc = FEATURE_DESCRIPTIONS.get(feat)
    if desc:
        return f"{indent}{feat}: {val}  # {desc}\n"
    return f"{indent}{feat}: {val}\n"


# ===========================
# Prompt Generation
# ===========================

def _render_v2_feature_block(full_features, selected_features, header_label):
    """Render an AMPLITUDE PATTERN + STATISTICAL FEATURES block for v2 format."""
    tap_amplitudes = full_features.get('tap_amplitudes', [])
    if tap_amplitudes and max(tap_amplitudes) > 0:
        max_amp = max(tap_amplitudes)
        tap_amplitudes_normalized = [round(a / max_amp, 2) for a in tap_amplitudes]
    else:
        tap_amplitudes_normalized = []

    text = "\n" + "=" * 35 + "\n"
    text += f"{header_label}\n"
    text += "=" * 35 + "\n"
    text += "AMPLITUDE PATTERN:\n"
    text += f"  tap_amplitudes_normalized (ratio to max): {tap_amplitudes_normalized}\n"
    text += "\nSTATISTICAL FEATURES:\n"
    for feat in selected_features:
        if feat in full_features:
            val = full_features[feat]
            if isinstance(val, float) and not np.isnan(val):
                val = round(val, 4)
            text += _fmt_line(feat, val)
    return text


_EXAMPLE_FEATURE_CACHE = {}


def _render_fewshot_examples(example_specs, csv_dir, selected_features, prompt_format):
    """Render few-shot example blocks from (score, csv_filename) tuples."""
    parts = []
    for score, csv_name in example_specs:
        csv_path = Path(csv_dir) / csv_name
        key = str(csv_path)
        if key not in _EXAMPLE_FEATURE_CACHE:
            feats = extract_full_features(csv_path)
            feats['num_taps'] = feats.get('num_peaks', 0)
            _EXAMPLE_FEATURE_CACHE[key] = feats
        ex_features = _EXAMPLE_FEATURE_CACHE[key]
        parts.append("\n───────────────\n")
        parts.append(f"EXAMPLE FOR SCORE {score}\n")
        parts.append("───────────────\n")
        if prompt_format == "v2":
            tap_amplitudes = ex_features.get('tap_amplitudes', [])
            if tap_amplitudes and max(tap_amplitudes) > 0:
                max_amp = max(tap_amplitudes)
                tap_norm = [round(a / max_amp, 2) for a in tap_amplitudes]
            else:
                tap_norm = []
            parts.append("AMPLITUDE PATTERN:\n")
            parts.append(f"  tap_amplitudes_normalized (ratio to max): {tap_norm}\n")
            parts.append("\nSTATISTICAL FEATURES:\n")
            for feat in selected_features:
                if feat in ex_features:
                    val = ex_features[feat]
                    if isinstance(val, float) and not np.isnan(val):
                        val = round(val, 4)
                    parts.append(_fmt_line(feat, val))
        else:
            for feat in selected_features:
                if feat in ex_features:
                    val = ex_features[feat]
                    if isinstance(val, float) and not np.isnan(val):
                        val = round(val, 4)
                    parts.append(_fmt_line(feat, val, indent=""))
        parts.append(f"\nThis example was rated as SCORE {score}.\n")
    return "".join(parts)


def generate_prompt_with_features(csv_path, base_prompt_path, selected_features=None,
                                   prompt_format="v2", fewshot_examples=None,
                                   example_csv_dir=None):
    """
    Generate prompt with dynamically selected features.

    Args:
        csv_path: Path to CSV file
        base_prompt_path: Path to base prompt file
        selected_features: List of feature names to include
        prompt_format: "v1" (flat list) or "v2" (AMPLITUDE PATTERN + STATISTICAL FEATURES)

    Returns:
        prompt: Complete prompt string
        features: Full feature dictionary
    """
    if selected_features is None:
        selected_features = LLMLASSO_SELECTED_FEATURES

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    full_features = extract_full_features(csv_path)
    full_features['num_taps'] = full_features.get('num_peaks', 0)

    examples_text = ""
    if fewshot_examples:
        ex_dir = example_csv_dir if example_csv_dir is not None else Path(csv_path).parent
        examples_text = _render_fewshot_examples(fewshot_examples, ex_dir,
                                                  selected_features, prompt_format)

    if prompt_format == "v2":
        feature_text = _render_v2_feature_block(full_features, selected_features,
                                                 header_label="SUBJECT TO SCORE")
    else:  # v1
        feature_text = "\n" + "="*35 + "\n"
        feature_text += "TARGET KINEMATIC FEATURES TO SCORE\n"
        feature_text += "="*35 + "\n"
        for feat in selected_features:
            if feat in full_features:
                val = full_features[feat]
                if isinstance(val, float) and not np.isnan(val):
                    val = round(val, 4)
                feature_text += _fmt_line(feat, val, indent="")

    return base_prompt + examples_text + feature_text, full_features


# ===========================
# Batch Prompt Generation
# ===========================

def batch_generate_prompts(csv_dir="../csv_files", output_dir="./generated_prompts",
                           base_prompt_path=None, selected_features=None,
                           prompt_format="v2"):
    """
    Batch generate prompts for all CSV files.

    Args:
        csv_dir: Directory containing CSV files
        output_dir: Directory for output prompt files
        base_prompt_path: Path to base prompt file
        selected_features: List of feature names to include
        prompt_format: "v1" or "v2"
    """
    if base_prompt_path is None:
        base_prompt_path = "./prompts/Prompt_fewshot_v2_amplitude.txt"
    if selected_features is None:
        selected_features = LLMLASSO_AMPLITUDE_FEATURES

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    results = {}
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}")
        try:
            prompt, features = generate_prompt_with_features(
                csv_path, base_prompt_path,
                selected_features=selected_features,
                prompt_format=prompt_format,
            )
            output_path = Path(output_dir) / f"{csv_name}_prompt.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(prompt)
            results[csv_name] = {
                'success': True,
                'features': {k: features.get(k) for k in selected_features},
                'prompt_path': str(output_path),
            }
            print(f"  ✓ Prompt saved: {output_path}\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            results[csv_name] = {'success': False, 'error': str(e)}

    print(f"\n✓ Complete! All prompts generated")
    return results


# ===========================
# Execution
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch generate prompts")
    parser.add_argument("--format", choices=["v1", "v2"], default="v2",
                        help="Prompt format: v1 (flat) or v2 (amplitude+statistical)")
    parser.add_argument("--prompt", default=None, help="Base prompt file path")
    parser.add_argument("--features", choices=["llmlasso", "amplitude"], default="amplitude",
                        help="Feature set to use")
    parser.add_argument("--output", default="./generated_prompts", help="Output directory")
    args = parser.parse_args()

    feat_set = LLMLASSO_AMPLITUDE_FEATURES if args.features == "amplitude" else LLMLASSO_SELECTED_FEATURES
    batch_generate_prompts(
        output_dir=args.output,
        base_prompt_path=args.prompt,
        selected_features=feat_set,
        prompt_format=args.format,
    )
