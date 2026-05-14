#!/usr/bin/env python3
"""
Unified Feature Extraction and Prompt Generation Script
Supports two prompt formats:
  v1: Flat feature list under "TARGET KINEMATIC FEATURES TO SCORE"
  v2: "AMPLITUDE PATTERN" (tap_amplitudes_normalized) + "STATISTICAL FEATURES"
"""

import numpy as np
from pathlib import Path

from run_feature_extraction_with_ratings import extract_full_features

# ===========================
# Constants
# ===========================

# LLM-LASSO (deepseek70b, η=3.0) 選出的 6 個特徵
LLMLASSO_SELECTED_FEATURES = [
    "finger_mvmnt_x_mean",
    "finger_mvmnt_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks",
]

# LLM-LASSO (amplitude, η=2.0) 選出的 8 個特徵
LLMLASSO_AMPLITUDE_FEATURES = [
    "finger_mvmnt_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks",
    "amplitude_quartile_range",
    "amplitude_min",
    "amplitude_entropy",
]

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
            text += f"  {feat}: {val}\n"
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
                    parts.append(f"  {feat}: {val}\n")
        else:
            for feat in selected_features:
                if feat in ex_features:
                    val = ex_features[feat]
                    if isinstance(val, float) and not np.isnan(val):
                        val = round(val, 4)
                    parts.append(f"{feat}: {val}\n")
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
                feature_text += f"{feat}: {val}\n"

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
