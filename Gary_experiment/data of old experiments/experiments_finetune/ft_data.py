"""
Data loading, feature serialization, and HuggingFace Dataset construction
for LLM fine-tuning.

Reuses rce_data.prepare_all_data() for identical CV splits and features.
Serializes 33 numerical features into structured natural language text.
"""

import sys
import numpy as np
from pathlib import Path
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments_rce"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments_traditional_ml"))

from ft_config import (
    UNIFIED_FEATURE_NAMES,
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    FEATURE_GROUPS,
    FEATURE_DESCRIPTIONS,
    SCORE_NUM_CLASSES,
)


def load_data():
    """Load data using the shared rce_data pipeline (same CV splits)."""
    from rce_data import prepare_all_data
    data = prepare_all_data()
    return {
        "all_features": data["all_features"],       # (541, 33)
        "all_ratings": data["all_ratings"],          # (541, 5)
        "majority_scores": data["majority_scores"],  # (541,)
        "cv_splits": data["cv_splits"],              # 5-fold
        "feature_names": data["feature_names"],       # 33 names
        "dataset_source": data["dataset_source"],     # "ext"/"user"
    }


def serialize_sample(features_array, feature_names):
    """Convert a single sample's 33 standardized features into structured text.

    Groups features by semantic category for better context.
    """
    feat_dict = dict(zip(feature_names, features_array))
    lines = []
    for group_name, group_features in FEATURE_GROUPS.items():
        lines.append(f"[{group_name}]")
        for fname in group_features:
            desc = FEATURE_DESCRIPTIONS[fname]
            val = feat_dict[fname]
            lines.append(f"- {desc}: {val:.4f}")
    return "\n".join(lines)


def format_chat_messages(features_array, label, feature_names):
    """Create a single chat message dict for training."""
    serialized = serialize_sample(features_array, feature_names)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(serialized_features=serialized)},
            {"role": "assistant", "content": str(int(label))},
        ]
    }


def build_train_dataset(features, labels, feature_names):
    """Build HuggingFace Dataset from features and labels for SFTTrainer."""
    examples = []
    for i in range(len(features)):
        example = format_chat_messages(features[i], labels[i], feature_names)
        examples.append(example)
    return Dataset.from_list(examples)


def build_inference_prompts(features, feature_names, tokenizer):
    """Build tokenized prompts for inference (without assistant response).

    Returns list of prompt strings ready for tokenizer encoding.
    """
    prompts = []
    for i in range(len(features)):
        serialized = serialize_sample(features[i], feature_names)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(serialized_features=serialized)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


def get_train_labels(strategy, train_ratings, train_majority):
    """Get training labels based on label strategy."""
    if strategy == "majority_vote":
        return train_majority.copy(), None
    elif strategy == "dawid_skene":
        from tml_dawid_skene import dawid_skene_em
        ds_result = dawid_skene_em(
            ratings=train_ratings,
            num_classes=SCORE_NUM_CLASSES,
            init_labels=train_majority,
        )
        return ds_result["estimated_labels"], ds_result
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
