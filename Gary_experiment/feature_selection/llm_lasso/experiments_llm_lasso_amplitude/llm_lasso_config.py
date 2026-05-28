"""
LLM-Lasso Feature Selection Configuration
Based on: Zhang et al. (2025) "LLM-Lasso: A Robust Framework for
Domain-Informed Feature Selection and Regularization"

Adapted for MDS-UPDRS Finger Tapping severity assessment.
"""

from pathlib import Path

# ===========================
# Paths
# ===========================

PROJECT_ROOT = Path(__file__).parent
EXPERIMENT_ROOT = PROJECT_ROOT.parent

# User dataset (53 sessions, 2 raters) — v2: thumb/index split + extended amplitude features
# Search upward from PROJECT_ROOT to support different directory structures across machines
def _find_dataset(name: str) -> Path:
    for ancestor in [PROJECT_ROOT, EXPERIMENT_ROOT, EXPERIMENT_ROOT.parent, EXPERIMENT_ROOT.parent.parent]:
        p = ancestor / name
        if p.exists():
            return p
    return EXPERIMENT_ROOT.parent / name  # fall back with original path so error is descriptive

USER_DATASET_PATH = _find_dataset("features_dataset_v2.csv")

# Output
OUTPUT_DIR = PROJECT_ROOT / "results"
SCORES_JSON = OUTPUT_DIR / "llm_feature_scores.json"
SELECTION_RESULTS = OUTPUT_DIR / "feature_selection_results.json"
PLOT_DIR = OUTPUT_DIR / "plots"

# ===========================
# Ollama / LLM Settings
# ===========================

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-r1:70b"
REQUEST_TIMEOUT = 300
NUM_PREDICT = 4096
DELAY_BETWEEN_CALLS = 5

# How many times to ask LLM for each feature (for stability)
LLM_SCORING_REPEATS = 5

# ===========================
# Feature Definitions
# ===========================

# Non-feature columns to exclude
EXCLUDE_COLS = {"hand", "filename", "Rating1", "Rating2", "Rating"}

# Human-readable descriptions for each feature (used in LLM prompt)
FEATURE_DESCRIPTIONS = {
    # Amplitude decrement
    "A_early":                            "mean amplitude of the first 3 taps",
    "A_late":                             "mean amplitude of the last 3 taps",
    "R_A_3_3":                            "ratio A_late / A_early (1=no decrement, <1=decrement)",
    "D_A_pct":                            "relative drop from A_early to A_late (0=none, 1=full collapse)",
    "D_A_max":                            "max relative amplitude drop across the sequence",
    "k_collapse":                         "tap index where amplitude first drops below 0.5*A_early (-1 if none)",
    "tilde_A_median":                     "median of amplitudes normalized by A_early",
    "beta_tilde_A":                       "linear slope of normalized amplitudes over tap index",
    "CV_tilde_A":                         "coefficient of variation of normalized amplitudes",
    "amplitude_MAD":                      "median absolute deviation of tap amplitudes",
    "amplitude_decrement_fitness_r2":     "R^2 of linear fit of amplitude vs tap index",
    "amplitude_decrement_slope":          "slope of amplitude decrement fit",
    "amplitude_decrement_end_to_mean":    "ratio last-tap amplitude / mean amplitude",
    "amplitude_decrement_last_to_first_half": "mean(last half) / mean(first half) of amplitudes",
    "amplitude_decrement_fit_min_degree": "min polynomial degree for amplitude decrement fit",
    # ITI / rhythm
    "I_ITI":                              "inter-tap-interval irregularity index",
    "I_ITI_norm":                         "ITI irregularity normalized by mean ITI",
    "ITI_MAD":                            "median absolute deviation of inter-tap intervals",
    # Halt / freeze / completion
    "S_halt":                             "total halt duration during the tapping (seconds)",
    "F_halt":                             "halt frequency (events per second)",
    "N_valid":                            "number of valid taps",
    "C_10":                               "1 if completed 10 taps, else 0",
    "T_10":                               "time to complete 10 taps (seconds)",
    "F_incomplete":                       "fraction of expected taps missing",
    "num_interruptions_norm":             "interruption events per second",
    "num_freeze_norm":                    "freeze events per second",
    # Opening / closing phase
    "V_open_median":                      "median opening-phase peak velocity in pixels/s (finger lifting)",
    "V_open_mean":                        "mean opening-phase peak velocity in pixels/s",
    "V_open_std":                         "std of opening-phase peak velocity across taps",
    "V_close_median":                     "median closing-phase peak velocity in pixels/s (finger tapping down)",
    "V_close_mean":                       "mean closing-phase peak velocity in pixels/s",
    "V_close_std":                        "std of closing-phase peak velocity across taps",
    "V_open_norm_median":                 "median opening-phase velocity normalized by distance range",
    "V_open_norm_mean":                   "mean opening-phase velocity normalized by distance range",
    "V_open_norm_std":                    "std of normalized opening-phase velocity",
    "V_close_norm_median":                "median closing-phase velocity normalized by distance range",
    "V_close_norm_mean":                  "mean closing-phase velocity normalized by distance range",
    "V_close_norm_std":                   "std of normalized closing-phase velocity",
    "T_open_median":                      "median opening-phase duration (seconds)",
    "T_open_mean":                        "mean opening-phase duration",
    "T_open_std":                         "std of opening-phase duration",
    "T_close_median":                     "median closing-phase duration",
    "T_close_mean":                       "mean closing-phase duration",
    "T_close_std":                        "std of closing-phase duration",
    "R_OC_median":                        "median ratio opening/closing duration",
    "R_OC_mean":                          "mean ratio opening/closing duration",
    "R_OC_std":                           "std of ratio opening/closing duration",
    "R_V_open":                           "opening-velocity decrement ratio (late/early taps)",
    "R_V_close":                          "closing-velocity decrement ratio (late/early taps)",
    # Period / frequency
    "aperiodicity":                       "aperiodicity score of tapping rhythm",
    "periodEntropy":                      "entropy of inter-tap period distribution",
    "periodVarianceNorm":                 "variance of periods normalized by mean period",
    "frequency_lr_fitness_r2":            "R^2 of linear fit of frequency vs tap index",
    "frequency_lr_slope":                 "slope of tap frequency over time (negative = slowing)",
    "frequency_fit_min_degree":           "min polynomial degree for frequency fit",
    # Finger movement — thumb
    "finger_mvmnt_thumb_x_median":        "median frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_x_mean":          "mean frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_x_max":           "max frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_x_stdev":         "std of frame-to-frame thumb x displacement",
    "finger_mvmnt_thumb_dist_stdev":      "std of frame-to-frame thumb trajectory distance",
    "finger_mvmnt_thumb_dist_quartile_range": "IQR of frame-to-frame thumb trajectory distance",
    # Finger movement — index
    "finger_mvmnt_index_x_median":        "median frame-to-frame index finger x displacement",
    "finger_mvmnt_index_x_mean":          "mean frame-to-frame index finger x displacement",
    "finger_mvmnt_index_x_max":           "max frame-to-frame index finger x displacement",
    "finger_mvmnt_index_x_stdev":         "std of frame-to-frame index finger x displacement",
    "finger_mvmnt_index_dist_stdev":      "std of frame-to-frame index finger trajectory distance",
    "finger_mvmnt_index_dist_quartile_range": "IQR of frame-to-frame index finger trajectory distance",
    # Normalized amplitude
    "amplitude_norm_median":              "median tap amplitude normalized by subject's full distance range",
    "amplitude_norm_mean":                "mean normalized tap amplitude",
    "amplitude_norm_min":                 "min normalized tap amplitude",
    "amplitude_norm_max":                 "max normalized tap amplitude",
    "amplitude_norm_stdev":               "std of normalized tap amplitudes",
    "amplitude_norm_quartile_range":      "IQR of normalized tap amplitudes",
    # Speed / acceleration (normalized)
    "speed_norm_median":                  "median tapping speed in Norm_Distance/s",
    "speed_norm_mean":                    "mean tapping speed in Norm_Distance/s",
    "speed_norm_max":                     "max tapping speed in Norm_Distance/s",
    "speed_norm_stdev":                   "std of tapping speed in Norm_Distance/s",
    "acceleration_norm_median":           "median tapping acceleration in Norm_Distance/s²",
    "acceleration_norm_mean":             "mean tapping acceleration in Norm_Distance/s²",
}

# ===========================
# LLM-Lasso Hyperparameters
# ===========================

# ?? search range (controls how much to trust LLM)
# ?? = 0 ??? standard LASSO; ?? large ??? heavily trust LLM
ETA_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# Number of features to select (for comparison with current 8)
N_FEATURES_TO_SELECT = [6, 8, 10, 12]

# Cross-validation settings
N_CV_FOLDS = 5
RANDOM_SEED = 42
