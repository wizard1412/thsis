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

# User dataset (53 sessions, 2 raters)
USER_DATASET_PATH = PROJECT_ROOT / "features_dataset.csv"

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
    # Movement features
    "finger_mvmnt_x_median":           "Median horizontal movement of thumb between consecutive frames",
    "finger_mvmnt_x_quartile_range":   "Interquartile range of horizontal thumb movement",
    "finger_mvmnt_x_mean":             "Mean horizontal thumb movement amplitude per frame",
    "finger_mvmnt_x_min":              "Minimum horizontal thumb movement (closest to stationary)",
    "finger_mvmnt_x_max":              "Maximum horizontal thumb movement in a single frame",
    "finger_mvmnt_x_stdev":            "Variability of horizontal thumb movement across frames",
    "finger_mvmnt_y_median":           "Median vertical movement of thumb between consecutive frames",
    "finger_mvmnt_y_quartile_range":   "Interquartile range of vertical thumb movement",
    "finger_mvmnt_y_mean":             "Mean vertical thumb movement amplitude per frame",
    "finger_mvmnt_y_min":              "Minimum vertical thumb movement",
    "finger_mvmnt_y_max":              "Maximum vertical thumb movement in a single frame",
    "finger_mvmnt_y_stdev":            "Variability of vertical thumb movement across frames",
    "finger_mvmnt_dist_median":        "Median Euclidean distance of thumb movement between frames",
    "finger_mvmnt_dist_quartile_range":"Interquartile range of thumb movement distance",
    "finger_mvmnt_dist_mean":          "Mean thumb-index distance (overall movement amplitude)",
    "finger_mvmnt_dist_min":           "Minimum thumb movement distance",
    "finger_mvmnt_dist_max":           "Maximum thumb movement distance in a single frame",
    "finger_mvmnt_dist_stdev":         "Variability of thumb movement distance",
    # Rhythm features
    "aperiodicity":                    "FFT power spectrum entropy measuring rhythm irregularity (higher = more irregular)",
    "periodEntropy":                   "Entropy of inter-tap interval distribution (higher = more complex rhythm)",
    "periodVarianceNorm":              "Normalized variance of tap periods (rhythm stability measure)",
    "numInterruptions":                "Count of interruptions/hesitations during tapping sequence",
    "numFreeze":                       "Count of freeze episodes (movement arrest) during tapping",
    "maxFreezeDuration":               "Duration of the longest freeze episode in seconds",
    # Period features
    "period_median":                   "Median inter-tap interval in seconds",
    "period_quartile_range":           "Interquartile range of tap intervals (rhythm consistency)",
    "period_mean":                     "Mean inter-tap interval in seconds (inversely related to tapping speed)",
    "period_min":                      "Shortest inter-tap interval in seconds",
    "period_max":                      "Longest inter-tap interval in seconds (may indicate hesitation)",
    "period_stdev":                    "Standard deviation of tap intervals (rhythm stability, higher = more unstable)",
    # Frequency features
    "frequency_median":                "Median tapping frequency in Hz",
    "frequency_quartile_range":        "Interquartile range of tapping frequency",
    "frequency_mean":                  "Mean tapping frequency in Hz (higher = faster tapping)",
    "frequency_min":                   "Minimum tapping frequency (slowest tapping moment)",
    "frequency_max":                   "Maximum tapping frequency (fastest tapping moment)",
    "frequency_stdev":                 "Variability of tapping frequency across the sequence",
    "frequency_lr_fitness_r2":         "R-squared of linear fit to frequency over time (frequency trend goodness)",
    "frequency_lr_slope":              "Slope of linear fit to frequency over time (negative = slowing down)",
    "frequency_fit_min_degree":        "Minimum polynomial degree to fit frequency trend",
    # Count features
    "num_peaks":                       "Total number of detected taps in the sequence",
    "num_interruptions_norm":          "Normalized count of interruptions (interruptions per tap)",
    "num_freeze_norm":                 "Normalized count of freeze episodes (freezes per tap)",
    # Speed features
    "speed_median":                    "Median finger movement speed in degrees/second",
    "speed_quartile_range":            "Interquartile range of movement speed",
    "speed_mean":                      "Mean finger movement speed (higher = more vigorous tapping)",
    "speed_min":                       "Minimum movement speed",
    "speed_max":                       "Maximum movement speed (peak velocity)",
    "speed_stdev":                     "Variability of movement speed",
    # Acceleration features
    "acceleration_median":             "Median finger movement acceleration",
    "acceleration_quartile_range":     "Interquartile range of acceleration",
    "acceleration_mean":               "Mean finger acceleration",
    "acceleration_min":                "Minimum acceleration (peak deceleration)",
    "acceleration_max":                "Maximum acceleration (peak acceleration)",
    "acceleration_stdev":              "Variability of acceleration across the sequence",
    # Amplitude features
    "amplitude_median":                "Median tap amplitude (finger-thumb distance at each peak)",
    "amplitude_quartile_range":        "Interquartile range of tap amplitudes (amplitude consistency)",
    "amplitude_mean":                  "Mean tap amplitude (overall tapping size)",
    "amplitude_min":                   "Minimum tap amplitude (smallest tap in sequence)",
    "amplitude_max":                   "Maximum tap amplitude (largest tap in sequence)",
    "amplitude_stdev":                 "Standard deviation of tap amplitudes (amplitude variability)",
    "amplitude_entropy":               "Entropy of tap amplitude distribution (complexity of amplitude pattern)",
    "amplitude_decrement_fitness_r2":  "R-squared of linear fit to amplitude over time (how well decrement is linear)",
    "amplitude_decrement_slope":       "Slope of linear fit to amplitude over time (negative = decrementing amplitude)",
    "amplitude_decrement_end_to_mean": "Ratio of final tap amplitude to mean amplitude (end decrement severity)",
    "amplitude_decrement_fit_min_degree": "Minimum polynomial degree to fit amplitude trend",
    "amplitude_decrement_last_to_first_half": "Ratio of second-half mean amplitude to first-half mean (progressive decrement)",
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
