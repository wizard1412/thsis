"""
LLM-Lasso Feature Selection Configuration
Based on: Zhang et al. (2025) "LLM-Lasso: A Robust Framework for
Domain-Informed Feature Selection and Regularization"

Adapted for MDS-UPDRS Finger Tapping severity assessment.
This variant compares CV-MAE selection vs CVE-area selection (original paper method).
"""

from pathlib import Path

# ===========================
# Paths
# ===========================

PROJECT_ROOT = Path(__file__).parent
EXPERIMENT_ROOT = PROJECT_ROOT.parent

# Reuse dataset and LLM scores from the rag experiment (71 features, amplitude scored)
USER_DATASET_PATH = EXPERIMENT_ROOT / "experiments_llm_lasso_rag" / "features_dataset.csv"
SCORES_JSON = EXPERIMENT_ROOT / "experiments_llm_lasso_rag" / "results" / "no_rag" / "llm_feature_scores.json"

# Output
OUTPUT_DIR = PROJECT_ROOT / "results"
SELECTION_RESULTS = OUTPUT_DIR / "feature_selection_results.json"
PLOT_DIR = OUTPUT_DIR / "plots"

# ===========================
# LLM-Lasso Hyperparameters
# ===========================

# eta search range — includes integers 0..5 to match original paper's max_imp_pow=5,
# plus finer intermediate values for comparison
ETA_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

# Cross-validation settings
N_CV_FOLDS = 5
RANDOM_SEED = 42

# Number of alphas in the regularization path (for CVE computation)
N_ALPHAS_PATH = 100

# ===========================
# Feature Definitions
# ===========================

EXCLUDE_COLS = {"hand", "filename", "Rating1", "Rating2", "Rating"}

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
}
