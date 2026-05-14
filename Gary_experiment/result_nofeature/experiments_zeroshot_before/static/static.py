import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

def calculate_comprehensive_statistics(file_path):
    """
    Calculates comprehensive statistics for AI vs Human validation study.
    Metrics included:
    1. Human-Human Reliability (Cohen's Kappa)
    2. AI Stability (Standard Deviation, Consistency Rate)
    3. AI Accuracy (MAE, Bias, Correlation)
    """
    
    # Load the dataset
    print(f"Loading file: {file_path}...")
    df = pd.read_csv(file_path)
    
    # Define column names based on your file structure
    # AI score columns are named '0' to '9'
    ai_score_cols = [str(i) for i in range(10)]
    doctor_1_col = 'label_Dr. Tan'
    doctor_2_col = 'label_Dr. Chien'
    doctor_avg_col = 'average_score_Doctors'
    
    # =========================================================
    # Part 1: Human-Human Reliability (Inter-rater Reliability)
    # =========================================================
    print("\n" + "="*40)
    print(" 1. Human-Human Reliability (Dr. Tan vs Dr. Chien)")
    print("="*40)
    
    # Filter valid rows where both doctors provided a score
    human_df = df.dropna(subset=[doctor_1_col, doctor_2_col])
    
    # Calculate Unweighted Cohen's Kappa
    kappa_unweighted = cohen_kappa_score(human_df[doctor_1_col], human_df[doctor_2_col])
    
    # Calculate Weighted Cohen's Kappa (Linear)
    # Weighted is better for ordinal data (e.g., score 0-4) because it accounts for the degree of disagreement
    kappa_weighted = cohen_kappa_score(human_df[doctor_1_col], human_df[doctor_2_col], weights='linear')
    
    print(f"Cohen's Kappa (Unweighted): {kappa_unweighted:.4f}")
    print(f"Cohen's Kappa (Linear Weighted): {kappa_weighted:.4f}")
    print("Note: Weighted Kappa is generally preferred for ordinal medical scores.")

    # =========================================================
    # Part 2: AI Stability (Internal Consistency)
    # =========================================================
    print("\n" + "="*40)
    print(" 2. AI Stability (Consistency across 10 runs)")
    print("="*40)
    
    # Calculate the standard deviation (SD) of the 10 AI runs for each video
    df['ai_std'] = df[ai_score_cols].std(axis=1)
    
    # Average SD across all videos (Lower is more stable)
    avg_ai_std = df['ai_std'].mean()
    
    # Perfect Consistency Rate
    # Count how many videos have SD == 0 (meaning all 10 scores were identical)
    perfect_cases = (df['ai_std'] == 0).sum()
    total_cases = len(df)
    perfect_rate = (perfect_cases / total_cases) * 100
    
    print(f"Average Standard Deviation (SD): {avg_ai_std:.4f}")
    print(f"Perfect Consistency Rate: {perfect_rate:.2f}% ({perfect_cases}/{total_cases} files)")

    # =========================================================
    # Part 3: AI Accuracy & Agreement (AI vs Human Average)
    # =========================================================
    print("\n" + "="*40)
    print(" 3. AI Accuracy (AI Mean vs Doctor Average)")
    print("="*40)
    
    # Calculate AI Mean Score
    df['ai_mean'] = df[ai_score_cols].mean(axis=1)
    
    # Filter valid rows for comparison
    valid_comparison = df.dropna(subset=['ai_mean', doctor_avg_col])
    
    # Mean Absolute Error (MAE)
    # The average absolute difference between AI score and Human score
    mae = mean_absolute_error(valid_comparison[doctor_avg_col], valid_comparison['ai_mean'])
    
    # Bias (Mean Difference)
    # Positive value = AI tends to score higher than humans
    # Negative value = AI tends to score lower than humans
    bias = np.mean(valid_comparison['ai_mean'] - valid_comparison[doctor_avg_col])
    
    # Pearson Correlation Coefficient
    correlation = valid_comparison['ai_mean'].corr(valid_comparison[doctor_avg_col])
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Bias (Mean Difference): {bias:.4f}")
    print(f"Pearson Correlation: {correlation:.4f}")

# Execute the function
# Ensure the CSV file is in the same directory
if __name__ == "__main__":
    file_name = 'scored_by_qwen_30b_promptC_10_251130.csv'
    try:
        calculate_comprehensive_statistics(file_name)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found. Please check the file path.")
        