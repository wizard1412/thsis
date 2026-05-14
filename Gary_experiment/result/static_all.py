import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

def analyze_and_plot_models(folder_path='.'):
    """
    Analyzes all CSV files in the specified folder and plots comparison charts.
    Assumes files contain columns '0' to '9' for AI scores and 'average_score_Doctors' for ground truth.
    """
    
    # 1. Find all CSV files in the folder
    # ---------------------------------------------------------
    file_pattern = os.path.join(folder_path, "*.csv")
    files = glob.glob(file_pattern)
    
    if not files:
        print("No CSV files found in the directory!")
        return
    
    summary_list = []
    
    print(f"Found {len(files)} files. Processing...")

    # 2. Process each file
    # ---------------------------------------------------------
    for file in files:
        try:
            df = pd.read_csv(file)
            
            # Extract Model Name from filename for the chart labels
            # e.g., 'C:/Data/GPT5_Result.csv' -> 'GPT5_Result'
            model_name = '_'.join(os.path.basename(file).replace('.csv', '').replace('scored_by_', '').split('_')[:3])
            
            # Identify score columns (0-9) and Ground Truth column
            ai_cols = [str(i) for i in range(10)]
            doctor_avg_col = 'average_score_Doctors'
            
            # Check if columns exist
            if not all(col in df.columns for col in ai_cols) or doctor_avg_col not in df.columns:
                print(f"Skipping {model_name}: Missing required columns.")
                continue

            # --- Calculate Metrics ---
            
            # Calculate AI Mean for each row
            df['ai_mean'] = df[ai_cols].mean(axis=1)
            
            # Filter valid rows (remove NaNs)
            valid_df = df.dropna(subset=[doctor_avg_col, 'ai_mean'])
            
            if len(valid_df) == 0:
                continue

            # A. Accuracy (MAE - Lower is better)
            mae = mean_absolute_error(valid_df[doctor_avg_col], valid_df['ai_mean'])
            
            # B. Correlation (Pearson - Higher is better)
            corr = valid_df[doctor_avg_col].corr(valid_df['ai_mean'])
            
            # C. Bias (Mean Difference - Closer to 0 is better)
            bias = np.mean(valid_df['ai_mean'] - valid_df[doctor_avg_col])
            
            # D. Stability (Average Standard Deviation - Lower is better)
            df['ai_std'] = df[ai_cols].std(axis=1)
            avg_stability = df['ai_std'].mean()

            # Append to list
            summary_list.append({
                'Model': model_name,
                'MAE (Accuracy)': mae,
                'Correlation': corr,
                'Bias': bias,
                'Stability (Avg SD)': avg_stability
            })
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_list)
    
    # Display the data table
    print("\n--- Model Comparison Summary ---")
    print(summary_df)

    # 3. Plotting
    # ---------------------------------------------------------
    if not summary_df.empty:
        # Set plot style
        sns.set(style="whitegrid")
        
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('AI Model Performance Comparison', fontsize=16)

        # Plot 1: MAE (Lower is better)
        sns.barplot(data=summary_df, x='Model', y='MAE (Accuracy)', hue='Model', ax=axes[0, 0], palette='Reds_r', legend=False)
        axes[0, 0].set_title('Mean Absolute Error (Lower is Better)')
        axes[0, 0].set_ylabel('MAE Score')

        # Plot 2: Correlation (Higher is better)
        sns.barplot(data=summary_df, x='Model', y='Correlation', hue='Model', ax=axes[0, 1], palette='Greens', legend=False)
        axes[0, 1].set_title('Pearson Correlation (Higher is Better)')
        axes[0, 1].set_ylabel('Correlation Coefficient')

        # Plot 3: Bias (Closer to 0 is better)
        sns.barplot(data=summary_df, x='Model', y='Bias', hue='Model', ax=axes[1, 0], palette='Blues', legend=False)
        axes[1, 0].set_title('Bias / Systematic Error (Closer to 0 is Better)')
        axes[1, 0].axhline(0, color='black', linewidth=1) # Add zero line

        # Plot 4: Stability (Lower is better)
        sns.barplot(data=summary_df, x='Model', y='Stability (Avg SD)', hue='Model', ax=axes[1, 1], palette='Purples_r', legend=False)
        axes[1, 1].set_title('Instability / Avg SD (Lower is Better)')
        axes[1, 1].set_ylabel('Standard Deviation')

        # Rotate x-axis labels for better readability
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        plt.tight_layout()
        plt.show()
        print("Comparison charts generated successfully.")
    else:
        print("No valid data found to plot.")

# Execute the function (Processes current directory by default)
analyze_and_plot_models()