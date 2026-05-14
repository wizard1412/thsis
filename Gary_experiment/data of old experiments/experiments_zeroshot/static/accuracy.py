#!/usr/bin/env python3
"""
Batch Generate Table 3.10 for All Models
This script processes all Excel files matching the pattern "scored_by_*.xlsx" 
and generates Table 3.10 metrics for each model with bootstrap confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_DIR = '.'  # Current directory (same folder as script)
OUTPUT_DIR = './results'  # Output directory (will be created if it doesn't exist)
FILE_PATTERN = 'scored_by_*.csv'  # Pattern to match input files
N_BOOTSTRAP = 1000  # Number of bootstrap samples for CI calculation
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def calculate_metrics(y_true, y_pred):
    """
    Calculate all metrics for a given set of predictions
    
    Args:
        y_true: Ground truth labels (concordant scores)
        y_pred: Predicted labels (final scores from median of 10 evaluations)
    
    Returns:
        Dictionary with all metrics
    """
    # Accuracy
    accuracy = (y_true == y_pred).mean() * 100
    
    # Acceptable Accuracy (within 1 point)
    acceptable_accuracy = (np.abs(y_true - y_pred) <= 1).mean() * 100
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Wasserstein Distance
    wd = stats.wasserstein_distance(y_true, y_pred)
    
    return {
        'Acc': accuracy,
        'AAcc': acceptable_accuracy,
        'MAE': mae,
        'RMSE': rmse,
        'WD': wd
    }

def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000):
    """
    Calculate bootstrap confidence intervals for all metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        Dictionary with point estimates and 95% CIs for all metrics
    """
    n_samples = len(y_true)
    
    # Storage for bootstrap samples
    bootstrap_results = {
        'Acc': [],
        'AAcc': [],
        'MAE': [],
        'RMSE': [],
        'WD': []
    }
    
    # Perform bootstrap sampling
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metrics for this bootstrap sample
        metrics = calculate_metrics(y_true_boot, y_pred_boot)
        
        for key, value in metrics.items():
            bootstrap_results[key].append(value)
    
    # Calculate point estimates and confidence intervals
    results = {}
    for metric_name in ['Acc', 'AAcc', 'MAE', 'RMSE', 'WD']:
        values = np.array(bootstrap_results[metric_name])
        
        # Point estimate (original data)
        point_estimate = calculate_metrics(y_true, y_pred)[metric_name]
        
        # 95% confidence interval (percentile method)
        ci_lower = np.percentile(values, 2.5)
        ci_upper = np.percentile(values, 97.5)
        
        results[metric_name] = {
            'point': point_estimate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return results

def format_metric(metric_dict, decimal_places=2):
    """
    Format metric with confidence interval
    
    Args:
        metric_dict: Dictionary with 'point', 'ci_lower', 'ci_upper'
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string
    """
    if decimal_places == 1:
        return f"{metric_dict['point']:.1f} [{metric_dict['ci_lower']:.1f}, {metric_dict['ci_upper']:.1f}]"
    else:
        return f"{metric_dict['point']:.2f} [{metric_dict['ci_lower']:.2f}, {metric_dict['ci_upper']:.2f}]"

def extract_model_name(filename):
    """
    Extract model name and experiment info from filename
    
    Args:
        filename: CSV file name (e.g., "scored_by_qwen_30b_promptC_10.csv" or "scored_by_ChatGPT_prompt3.csv")
    
    Returns:
        Dictionary with model info
    """
    # Remove "scored_by_" prefix and ".csv" suffix
    name = filename.replace('scored_by_', '').replace('.csv', '')
    
    # Split into components
    parts = name.split('_')
    
    # Try to identify prompt from the parts
    prompt = 'unknown'
    for part in parts:
        if 'prompt' in part.lower():
            prompt = part
            break
    
    # Create a clean display name
    if prompt != 'unknown':
        # Remove prompt part and use the rest as model name
        model_parts = [p for p in parts if p != prompt and not p.isdigit()]
        model_name = '_'.join(model_parts)
    else:
        # Use all non-digit parts as model name
        model_parts = [p for p in parts if not p.isdigit()]
        model_name = '_'.join(model_parts)
    
    # Find evaluation count if it exists
    eval_count = 'unknown'
    for part in reversed(parts):
        if part.isdigit():
            eval_count = part
            break
    
    display_name = f"{model_name.upper()}"
    if prompt != 'unknown':
        display_name += f" {prompt.upper()}"
    
    return {
        'full_name': name,
        'model': model_name,
        'prompt': prompt,
        'eval_count': eval_count,
        'display_name': display_name
    }

def process_single_file(file_path):
    """
    Process a single CSV file and calculate metrics
    
    Args:
        file_path: Path to CSV file
    
    Returns:
        Dictionary with results or None if processing fails
    """
    try:
        # Load data from CSV
        df = pd.read_csv(file_path)
        
        # Extract score columns (0-9 can be either integers or strings)
        # Try both integer and string column names
        score_columns = []
        for i in range(10):
            if i in df.columns:
                score_columns.append(i)
            elif str(i) in df.columns:
                score_columns.append(str(i))
            else:
                print(f"  ✗ Missing score column: {i}")
                return None
        
        if len(score_columns) != 10:
            print(f"  ✗ Expected 10 score columns, found {len(score_columns)}")
            return None
        
        # Verify required doctor label columns exist
        if 'label_Dr. Tan' not in df.columns or 'label_Dr. Chien' not in df.columns:
            print(f"  ✗ Missing required doctor label columns")
            return None
        
        # Get doctor labels
        df['label_Dr_Tan'] = df['label_Dr. Tan']
        df['label_Dr_Chien'] = df['label_Dr. Chien']
        
        # Identify concordant sessions
        df['concordant'] = df['label_Dr_Tan'] == df['label_Dr_Chien']
        df['concordant_score'] = df.apply(
            lambda row: row['label_Dr_Tan'] if row['concordant'] else np.nan, 
            axis=1
        )
        
        # Filter for concordant sessions only
        concordant_df = df[df['concordant']].copy()
        
        if len(concordant_df) == 0:
            print(f"  ✗ No concordant sessions found")
            return None
        
        # Calculate final score (median of 10 evaluations, rounded to nearest integer)
        concordant_df['final_score'] = concordant_df[score_columns].median(axis=1).round()
        
        # Extract arrays for calculation
        y_true = concordant_df['concordant_score'].values
        y_pred = concordant_df['final_score'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            print(f"  ✗ No valid data after removing NaN values")
            return None
        
        # Calculate metrics with bootstrap confidence intervals
        results = bootstrap_metrics(y_true, y_pred, n_bootstrap=N_BOOTSTRAP)
        
        # Get score distribution
        score_distribution = concordant_df['concordant_score'].value_counts().sort_index().to_dict()
        
        return {
            'total_sessions': len(df),
            'concordant_sessions': len(y_true),
            'score_distribution': score_distribution,
            'metrics': results,
            'y_true': y_true,
            'y_pred': y_pred,
            'concordant_df': concordant_df
        }
        
    except Exception as e:
        print(f"  ✗ Error processing file: {e}")
        return None

def generate_combined_table(all_results):
    """
    Generate a combined table with all models
    
    Args:
        all_results: Dictionary with results for all models
    
    Returns:
        DataFrame with combined results
    """
    table_data = []
    
    for model_name, result in all_results.items():
        if result is not None:
            metrics = result['metrics']
            row = {
                'Model': model_name,
                'N': result['concordant_sessions'],
                'Acc (%)': format_metric(metrics['Acc'], 1),
                'AAcc (%)': format_metric(metrics['AAcc'], 1),
                'MAE': format_metric(metrics['MAE'], 2),
                'RMSE': format_metric(metrics['RMSE'], 2),
                'WD': format_metric(metrics['WD'], 2)
            }
            table_data.append(row)
    
    return pd.DataFrame(table_data)

def save_detailed_report(all_results, output_file):
    """
    Save detailed report with all models
    
    Args:
        all_results: Dictionary with results for all models
        output_file: Path to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("TABLE 3.10 - COMPREHENSIVE RESULTS FOR ALL MODELS\n")
        f.write("="*100 + "\n\n")
        
        for model_name, result in all_results.items():
            if result is None:
                f.write(f"\n{model_name}: PROCESSING FAILED\n")
                f.write("-"*100 + "\n")
                continue
            
            f.write(f"\n{'='*100}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'='*100}\n\n")
            
            f.write(f"Total sessions: {result['total_sessions']}\n")
            f.write(f"Concordant sessions: {result['concordant_sessions']}\n\n")
            
            f.write("Score distribution:\n")
            for score, count in sorted(result['score_distribution'].items()):
                f.write(f"  Score {int(score)}: {count} sessions\n")
            
            f.write("\nMetrics (Point Estimate [95% CI]):\n")
            f.write("-"*100 + "\n")
            
            metrics = result['metrics']
            f.write(f"Accuracy (Acc):             {format_metric(metrics['Acc'], 1)} %\n")
            f.write(f"Acceptable Accuracy (AAcc): {format_metric(metrics['AAcc'], 1)} %\n")
            f.write(f"MAE:                        {format_metric(metrics['MAE'], 2)}\n")
            f.write(f"RMSE:                       {format_metric(metrics['RMSE'], 2)}\n")
            f.write(f"Wasserstein Distance (WD):  {format_metric(metrics['WD'], 2)}\n")
            
            # Session-by-session comparison
            f.write("\nSession-by-Session Comparison:\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Session':<50} {'True':<6} {'Pred':<6} {'Error':<6}\n")
            f.write("-"*100 + "\n")
            
            concordant_df = result['concordant_df']
            for idx, row in concordant_df.iterrows():
                if 'filename' in row:
                    filename = row['filename'][:45] + "..." if len(row['filename']) > 45 else row['filename']
                else:
                    filename = f"Session_{idx}"
                
                true_score = int(row['concordant_score'])
                pred_score = int(row['final_score'])
                error = abs(true_score - pred_score)
                f.write(f"{filename:<50} {true_score:<6} {pred_score:<6} {error:<6}\n")
            
            f.write("\n")

def main():
    """Main function to process all CSV files"""
    
    print("="*100)
    print("BATCH PROCESSING: GENERATE TABLE 3.10 FOR ALL MODELS")
    print("="*100)
    
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files matching the pattern
    input_path = Path(INPUT_DIR)
    csv_files = list(input_path.glob(FILE_PATTERN))
    
    if len(csv_files) == 0:
        print(f"\n✗ No files found matching pattern '{FILE_PATTERN}' in {INPUT_DIR}")
        print(f"  Please ensure your files follow the naming convention: scored_by_<model>_<prompt>_<count>.csv")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:\n")
    
    all_results = {}
    
    # Process each file
    for i, file_path in enumerate(csv_files, 1):
        filename = file_path.name
        model_info = extract_model_name(filename)
        model_display_name = model_info['display_name']
        
        print(f"[{i}/{len(csv_files)}] Processing: {filename}")
        print(f"  Model: {model_display_name}")
        
        # Process the file
        result = process_single_file(file_path)
        
        if result is not None:
            all_results[model_display_name] = result
            print(f"  ✓ Concordant sessions: {result['concordant_sessions']}")
            print(f"  ✓ Accuracy: {result['metrics']['Acc']['point']:.1f}%")
        
        print()
    
    if len(all_results) == 0:
        print("✗ No files were successfully processed")
        return None, None
    
    print("="*100)
    print(f"Successfully processed {len(all_results)} model(s)")
    print("="*100)
    
    # Generate combined table
    print("\nGenerating combined table...")
    combined_table = generate_combined_table(all_results)
    
    # Display combined table
    print("\n" + "="*100)
    print("COMBINED TABLE 3.10 - ALL MODELS")
    print("="*100)
    print(combined_table.to_string(index=False))
    print("="*100)
    
    # Save combined table
    combined_csv_path = Path(OUTPUT_DIR) / 'table3_10_all_models.csv'
    combined_table.to_csv(combined_csv_path, index=False)
    print(f"\n✓ Combined table saved to: {combined_csv_path}")
    
    # Save detailed report
    detailed_report_path = Path(OUTPUT_DIR) / 'table3_10_all_models_detailed.txt'
    save_detailed_report(all_results, detailed_report_path)
    print(f"✓ Detailed report saved to: {detailed_report_path}")
    
    # Save individual model tables
    print(f"\nSaving individual model results...")
    for model_name, result in all_results.items():
        if result is not None:
            # Clean model name for filename
            safe_model_name = model_name.replace(' ', '_').replace('/', '_')
            
            # Create individual table
            individual_table = pd.DataFrame([{
                'Model': model_name,
                'N': result['concordant_sessions'],
                'Acc (%)': format_metric(result['metrics']['Acc'], 1),
                'AAcc (%)': format_metric(result['metrics']['AAcc'], 1),
                'MAE': format_metric(result['metrics']['MAE'], 2),
                'RMSE': format_metric(result['metrics']['RMSE'], 2),
                'WD': format_metric(result['metrics']['WD'], 2)
            }])
            
            # Save individual CSV
            individual_csv_path = Path(OUTPUT_DIR) / f'table3_10_{safe_model_name}.csv'
            individual_table.to_csv(individual_csv_path, index=False)
            print(f"  ✓ {model_name}: {individual_csv_path}")
    
    print("\n" + "="*100)
    print("✓ ALL PROCESSING COMPLETE!")
    print("="*100)
    
    return combined_table, all_results

if __name__ == "__main__":
    result = main()
    if result is not None:
        combined_table, all_results = result