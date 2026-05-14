import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import glob
import os

# Set plot style
sns.set_theme(style="white")

def extract_model_name(filename):
    """
    從檔名中提取模型名稱作為圖表標題
    預期格式: ...scored_by_<ModelName>_...
    或是: ...scored_by_<ModelName>.csv
    邏輯: 抓取 'scored_by_' 與下一個 '_' 之間的字串
    """
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    
    # 尋找 'scored_by_' 關鍵字
    if 'scored_by_' in name_no_ext:
        # 取得 'scored_by_' 之後的部分
        parts = name_no_ext.split('scored_by_', 1)
        if len(parts) > 1:
            suffix = parts[1]
            
            # 嘗試以底線分割，取第一部分作為模型名
            if '_' in suffix:
                model_name = suffix.split('_')[0]
            else:
                # 如果沒有底線，就取剩下全部
                model_name = suffix
            
            # 如果提取出的名稱有效，回傳
            if model_name.strip():
                return model_name
    
    # 如果找不到格式，回傳 Unknown
    return "Unknown_Model"

def compute_percentage_matrix(pairs, max_score=4):
    """
    Computes a confusion matrix normalized by percentage.
    Now: Row (Y-axis) = s1, Column (X-axis) = s2.
    """
    matrix = np.zeros((max_score + 1, max_score + 1))
    valid_count = 0
    
    for s1, s2 in pairs:
        if not np.isnan(s1) and not np.isnan(s2):
            r, c = int(s1), int(s2) 
            if 0 <= r <= max_score and 0 <= c <= max_score:
                matrix[r, c] += 1
                valid_count += 1
                
    if valid_count > 0:
        matrix = (matrix / valid_count) * 100
        
    return matrix

def plot_heatmap(matrix, title, xlabel, ylabel, dir_png, dir_pdf, base_filename):
    """
    Plots and saves the heatmap to separate PNG and PDF folders.
    """
    plt.figure(figsize=(6, 5))
    
    # Draw heatmap
    ax = sns.heatmap(matrix, annot=True, fmt='.1f', cmap='OrRd', 
                     cbar_kws={'label': 'Percentage (%)'},
                     linewidths=.5, linecolor='gray',
                     vmin=0, vmax=100) # Fixed scale 0-100%
    
    ax.set_title(title, pad=20, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    plt.tight_layout()
    
    # 1. Save as PNG
    png_path = os.path.join(dir_png, f"{base_filename}.png")
    plt.savefig(png_path, dpi=300, format='png')
    print(f"  -> Saved PNG: {png_path}")
    
    # 2. Save as PDF
    pdf_path = os.path.join(dir_pdf, f"{base_filename}.pdf")
    plt.savefig(pdf_path, dpi=300, format='pdf')
    print(f"  -> Saved PDF: {pdf_path}")
    
    plt.close() # Close figure to free memory

def generate_doctors_heatmap(df, dir_png, dir_pdf):
    """
    Generates the doctors' agreement heatmap (only once).
    Axes Swapped: Y = Dr. Tan, X = Dr. Chien
    """
    col_dr1 = 'label_Dr. Tan'
    col_dr2 = 'label_Dr. Chien'
    
    if col_dr1 in df.columns and col_dr2 in df.columns:
        print("\nChecking for Doctors' data...")
        # s1 = Tan, s2 = Chien
        doc_pairs = list(zip(df[col_dr1], df[col_dr2]))
        # Filter pairs where both are not NaN
        doc_pairs = [(x, y) for x, y in doc_pairs if not np.isnan(x) and not np.isnan(y)]
        
        if doc_pairs:
            doc_matrix = compute_percentage_matrix(doc_pairs)
            
            plot_heatmap(doc_matrix, 
                         title="Two neurologists", 
                         xlabel="Dr. Chien",  # X-axis
                         ylabel="Dr. Tan",    # Y-axis
                         dir_png=dir_png,
                         dir_pdf=dir_pdf,
                         base_filename="Gary2025_heatmap_doctors_combined")
            return True
    return False

def process_file_model_consistency(csv_file, dir_png, dir_pdf):
    """
    Process a single CSV file to generate ONLY Model Consistency heatmaps.
    Axes Swapped: Y = Alpha, X = Beta
    """
    print(f"\nProcessing file: {csv_file}...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read {csv_file}: {e}")
        return None

    # Create a base name for output files (remove .csv extension)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # ★★★ 1. 提取模型名稱作為標題 ★★★
    model_name_title = extract_model_name(csv_file)
    print(f"  Found Model Name (for Title): {model_name_title}")

    # ------------------------------------------
    # Model Consistency (Self-Consistency)
    # ------------------------------------------
    score_cols = [str(i) for i in range(10)]
    existing_cols = [c for c in score_cols if c in df.columns]
    
    if len(existing_cols) >= 2:
        model_pairs = []
        for _, row in df.iterrows():
            scores = row[existing_cols].values
            valid_scores = [s for s in scores if not np.isnan(s)]
            
            # Generate pairwise combinations (nC2)
            if len(valid_scores) >= 2:
                pairs = list(itertools.combinations(valid_scores, 2))
                model_pairs.extend(pairs)
        
        if model_pairs:
            model_matrix = compute_percentage_matrix(model_pairs)
            
            # ★★★ 2. 標題使用模型名稱，但檔名維持原樣 ★★★
            plot_heatmap(model_matrix, 
                         title=model_name_title,     # 圖表標題：使用提取的模型名稱
                         xlabel="Evaluation Beta",   # X-axis
                         ylabel="Evaluation Alpha",  # Y-axis
                         dir_png=dir_png,
                         dir_pdf=dir_pdf,
                         base_filename=f"Gary2025_heatmap_model_{base_name}") # 檔名：維持原樣
        else:
            print(f"  No valid score pairs found in {csv_file}")
    else:
        print(f"  Skipping model consistency: Columns 0-9 not found in {csv_file}")
        
    return df

def main():
    # Configuration
    base_output_dir = "heatmap_results"
    dir_png = os.path.join(base_output_dir, "png")
    dir_pdf = os.path.join(base_output_dir, "pdf")
    
    # Create output directories
    for d in [dir_png, dir_pdf]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
        else:
            print(f"Directory exists: {d}")

    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    print(f"Found {len(csv_files)} CSV file(s).")
    
    doctors_plot_done = False
    
    # Process each file
    for csv_file in csv_files:
        # 1. Generate Model Consistency Heatmap for this file
        df = process_file_model_consistency(csv_file, dir_png, dir_pdf)
        
        # 2. Generate Doctors' Heatmap ONLY ONCE
        if not doctors_plot_done and df is not None:
            if generate_doctors_heatmap(df, dir_png, dir_pdf):
                doctors_plot_done = True
        
    print(f"\nAll tasks completed. Results are in '{base_output_dir}/'.")

if __name__ == "__main__":
    main()