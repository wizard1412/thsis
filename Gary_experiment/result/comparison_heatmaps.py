import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from sklearn.metrics import mean_absolute_error, cohen_kappa_score, roc_auc_score
from scipy.stats import wasserstein_distance

# Set plot style
sns.set_theme(style="white")

def extract_model_name(filename):
    """Extract model name from filename."""
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]

    if 'scored_by_' in name_no_ext:
        parts = name_no_ext.split('scored_by_', 1)
        if len(parts) > 1:
            suffix = parts[1]
            parts_suffix = suffix.split('_')
            if len(parts_suffix) >= 3:
                model_name = '_'.join(parts_suffix[:3])
            else:
                model_name = suffix
            if model_name.strip():
                return model_name
    return "Unknown_Model"

def compute_percentage_matrix(pairs, max_score=4):
    """Compute confusion matrix normalized by percentage."""
    matrix = np.zeros((max_score + 1, max_score + 1))
    valid_count = 0

    for s1, s2 in pairs:
        if not np.isnan(s1) and not np.isnan(s2):
            r, c = int(round(s1)), int(round(s2))
            if 0 <= r <= max_score and 0 <= c <= max_score:
                matrix[r, c] += 1
                valid_count += 1

    if valid_count > 0:
        matrix = (matrix / valid_count) * 100

    return matrix, valid_count

def calculate_aauc(y_true, y_pred, max_score=4):
    """
    Calculate Average AUC (AAUC) using one-vs-rest approach.
    For ordinal classification with scores 0-4.
    """
    try:
        y_true_int = [int(round(x)) for x in y_true]
        y_pred_rounded = [int(round(x)) for x in y_pred]

        aucs = []
        for threshold in range(max_score + 1):
            # Binary: >= threshold vs < threshold
            y_true_bin = [1 if y >= threshold else 0 for y in y_true_int]
            y_pred_bin = [1 if y >= threshold else 0 for y in y_pred_rounded]

            # Check if both classes present
            if len(set(y_true_bin)) > 1:
                auc = roc_auc_score(y_true_bin, y_pred_bin)
                aucs.append(auc)

        if aucs:
            return np.mean(aucs)
        return np.nan
    except:
        return np.nan

def calculate_wasserstein(y_true, y_pred):
    """
    Calculate Wasserstein Distance (Earth Mover's Distance).
    Lower is better - measures the "cost" of transforming one distribution to another.
    """
    try:
        return wasserstein_distance(y_true, y_pred)
    except:
        return np.nan

def calculate_metrics(pairs):
    """Calculate MAE, Correlation, Weighted Kappa, AAUC, and WD for a set of pairs."""
    valid_pairs = [(s1, s2) for s1, s2 in pairs if not np.isnan(s1) and not np.isnan(s2)]

    if len(valid_pairs) < 2:
        return {'mae': None, 'corr': None, 'kappa': None, 'aauc': None, 'wd': None}

    s1_vals = np.array([p[0] for p in valid_pairs])
    s2_vals = np.array([p[1] for p in valid_pairs])

    # MAE
    mae = mean_absolute_error(s1_vals, s2_vals)

    # Correlation
    corr = np.corrcoef(s1_vals, s2_vals)[0, 1]

    # Weighted Kappa (quadratic)
    s1_int = [int(round(x)) for x in s1_vals]
    s2_int = [int(round(x)) for x in s2_vals]
    try:
        kappa = cohen_kappa_score(s1_int, s2_int, weights='quadratic')
    except:
        kappa = np.nan

    # AAUC
    aauc = calculate_aauc(s1_vals, s2_vals)

    # Wasserstein Distance
    wd = calculate_wasserstein(s1_vals, s2_vals)

    return {'mae': mae, 'corr': corr, 'kappa': kappa, 'aauc': aauc, 'wd': wd}

def plot_single_heatmap(matrix, title, xlabel, ylabel, metrics_dict, ax, cmap='OrRd'):
    """Plot a single heatmap on given axes with all metrics."""
    sns.heatmap(matrix, annot=True, fmt='.1f', cmap=cmap,
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=.5, linecolor='gray',
                vmin=0, vmax=50, ax=ax)

    # Format metrics text (2 lines)
    if metrics_dict['mae'] is not None:
        line1 = f"MAE={metrics_dict['mae']:.2f}, r={metrics_dict['corr']:.2f}, κ={metrics_dict['kappa']:.2f}"
        line2 = f"AAUC={metrics_dict['aauc']:.2f}, WD={metrics_dict['wd']:.2f}"
        metrics_text = f"{line1}\n{line2}"
    else:
        metrics_text = "N/A"

    ax.set_title(f"{title}\n{metrics_text}", fontsize=9, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

def generate_comparison_figure(csv_file, dir_png, dir_pdf):
    """
    Generate a 2x2 comparison figure with:
    1. Model vs. Doc Tan
    2. Model vs. Doc Chien
    3. Model vs. Consensus
    4. Doc Tan vs. Doc Chien (Human Baseline)
    """
    print(f"\nProcessing: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"  Failed to read file: {e}")
        return None

    model_name = extract_model_name(csv_file)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Column definitions
    score_cols = [str(i) for i in range(10)]
    col_tan = 'label_Dr. Tan'
    col_chien = 'label_Dr. Chien'

    # Check required columns
    existing_score_cols = [c for c in score_cols if c in df.columns]
    if len(existing_score_cols) < 1:
        print("  Missing model score columns (0-9)")
        return None
    if col_tan not in df.columns or col_chien not in df.columns:
        print("  Missing doctor columns")
        return None

    # Calculate model mean for each row
    df['model_mean'] = df[existing_score_cols].mean(axis=1)

    # Calculate consensus (average of both doctors)
    df['consensus'] = (df[col_tan] + df[col_chien]) / 2

    # Prepare pairs for each comparison
    pairs_model_tan = list(zip(df['model_mean'], df[col_tan]))
    pairs_model_chien = list(zip(df['model_mean'], df[col_chien]))
    pairs_model_consensus = list(zip(df['model_mean'], df['consensus']))
    pairs_tan_chien = list(zip(df[col_tan], df[col_chien]))

    # Calculate matrices and metrics
    matrix_model_tan, n1 = compute_percentage_matrix(pairs_model_tan)
    matrix_model_chien, n2 = compute_percentage_matrix(pairs_model_chien)
    matrix_model_consensus, n3 = compute_percentage_matrix(pairs_model_consensus)
    matrix_tan_chien, n4 = compute_percentage_matrix(pairs_tan_chien)

    metrics1 = calculate_metrics(pairs_model_tan)
    metrics2 = calculate_metrics(pairs_model_chien)
    metrics3 = calculate_metrics(pairs_model_consensus)
    metrics4 = calculate_metrics(pairs_tan_chien)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # Plot 1: Model vs. Doc Tan
    plot_single_heatmap(matrix_model_tan,
                        f"Model vs. Dr. Tan (n={n1})",
                        "Dr. Tan", f"{model_name} (Mean)",
                        metrics1, axes[0, 0], 'Blues')

    # Plot 2: Model vs. Doc Chien
    plot_single_heatmap(matrix_model_chien,
                        f"Model vs. Dr. Chien (n={n2})",
                        "Dr. Chien", f"{model_name} (Mean)",
                        metrics2, axes[0, 1], 'Greens')

    # Plot 3: Model vs. Consensus
    plot_single_heatmap(matrix_model_consensus,
                        f"Model vs. Consensus (n={n3})",
                        "Consensus (Avg)", f"{model_name} (Mean)",
                        metrics3, axes[1, 0], 'Purples')

    # Plot 4: Doc Tan vs. Doc Chien (Human Baseline)
    plot_single_heatmap(matrix_tan_chien,
                        f"Human Baseline (n={n4})",
                        "Dr. Chien", "Dr. Tan",
                        metrics4, axes[1, 1], 'OrRd')

    plt.tight_layout()

    # Save figures
    png_path = os.path.join(dir_png, f"comparison_{base_name}.png")
    pdf_path = os.path.join(dir_pdf, f"comparison_{base_name}.pdf")

    plt.savefig(png_path, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')

    print(f"  -> Saved: {png_path}")
    print(f"  -> Saved: {pdf_path}")

    plt.close()

    # Return metrics summary
    return {
        'Model': model_name,
        # Model vs Tan
        'vs_Tan_MAE': metrics1['mae'],
        'vs_Tan_r': metrics1['corr'],
        'vs_Tan_κ': metrics1['kappa'],
        'vs_Tan_AAUC': metrics1['aauc'],
        'vs_Tan_WD': metrics1['wd'],
        # Model vs Chien
        'vs_Chien_MAE': metrics2['mae'],
        'vs_Chien_r': metrics2['corr'],
        'vs_Chien_κ': metrics2['kappa'],
        'vs_Chien_AAUC': metrics2['aauc'],
        'vs_Chien_WD': metrics2['wd'],
        # Model vs Consensus
        'vs_Consensus_MAE': metrics3['mae'],
        'vs_Consensus_r': metrics3['corr'],
        'vs_Consensus_κ': metrics3['kappa'],
        'vs_Consensus_AAUC': metrics3['aauc'],
        'vs_Consensus_WD': metrics3['wd'],
        # Human Baseline
        'Human_MAE': metrics4['mae'],
        'Human_r': metrics4['corr'],
        'Human_κ': metrics4['kappa'],
        'Human_AAUC': metrics4['aauc'],
        'Human_WD': metrics4['wd'],
    }

def generate_summary_barplot(summary_df, dir_png, dir_pdf):
    """Generate bar plots comparing all models across metrics."""
    if summary_df.empty:
        return

    models = summary_df['Model'].tolist()
    n_models = len(models)

    # Dynamic sizing based on number of models
    fig_width = max(16, 2 + n_models * 1.5)
    fig_height = max(12, 10 + n_models * 0.3)

    # Adjust bar width based on number of models
    if n_models <= 3:
        width = 0.2
    elif n_models <= 6:
        width = 0.18
    elif n_models <= 10:
        width = 0.15
    else:
        width = 0.12

    # Adjust rotation based on number of models
    rotation = 45 if n_models > 5 else 25

    fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))

    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']
    labels = ['vs Dr. Tan', 'vs Dr. Chien', 'vs Consensus', 'Human Baseline']

    x = np.arange(n_models)

    def plot_metric_bars(ax, metric_cols, ylabel, title, legend_loc='upper right'):
        for i, (col, color, label) in enumerate(zip(metric_cols, colors, labels)):
            vals = summary_df[col].values
            ax.bar(x + i*width, vals, width, label=label, color=color, alpha=0.8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x + 1.5*width)
        ax.set_xticklabels(models, rotation=rotation, ha='right', fontsize=8)
        ax.legend(loc=legend_loc, fontsize=6)
        ax.tick_params(axis='y', labelsize=8)

    # Plot 1: MAE Comparison (Lower is better)
    plot_metric_bars(axes[0, 0],
                     ['vs_Tan_MAE', 'vs_Chien_MAE', 'vs_Consensus_MAE', 'Human_MAE'],
                     'MAE (Lower is better)', 'Mean Absolute Error', 'upper right')

    # Plot 2: Correlation Comparison (Higher is better)
    plot_metric_bars(axes[0, 1],
                     ['vs_Tan_r', 'vs_Chien_r', 'vs_Consensus_r', 'Human_r'],
                     'Correlation (Higher is better)', 'Pearson Correlation', 'lower right')

    # Plot 3: Kappa Comparison (Higher is better)
    plot_metric_bars(axes[0, 2],
                     ['vs_Tan_κ', 'vs_Chien_κ', 'vs_Consensus_κ', 'Human_κ'],
                     'Weighted Kappa (Higher is better)', "Cohen's Weighted Kappa", 'lower right')

    # Plot 4: AAUC Comparison (Higher is better)
    plot_metric_bars(axes[1, 0],
                     ['vs_Tan_AAUC', 'vs_Chien_AAUC', 'vs_Consensus_AAUC', 'Human_AAUC'],
                     'AAUC (Higher is better)', 'Average AUC (One-vs-Rest)', 'lower right')

    # Plot 5: WD Comparison (Lower is better)
    plot_metric_bars(axes[1, 1],
                     ['vs_Tan_WD', 'vs_Chien_WD', 'vs_Consensus_WD', 'Human_WD'],
                     'Wasserstein Distance (Lower is better)', 'Wasserstein Distance', 'upper right')

    # Plot 6: Summary Table (Model vs Consensus only)
    ax = axes[1, 2]
    ax.axis('off')

    # Create summary table (Model vs Consensus)
    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['vs_Consensus_MAE']:.2f}",
            f"{row['vs_Consensus_r']:.2f}",
            f"{row['vs_Consensus_κ']:.2f}",
            f"{row['vs_Consensus_AAUC']:.2f}",
            f"{row['vs_Consensus_WD']:.2f}",
        ])

    # Add human baseline row
    table_data.append([
        'Human',
        f"{summary_df['Human_MAE'].iloc[0]:.2f}",
        f"{summary_df['Human_r'].iloc[0]:.2f}",
        f"{summary_df['Human_κ'].iloc[0]:.2f}",
        f"{summary_df['Human_AAUC'].iloc[0]:.2f}",
        f"{summary_df['Human_WD'].iloc[0]:.2f}",
    ])

    # Adjust table font size based on number of models
    table_fontsize = max(6, 10 - n_models * 0.3)
    row_height = max(1.2, 1.8 - n_models * 0.05)

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'MAE', 'r', 'κ', 'AAUC', 'WD'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(table_fontsize)
    table.scale(1.0, row_height)
    ax.set_title('Model vs Consensus Summary', fontsize=10, pad=10)

    plt.tight_layout()

    # Save
    png_path = os.path.join(dir_png, "summary_comparison.png")
    pdf_path = os.path.join(dir_pdf, "summary_comparison.pdf")

    plt.savefig(png_path, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')

    print(f"\nSummary bar plot saved: {png_path}")
    plt.close()

    # Generate separate detailed tables for many models
    generate_detailed_tables(summary_df, dir_png, dir_pdf)


def generate_detailed_tables(summary_df, dir_png, dir_pdf):
    """Generate detailed comparison tables as separate figures."""
    n_models = len(summary_df)

    # Calculate figure height based on number of models
    fig_height = max(8, 3 + n_models * 0.5)

    # Table 1: Model vs Consensus (all metrics)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    ax.set_title('Detailed Metrics: Model vs Consensus', fontsize=14, fontweight='bold', pad=20)

    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['vs_Consensus_MAE']:.3f}",
            f"{row['vs_Consensus_r']:.3f}",
            f"{row['vs_Consensus_κ']:.3f}",
            f"{row['vs_Consensus_AAUC']:.3f}",
            f"{row['vs_Consensus_WD']:.3f}",
        ])

    # Add human baseline
    table_data.append([
        'Human Baseline',
        f"{summary_df['Human_MAE'].iloc[0]:.3f}",
        f"{summary_df['Human_r'].iloc[0]:.3f}",
        f"{summary_df['Human_κ'].iloc[0]:.3f}",
        f"{summary_df['Human_AAUC'].iloc[0]:.3f}",
        f"{summary_df['Human_WD'].iloc[0]:.3f}",
    ])

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'MAE ↓', 'Correlation ↑', 'Kappa ↑', 'AAUC ↑', 'WD ↓'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight header row
    for j in range(6):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight human baseline row
    for j in range(6):
        table[(n_models + 1, j)].set_facecolor('#FFC000')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_png, "table_vs_consensus.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_pdf, "table_vs_consensus.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # Table 2: Model vs Dr. Tan
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    ax.set_title('Detailed Metrics: Model vs Dr. Tan', fontsize=14, fontweight='bold', pad=20)

    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['vs_Tan_MAE']:.3f}",
            f"{row['vs_Tan_r']:.3f}",
            f"{row['vs_Tan_κ']:.3f}",
            f"{row['vs_Tan_AAUC']:.3f}",
            f"{row['vs_Tan_WD']:.3f}",
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'MAE ↓', 'Correlation ↑', 'Kappa ↑', 'AAUC ↑', 'WD ↓'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for j in range(6):
        table[(0, j)].set_facecolor('#1f77b4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_png, "table_vs_tan.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_pdf, "table_vs_tan.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # Table 3: Model vs Dr. Chien
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis('off')
    ax.set_title('Detailed Metrics: Model vs Dr. Chien', fontsize=14, fontweight='bold', pad=20)

    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Model'],
            f"{row['vs_Chien_MAE']:.3f}",
            f"{row['vs_Chien_r']:.3f}",
            f"{row['vs_Chien_κ']:.3f}",
            f"{row['vs_Chien_AAUC']:.3f}",
            f"{row['vs_Chien_WD']:.3f}",
        ])

    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'MAE ↓', 'Correlation ↑', 'Kappa ↑', 'AAUC ↑', 'WD ↓'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.35, 0.13, 0.13, 0.13, 0.13, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for j in range(6):
        table[(0, j)].set_facecolor('#2ca02c')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_png, "table_vs_chien.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_pdf, "table_vs_chien.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    # Table 4: Complete comparison table (all comparisons side by side)
    fig, ax = plt.subplots(figsize=(20, fig_height + 2))
    ax.axis('off')
    ax.set_title('Complete Metrics Comparison (All Targets)', fontsize=14, fontweight='bold', pad=20)

    table_data = []
    for _, row in summary_df.iterrows():
        table_data.append([
            row['Model'],
            # vs Tan
            f"{row['vs_Tan_MAE']:.2f}",
            f"{row['vs_Tan_κ']:.2f}",
            # vs Chien
            f"{row['vs_Chien_MAE']:.2f}",
            f"{row['vs_Chien_κ']:.2f}",
            # vs Consensus
            f"{row['vs_Consensus_MAE']:.2f}",
            f"{row['vs_Consensus_κ']:.2f}",
        ])

    # Add human baseline
    table_data.append([
        'Human Baseline',
        f"{summary_df['Human_MAE'].iloc[0]:.2f}",
        f"{summary_df['Human_κ'].iloc[0]:.2f}",
        '-', '-',
        '-', '-',
    ])

    col_labels = ['Model',
                  'Tan MAE↓', 'Tan κ↑',
                  'Chien MAE↓', 'Chien κ↑',
                  'Cons MAE↓', 'Cons κ↑']

    table = ax.table(cellText=table_data,
                     colLabels=col_labels,
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.28, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Color header
    header_colors = ['#333333', '#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#9467bd', '#9467bd']
    for j, color in enumerate(header_colors):
        table[(0, j)].set_facecolor(color)
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight human baseline
    for j in range(7):
        table[(n_models + 1, j)].set_facecolor('#FFC000')

    plt.tight_layout()
    plt.savefig(os.path.join(dir_png, "table_complete.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(dir_pdf, "table_complete.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Detailed tables saved: table_vs_consensus, table_vs_tan, table_vs_chien, table_complete")

def generate_metrics_reference_figure(dir_png, dir_pdf):
    """Generate a reference figure explaining each metric's formula, range, and direction."""

    metrics_info = [
        {
            'abbrev': 'MAE',
            'name': 'Mean Absolute Error',
            'formula': r'$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$',
            'range': 'Range:  [0, +\u221e)',
            'direction': 'Lower is Better',
            'dir_symbol': '\u2193',
            'dir_color': '#d62728',
            'bg_color': '#fff5f5',
            'border_color': '#e8aaaa',
            'description': 'Average absolute difference between\npredicted and true scores.',
            'note': 'Penalizes all errors equally.',
        },
        {
            'abbrev': 'r',
            'name': 'Pearson Correlation',
            'formula': r'$r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$',
            'range': 'Range:  [\u22121, +1]',
            'direction': 'Higher is Better',
            'dir_symbol': '\u2191',
            'dir_color': '#2ca02c',
            'bg_color': '#f5fff5',
            'border_color': '#96d896',
            'description': 'Linear correlation between\npredicted and true scores.',
            'note': '1 = perfect, 0 = no linear relation.',
        },
        {
            'abbrev': '\u03ba',
            'name': "Cohen's Weighted Kappa",
            'formula': r'$\kappa = \frac{P_o - P_e}{1 - P_e}$',
            'range': 'Range:  (\u2212\u221e, +1]',
            'direction': 'Higher is Better',
            'dir_symbol': '\u2191',
            'dir_color': '#2ca02c',
            'bg_color': '#f5fff5',
            'border_color': '#96d896',
            'description': 'Agreement beyond chance; quadratic\nweights penalize larger disagreements.\n'
                           r'$P_o$=weighted observed, $P_e$=weighted expected.',
            'note': '1 = perfect, 0 = chance level.',
        },
        {
            'abbrev': 'AAUC',
            'name': 'Average AUC  (One-vs-Rest)',
            'formula': r'$\mathrm{AAUC} = \frac{1}{|T|}\sum_{t \in T} \mathrm{AUC}(y{\geq}t,\;\hat{y}{\geq}t)$',
            'range': 'Range:  [0, 1]',
            'direction': 'Higher is Better',
            'dir_symbol': '\u2191',
            'dir_color': '#2ca02c',
            'bg_color': '#f5fff5',
            'border_color': '#96d896',
            'description': 'Mean AUC across all ordinal thresholds\n(one-vs-rest binarization per threshold t).',
            'note': '1 = perfect, 0.5 = random classifier.',
        },
        {
            'abbrev': 'WD',
            'name': 'Wasserstein Distance',
            'formula': r'$\mathrm{WD}(P,Q) = \inf_{\gamma \in \Pi(P,Q)}\,E_{(x,y)\sim\gamma}[\,|x-y|\,]$',
            'range': 'Range:  [0, +\u221e)',
            'direction': 'Lower is Better',
            'dir_symbol': '\u2193',
            'dir_color': '#d62728',
            'bg_color': '#fff5f5',
            'border_color': '#e8aaaa',
            'description': "Earth Mover's Distance:\nminimum 'cost' to transform\npredicted into true distribution.",
            'note': '0 = identical distributions.',
        },
    ]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor('#f4f4f8')

    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.32,
                          left=0.03, right=0.97, top=0.92, bottom=0.04)

    card_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    for info, (row, col) in zip(metrics_info, card_positions):
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(info['bg_color'])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(info['border_color'])
            spine.set_linewidth(2.0)

        # Direction badge (upper right)
        badge_label = f"{info['dir_symbol']}  {info['direction']}"
        ax.text(0.97, 0.95, badge_label,
                transform=ax.transAxes,
                fontsize=9.5, fontweight='bold', color='white',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor=info['dir_color'], edgecolor='none', alpha=0.92))

        # Abbreviation (large, monospace)
        ax.text(0.06, 0.93, info['abbrev'],
                transform=ax.transAxes,
                fontsize=26, fontweight='bold',
                color='#1a1a1a', ha='left', va='top',
                fontfamily='monospace')

        # Full name (italic, smaller)
        ax.text(0.06, 0.74, info['name'],
                transform=ax.transAxes,
                fontsize=9, color='#555555',
                ha='left', va='top', style='italic')

        # Horizontal divider 1
        ax.axhline(y=0.66, xmin=0.04, xmax=0.96, color='#cccccc', linewidth=0.9)

        # Formula (centred)
        ax.text(0.50, 0.61, info['formula'],
                transform=ax.transAxes,
                fontsize=11.5, color='#111111',
                ha='center', va='top')

        # Range
        ax.text(0.06, 0.38, info['range'],
                transform=ax.transAxes,
                fontsize=9.5, color='#333333',
                ha='left', va='top', fontweight='bold')

        # Horizontal divider 2
        ax.axhline(y=0.31, xmin=0.04, xmax=0.96, color='#cccccc', linewidth=0.9)

        # Description
        ax.text(0.06, 0.28, info['description'],
                transform=ax.transAxes,
                fontsize=8.5, color='#333333',
                ha='left', va='top')

        # Note (footer)
        ax.text(0.06, 0.04, f"Note: {info['note']}",
                transform=ax.transAxes,
                fontsize=7.5, color='#888888',
                ha='left', va='bottom', style='italic')

    # Summary / legend card (bottom-right slot)
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_facecolor('#f0f4ff')
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    for spine in ax_leg.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#aabbee')
        spine.set_linewidth(2.0)

    ax_leg.text(0.50, 0.94, 'Direction Summary',
                transform=ax_leg.transAxes,
                fontsize=13, fontweight='bold',
                color='#1a1a1a', ha='center', va='top')

    # Green badge
    ax_leg.text(0.50, 0.78,
                '\u2191  Higher is Better',
                transform=ax_leg.transAxes,
                fontsize=13, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor='#2ca02c', edgecolor='none'))

    ax_leg.text(0.50, 0.63,
                'r        \u03ba        AAUC',
                transform=ax_leg.transAxes,
                fontsize=15, fontweight='bold',
                color='#2ca02c', ha='center', va='center')

    ax_leg.axhline(y=0.52, xmin=0.08, xmax=0.92, color='#bbbbcc', linewidth=1.2)

    # Red badge
    ax_leg.text(0.50, 0.41,
                '\u2193  Lower is Better',
                transform=ax_leg.transAxes,
                fontsize=13, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.45',
                          facecolor='#d62728', edgecolor='none'))

    ax_leg.text(0.50, 0.26,
                'MAE            WD',
                transform=ax_leg.transAxes,
                fontsize=15, fontweight='bold',
                color='#d62728', ha='center', va='center')

    ax_leg.axhline(y=0.15, xmin=0.08, xmax=0.92, color='#bbbbcc', linewidth=1.2)

    ax_leg.text(0.50, 0.08,
                'Scoring scale: 0 \u2013 4  (ordinal)',
                transform=ax_leg.transAxes,
                fontsize=9, color='#555577',
                ha='center', va='bottom', style='italic')

    # Save
    png_path = os.path.join(dir_png, "metrics_reference.png")
    pdf_path = os.path.join(dir_pdf, "metrics_reference.pdf")
    plt.savefig(png_path, dpi=300, format='png', bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f"  Metrics reference figure saved: {png_path}")
    plt.close()


def main():
    # Configuration
    base_output_dir = "comparison_results"
    dir_png = os.path.join(base_output_dir, "png")
    dir_pdf = os.path.join(base_output_dir, "pdf")

    # Create output directories
    for d in [dir_png, dir_pdf]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # Generate metrics reference figure (standalone, no data needed)
    generate_metrics_reference_figure(dir_png, dir_pdf)

    # Find all CSV files
    csv_files = glob.glob("*.csv")

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    print(f"Found {len(csv_files)} CSV file(s).")

    # Process each file and collect metrics
    summary_list = []

    for csv_file in csv_files:
        metrics = generate_comparison_figure(csv_file, dir_png, dir_pdf)
        if metrics:
            summary_list.append(metrics)

    # Generate summary comparison
    if summary_list:
        summary_df = pd.DataFrame(summary_list)

        # Print summary table
        print("\n" + "="*100)
        print("METRICS SUMMARY")
        print("="*100)

        # Print in a more readable format
        for _, row in summary_df.iterrows():
            print(f"\n{row['Model']}:")
            print(f"  vs Dr. Tan:      MAE={row['vs_Tan_MAE']:.3f}, r={row['vs_Tan_r']:.3f}, κ={row['vs_Tan_κ']:.3f}, AAUC={row['vs_Tan_AAUC']:.3f}, WD={row['vs_Tan_WD']:.3f}")
            print(f"  vs Dr. Chien:    MAE={row['vs_Chien_MAE']:.3f}, r={row['vs_Chien_r']:.3f}, κ={row['vs_Chien_κ']:.3f}, AAUC={row['vs_Chien_AAUC']:.3f}, WD={row['vs_Chien_WD']:.3f}")
            print(f"  vs Consensus:    MAE={row['vs_Consensus_MAE']:.3f}, r={row['vs_Consensus_r']:.3f}, κ={row['vs_Consensus_κ']:.3f}, AAUC={row['vs_Consensus_AAUC']:.3f}, WD={row['vs_Consensus_WD']:.3f}")

        # Print human baseline once
        print(f"\nHuman Baseline (Tan vs Chien):")
        print(f"  MAE={summary_df['Human_MAE'].iloc[0]:.3f}, r={summary_df['Human_r'].iloc[0]:.3f}, κ={summary_df['Human_κ'].iloc[0]:.3f}, AAUC={summary_df['Human_AAUC'].iloc[0]:.3f}, WD={summary_df['Human_WD'].iloc[0]:.3f}")

        # Save summary CSV
        summary_csv_path = os.path.join(base_output_dir, "metrics_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nMetrics saved to: {summary_csv_path}")

        # Generate summary bar plot
        generate_summary_barplot(summary_df, dir_png, dir_pdf)

    print(f"\nAll tasks completed. Results in '{base_output_dir}/'.")

if __name__ == "__main__":
    main()
