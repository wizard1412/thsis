#!/usr/bin/env python3
"""
Generate Performance Summary Table for Dallen vs Gary (This Work) Comparison.
Computes: ACC, AAcc, MAE, RMSE, Pearson Corr, Weighted Kappa, WD, Stability.
Uses consensus = (Dr.Tan + Dr.Chien) / 2 as ground truth.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR          # CSV files in static/
OUTPUT_DIR = SCRIPT_DIR / 'comparison_results'
FILE_PATTERN = 'scored_by_*.csv'
N_BOOTSTRAP = 2000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ── Metric functions ───────────────────────────────────────────────────────
def weighted_kappa(y_true, y_pred, num_classes=5):
    """Quadratic weighted Cohen's kappa."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                            labels=list(range(num_classes)))


def compute_metrics(y_true, y_pred, scores_matrix=None):
    """Compute all metrics for one model."""
    # Consensus ground truth (already passed in)
    y_true_int = np.clip(np.round(y_true).astype(int), 0, 4)
    y_pred_int = np.clip(np.round(y_pred).astype(int), 0, 4)

    acc = np.mean(y_true_int == y_pred_int) * 100
    aacc = np.mean(np.abs(y_true_int - y_pred_int) <= 1) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan
    kappa = weighted_kappa(y_true_int, y_pred_int)
    wd = stats.wasserstein_distance(y_true, y_pred)

    # Stability = mean of per-session std across 10 runs
    if scores_matrix is not None:
        stab = scores_matrix.std(axis=1).mean()
    else:
        stab = np.nan

    return {
        'ACC': acc, 'AAcc': aacc,
        'MAE': mae, 'RMSE': rmse,
        'Corr': corr, 'Kappa': kappa,
        'WD': wd, 'Stability': stab
    }


def bootstrap_ci(y_true, y_pred, scores_matrix=None,
                 n_boot=2000, ci=95):
    """Bootstrap 95 % CI for every metric."""
    n = len(y_true)
    boot = {k: [] for k in ['ACC', 'AAcc', 'MAE', 'RMSE',
                              'Corr', 'Kappa', 'WD', 'Stability']}
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        sm = scores_matrix.iloc[idx] if scores_matrix is not None else None
        m = compute_metrics(y_true[idx], y_pred[idx], sm)
        for k, v in m.items():
            boot[k].append(v)

    alpha = (100 - ci) / 2
    result = {}
    point = compute_metrics(y_true, y_pred, scores_matrix)
    for k in boot:
        arr = np.array(boot[k])
        result[k] = {
            'point': point[k],
            'lo': np.nanpercentile(arr, alpha),
            'hi': np.nanpercentile(arr, 100 - alpha)
        }
    return result


# ── Process a single CSV ──────────────────────────────────────────────────
def process_file(csv_path):
    df = pd.read_csv(csv_path)

    # Score columns 0-9
    score_cols = []
    for i in range(10):
        if str(i) in df.columns:
            score_cols.append(str(i))
        elif i in df.columns:
            score_cols.append(i)
    if len(score_cols) != 10:
        print(f"  [FAIL] Cannot find 10 score columns in {csv_path.name}")
        return None

    # Doctor labels
    if 'label_Dr. Tan' not in df.columns or 'label_Dr. Chien' not in df.columns:
        print(f"  [FAIL] Missing doctor label columns in {csv_path.name}")
        return None

    dr_tan = df['label_Dr. Tan']
    dr_chien = df['label_Dr. Chien']

    # Consensus = average of two raters
    consensus = (dr_tan + dr_chien) / 2.0

    # Predicted = median of 10 runs, round & clip to 0-4
    scores_matrix = df[score_cols].apply(pd.to_numeric, errors='coerce')
    predicted = scores_matrix.median(axis=1).round().clip(0, 4)

    # Drop rows where consensus or predicted is NaN
    mask = consensus.notna() & predicted.notna()
    consensus = consensus[mask].values
    predicted = predicted[mask].values
    scores_matrix = scores_matrix[mask]

    if len(consensus) == 0:
        return None

    metrics = bootstrap_ci(consensus, predicted, scores_matrix,
                           n_boot=N_BOOTSTRAP)
    return {
        'n': len(consensus),
        'metrics': metrics,
        'consensus': consensus,
        'predicted': predicted
    }


# ── Human baseline ────────────────────────────────────────────────────────
def compute_human_baseline(csv_path):
    """Inter-rater: Dr.Tan vs Dr.Chien."""
    df = pd.read_csv(csv_path)
    dr_tan = df['label_Dr. Tan']
    dr_chien = df['label_Dr. Chien']
    mask = dr_tan.notna() & dr_chien.notna()
    t = dr_tan[mask].values.astype(float)
    c = dr_chien[mask].values.astype(float)
    return bootstrap_ci(t, c, n_boot=N_BOOTSTRAP)


# ── Formatting helpers ────────────────────────────────────────────────────
def fmt(m, dec=2, pct=False):
    p = m['point']
    if pct:
        return f"{p:.1f}%"
    return f"{p:.{dec}f}"


def fmt_ci(m, dec=2, pct=False):
    if pct:
        return f"{m['point']:.1f}% [{m['lo']:.1f}, {m['hi']:.1f}]"
    return f"{m['point']:.{dec}f} [{m['lo']:.{dec}f}, {m['hi']:.{dec}f}]"


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("Performance Summary Table - Dallen vs This Work")
    print("=" * 80)

    csv_files = sorted(INPUT_DIR.glob(FILE_PATTERN))
    if not csv_files:
        print(f"No files matching {FILE_PATTERN} in {INPUT_DIR}")
        return

    print(f"Found {len(csv_files)} CSV file(s):\n")

    results = {}
    first_csv = None  # for human baseline

    for f in csv_files:
        name = f.stem.replace('scored_by_', '')
        print(f"  Processing: {f.name}  →  {name}")
        res = process_file(f)
        if res is not None:
            results[name] = res
            if first_csv is None:
                first_csv = f
            print(f"    n={res['n']}, ACC={fmt(res['metrics']['ACC'], pct=True)}, "
                  f"MAE={fmt(res['metrics']['MAE'])}")
        else:
            print(f"    [FAIL] skipped")

    # Human baseline (use first CSV for doctor labels)
    if first_csv is not None:
        print(f"\n  Computing human baseline from {first_csv.name}")
        human = compute_human_baseline(first_csv)
        results['Human Baseline'] = {'n': '-', 'metrics': human}

    # ── Print table ────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    header = f"{'Model':<25} {'ACC':>8} {'AAcc':>8} {'MAE':>8} {'RMSE':>8} {'Corr':>8} {'Kappa':>8} {'WD':>8} {'Stab':>8}"
    print(header)
    print("-" * 120)
    for name, res in results.items():
        m = res['metrics']
        stab_str = fmt(m['Stability']) if not np.isnan(m['Stability']['point']) else '-'
        print(f"{name:<25} {fmt(m['ACC'], pct=True):>8} {fmt(m['AAcc'], pct=True):>8} "
              f"{fmt(m['MAE']):>8} {fmt(m['RMSE']):>8} {fmt(m['Corr']):>8} "
              f"{fmt(m['Kappa']):>8} {fmt(m['WD']):>8} {stab_str:>8}")
    print("=" * 120)

    # ── Print table with CI ────────────────────────────────────────────
    print("\n\nWith 95% Bootstrap CI:")
    print("=" * 180)
    for name, res in results.items():
        m = res['metrics']
        print(f"\n{name}:")
        for k in ['ACC', 'AAcc', 'MAE', 'RMSE', 'Corr', 'Kappa', 'WD', 'Stability']:
            pct = k in ('ACC', 'AAcc')
            if k == 'Stability' and np.isnan(m[k]['point']):
                print(f"  {k:<12}: -")
            else:
                print(f"  {k:<12}: {fmt_ci(m[k], pct=pct)}")

    # ── Save to CSV ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, res in results.items():
        m = res['metrics']
        rows.append({
            'Model': name,
            'n': res['n'],
            'ACC(%)': round(m['ACC']['point'], 1),
            'AAcc(%)': round(m['AAcc']['point'], 1),
            'MAE': round(m['MAE']['point'], 3),
            'RMSE': round(m['RMSE']['point'], 3),
            'Corr': round(m['Corr']['point'], 3),
            'Kappa': round(m['Kappa']['point'], 3),
            'WD': round(m['WD']['point'], 3),
            'Stability': round(m['Stability']['point'], 3) if not np.isnan(m['Stability']['point']) else ''
        })
    out_csv = OUTPUT_DIR / 'performance_summary_dallen_vs_gary.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[OK] Saved to {out_csv}")


if __name__ == '__main__':
    main()
