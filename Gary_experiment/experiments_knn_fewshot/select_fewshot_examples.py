#!/usr/bin/env python3
"""
Stratified kNN Few-Shot Example Selector
==========================================
對每個受試者，從資料集中為每個分數等級（0-3）各選出
特徵最相似（歐氏距離最近）的樣本，並輸出選擇結果。

執行方式:
  python select_fewshot_examples.py              # 印出全部受試者的選例結果
  python select_fewshot_examples.py --save       # 額外儲存成 CSV
  python select_fewshot_examples.py --subject 553661824_20200629_Hands_R_7-_L.mp4FT
                                                 # 只看某一個受試者
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from run_feature_extraction_with_ratings import extract_full_features

# ===========================
# 設定
# ===========================
CSV_DIR      = "../csv_files"
TEMPLATE_CSV = "scored_by_ChatGPT_promptC_template.csv"

LLMLASSO_FEATURES = [
    "finger_mvmnt_x_mean",
    "finger_mvmnt_x_max",
    "periodEntropy",
    "period_quartile_range",
    "period_min",
    "num_peaks",
]


# ===========================
# 資料集建立
# ===========================

def build_dataset(csv_dir=CSV_DIR, template_csv=TEMPLATE_CSV):
    tpl = pd.read_csv(template_csv, encoding='utf-8-sig')

    label_map = {}
    for _, row in tpl.iterrows():
        fname = str(row.get('filename', '')).strip().replace('.csv', '')
        if not fname:
            continue
        for col in ['average_score_Doctors', 'Rating']:
            val = row.get(col, None)
            if val is not None and pd.notna(val) and val != '':
                try:
                    label_map[fname] = min(int(np.round(float(val))), 3)
                except (ValueError, TypeError):
                    pass
                break

    rows = []
    all_csv_stems = set()
    for csv_path in sorted(Path(csv_dir).glob("*.csv")):
        stem = csv_path.stem
        all_csv_stems.add(stem)
        if stem not in label_map:
            print(f"  [SKIP] no label: {stem}")
            continue
        try:
            feat = extract_full_features(csv_path)
            feat['filename'] = stem
            feat['num_taps'] = feat.get('num_peaks', 0)
            feat['label']    = label_map[stem]
            rows.append(feat)
        except Exception as e:
            print(f"  [Warning] {stem}: {e}")

    df_raw = pd.DataFrame(rows)
    df = df_raw.dropna(subset=LLMLASSO_FEATURES + ['label'])

    # 顯示因 NaN 被移除的受試者
    dropped = df_raw[~df_raw.index.isin(df.index)]
    for _, r in dropped.iterrows():
        nan_feats = [f for f in LLMLASSO_FEATURES if pd.isna(r.get(f))]
        print(f"  [DROP] NaN features: {r['filename']} — {nan_feats}")

    df['label'] = df['label'].astype(int)
    return df


# ===========================
# kNN 選例
# ===========================

def select_examples(test_row, all_df, scaler, exclude_filename=None):
    """
    回傳 list of dict，每個分數等級一筆：
      score, filename, distance, feature_values
    """
    df = all_df.copy()
    if exclude_filename:
        df = df[df['filename'] != exclude_filename]

    X        = scaler.transform(df[LLMLASSO_FEATURES].values)
    test_raw = np.array([test_row.get(f, np.nan) for f in LLMLASSO_FEATURES], dtype=float)
    test_raw = np.nan_to_num(test_raw)
    test_vec = scaler.transform([test_raw])[0]

    results = []
    for score in sorted(df['label'].unique()):
        mask      = (df['label'] == score).values
        subset_X  = X[mask]
        subset_df = df[mask].reset_index(drop=True)

        dists  = np.linalg.norm(subset_X - test_vec, axis=1)
        best_i = int(np.argmin(dists))
        row    = subset_df.iloc[best_i]

        results.append({
            'score'    : score,
            'filename' : row['filename'],
            'distance' : round(float(dists[best_i]), 4),
            **{f: round(float(row[f]), 4) for f in LLMLASSO_FEATURES},
        })

    return results


# ===========================
# 主程式
# ===========================

def print_candidates(df):
    """列出每個分數等級的全部候選樣本及其特徵值"""
    print("=" * 70)
    print("CANDIDATES PER SCORE LEVEL")
    print("=" * 70)
    for score in sorted(df['label'].unique()):
        subset = df[df['label'] == score].reset_index(drop=True)
        print(f"\n[Score {score}]  ({len(subset)} subjects)")
        print(f"  {'filename':<45} " + "  ".join(f"{f[:10]:<10}" for f in LLMLASSO_FEATURES))
        print(f"  {'-'*45} " + "  ".join("-"*10 for _ in LLMLASSO_FEATURES))
        for _, row in subset.iterrows():
            vals = "  ".join(f"{round(float(row[f]),3):<10}" for f in LLMLASSO_FEATURES)
            print(f"  {row['filename'][:45]:<45} {vals}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', '-s', default=None,
                        help='只顯示特定受試者（stem 名稱）')
    parser.add_argument('--save', action='store_true',
                        help='將結果儲存成 knn_selection_results.csv')
    parser.add_argument('--candidates', '-c', action='store_true',
                        help='列出每個分數等級的全部候選樣本')
    args = parser.parse_args()

    print("Building dataset...")
    df     = build_dataset()
    scaler = StandardScaler().fit(df[LLMLASSO_FEATURES].values)
    print(f"  {len(df)} subjects loaded (labels: {sorted(df['label'].unique())})\n")

    if args.candidates:
        print_candidates(df)
        return

    subjects = df[df['filename'] == args.subject] if args.subject else df

    all_rows = []
    for _, test_row in subjects.iterrows():
        subj = test_row['filename']
        label = int(test_row['label'])

        examples = select_examples(test_row, df, scaler, exclude_filename=subj)

        print(f"{'='*60}")
        print(f"Subject : {subj}  (label={label})")
        print(f"{'-'*60}")
        header = f"{'Score':<6} {'Distance':<10} " + "  ".join(f"{f[:14]:<14}" for f in LLMLASSO_FEATURES)
        print(header)
        print('─' * len(header))
        for ex in examples:
            vals = "  ".join(f"{ex[f]:<14}" for f in LLMLASSO_FEATURES)
            marker = " <-- target" if ex['score'] == label else ""
            print(f"  {ex['score']:<4} {ex['distance']:<10} {vals}  {ex['filename'][:30]}{marker}")
        print()

        # 順序是否單調（entropy 和 period_min 應隨 score 增加；amplitude 應隨 score 減少）
        entropies  = [ex['periodEntropy'] for ex in examples]
        amplitudes = [ex['finger_mvmnt_x_mean'] for ex in examples]
        mono_e = all(entropies[i] <= entropies[i+1] for i in range(len(entropies)-1))
        mono_a = all(amplitudes[i] >= amplitudes[i+1] for i in range(len(amplitudes)-1))
        print(f"  periodEntropy monotone up  : {'OK' if mono_e else 'NG'}  ({[round(v,3) for v in entropies]})")
        print(f"  amplitude     monotone down: {'OK' if mono_a else 'NG'}  ({[round(v,3) for v in amplitudes]})")
        print()

        for ex in examples:
            all_rows.append({'subject': subj, 'subject_label': label, **ex})

    if args.save:
        out = pd.DataFrame(all_rows)
        out.to_csv('knn_selection_results.csv', index=False)
        print(f"Saved to knn_selection_results.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
