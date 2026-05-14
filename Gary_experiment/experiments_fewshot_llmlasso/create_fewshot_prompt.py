#!/usr/bin/env python3
"""
創建 Few-Shot Prompt：使用 LLMLasso 挑選出來的動態特徵為每個分數提取範例
"""

import pandas as pd
import numpy as np
from pathlib import Path

from generate_prompt_with_features import LLMLASSO_SELECTED_FEATURES
from run_feature_extraction_with_ratings import extract_full_features

def create_fewshot_prompt(
    scored_csv="scored_by_ChatGPT_promptC_template.csv",
    csv_dir="./csv_files",
    base_prompt_path="Prompt_zeroshot.txt",
    output_prompt_path="Prompt_fewshot.txt"
):
    """
    創建包含 few-shot 範例的 prompt，使用 scored CSV 中標記的範例

    Args:
        scored_csv: 含有 example 標記欄位的 CSV 路徑
        csv_dir: 個別CSV檔案目錄
        base_prompt_path: 基礎Prompt檔案路徑
        output_prompt_path: 輸出的新Prompt檔案路徑
    """
    
    # 取得腳本所在目錄，轉換為絕對路徑，確保跨資料夾執行時不會找不到檔案
    script_dir = Path(__file__).parent
    scored_csv = script_dir / scored_csv
    csv_dir = script_dir / csv_dir
    base_prompt_path = script_dir / base_prompt_path
    output_prompt_path = script_dir / output_prompt_path

    # 1. 讀取基礎 Prompt
    print("="*80)
    print("創建 Few-Shot Prompt")
    print("="*80)
    print(f"\n讀取基礎 Prompt: {base_prompt_path}")

    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        base_prompt = f.read()

    # 2. 讀取 scored CSV，篩選標記的範例
    print(f"讀取標記範例: {scored_csv}")
    scored_df = pd.read_csv(scored_csv)
    example_rows = scored_df[scored_df['example'] == 'Y']
    print(f"找到 {len(example_rows)} 個標記範例")

    # 3. 為每個標記的範例生成文字
    examples_text = "\n\n" + "="*35 + "\n"
    examples_text += "FEW-SHOT EXAMPLES\n"
    examples_text += "="*35 + "\n"
    examples_text += "Below are examples for each score to help you understand the scoring criteria:\n"

    for _, row in example_rows.sort_values('average_score_Doctors').iterrows():
        filename = row['filename']  # includes .csv extension
        score = int(row['average_score_Doctors'])

        print(f"\n處理分數 {score}...")
        print(f"  選擇範例: {filename}")

        # 開始寫入該分數的範例
        examples_text += f"\n{'─'*15}\n"
        examples_text += f"EXAMPLE FOR SCORE {score}\n"
        examples_text += f"{'─'*15}"

        # 從個別CSV提取詳細特徵
        csv_path = Path(csv_dir) / filename

        if csv_path.exists():
            try:
                full_features = extract_full_features(csv_path)
                num_taps = full_features.get('num_peaks', 0)

                examples_text += "\nKINEMATIC FEATURES :\n"
                examples_text += f"  Number of taps: {num_taps}\n"
                
                for feat in LLMLASSO_SELECTED_FEATURES:
                    if feat in full_features:
                        val = full_features[feat]
                        if isinstance(val, float) and not np.isnan(val):
                            val = round(val, 4)
                        examples_text += f"  {feat}: {val}\n"

                print(f"  ✓ 成功提取特徵 ({num_taps} taps)")

            except Exception as e:
                print(f"  ⚠ 提取特徵時發生錯誤: {e}")
                continue
        else:
            print(f"  ⚠ CSV檔案不存在: {csv_path}")
            continue

        examples_text += f"\nThis example was rated as SCORE {score}.\n"

    # 4. 組合成完整的 Prompt
    full_prompt = base_prompt + examples_text

    # 5. 儲存新的 Prompt
    print(f"\n儲存新的 Prompt 到: {output_prompt_path}")
    with open(output_prompt_path, 'w', encoding='utf-8') as f:
        f.write(full_prompt)

    print("\n" + "="*80)
    print("✓ 完成！Few-Shot Prompt 已創建")
    print("="*80)
    print(f"輸出檔案: {output_prompt_path}")

    return full_prompt


if __name__ == "__main__":
    create_fewshot_prompt()
