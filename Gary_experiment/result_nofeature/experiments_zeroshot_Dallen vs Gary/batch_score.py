#!/usr/bin/env python3
"""
完整批次評分系統
功能: 
1. 提取 CSV 特徵
2. 生成包含特徵的 Prompt
3. 呼叫 Ollama 進行評分
4. 儲存結果
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import requests
import json
import re
import time

# ===========================
# 配置
# ===========================

# Ollama 設定
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "deepseek-r1:70b"  # 你的模型

# 目錄設定
CSV_DIR = "./csv_files"      # 放你的 CSV 檔案
OUTPUT_DIR = "./results"     # 輸出結果
BASE_PROMPT_PATH = "Prompt.txt"  # 基礎 Prompt

# 每個 CSV 評估次數
EVALUATION_TIMES = 10

# 每次評估之間的延遲(秒)
DELAY_BETWEEN_CALLS = 25

# ===========================
# 特徵提取
# ===========================

def extract_features_from_csv(csv_path):
    """
    使用 data_feature.py 的邏輯提取特徵
    """
    data = pd.read_csv(csv_path)
    
    # 計算距離
    data['Distance'] = np.sqrt(
        (data['X_thumb'] - data['X_index'])**2 + 
        (data['Y_thumb'] - data['Y_index'])**2
    )
    data['Distance_Norm'] = (data['Distance'] - np.min(data['Distance'])) / \
                            (np.max(data['Distance']) - np.min(data['Distance']))
    
    # 偵測峰值
    peaks, _ = find_peaks(
        data['Distance_Norm'], 
        height=0.2,
        distance=50,
        prominence=0.15,
        width=1,
        plateau_size=[1, 40]
    )
    
    # 計算三個基本特徵
    num_taps = len(peaks)
    tap_periods = np.diff(data['Frame'].iloc[peaks])
    tap_amplitudes = data['Distance'].iloc[peaks]
    
    return num_taps, tap_periods, tap_amplitudes

# ===========================
# Prompt 生成
# ===========================

def generate_prompt_with_features(csv_path, base_prompt_path=BASE_PROMPT_PATH):
    """
    生成包含特徵的 Prompt
    """
    # 1. 讀取基礎 Prompt
    with open(base_prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # 2. 提取特徵
    num_taps, tap_periods, tap_amplitudes = extract_features_from_csv(csv_path)
    
    # 3. 準備特徵字典
    features = {
        'num_taps': int(num_taps),
        'tap_periods': tap_periods.tolist(),
        'tap_amplitudes': tap_amplitudes.tolist()
    }
    
    # 4. 將特徵添加到 prompt 後面
    feature_text = "\n#EXTRACTED FEATURES\n"
    feature_text += f"Number of taps: {features['num_taps']}\n"
    feature_text += f"Tap periods (frames): {features['tap_periods']}\n"
    feature_text += f"Tap amplitudes (pixels): {features['tap_amplitudes']}\n"
    
    enhanced_prompt = prompt + feature_text
    
    return enhanced_prompt, features

# ===========================
# LLM 呼叫
# ===========================

def call_ollama(prompt, model=MODEL):
    """
    呼叫 Ollama API
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        return result['response']
    except Exception as e:
        print(f"    Ollama 呼叫失敗: {e}")
        return None

def parse_score(response):
    """
    從 LLM 回應中提取分數
    尋找最後一行的 0-4 分數
    """
    if response is None:
        return None
    
    # 找最後一行的數字
    lines = response.strip().split('\n')
    for line in reversed(lines):
        # 尋找 0-4 的數字
        match = re.search(r'\b([0-4])\b', line)
        if match:
            return int(match.group(1))
    
    return None

# ===========================
# 儲存結果
# ===========================

def save_results(all_results, output_dir=OUTPUT_DIR):
    """
    儲存結果到檔案
    """
    # 準備可序列化的結果
    serializable_results = {}
    for csv_name, result in all_results.items():
        serializable_results[csv_name] = {
            'scores': result['scores'],
            'valid_scores': result['valid_scores'],
            'mean_score': float(result['mean_score']) if result['mean_score'] is not None else None,
            'std_score': float(result['std_score']) if result['std_score'] is not None else None,
            'mode_score': result['mode_score'],
            'features': result['features']
        }
    
    # 儲存 JSON
    output_file = Path(output_dir) / "all_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    # 儲存詳細回應
    responses_file = Path(output_dir) / "detailed_responses.txt"
    with open(responses_file, 'w', encoding='utf-8') as f:
        for csv_name, result in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{csv_name}\n")
            f.write(f"{'='*60}\n\n")
            
            for i, (score, response) in enumerate(zip(result['scores'], result['responses']), 1):
                f.write(f"--- 評估 {i} (分數: {score}) ---\n")
                f.write(response if response else "無回應")
                f.write("\n\n")
    
    return output_file, responses_file

# ===========================
# 單一 CSV 處理
# ===========================

def process_single_csv(csv_path, evaluation_num):
    """
    處理單一 CSV 的一次評估
    """
    # 1. 生成 prompt
    prompt, features = generate_prompt_with_features(csv_path)
    
    # 2. 呼叫 Ollama
    response = call_ollama(prompt)
    
    # 3. 解析分數
    score = parse_score(response)
    
    return {
        'score': score,
        'response': response,
        'features': features
    }

# ===========================
# 批次處理主流程
# ===========================

def batch_process_all():
    """
    批次處理所有 CSV 檔案
    每次評估完成後立即儲存，避免中途中斷導致資料遺失
    """
    # 建立輸出目錄
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # 找到所有 CSV
    csv_files = list(Path(CSV_DIR).glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"錯誤: 在 {CSV_DIR} 中找不到 CSV 檔案")
        return
    
    print(f"找到 {len(csv_files)} 個 CSV 檔案")
    print(f"每個檔案將評估 {EVALUATION_TIMES} 次\n")
    
    all_results = {}
    
    # 處理每個 CSV
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"\n{'='*60}")
        print(f"[{i}/{len(csv_files)}] 處理: {csv_name}")
        print(f"{'='*60}")
        
        scores = []
        responses = []
        features_list = []
        
        # 每個 CSV 評估多次
        for eval_num in range(1, EVALUATION_TIMES + 1):
            print(f"  評估 {eval_num}/{EVALUATION_TIMES}...", end=' ', flush=True)
            
            try:
                result = process_single_csv(csv_path, eval_num)
                
                scores.append(result['score'])
                responses.append(result['response'])
                features_list.append(result['features'])
                
                if result['score'] is not None:
                    print(f"分數: {result['score']}", end=' ')
                else:
                    print("分數解析失敗", end=' ')
                
            except Exception as e:
                print(f"錯誤: {e}", end=' ')
                scores.append(None)
                responses.append(str(e))
                features_list.append(None)
            
            # ★★★ 每次評估完成後立即存檔 ★★★
            # 計算當前統計
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                std_score = np.std(valid_scores)
                mode_score = max(set(valid_scores), key=valid_scores.count)
            else:
                mean_score = None
                std_score = None
                mode_score = None
            
            # 更新結果字典
            all_results[csv_name] = {
                'scores': scores,
                'valid_scores': valid_scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'mode_score': mode_score,
                'responses': responses,
                'features': features_list[0] if features_list else None
            }
            
            # 立即儲存到檔案
            try:
                save_results(all_results)
                print(f"✓", flush=True)
            except Exception as e:
                print(f"⚠儲存失敗: {e}", flush=True)
            
            # 避免太快
            if eval_num < EVALUATION_TIMES:
                time.sleep(DELAY_BETWEEN_CALLS)
        
        # 顯示摘要
        result = all_results[csv_name]
        print(f"\n  完成! 統計:")
        print(f"    有效評分: {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        if result['valid_scores']:
            print(f"    平均分數: {result['mean_score']:.2f} ± {result['std_score']:.2f}")
            print(f"    眾數分數: {result['mode_score']}")
            print(f"    所有分數: {result['valid_scores']}")
    
    # 最終儲存
    output_file, responses_file = save_results(all_results)
    # 最終儲存
    output_file, responses_file = save_results(all_results)
    
    print(f"\n{'='*60}")
    print(f"✓ 完成! 結果已儲存至: {output_file}")
    print(f"✓ 詳細回應已儲存至: {responses_file}")
    print(f"{'='*60}")
    
    # 顯示摘要表格
    print(f"\n{'='*60}")
    print("評分摘要")
    print(f"{'='*60}")
    print(f"{'檔案名稱':<40} {'平均分數':<12} {'眾數':<8} {'有效/總數'}")
    print(f"{'-'*60}")
    
    for csv_name, result in all_results.items():
        name_short = csv_name[:37] + "..." if len(csv_name) > 40 else csv_name
        if result['mean_score'] is not None:
            print(f"{name_short:<40} {result['mean_score']:>6.2f}±{result['std_score']:.2f}   "
                  f"{result['mode_score']:>4}     {len(result['valid_scores'])}/{EVALUATION_TIMES}")
        else:
            print(f"{name_short:<40} {'N/A':<12} {'N/A':<8} {len(result['valid_scores'])}/{EVALUATION_TIMES}")
    
    return all_results

# ===========================
# 執行
# ===========================

if __name__ == "__main__":
    print("="*60)
    print("手指敲擊自動評分系統")
    print("="*60)
    print(f"模型: {MODEL}")
    print(f"CSV 目錄: {CSV_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"每個檔案評估次數: {EVALUATION_TIMES}")
    print("="*60 + "\n")
    
    # 檢查基礎 Prompt 是否存在
    if not Path(BASE_PROMPT_PATH).exists():
        print(f"錯誤: 找不到 {BASE_PROMPT_PATH}")
        print("請確保 Prompt3.txt 在當前目錄")
        exit(1)
    
    # 執行批次處理
    batch_process_all()