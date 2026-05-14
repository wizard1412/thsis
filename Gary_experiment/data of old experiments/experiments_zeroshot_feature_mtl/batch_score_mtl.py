#!/usr/bin/env python3
"""
MTL 增強的批次評分系統

在原有的 LLM zero-shot 評分流程中，注入 FTDualNet MTL 模型的預測結果，
讓 LLM 參考 ML 模型的分析來做最終判斷。

流程：
1. 從 CSV 提取動力學特徵
2. FTDualNet 預測分數 + 信心度 + 診斷機率
3. 將 MTL 分析結果注入 Prompt
4. 呼叫 LLM (Ollama) 做最終評分
5. 儲存結果
"""

import sys
import numpy as np
from pathlib import Path
import requests
import json
import re
import time
import logging
from datetime import datetime

# 加入原始模組路徑
ZEROSHOT_FEATURE_DIR = Path(__file__).parent.parent / "experiments_zeroshot_feature"
sys.path.insert(0, str(ZEROSHOT_FEATURE_DIR))

from generate_prompt_with_features import generate_prompt_with_features
from mtl_predict import load_trained_model, load_scaler, predict_single, format_mtl_output
from mtl_config import (
    OLLAMA_URL, LLM_MODEL, EVALUATION_TIMES, DELAY_BETWEEN_CALLS,
    REQUEST_TIMEOUT, MAX_RETRIES, NUM_PREDICT,
    OUTPUT_DIR, LOG_DIR, CSV_DIR, BASE_PROMPT_PATH,
)


# ===========================
# 日誌設定
# ===========================

def setup_logging():
    """設定日誌系統"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"batch_score_mtl_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"日誌檔案: {log_file}")
    return log_file


# ===========================
# LLM 呼叫
# ===========================

def call_ollama(prompt, model=LLM_MODEL, retry_count=0):
    """呼叫 Ollama API"""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": NUM_PREDICT},
    }

    start_time = time.time()

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()

        if "response" not in result:
            raise ValueError("回應格式無效")

        response_text = result["response"]
        elapsed = time.time() - start_time
        logging.info(f"LLM 回應完成，耗時 {elapsed:.1f} 秒，長度 {len(response_text)} 字元")

        if len(response_text) == 0 and retry_count < MAX_RETRIES - 1:
            logging.warning("空回應，重試中...")
            time.sleep(10)
            return call_ollama(prompt, model, retry_count + 1)

        return response_text

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logging.error(f"請求失敗: {e}")
        if retry_count < MAX_RETRIES - 1:
            time.sleep(10)
            return call_ollama(prompt, model, retry_count + 1)
        return None

    except Exception as e:
        logging.error(f"未預期的錯誤: {e}")
        if retry_count < MAX_RETRIES - 1:
            time.sleep(10)
            return call_ollama(prompt, model, retry_count + 1)
        return None


def parse_score(response):
    """從 LLM 回應中提取 0-4 分數"""
    if response is None:
        return None
    lines = response.strip().split("\n")
    for line in reversed(lines):
        match = re.search(r"\b([0-4])\b", line)
        if match:
            return int(match.group(1))
    return None


# ===========================
# 單筆 CSV 處理（含 MTL 注入）
# ===========================

def process_single_csv_with_mtl(csv_path, mtl_model, mtl_scaler, mtl_device):
    """
    處理單個 CSV：提取特徵 + MTL 預測 + 注入 Prompt + LLM 評分
    """
    logging.info(f"處理 CSV: {csv_path.name}")

    # 1. 產生原始 Prompt（含特徵）
    prompt, features = generate_prompt_with_features(csv_path, str(BASE_PROMPT_PATH))

    # 2. MTL 模型預測
    try:
        mtl_prediction = predict_single(mtl_model, csv_path, mtl_scaler, mtl_device)
        mtl_text = format_mtl_output(mtl_prediction)
        logging.info(
            f"MTL 預測: 分數={mtl_prediction['predicted_score']}, "
            f"信心度={mtl_prediction['confidence']:.2f}"
        )
    except Exception as e:
        logging.warning(f"MTL 預測失敗: {e}，僅使用原始 Prompt")
        mtl_text = ""
        mtl_prediction = None

    # 3. 注入 MTL 分析到 Prompt
    enhanced_prompt = prompt + mtl_text

    # 4. 呼叫 LLM
    response = call_ollama(enhanced_prompt)
    score = parse_score(response)

    if score is not None:
        logging.info(f"LLM 評分: {score}")
    else:
        logging.warning("分數解析失敗")

    return {
        "score": score,
        "response": response,
        "features": features,
        "mtl_prediction": mtl_prediction,
    }


# ===========================
# 儲存結果
# ===========================

def save_results(all_results, output_dir=None):
    """儲存結果到檔案"""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    serializable = {}
    for csv_name, result in all_results.items():
        mtl_pred = result.get("mtl_prediction")
        mtl_info = None
        if mtl_pred:
            mtl_info = {
                "predicted_score": mtl_pred["predicted_score"],
                "score_probs": mtl_pred["score_probs"],
                "confidence": float(mtl_pred["confidence"]),
                "diag_prob": float(mtl_pred["diag_prob"]),
            }

        serializable[csv_name] = {
            "scores": result["scores"],
            "valid_scores": result["valid_scores"],
            "mean_score": float(result["mean_score"]) if result["mean_score"] is not None else None,
            "std_score": float(result["std_score"]) if result["std_score"] is not None else None,
            "mode_score": result["mode_score"],
            "mtl_prediction": mtl_info,
            "features": result["features"],
        }

    output_file = Path(output_dir) / "all_results_mtl.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return output_file


# ===========================
# 批次處理主流程
# ===========================

def batch_process_all():
    """批次處理所有 CSV 檔案（含 MTL 增強）"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 載入 MTL 模型
    logging.info("載入 FTDualNet 模型...")
    try:
        mtl_model, mtl_device = load_trained_model()
        mtl_scaler = load_scaler()
        logging.info("MTL 模型載入成功")
        mtl_enabled = True
    except Exception as e:
        logging.warning(f"MTL 模型載入失敗: {e}")
        logging.warning("將使用不含 MTL 的原始模式")
        mtl_model = mtl_scaler = mtl_device = None
        mtl_enabled = False

    # 尋找 CSV 檔案
    csv_files = list(Path(CSV_DIR).glob("*.csv"))
    if not csv_files:
        logging.error(f"在 {CSV_DIR} 中找不到 CSV 檔案")
        return

    logging.info(f"找到 {len(csv_files)} 個 CSV 檔案")
    print(f"找到 {len(csv_files)} 個 CSV 檔案")
    print(f"每個檔案評估 {EVALUATION_TIMES} 次")
    print(f"MTL 增強: {'啟用' if mtl_enabled else '停用'}\n")

    all_results = {}

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"\n{'='*60}")
        print(f"[{i}/{len(csv_files)}] 處理: {csv_name}")
        print(f"{'='*60}")

        scores = []
        responses = []
        mtl_prediction = None

        for eval_num in range(1, EVALUATION_TIMES + 1):
            print(f"  評估 {eval_num}/{EVALUATION_TIMES}...", end=" ", flush=True)

            try:
                if mtl_enabled:
                    result = process_single_csv_with_mtl(
                        csv_path, mtl_model, mtl_scaler, mtl_device
                    )
                    if mtl_prediction is None:
                        mtl_prediction = result["mtl_prediction"]
                else:
                    prompt, features = generate_prompt_with_features(
                        csv_path, str(BASE_PROMPT_PATH)
                    )
                    response = call_ollama(prompt)
                    result = {
                        "score": parse_score(response),
                        "response": response,
                        "features": features,
                    }

                scores.append(result["score"])
                responses.append(result["response"])

                if result["score"] is not None:
                    print(f"分數: {result['score']}", end=" ")
                else:
                    print("解析失敗", end=" ")

            except Exception as e:
                logging.error(f"評估 {eval_num} 失敗: {e}")
                scores.append(None)
                responses.append(str(e))

            # 即時儲存
            valid_scores = [s for s in scores if s is not None]
            all_results[csv_name] = {
                "scores": scores,
                "valid_scores": valid_scores,
                "mean_score": np.mean(valid_scores) if valid_scores else None,
                "std_score": np.std(valid_scores) if valid_scores else None,
                "mode_score": max(set(valid_scores), key=valid_scores.count) if valid_scores else None,
                "responses": responses,
                "features": result.get("features"),
                "mtl_prediction": mtl_prediction,
            }

            try:
                save_results(all_results)
                print("v", flush=True)
            except Exception as e:
                print(f"! 儲存失敗: {e}", flush=True)

            if eval_num < EVALUATION_TIMES:
                time.sleep(DELAY_BETWEEN_CALLS)

        # 摘要
        r = all_results[csv_name]
        print(f"\n  完成！統計:")
        print(f"    有效分數: {len(r['valid_scores'])}/{EVALUATION_TIMES}")
        if r["valid_scores"]:
            print(f"    平均分數: {r['mean_score']:.2f} +/- {r['std_score']:.2f}")
            print(f"    眾數: {r['mode_score']}")
        if mtl_prediction:
            print(f"    MTL 預測: {mtl_prediction['predicted_score']} (信心度 {mtl_prediction['confidence']:.2f})")

    # 最終儲存
    output_file = save_results(all_results)
    print(f"\n{'='*60}")
    print(f"完成！結果已儲存至: {output_file}")
    print(f"{'='*60}")


# ===========================
# 主程式
# ===========================

if __name__ == "__main__":
    log_file = setup_logging()

    print("=" * 60)
    print("FTDualNet MTL 增強手指敲擊自動評分系統")
    print("=" * 60)
    print(f"LLM 模型: {LLM_MODEL}")
    print(f"CSV 目錄: {CSV_DIR}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print(f"每檔評估次數: {EVALUATION_TIMES}")
    print(f"日誌檔案: {log_file}")
    print("=" * 60 + "\n")

    # 檢查 Ollama 連線
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        print("Ollama 連線正常\n")
    except Exception as e:
        print(f"錯誤：無法連線到 Ollama: {e}")
        print("請確認 Ollama 已啟動")
        exit(1)

    batch_process_all()
