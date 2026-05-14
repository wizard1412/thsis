#!/usr/bin/env python3
"""
特徵提取系統 - 簡化版 (僅 Denoised 版本)
包含特徵提取核心邏輯與醫生評分整合

特徵類別:
1. 頻率特徵 (Frequency)
2. 週期特徵 (Period)
3. 速度特徵 (Speed)
4. 加速度特徵 (Acceleration)
5. 韻律特徵 (Rhythm)
6. 手指移動特徵 (Finger Movement)

注意: 振幅特徵已移除 (因角度計算方式與原始 ROC-HCI 不同)
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy
from pathlib import Path

# ===========================
# 配置
# ===========================

# CSV 檔案目錄
CSV_DIR = "./csv_files"

# 輸出檔案路徑
OUTPUT_PATH = "./features_dataset.csv"

# 醫生評分檔案路徑 (可選)
RATINGS_FILE = Path(__file__).parent / "scored_by_ChatGPT_promptC_template.csv"

# 如果沒有評分檔案，設為 None
# RATINGS_FILE = None

# ===========================
# 常數設定
# ===========================

FPS = 240  # 影片幀率
SPEED_THRESHOLD = 50  # 中斷判定閾值 (degrees/sec)
INTERRUPTION_DURATION = 0.20  # 中斷判定時長 (秒) - 與原始 ROC-HCI 一致
FREEZE_DURATION = 0.30  # 凍結判定時長 (秒)


# ===========================
# 輔助函數
# ===========================

def safe_stats(arr):
    """安全計算統計量，處理空陣列"""
    if len(arr) == 0:
        return {
            'median': np.nan,
            'quartile_range': np.nan,
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan,
            'stdev': np.nan
        }
    return {
        'median': np.median(arr),
        'quartile_range': np.percentile(arr, 75) - np.percentile(arr, 25),
        'mean': np.mean(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'stdev': np.std(arr)
    }


def calculate_entropy(arr, bins=10, range_vals=None):
    """計算分佈熵"""
    if len(arr) == 0:
        return np.nan
    hist, _ = np.histogram(arr, bins=bins, range=range_vals, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0
    return entropy(hist)


def denoise_signal(signal, window=11, polyorder=3):
    """使用 Savitzky-Golay 濾波器去噪"""
    if len(signal) < window:
        return signal
    return savgol_filter(signal, window, polyorder)


# ===========================
# 主要特徵提取函數
# ===========================

def extract_full_features(csv_path, fps=FPS):
    """
    從 CSV 提取完整特徵集 (僅 Denoised 版本)

    Args:
        csv_path: CSV 檔案路徑
        fps: 影片幀率 (預設 240)

    Returns:
        features: 特徵字典
    """
    # 讀取資料
    data = pd.read_csv(csv_path)

    # ===========================
    # 1. 基礎距離計算
    # ===========================

    # 計算手指間距離 (像素)
    data['Distance'] = np.sqrt(
        (data['X_thumb'] - data['X_index'])**2 +
        (data['Y_thumb'] - data['Y_index'])**2
    )

    # 正規化距離
    dist_min = data['Distance'].min()
    dist_max = data['Distance'].max()
    dist_range = dist_max - dist_min if dist_max > dist_min else 1
    data['Distance_Norm'] = (data['Distance'] - dist_min) / dist_range

    # 計算角度 (將像素距離轉換為近似角度，假設最大張開約 90 度)
    max_distance = data['Distance'].max()
    data['Angle'] = (data['Distance'] / max_distance) * 90

    # ===========================
    # 2. 去噪訊號
    # ===========================

    distance_denoised = denoise_signal(data['Distance'].values)
    angle_denoised = denoise_signal(data['Angle'].values)

    # ===========================
    # 3. 速度計算 (degrees/second)
    # ===========================

    dt = 1.0 / fps
    # 保留方向的速度 (與原始 ROC-HCI 一致)
    speed_raw = np.gradient(angle_denoised, dt)
    speed = speed_raw  # 統計時才取絕對值

    # ===========================
    # 4. 加速度計算 (degrees/second²)
    # ===========================

    acceleration = np.gradient(speed, dt)

    # ===========================
    # 5. 峰值檢測 (Taps)
    # ===========================

    distance_norm = (distance_denoised - distance_denoised.min()) / (distance_denoised.max() - distance_denoised.min() + 1e-10)

    peaks, _ = find_peaks(
        distance_norm,
        height=0.2,
        distance=int(fps * 0.15),  # 最小間隔 0.15 秒 (與原始 ROC-HCI 一致)
        prominence=0.15,
        width=1
    )

    num_peaks = len(peaks)

    # ===========================
    # 6. 週期/頻率特徵
    # ===========================

    if num_peaks > 1:
        periods_frames = np.diff(peaks)
        periods_seconds = periods_frames / fps
        frequencies = 1.0 / periods_seconds
        frequencies = frequencies[np.isfinite(frequencies)]
    else:
        periods_seconds = np.array([])
        frequencies = np.array([])

    period_stats = safe_stats(periods_seconds)
    frequency_stats = safe_stats(frequencies)

    # 週期變異 (正規化)
    if len(periods_seconds) > 0 and np.max(periods_seconds) > 0:
        periodVarianceNorm = np.var(periods_seconds) / np.max(periods_seconds)
    else:
        periodVarianceNorm = np.nan

    # 週期熵
    periodEntropy = calculate_entropy(periods_seconds, bins=50, range_vals=(0, 2))

    # 頻率趨勢 (線性迴歸)
    if len(frequencies) >= 3:
        x = np.arange(len(frequencies))
        slope, intercept = np.polyfit(x, frequencies, 1)
        frequency_lr_slope = slope

        y_pred = slope * x + intercept
        ss_res = np.sum((frequencies - y_pred) ** 2)
        ss_tot = np.sum((frequencies - np.mean(frequencies)) ** 2)
        frequency_lr_fitness_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        frequency_fit_min_degree = 1
    else:
        frequency_lr_slope = np.nan
        frequency_lr_fitness_r2 = np.nan
        frequency_fit_min_degree = np.nan

    # ===========================
    # 8. 韻律特徵 (Rhythm)
    # ===========================

    # 非週期性 (aperiodicity) - FFT 功率譜熵 (使用完整頻譜，與原始一致)
    if len(distance_denoised) > 10:
        fft_result = np.fft.fft(distance_denoised)
        power_spectrum = np.abs(fft_result) ** 2
        power_spectrum_norm = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else power_spectrum
        aperiodicity = entropy(power_spectrum_norm[power_spectrum_norm > 0])
    else:
        aperiodicity = np.nan

    # 中斷計數 (速度低於閾值且持續超過 INTERRUPTION_DURATION)
    abs_speed = np.abs(speed)
    interruption_frames = int(INTERRUPTION_DURATION * fps)
    numInterruptions = 0
    current_low_speed_len = 0

    for i in range(len(abs_speed)):
        if abs_speed[i] <= SPEED_THRESHOLD:
            current_low_speed_len += 1
        else:
            if current_low_speed_len >= interruption_frames:
                numInterruptions += 1
            current_low_speed_len = 0

    # 凍結計數 (持續低速超過 FREEZE_DURATION)
    freeze_frames = int(FREEZE_DURATION * fps)
    numFreeze = 0
    maxFreezeDuration = 0
    current_freeze_len = 0

    for i in range(len(abs_speed)):
        if abs_speed[i] <= SPEED_THRESHOLD:
            current_freeze_len += 1
            maxFreezeDuration = max(maxFreezeDuration, current_freeze_len)
        else:
            if current_freeze_len >= freeze_frames:
                numFreeze += 1
            current_freeze_len = 0

    if current_freeze_len >= freeze_frames:
        numFreeze += 1

    maxFreezeDuration = maxFreezeDuration / fps

    # 正規化中斷與凍結數量
    num_interruptions_norm = numInterruptions / num_peaks if num_peaks > 0 else np.nan
    num_freeze_norm = numFreeze / num_peaks if num_peaks > 0 else np.nan

    # ===========================
    # 9. 速度統計 (統計時取絕對值)
    # ===========================

    speed_stats = safe_stats(np.abs(speed))

    # ===========================
    # 10. 加速度統計 (統計時取絕對值)
    # ===========================

    accel_stats = safe_stats(np.abs(acceleration))

    # ===========================
    # 11. 手指移動特徵
    # ===========================

    thumb_mvmnt_x = np.abs(np.diff(data['X_thumb'].values))
    thumb_mvmnt_y = np.abs(np.diff(data['Y_thumb'].values))
    thumb_mvmnt_dist = np.sqrt(thumb_mvmnt_x**2 + thumb_mvmnt_y**2)

    thumb_mvmnt_x_stats = safe_stats(thumb_mvmnt_x)
    thumb_mvmnt_y_stats = safe_stats(thumb_mvmnt_y)
    thumb_mvmnt_dist_stats = safe_stats(thumb_mvmnt_dist)

    # ===========================
    # 組合所有特徵
    # ===========================

    features = {
        # 手指移動特徵
        'finger_mvmnt_x_median': thumb_mvmnt_x_stats['median'],
        'finger_mvmnt_x_quartile_range': thumb_mvmnt_x_stats['quartile_range'],
        'finger_mvmnt_x_mean': thumb_mvmnt_x_stats['mean'],
        'finger_mvmnt_x_min': thumb_mvmnt_x_stats['min'],
        'finger_mvmnt_x_max': thumb_mvmnt_x_stats['max'],
        'finger_mvmnt_x_stdev': thumb_mvmnt_x_stats['stdev'],

        'finger_mvmnt_y_median': thumb_mvmnt_y_stats['median'],
        'finger_mvmnt_y_quartile_range': thumb_mvmnt_y_stats['quartile_range'],
        'finger_mvmnt_y_mean': thumb_mvmnt_y_stats['mean'],
        'finger_mvmnt_y_min': thumb_mvmnt_y_stats['min'],
        'finger_mvmnt_y_max': thumb_mvmnt_y_stats['max'],
        'finger_mvmnt_y_stdev': thumb_mvmnt_y_stats['stdev'],

        'finger_mvmnt_dist_median': thumb_mvmnt_dist_stats['median'],
        'finger_mvmnt_dist_quartile_range': thumb_mvmnt_dist_stats['quartile_range'],
        'finger_mvmnt_dist_mean': thumb_mvmnt_dist_stats['mean'],
        'finger_mvmnt_dist_min': thumb_mvmnt_dist_stats['min'],
        'finger_mvmnt_dist_max': thumb_mvmnt_dist_stats['max'],
        'finger_mvmnt_dist_stdev': thumb_mvmnt_dist_stats['stdev'],

        # 韻律特徵
        'aperiodicity': aperiodicity,
        'periodEntropy': periodEntropy,
        'periodVarianceNorm': periodVarianceNorm,
        'numInterruptions': numInterruptions,
        'numFreeze': numFreeze,
        'maxFreezeDuration': maxFreezeDuration,

        # 週期特徵
        'period_median': period_stats['median'],
        'period_quartile_range': period_stats['quartile_range'],
        'period_mean': period_stats['mean'],
        'period_min': period_stats['min'],
        'period_max': period_stats['max'],
        'period_stdev': period_stats['stdev'],

        # 頻率特徵
        'frequency_median': frequency_stats['median'],
        'frequency_quartile_range': frequency_stats['quartile_range'],
        'frequency_mean': frequency_stats['mean'],
        'frequency_min': frequency_stats['min'],
        'frequency_max': frequency_stats['max'],
        'frequency_stdev': frequency_stats['stdev'],
        'frequency_lr_fitness_r2': frequency_lr_fitness_r2,
        'frequency_lr_slope': frequency_lr_slope,
        'frequency_fit_min_degree': frequency_fit_min_degree,

        # 峰值數量
        'num_peaks': num_peaks,

        # 正規化中斷與凍結
        'num_interruptions_norm': num_interruptions_norm,
        'num_freeze_norm': num_freeze_norm,

        # 速度特徵
        'speed_median': speed_stats['median'],
        'speed_quartile_range': speed_stats['quartile_range'],
        'speed_mean': speed_stats['mean'],
        'speed_min': speed_stats['min'],
        'speed_max': speed_stats['max'],
        'speed_stdev': speed_stats['stdev'],

        # 加速度特徵
        'acceleration_median': accel_stats['median'],
        'acceleration_quartile_range': accel_stats['quartile_range'],
        'acceleration_mean': accel_stats['mean'],
        'acceleration_min': accel_stats['min'],
        'acceleration_max': accel_stats['max'],
        'acceleration_stdev': accel_stats['stdev'],

        # 元數據
        'hand': '',
        'filename': '',

        # 醫生評分
        'Rating1': np.nan,
        'Rating2': np.nan,
        'Rating': np.nan,
    }

    return features


def extract_features_summary(features):
    """
    生成特徵摘要文字 (用於 LLM Prompt)
    """
    summary = []
    summary.append("=" * 50)
    summary.append("EXTRACTED KINEMATIC FEATURES")
    summary.append("=" * 50)

    summary.append(f"\n[Basic Info]")
    summary.append(f"  Number of taps: {features['num_peaks']}")

    summary.append(f"\n[Frequency (Hz)]")
    if not np.isnan(features['frequency_mean']):
        summary.append(f"  Mean: {features['frequency_mean']:.2f}")
        summary.append(f"  Std: {features['frequency_stdev']:.2f}")
    if not np.isnan(features['frequency_lr_slope']):
        summary.append(f"  Trend (slope): {features['frequency_lr_slope']:.4f}")

    summary.append(f"\n[Speed (degrees/sec)]")
    summary.append(f"  Mean: {features['speed_mean']:.2f}")
    summary.append(f"  Max: {features['speed_max']:.2f}")

    summary.append(f"\n[Rhythm]")
    if not np.isnan(features['periodVarianceNorm']):
        summary.append(f"  Period variance (norm): {features['periodVarianceNorm']:.4f}")
    if not np.isnan(features['aperiodicity']):
        summary.append(f"  Aperiodicity: {features['aperiodicity']:.4f}")
    summary.append(f"  Interruptions: {features['numInterruptions']}")
    summary.append(f"  Freezes: {features['numFreeze']}")

    return "\n".join(summary)


# ===========================
# 批次處理
# ===========================

def batch_extract_features(csv_dir="./csv_files", output_path="./features_dataset.csv", ratings_file=None):
    """
    批次提取所有 CSV 的特徵並儲存為資料集
    """
    csv_files = list(Path(csv_dir).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    ratings_df = None
    rating_column_map = {}
    if ratings_file and Path(ratings_file).exists():
        ratings_df = pd.read_csv(ratings_file)
        print(f"Loaded ratings from: {ratings_file}")

        if 'label_Dr. Tan' in ratings_df.columns:
            rating_column_map['Rating1'] = 'label_Dr. Tan'
        if 'label_Dr. Chien' in ratings_df.columns:
            rating_column_map['Rating2'] = 'label_Dr. Chien'
        if 'average_score_Doctors' in ratings_df.columns:
            rating_column_map['Rating'] = 'average_score_Doctors'
        for col in ['Rating1', 'Rating2', 'Rating']:
            if col in ratings_df.columns:
                rating_column_map[col] = col

        if rating_column_map:
            print(f"Detected rating columns: {list(rating_column_map.keys())}\n")

    all_features = []

    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] Processing: {csv_name}...", end=" ")

        try:
            features = extract_full_features(csv_path)
            features['filename'] = csv_name

            # 從檔名提取手的資訊 (格式: XXXXX_YYYYMMDD_Hands_L/R_XX_相機位置.mp4FT)
            # 第一個 L/R 是手，最後的 L/R 是相機位置
            if '_Hands_L_' in csv_name:
                features['hand'] = 'L'
            elif '_Hands_R_' in csv_name:
                features['hand'] = 'R'
            else:
                features['hand'] = ''

            if ratings_df is not None:
                matching_row = ratings_df[
                    (ratings_df['filename'] == csv_name) |
                    (ratings_df['filename'] == csv_name + '.csv')
                ]
                if not matching_row.empty:
                    row = matching_row.iloc[0]
                    for target_col, source_col in rating_column_map.items():
                        if source_col in row:
                            value = row[source_col]
                            if pd.notna(value) and value != '':
                                features[target_col] = value

            all_features.append(features)
            print(f"OK ({features['num_peaks']} taps, hand: {features['hand']})")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    print(f"  Shape: {df.shape}")

    return df


# ===========================
# 執行特徵提取
# ===========================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("特徵提取系統 (Denoised Only)")
    print("=" * 60 + "\n")

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Extracting features from: {csv_file}\n")
        features = extract_full_features(csv_file)
        summary = extract_features_summary(features)
        print(summary)
        print(f"\n\nTotal features extracted: {len(features)}")
    else:
        ratings_file = RATINGS_FILE
        if ratings_file and not Path(ratings_file).exists():
            print(f"Warning: Ratings file not found: {ratings_file}")
            print("Proceeding without ratings...\n")
            ratings_file = None

        df = batch_extract_features(
            csv_dir=CSV_DIR,
            output_path=OUTPUT_PATH,
            ratings_file=ratings_file
        )

        print("\n" + "=" * 60)
        print("完成！")
        print("=" * 60)
        print(f"\n總共處理檔案數: {len(df)}")
        print(f"特徵數量: {len(df.columns)}")

        if 'Rating1' in df.columns:
            print(f"有 Rating1 的檔案數: {df['Rating1'].notna().sum()}")
        if 'Rating2' in df.columns:
            print(f"有 Rating2 的檔案數: {df['Rating2'].notna().sum()}")

        print(f"\n特徵欄位:")
        for i, col in enumerate(df.columns, 1):
            if col not in ['filename', 'hand', 'Rating1', 'Rating2', 'Rating']:
                print(f"  {i:2d}. {col}")

        print(f"\n結果已儲存至: {OUTPUT_PATH}")
