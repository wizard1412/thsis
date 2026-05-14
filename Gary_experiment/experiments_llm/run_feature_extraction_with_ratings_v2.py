#!/usr/bin/env python3
"""
特徵提取 v2 - 含進階 MDS-UPDRS 3.4 特徵

相對於 run_feature_extraction_with_ratings.py 的差異:
  1. Interrupt/Freeze 改為 ITI-based 判定（符合 MDS-UPDRS 臨床定義）
  2. 手指移動特徵拆分為 thumb / index，並加上明確前綴
  3. 速度特徵分為 pixel-based (speed_*) 與 Norm_Distance-based (speed_norm_*)
  4. 新增振幅正規化特徵: amplitude_norm_*
  5. Phase 特徵加入正規化速度: V_open_norm_*, V_close_norm_*
  6. 新增第一波高優先特徵:
     - Amplitude decrement: A_early, A_late, R_A_3_3, D_A_pct, D_A_max, k_collapse
     - Re-normalized amplitude: tilde_A_median, beta_tilde_A, CV_tilde_A
     - ITI rhythm: I_ITI, I_ITI_norm, amplitude_MAD/skew/kurtosis, ITI_MAD/skew/kurtosis
     - Halt: S_halt, F_halt
     - Task completion: N_valid, C_10, T_10, F_incomplete
     - Opening/Closing phase: V_open/V_close/T_open/T_close/R_OC + R_V_open/R_V_close
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy, skew as scipy_skew, kurtosis as scipy_kurtosis
from pathlib import Path

# 共用 helper 從原始模組 import
from run_feature_extraction_with_ratings import (
    safe_stats,
    calculate_entropy,
    denoise_signal,
    FPS,
)

# ITI-based 中斷/凍結判定閾值（符合 MDS-UPDRS 臨床定義）
INTERRUPTION_ITI_MULTIPLIER = 2.0  # ITI > 2× median ITI = 中斷
FREEZE_ITI_MULTIPLIER = 3.0        # ITI > 3× median ITI = 凍結

# ===========================
# 配置
# ===========================

CSV_DIR      = "./csv_files"
OUTPUT_PATH  = "./features_dataset_v2.csv"
RATINGS_FILE = Path(__file__).parent / "scored_by_ChatGPT_promptC_template.csv"

# k_collapse 的門檻: A_k < COLLAPSE_ETA * A_early
COLLAPSE_ETA = 0.5


# ===========================
# 小工具
# ===========================

def mad(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def safe_skew(arr):
    arr = np.asarray(arr, dtype=float)
    return float(scipy_skew(arr)) if len(arr) >= 3 else np.nan


def safe_kurt(arr):
    arr = np.asarray(arr, dtype=float)
    return float(scipy_kurtosis(arr)) if len(arr) >= 4 else np.nan


def decrement_ratio_3_3(arr):
    """median(last 3) / median(first 3); NaN if <6 or early median == 0"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 6:
        return np.nan
    early = np.median(arr[:3])
    late  = np.median(arr[-3:])
    if early <= 0:
        return np.nan
    return float(late / early)


# ===========================
# 進階振幅特徵
# ===========================

def compute_amplitude_advanced(tap_amplitudes):
    result = {
        "A_early": np.nan, "A_late": np.nan,
        "R_A_3_3": np.nan, "D_A_pct": np.nan,
        "D_A_max": np.nan, "k_collapse": np.nan,
        "tilde_A_median": np.nan, "beta_tilde_A": np.nan, "CV_tilde_A": np.nan,
        "amplitude_MAD": np.nan, "amplitude_skew": np.nan, "amplitude_kurtosis": np.nan,
    }
    arr = np.asarray(tap_amplitudes, dtype=float)
    arr = arr[np.isfinite(arr)]
    K = len(arr)
    if K == 0:
        return result

    result["amplitude_MAD"]      = mad(arr)
    result["amplitude_skew"]     = safe_skew(arr)
    result["amplitude_kurtosis"] = safe_kurt(arr)

    a_max = float(np.max(arr))
    a_min = float(np.min(arr))
    if a_max > 0:
        result["D_A_max"] = float(1 - a_min / a_max)

    if K >= 3:
        A_early = float(np.median(arr[:3]))
        A_late  = float(np.median(arr[-3:]))
        result["A_early"] = A_early
        result["A_late"]  = A_late

        if A_early > 0:
            R_A = A_late / A_early
            result["R_A_3_3"] = float(R_A)
            result["D_A_pct"] = float(100 * (1 - R_A))

            threshold = COLLAPSE_ETA * A_early
            below = np.where(arr < threshold)[0]
            result["k_collapse"] = int(below[0] + 1) if len(below) > 0 else np.nan

            tilde_A = arr / A_early
            result["tilde_A_median"] = float(np.median(tilde_A))
            if K >= 2:
                k_idx = np.arange(K, dtype=float)
                slope, _ = np.polyfit(k_idx, tilde_A, 1)
                result["beta_tilde_A"] = float(slope)
            tm = float(np.mean(tilde_A))
            ts = float(np.std(tilde_A))
            result["CV_tilde_A"] = float(ts / tm) if tm > 0 else np.nan

    return result


# ===========================
# 進階 ITI 特徵
# ===========================

def compute_iti_advanced(iti_seconds):
    result = {
        "I_ITI": np.nan, "I_ITI_norm": np.nan,
        "ITI_MAD": np.nan, "ITI_skew": np.nan, "ITI_kurtosis": np.nan,
    }
    arr = np.asarray(iti_seconds, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return result

    result["ITI_MAD"]      = mad(arr)
    result["ITI_skew"]     = safe_skew(arr)
    result["ITI_kurtosis"] = safe_kurt(arr)

    if len(arr) >= 2:
        diffs = np.abs(np.diff(arr))
        I_ITI = float(np.mean(diffs))
        result["I_ITI"] = I_ITI
        mean_iti = float(np.mean(arr))
        if mean_iti > 0:
            result["I_ITI_norm"] = float(I_ITI / mean_iti)

    return result


# ===========================
# Halt 聚合特徵
# ===========================

def compute_halt_advanced(periods_seconds, total_duration):
    result = {"S_halt": np.nan, "F_halt": np.nan}
    arr = np.asarray(periods_seconds, dtype=float)
    if len(arr) < 2 or total_duration <= 0:
        return result
    median_iti = float(np.median(arr))
    if median_iti <= 0:
        return result
    interruption_mask = arr > INTERRUPTION_ITI_MULTIPLIER * median_iti
    S_halt = float(np.sum(arr[interruption_mask]))
    result["S_halt"] = S_halt
    result["F_halt"] = float(S_halt / total_duration)
    return result


# ===========================
# Task completion
# ===========================

def compute_task_completion(peaks, fps):
    N_valid = int(len(peaks))
    C_10 = int(N_valid >= 10)
    T_10 = float((peaks[9] - peaks[0]) / fps) if N_valid >= 10 else np.nan
    F_incomplete = float(max(0.0, 1.0 - N_valid / 10.0))
    return {
        "N_valid": N_valid,
        "C_10": C_10,
        "T_10": T_10,
        "F_incomplete": F_incomplete,
    }


# ===========================
# Opening / Closing phase
# ===========================

def _empty_phase_dict():
    keys = []
    for prefix in ["V_open", "V_close", "V_open_norm", "V_close_norm",
                   "T_open", "T_close", "R_OC"]:
        for stat in ["median", "mean", "std"]:
            keys.append(f"{prefix}_{stat}")
    keys.extend(["R_V_open", "R_V_close"])
    return {k: np.nan for k in keys}


def compute_phase_features(distance_denoised, peaks, valleys_all, fps):
    """
    每拍 tap k 的 phase 拆分:
      Opening phase O_k: valley_before(k) -> peak(k)        (距離遞增)
      Closing phase C_k: peak(k) -> valley_after(k)         (距離遞減)

    每拍內計算:
      V_open_k  = max   v  during O_k   (距離對時間導數的峰值, 正)
      V_close_k = |min  v| during C_k   (距離導數最負值的絕對值)
      T_open_k  = 秒
      T_close_k = 秒
      R_OC_k    = T_open_k / T_close_k

    之後聚合 (median / mean / std) 並計算 decrement ratio:
      R_V_open  = median(V_open [-3:]) / median(V_open [:3])
      R_V_close = median(V_close[-3:]) / median(V_close[:3])
    """
    result = _empty_phase_dict()
    if len(peaks) < 2 or len(valleys_all) == 0:
        return result

    dt = 1.0 / fps
    V_opens, V_closes, T_opens, T_closes, R_OCs = [], [], [], [], []

    for p in peaks:
        left_v  = valleys_all[valleys_all < p]
        right_v = valleys_all[valleys_all > p]
        if len(left_v) == 0 or len(right_v) == 0:
            continue
        v_before = int(left_v[-1])
        v_after  = int(right_v[0])

        T_open  = (p - v_before) / fps
        T_close = (v_after - p)  / fps
        if T_open <= 0 or T_close <= 0:
            continue

        # Opening: 距離遞增 → 導數正 → 取 max
        seg_open = distance_denoised[v_before:p + 1]
        V_open_k = float(np.max(np.diff(seg_open) / dt)) if len(seg_open) >= 2 else np.nan

        # Closing: 距離遞減 → 導數負 → 取 |min|
        seg_close = distance_denoised[p:v_after + 1]
        V_close_k = float(np.abs(np.min(np.diff(seg_close) / dt))) if len(seg_close) >= 2 else np.nan

        V_opens.append(V_open_k)
        V_closes.append(V_close_k)
        T_opens.append(T_open)
        T_closes.append(T_close)
        R_OCs.append(T_open / T_close)

    V_opens  = np.array([v for v in V_opens  if np.isfinite(v)])
    V_closes = np.array([v for v in V_closes if np.isfinite(v)])
    T_opens  = np.asarray(T_opens,  dtype=float)
    T_closes = np.asarray(T_closes, dtype=float)
    R_OCs    = np.asarray(R_OCs,    dtype=float)

    dist_range = float(distance_denoised.max() - distance_denoised.min())
    V_opens_norm  = V_opens  / dist_range if dist_range > 0 else np.full_like(V_opens,  np.nan)
    V_closes_norm = V_closes / dist_range if dist_range > 0 else np.full_like(V_closes, np.nan)

    def stats3(arr, prefix):
        if len(arr) == 0:
            return {f"{prefix}_median": np.nan,
                    f"{prefix}_mean":   np.nan,
                    f"{prefix}_std":    np.nan}
        return {
            f"{prefix}_median": float(np.median(arr)),
            f"{prefix}_mean":   float(np.mean(arr)),
            f"{prefix}_std":    float(np.std(arr)),
        }

    result.update(stats3(V_opens,       "V_open"))
    result.update(stats3(V_closes,      "V_close"))
    result.update(stats3(V_opens_norm,  "V_open_norm"))
    result.update(stats3(V_closes_norm, "V_close_norm"))
    result.update(stats3(T_opens,  "T_open"))
    result.update(stats3(T_closes, "T_close"))
    result.update(stats3(R_OCs,    "R_OC"))

    result["R_V_open"]  = decrement_ratio_3_3(V_opens)
    result["R_V_close"] = decrement_ratio_3_3(V_closes)

    return result


# ===========================
# 主特徵提取函數 (v2 = base 修 bug + 新特徵)
# ===========================

def extract_full_features_v2(csv_path, fps=FPS):
    data = pd.read_csv(csv_path)

    # 1. 距離
    data['Distance'] = np.sqrt(
        (data['X_thumb'] - data['X_index'])**2 +
        (data['Y_thumb'] - data['Y_index'])**2
    )
    d_min = data['Distance'].min()
    d_max = data['Distance'].max()
    data['Norm_Distance'] = (data['Distance'] - d_min) / (d_max - d_min + 1e-10)

    # 2. 去噪
    distance_denoised  = denoise_signal(data['Distance'].values)
    norm_dist_denoised = denoise_signal(data['Norm_Distance'].values)

    # 3. 速度與加速度
    dt = 1.0 / fps
    speed        = np.gradient(norm_dist_denoised, dt)   # Norm_Distance/秒 → speed_norm_*
    acceleration = np.gradient(speed, dt)

    # 原始 pixel-based 速度/加速度
    speed_dist = np.gradient(distance_denoised, dt)       # pixels/秒 → speed_*
    accel_dist = np.gradient(speed_dist, dt)

    # 4. 峰值 / 谷值
    distance_norm = (distance_denoised - distance_denoised.min()) / \
                    (distance_denoised.max() - distance_denoised.min() + 1e-10)
    peaks, _ = find_peaks(distance_norm, height=0.2,
                          distance=int(fps * 0.15), prominence=0.15, width=1)
    valleys, _ = find_peaks(-distance_norm,
                            distance=int(fps * 0.15), prominence=0.15, width=1)
    num_peaks = len(peaks)

    # 5. 週期/頻率
    if num_peaks > 1:
        periods_seconds = np.diff(peaks) / fps
        frequencies = 1.0 / periods_seconds
        frequencies = frequencies[np.isfinite(frequencies)]
    else:
        periods_seconds = np.array([])
        frequencies = np.array([])

    period_stats    = safe_stats(periods_seconds)
    frequency_stats = safe_stats(frequencies)

    if len(periods_seconds) > 0 and np.max(periods_seconds) > 0:
        periodVarianceNorm = np.var(periods_seconds) / np.max(periods_seconds)
    else:
        periodVarianceNorm = np.nan
    periodEntropy = calculate_entropy(periods_seconds, bins=50, range_vals=(0, 2))

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
        frequency_lr_slope       = np.nan
        frequency_lr_fitness_r2  = np.nan
        frequency_fit_min_degree = np.nan

    # 6. Aperiodicity (FFT power spectrum entropy)
    if len(distance_denoised) > 10:
        fft_result = np.fft.fft(distance_denoised)
        power_spectrum = np.abs(fft_result) ** 2
        ps_norm = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else power_spectrum
        aperiodicity = entropy(ps_norm[ps_norm > 0])
    else:
        aperiodicity = np.nan

    # 7. Interruption / Freeze：ITI-based（符合 MDS-UPDRS 臨床定義）
    if len(periods_seconds) >= 2:
        median_iti = float(np.median(periods_seconds))
        if median_iti > 0:
            interruption_mask = periods_seconds > INTERRUPTION_ITI_MULTIPLIER * median_iti
            freeze_mask       = periods_seconds > FREEZE_ITI_MULTIPLIER * median_iti
            numInterruptions  = int(np.sum(interruption_mask))
            numFreeze         = int(np.sum(freeze_mask))
            maxFreezeDuration = float(np.max(periods_seconds[freeze_mask])) if numFreeze > 0 else 0.0
        else:
            numInterruptions, numFreeze, maxFreezeDuration = 0, 0, 0.0
    else:
        numInterruptions, numFreeze, maxFreezeDuration = 0, 0, 0.0

    num_interruptions_norm = numInterruptions / num_peaks if num_peaks > 0 else np.nan
    num_freeze_norm        = numFreeze        / num_peaks if num_peaks > 0 else np.nan

    # 8. 速度 / 加速度統計
    speed_stats      = safe_stats(np.abs(speed))        # Norm_Distance/秒 → speed_norm_*
    accel_stats      = safe_stats(np.abs(acceleration))
    speed_dist_stats = safe_stats(np.abs(speed_dist))   # pixels/秒 → speed_*
    accel_dist_stats = safe_stats(np.abs(accel_dist))

    # 9. 手指移動 — Thumb & Index
    thumb_x = np.abs(np.diff(data['X_thumb'].values))
    thumb_y = np.abs(np.diff(data['Y_thumb'].values))
    thumb_d = np.sqrt(thumb_x**2 + thumb_y**2)
    thumb_x_stats = safe_stats(thumb_x)
    thumb_y_stats = safe_stats(thumb_y)
    thumb_d_stats = safe_stats(thumb_d)

    index_x = np.abs(np.diff(data['X_index'].values))
    index_y = np.abs(np.diff(data['Y_index'].values))
    index_d = np.sqrt(index_x**2 + index_y**2)
    index_x_stats = safe_stats(index_x)
    index_y_stats = safe_stats(index_y)
    index_d_stats = safe_stats(index_d)

    # 10. 振幅 (逐拍 + 既有 decrement)
    if num_peaks >= 2:
        tap_amplitudes = np.empty(num_peaks, dtype=float)
        for i, p in enumerate(peaks):
            left  = valleys[valleys < p]
            right = valleys[valleys > p]
            lv = distance_denoised[left[-1]]  if len(left)  else None
            rv = distance_denoised[right[0]]  if len(right) else None
            if lv is not None and rv is not None:
                base = (lv + rv) / 2
            elif lv is not None:
                base = lv
            else:
                base = rv
            tap_amplitudes[i] = distance_denoised[p] - base
        tap_amplitudes = np.clip(tap_amplitudes, 0, None)

        d_range = d_max - d_min
        tap_amplitudes_norm = tap_amplitudes / (d_range + 1e-10)

        amplitude_stats      = safe_stats(tap_amplitudes)
        amplitude_norm_stats = safe_stats(tap_amplitudes_norm)
        amplitude_entropy_val = calculate_entropy(tap_amplitudes, bins=10)
        amplitude_cv = (amplitude_stats['stdev'] / amplitude_stats['mean']
                        if amplitude_stats['mean'] > 0 else np.nan)

        tap_idx = np.arange(len(tap_amplitudes), dtype=float)
        amp_slope, amp_intercept = np.polyfit(tap_idx, tap_amplitudes, 1)
        amp_pred = amp_slope * tap_idx + amp_intercept
        amp_ss_res = np.sum((tap_amplitudes - amp_pred) ** 2)
        amp_ss_tot = np.sum((tap_amplitudes - np.mean(tap_amplitudes)) ** 2)
        amplitude_decrement_slope          = amp_slope
        amplitude_decrement_fitness_r2     = 1 - (amp_ss_res / amp_ss_tot) if amp_ss_tot > 0 else np.nan
        amplitude_decrement_end_to_mean    = (tap_amplitudes[-1] / np.mean(tap_amplitudes)
                                              if np.mean(tap_amplitudes) > 0 else np.nan)
        amplitude_decrement_fit_min_degree = 1

        half = max(len(tap_amplitudes) // 2, 1)
        first_half_mean  = np.mean(tap_amplitudes[:half])
        second_half_mean = np.mean(tap_amplitudes[half:]) if len(tap_amplitudes) > half else np.nan
        amplitude_decrement_last_to_first_half = (second_half_mean / first_half_mean
                                                  if first_half_mean > 0 and not np.isnan(second_half_mean)
                                                  else np.nan)
    else:
        tap_amplitudes                          = np.array([])
        tap_amplitudes_norm                     = np.array([])
        amplitude_stats                         = safe_stats(np.array([]))
        amplitude_norm_stats                    = safe_stats(np.array([]))
        amplitude_entropy_val                   = np.nan
        amplitude_cv                            = np.nan
        amplitude_decrement_slope               = np.nan
        amplitude_decrement_fitness_r2          = np.nan
        amplitude_decrement_end_to_mean         = np.nan
        amplitude_decrement_fit_min_degree      = np.nan
        amplitude_decrement_last_to_first_half  = np.nan

    # === 新特徵 ===
    amp_adv    = compute_amplitude_advanced(tap_amplitudes)
    iti_adv    = compute_iti_advanced(periods_seconds)
    total_dur  = len(distance_denoised) / fps
    halt_adv   = compute_halt_advanced(periods_seconds, total_dur)
    task_comp  = compute_task_completion(peaks, fps)
    phase_feat = compute_phase_features(distance_denoised, peaks, valleys, fps)

    # 組合
    features = {
        # 手指移動 — Thumb
        'finger_mvmnt_thumb_x_median':         thumb_x_stats['median'],
        'finger_mvmnt_thumb_x_quartile_range': thumb_x_stats['quartile_range'],
        'finger_mvmnt_thumb_x_mean':           thumb_x_stats['mean'],
        'finger_mvmnt_thumb_x_min':            thumb_x_stats['min'],
        'finger_mvmnt_thumb_x_max':            thumb_x_stats['max'],
        'finger_mvmnt_thumb_x_stdev':          thumb_x_stats['stdev'],
        'finger_mvmnt_thumb_y_median':         thumb_y_stats['median'],
        'finger_mvmnt_thumb_y_quartile_range': thumb_y_stats['quartile_range'],
        'finger_mvmnt_thumb_y_mean':           thumb_y_stats['mean'],
        'finger_mvmnt_thumb_y_min':            thumb_y_stats['min'],
        'finger_mvmnt_thumb_y_max':            thumb_y_stats['max'],
        'finger_mvmnt_thumb_y_stdev':          thumb_y_stats['stdev'],
        'finger_mvmnt_thumb_dist_median':         thumb_d_stats['median'],
        'finger_mvmnt_thumb_dist_quartile_range': thumb_d_stats['quartile_range'],
        'finger_mvmnt_thumb_dist_mean':           thumb_d_stats['mean'],
        'finger_mvmnt_thumb_dist_min':            thumb_d_stats['min'],
        'finger_mvmnt_thumb_dist_max':            thumb_d_stats['max'],
        'finger_mvmnt_thumb_dist_stdev':          thumb_d_stats['stdev'],

        # 手指移動 — Index
        'finger_mvmnt_index_x_median':         index_x_stats['median'],
        'finger_mvmnt_index_x_quartile_range': index_x_stats['quartile_range'],
        'finger_mvmnt_index_x_mean':           index_x_stats['mean'],
        'finger_mvmnt_index_x_min':            index_x_stats['min'],
        'finger_mvmnt_index_x_max':            index_x_stats['max'],
        'finger_mvmnt_index_x_stdev':          index_x_stats['stdev'],
        'finger_mvmnt_index_y_median':         index_y_stats['median'],
        'finger_mvmnt_index_y_quartile_range': index_y_stats['quartile_range'],
        'finger_mvmnt_index_y_mean':           index_y_stats['mean'],
        'finger_mvmnt_index_y_min':            index_y_stats['min'],
        'finger_mvmnt_index_y_max':            index_y_stats['max'],
        'finger_mvmnt_index_y_stdev':          index_y_stats['stdev'],
        'finger_mvmnt_index_dist_median':         index_d_stats['median'],
        'finger_mvmnt_index_dist_quartile_range': index_d_stats['quartile_range'],
        'finger_mvmnt_index_dist_mean':           index_d_stats['mean'],
        'finger_mvmnt_index_dist_min':            index_d_stats['min'],
        'finger_mvmnt_index_dist_max':            index_d_stats['max'],
        'finger_mvmnt_index_dist_stdev':          index_d_stats['stdev'],

        # Rhythm
        'aperiodicity':       aperiodicity,
        'periodEntropy':      periodEntropy,
        'periodVarianceNorm': periodVarianceNorm,
        'numInterruptions':   numInterruptions,
        'numFreeze':          numFreeze,
        'maxFreezeDuration':  maxFreezeDuration,

        # Period
        'period_median':         period_stats['median'],
        'period_quartile_range': period_stats['quartile_range'],
        'period_mean':           period_stats['mean'],
        'period_min':            period_stats['min'],
        'period_max':            period_stats['max'],
        'period_stdev':          period_stats['stdev'],

        # Frequency
        'frequency_median':          frequency_stats['median'],
        'frequency_quartile_range':  frequency_stats['quartile_range'],
        'frequency_mean':            frequency_stats['mean'],
        'frequency_min':             frequency_stats['min'],
        'frequency_max':             frequency_stats['max'],
        'frequency_stdev':           frequency_stats['stdev'],
        'frequency_lr_fitness_r2':   frequency_lr_fitness_r2,
        'frequency_lr_slope':        frequency_lr_slope,
        'frequency_fit_min_degree':  frequency_fit_min_degree,

        'num_peaks':               num_peaks,
        'num_interruptions_norm':  num_interruptions_norm,
        'num_freeze_norm':         num_freeze_norm,

        # Speed / Acceleration (pixels/秒)
        'speed_median':         speed_dist_stats['median'],
        'speed_quartile_range': speed_dist_stats['quartile_range'],
        'speed_mean':           speed_dist_stats['mean'],
        'speed_min':            speed_dist_stats['min'],
        'speed_max':            speed_dist_stats['max'],
        'speed_stdev':          speed_dist_stats['stdev'],
        'acceleration_median':         accel_dist_stats['median'],
        'acceleration_quartile_range': accel_dist_stats['quartile_range'],
        'acceleration_mean':           accel_dist_stats['mean'],
        'acceleration_min':            accel_dist_stats['min'],
        'acceleration_max':            accel_dist_stats['max'],
        'acceleration_stdev':          accel_dist_stats['stdev'],

        # Speed / Acceleration (Norm_Distance/秒)
        'speed_norm_median':         speed_stats['median'],
        'speed_norm_quartile_range': speed_stats['quartile_range'],
        'speed_norm_mean':           speed_stats['mean'],
        'speed_norm_min':            speed_stats['min'],
        'speed_norm_max':            speed_stats['max'],
        'speed_norm_stdev':          speed_stats['stdev'],
        'acceleration_norm_median':         accel_stats['median'],
        'acceleration_norm_quartile_range': accel_stats['quartile_range'],
        'acceleration_norm_mean':           accel_stats['mean'],
        'acceleration_norm_min':            accel_stats['min'],
        'acceleration_norm_max':            accel_stats['max'],
        'acceleration_norm_stdev':          accel_stats['stdev'],

        # Amplitude (pixels)
        'amplitude_median':         amplitude_stats['median'],
        'amplitude_quartile_range': amplitude_stats['quartile_range'],
        'amplitude_mean':           amplitude_stats['mean'],
        'amplitude_min':            amplitude_stats['min'],
        'amplitude_max':            amplitude_stats['max'],
        'amplitude_stdev':          amplitude_stats['stdev'],
        'amplitude_cv':             amplitude_cv,
        'amplitude_entropy':        amplitude_entropy_val,

        # Amplitude (正規化: ÷ (d_max − d_min))
        'amplitude_norm_median':         amplitude_norm_stats['median'],
        'amplitude_norm_quartile_range': amplitude_norm_stats['quartile_range'],
        'amplitude_norm_mean':           amplitude_norm_stats['mean'],
        'amplitude_norm_min':            amplitude_norm_stats['min'],
        'amplitude_norm_max':            amplitude_norm_stats['max'],
        'amplitude_norm_stdev':          amplitude_norm_stats['stdev'],

        # Amplitude decrement
        'amplitude_decrement_fitness_r2':         amplitude_decrement_fitness_r2,
        'amplitude_decrement_slope':              amplitude_decrement_slope,
        'amplitude_decrement_end_to_mean':        amplitude_decrement_end_to_mean,
        'amplitude_decrement_fit_min_degree':     amplitude_decrement_fit_min_degree,
        'amplitude_decrement_last_to_first_half': amplitude_decrement_last_to_first_half,

        # Per-tap amplitudes (for "AMPLITUDE PATTERN" block in prompts)
        'tap_amplitudes': tap_amplitudes.tolist() if len(tap_amplitudes) >= 2 else [],

        # === v2 新增 ===
        **amp_adv,
        **iti_adv,
        **halt_adv,
        **task_comp,
        **phase_feat,

        # Metadata
        'hand':     '',
        'filename': '',
        'Rating1':  np.nan,
        'Rating2':  np.nan,
        'Rating':   np.nan,
    }

    return features


# ===========================
# 批次處理
# ===========================

def batch_extract(csv_dir=CSV_DIR, output_path=OUTPUT_PATH, ratings_file=RATINGS_FILE):
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

    all_features = []
    for i, csv_path in enumerate(csv_files, 1):
        csv_name = csv_path.stem
        print(f"[{i}/{len(csv_files)}] {csv_name}...", end=" ")
        try:
            feats = extract_full_features_v2(csv_path)
            feats['filename'] = csv_name
            if '_Hands_L_' in csv_name:
                feats['hand'] = 'L'
            elif '_Hands_R_' in csv_name:
                feats['hand'] = 'R'
            else:
                feats['hand'] = ''

            if ratings_df is not None:
                m = ratings_df[
                    (ratings_df['filename'] == csv_name) |
                    (ratings_df['filename'] == csv_name + '.csv')
                ]
                if not m.empty:
                    row = m.iloc[0]
                    for tgt, src in rating_column_map.items():
                        if src in row:
                            val = row[src]
                            if pd.notna(val) and val != '':
                                feats[tgt] = val

            all_features.append(feats)
            print(f"OK ({feats['num_peaks']} taps)")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    df = pd.DataFrame(all_features)
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to: {output_path}")
    print(f"  Shape: {df.shape}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("特徵提取 v2 (bug-fixed + 第一波 + opening/closing phase)")
    print("=" * 60 + "\n")
    df = batch_extract()
    print("\nDone!")
