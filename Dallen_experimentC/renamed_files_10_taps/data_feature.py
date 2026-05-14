import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os
import glob

# 設定路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = current_dir  # 資料所在資料夾
output_folder = os.path.join(current_dir, "output_results")  # 輸出資料夾

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 尋找所有CSV檔案
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

# 儲存所有結果的列表
all_results = []

# 處理每個CSV檔案
for csv_file in csv_files:
    try:
        # 讀取資料
        data = pd.read_csv(csv_file)
        file_name = os.path.basename(csv_file)
        file_name_without_ext = os.path.splitext(file_name)[0]
        
        print(f"\n處理檔案: {file_name}")
        
        # 計算拇指和食指之間的距離
        data['Distance'] = np.sqrt((data['X_thumb'] - data['X_index'])**2 + 
                                   (data['Y_thumb'] - data['Y_index'])**2)
        data['Distance_Norm'] = (data['Distance'] - np.min(data['Distance'])) / \
                                (np.max(data['Distance']) - np.min(data['Distance']))
        
        # === 改良的peak偵測 ===
        # 使用多個參數來避免誤判，特別是停留時間過長的情況
        peaks, properties = find_peaks(
            data['Distance_Norm'], 
            height=0.2,           # 最小高度閾值
            distance=50,          # peaks之間最小距離（保持你原本的設定）
            prominence=0.15,      # peak必須明顯突出，避免平台誤判
            width=1,              # peak的最小寬度
            plateau_size=[1, 40]  # 允許1-40個frame的平台，超過視為同一個peak
        )
        
        # 額外的後處理：確保peaks之間有足夠的間距
        # 這可以進一步避免因為手指停留造成的重複偵測
        def filter_close_peaks(peaks, signal, min_distance=50):
            """過濾掉距離太近的peaks，保留振幅較大的"""
            if len(peaks) <= 1:
                return peaks
            
            filtered_peaks = [peaks[0]]
            for i in range(1, len(peaks)):
                if peaks[i] - filtered_peaks[-1] >= min_distance:
                    filtered_peaks.append(peaks[i])
                else:
                    # 如果距離太近，保留振幅較大的那個
                    if signal[peaks[i]] > signal[filtered_peaks[-1]]:
                        filtered_peaks[-1] = peaks[i]
            
            return np.array(filtered_peaks)
        
        # 應用過濾
        # peaks = filter_close_peaks(peaks, data['Distance_Norm'].values, min_distance=50)
        
        # 計算點擊的週期和振幅
        tap_periods = np.diff(data['Frame'].iloc[peaks])
        tap_amplitudes = data['Distance'].iloc[peaks]
        
        # 點擊次數
        num_taps = len(peaks)
        
        # 計算統計資料
        avg_period = np.mean(tap_periods) if len(tap_periods) > 0 else 0
        std_period = np.std(tap_periods) if len(tap_periods) > 0 else 0
        avg_amplitude = np.mean(tap_amplitudes) if len(tap_amplitudes) > 0 else 0
        std_amplitude = np.std(tap_amplitudes) if len(tap_amplitudes) > 0 else 0
        
        print(f"檢測到的點擊次數: {num_taps}")
        print(f"平均週期: {avg_period:.2f} ± {std_period:.2f} frames")
        print(f"平均振幅: {avg_amplitude:.4f} ± {std_amplitude:.4f}")
        
        # 儲存結果
        all_results.append({
            'File': file_name,
            'Num_Taps': num_taps,
            'Avg_Period': avg_period,
            'Std_Period': std_period,
            'Avg_Amplitude': avg_amplitude,
            'Std_Amplitude': std_amplitude,
            'Tap_Periods': tap_periods.tolist() if len(tap_periods) > 0 else [],
            'Tap_Amplitudes': tap_amplitudes.tolist() if len(tap_amplitudes) > 0 else []
        })
        
        # 繪製圖表
        plt.figure(figsize=(15, 5))
        plt.plot(data['Frame'], data['Distance_Norm'], label='Distance between thumb and index finger', linewidth=1.5)
        plt.plot(data['Frame'].iloc[peaks], data['Distance_Norm'].iloc[peaks], 'rx', 
                markersize=12, markeredgewidth=2.5, label='Detected Taps')
        plt.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='Threshold')
        plt.xlabel('Frame', fontsize=12)
        plt.ylabel('Distance_Norm', fontsize=12)
        plt.title(f'Finger Tapping Test: {file_name}\nDetected Taps: {num_taps} | Avg Period: {avg_period:.2f} frames', 
                 fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 儲存圖表
        plot_path = os.path.join(output_folder, f"{file_name_without_ext}_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"圖表已儲存: {plot_path}")
        
        # 儲存詳細資料到CSV
        detail_data = pd.DataFrame({
            'Frame': data['Frame'],
            'Distance': data['Distance'],
            'Distance_Norm': data['Distance_Norm']
        })
        
        # 標記哪些frame是peak
        detail_data['Is_Peak'] = False
        detail_data.loc[peaks, 'Is_Peak'] = True
        
        detail_path = os.path.join(output_folder, f"{file_name_without_ext}_processed.csv")
        detail_data.to_csv(detail_path, index=False)
        
    except Exception as e:
        print(f"處理 {file_name} 時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# 儲存總結報告
if all_results:
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_folder, "summary_report.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n總結報告已儲存: {summary_path}")
    
    # 顯示總結
    print("\n=== 處理總結 ===")
    print(summary_df[['File', 'Num_Taps', 'Avg_Period', 'Avg_Amplitude']])
    
    # 繪製總結圖表
    if len(all_results) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 點擊次數比較
        axes[0].bar(range(len(summary_df)), summary_df['Num_Taps'])
        axes[0].set_xlabel('File Index')
        axes[0].set_ylabel('Number of Taps')
        axes[0].set_title('Number of Taps Comparison')
        axes[0].grid(True, alpha=0.3)
        
        # 平均週期比較
        axes[1].bar(range(len(summary_df)), summary_df['Avg_Period'])
        axes[1].set_xlabel('File Index')
        axes[1].set_ylabel('Average Period (frames)')
        axes[1].set_title('Average Period Comparison')
        axes[1].grid(True, alpha=0.3)
        
        # 平均振幅比較
        axes[2].bar(range(len(summary_df)), summary_df['Avg_Amplitude'])
        axes[2].set_xlabel('File Index')
        axes[2].set_ylabel('Average Amplitude')
        axes[2].set_title('Average Amplitude Comparison')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        comparison_plot_path = os.path.join(output_folder, "comparison_plot.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"比較圖表已儲存: {comparison_plot_path}")
else:
    print("\n未找到任何CSV檔案或處理失敗")

print(f"\n所有結果已儲存至: {output_folder}")