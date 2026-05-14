"""
Figure 3.2 生成器 - 簡化版本 (自動讀取檔名中的模型名稱)
用於生成類似論文 Figure 3.2 的堆疊長條圖

圖表說明：
- 顯示 LLM 對不同 concordant score (0-3) 的預測分布
- 藍色：預測分數與 concordant score 一致 (正確)
- 黑色：預測分數與 concordant score 不一致 (錯誤)
- 橘色/黃色：範例 session 的預測分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # 導入正則表達式模組

# 設置繪圖風格
sns.set_theme(style="whitegrid")

# ===========================
# 配置參數
# ===========================

# 範例 session ID (在論文 Exp D 中使用的 few-shot 範例)
EXAMPLE_SESSION_IDS = [2, 17, 46, 52]

# 顏色設定 (符合 Figure 3.2)
COLOR_MAP = {
    'match': '#4169E1',      # 藍色 - 預測正確
    'mismatch': 'black',     # 黑色 - 預測錯誤
    'example': '#FFA500'     # 橘色 - 範例 session
}

# 輸出資料夾
OUTPUT_DIR = "distribution_of_predicted_score"


# ===========================
# 資料處理函數
# ===========================

def load_and_process_data(csv_file):
    """
    讀取並處理 CSV 資料
    """
    print(f"正在讀取: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"  成功讀取 {len(df)} 列資料")
        return df
    except Exception as e:
        print(f"  錯誤: 無法讀取檔案 - {e}")
        return None


def extract_model_name(filename):
    """
    從檔名中提取模型名稱
    預期格式: ...scored_by_<ModelName>_...
    或是: ...scored_by_<ModelName>.csv
    """
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    
    # 尋找 'scored_by_' 關鍵字
    if 'scored_by_' in name_no_ext:
        # 取得 'scored_by_' 之後的部分
        # split 的參數 1 表示只分割一次，取後面的部分
        parts = name_no_ext.split('scored_by_', 1)
        if len(parts) > 1:
            suffix = parts[1]
            
            # 根據使用者的描述 "scored_by_--------_"，模型名可能被下一個底線包圍
            # 嘗試以底線分割，取第一部分
            if '_' in suffix:
                model_name = suffix.split('_')[0]
            else:
                # 如果沒有底線，就取剩下全部
                model_name = suffix
            
            # 如果提取出的名稱是空的 (例如 scored_by__test)，回傳預設值
            if model_name.strip():
                return model_name
    
    # 如果找不到格式，回傳原始檔名或 Unknown
    return "Unknown_Model"


def extract_plot_data(df):
    """
    提取繪圖所需資料
    """
    
    # 欄位名稱
    col_dr1 = 'label_Dr. Tan'
    col_dr2 = 'label_Dr. Chien'
    
    # 檢查必要欄位
    if col_dr1 not in df.columns or col_dr2 not in df.columns:
        print(f"  錯誤: 找不到醫生評分欄位")
        return None
    
    # 預測欄位 (0-9)
    pred_cols = [str(i) for i in range(10)]
    valid_pred_cols = [c for c in pred_cols if c in df.columns]
    
    if len(valid_pred_cols) == 0:
        print(f"  錯誤: 找不到預測欄位 (0-9)")
        return None
    
    print(f"  找到 {len(valid_pred_cols)} 個預測欄位")
    
    # 轉換為數值型態
    for col in valid_pred_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df[col_dr1] = pd.to_numeric(df[col_dr1], errors='coerce')
    df[col_dr2] = pd.to_numeric(df[col_dr2], errors='coerce')
    
    # 篩選 concordant sessions (兩位醫生評分一致)
    df_clean = df.dropna(subset=[col_dr1, col_dr2])
    df_concordant = df_clean[df_clean[col_dr1] == df_clean[col_dr2]].copy()
    
    if df_concordant.empty:
        print(f"  警告: 沒有找到 concordant sessions")
        return None
    
    print(f"  找到 {len(df_concordant)} 個 concordant sessions")
    
    # 建立 Ground Truth 欄位
    df_concordant['GT'] = df_concordant[col_dr1].astype(int)
    
    # 建立 session ID
    if 'session' in df.columns:
        df_concordant['sess_id'] = pd.to_numeric(
            df_concordant['session'], errors='coerce'
        ).fillna(-1).astype(int)
    else:
        df_concordant['sess_id'] = df_concordant.index + 1
    
    # 初始化統計資料結構
    plot_data = {}
    target_gt_scores = [0, 1, 2, 3]
    
    for gt in target_gt_scores:
        # 篩選此 GT 分數的 sessions
        sessions_gt = df_concordant[df_concordant['GT'] == gt]
        
        # 初始化計數 (預測分數 0-4)
        counts = pd.DataFrame(0, index=range(5), 
                            columns=['example', 'match', 'mismatch'])
        
        if sessions_gt.empty:
            plot_data[gt] = counts
            continue
        
        # 統計每個 session 的預測
        for _, row in sessions_gt.iterrows():
            sess_id = row['sess_id']
            is_example = sess_id in EXAMPLE_SESSION_IDS
            
            # 取得所有預測並轉換為 float 陣列
            preds = row[valid_pred_cols].values
            # 確保是 numpy array 並轉換為 float
            preds = np.array(preds, dtype=float)
            # 過濾 NaN 值
            preds = preds[~np.isnan(preds)]
            
            for p in preds:
                try:
                    p_int = int(round(p))
                except:
                    continue
                
                # 只統計 0-4 的預測
                if 0 <= p_int <= 4:
                    if is_example:
                        counts.loc[p_int, 'example'] += 1
                    elif p_int == gt:
                        counts.loc[p_int, 'match'] += 1
                    else:
                        counts.loc[p_int, 'mismatch'] += 1
        
        plot_data[gt] = counts
        print(f"  GT={gt}: {len(sessions_gt)} sessions, "
              f"{counts.values.sum():.0f} 總預測數")
    
    return plot_data


# ===========================
# 繪圖函數
# ===========================

def create_figure(plot_data, title):
    """
    創建 Figure 3.2 樣式的圖表
    
    參數:
        plot_data: 由 extract_plot_data() 回傳的資料
        title: 圖表標題 (模型名稱)
    
    回傳:
        matplotlib figure 物件
    """
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    
    # 使用傳入的 title (從檔名提取的模型名稱) 作為大標題
    fig.suptitle(title, fontsize=16, y=1.02)
    
    gt_scores = [0, 1, 2, 3]
    
    # 計算 Y 軸範圍
    max_y = 0
    for counts in plot_data.values():
        current_max = counts.sum(axis=1).max()
        if current_max > max_y:
            max_y = current_max
    
    y_limit = max(max_y * 1.2, 10)
    
    # 繪製每個子圖
    for i, gt in enumerate(gt_scores):
        ax = axes[i]
        data = plot_data[gt]
        
        x = np.arange(5)  # 預測分數 0-4
        y_match = data['match'].values
        y_mismatch = data['mismatch'].values
        y_example = data['example'].values
        
        # 繪製堆疊長條圖
        ax.bar(x, y_match, color=COLOR_MAP['match'], 
               label='Scores same as concordant score')
        ax.bar(x, y_mismatch, bottom=y_match, 
               color=COLOR_MAP['mismatch'], label='Scores different from concordant score')
        ax.bar(x, y_example, bottom=y_match + y_mismatch, 
               color=COLOR_MAP['example'], label='Scores of example session')
        
        # 設定圖表格式
        ax.set_title(f"Concordant Score {gt}", fontsize=14)
        ax.set_xlabel("Predicted Score", fontsize=12)
        ax.set_xticks(x)
        ax.set_ylim(0, y_limit)
        ax.grid(axis='y', alpha=0.3)
        
        # 只在第一個子圖顯示 Y 軸標籤和圖例
        if i == 0:
            ax.set_ylabel("Counts", fontsize=12)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        # 在長條圖頂端顯示百分比
        total_predictions = data.values.sum()
        total_per_bar = y_match + y_mismatch + y_example
        
        for idx, total in enumerate(total_per_bar):
            if total_predictions > 0:
                pct = (total / total_predictions) * 100
                label_y = total + (y_limit * 0.02)
                ax.text(idx, label_y, f"{pct:.1f}%", 
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


# ===========================
# 主程式
# ===========================

def main():
    """
    主程式流程
    """
    print("=" * 60)
    print("Figure 3.2 生成器 (自動提取模型名稱)")
    print("=" * 60)
    
    # 建立輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n輸出資料夾: {OUTPUT_DIR}/")
    
    # 尋找 CSV 檔案
    import glob
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("\n錯誤: 當前目錄下找不到 CSV 檔案")
        return
    
    print(f"\n找到 {len(csv_files)} 個 CSV 檔案\n")
    
    success_count = 0
    
    for csv_file in csv_files:
        print("-" * 60)
        
        # 自動從檔名提取模型名稱
        model_name = extract_model_name(csv_file)
        print(f"檔案: {csv_file}")
        print(f"提取的模型名稱: {model_name}")
        
        df = load_and_process_data(csv_file)
        if df is None:
            continue
        
        plot_data = extract_plot_data(df)
        if plot_data is None:
            continue
        
        # 生成圖表，傳入提取出的模型名稱作為標題
        fig = create_figure(plot_data, title=model_name)
        
        # 儲存檔案 (檔名也包含模型名稱，避免重複)
        # 清理模型名稱中的特殊字元以便作為檔名
        safe_model_name = model_name.replace(" ", "_").replace(":", "")
        
        png_path = os.path.join(OUTPUT_DIR, f"Gary2025_distribution_of_predicted_scored_by_{safe_model_name}.png")
        pdf_path = os.path.join(OUTPUT_DIR, f"Gary2025_distribution_of_predicted_scored_by_{safe_model_name}.pdf")
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 已儲存: {png_path}")
        
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 已儲存: {pdf_path}")
        
        plt.close(fig)
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"完成! 成功處理 {success_count}/{len(csv_files)} 個檔案")
    print("=" * 60)


if __name__ == "__main__":
    main()