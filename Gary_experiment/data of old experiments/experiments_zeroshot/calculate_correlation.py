import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 設定檔案路徑
    base_dir = r'c:\Users\user\OneDrive\文件\研究所\thesis\Gary_experiment\experiments_zeroshot'
    file_path = os.path.join(base_dir, 'features_dataset.csv')
    output_path = os.path.join(base_dir, 'correlation_results.csv')
    
    # 讀取 CSV 檔案
    try:
        df = pd.read_csv(file_path)
        print(f"成功讀取檔案: {file_path}")
    except FileNotFoundError:
        print(f"找不到檔案: {file_path}")
        return

    # 確保評分欄位是數值型別 (處理可能的非數值字符)
    df['Rating1'] = pd.to_numeric(df['Rating1'], errors='coerce')
    df['Rating2'] = pd.to_numeric(df['Rating2'], errors='coerce')

    # 1. 計算兩個醫生間的平均值
    # 注意：我們重新計算平均值以避免原始 'Rating' 欄位可能有缺漏值的問題
    df['Rating_Mean'] = (df['Rating1'] + df['Rating2']) / 2.0

    # 準備存放結果的列表
    results = []

    # 定義要排除的欄位 (非特徵欄位)
    exclude_cols = {'hand', 'filename', 'Rating1', 'Rating2', 'Rating', 'Rating_Mean'}
    
    # 針對每個特徵欄位計算 Pearson Correlation
    for col in df.columns:
        # 檢查是否為數值欄位且不在排除名單中
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            # 計算 Pearson Correlation
            corr_mean = df[col].corr(df['Rating_Mean'], method='pearson')
            corr_r1 = df[col].corr(df['Rating1'], method='pearson')
            corr_r2 = df[col].corr(df['Rating2'], method='pearson')
            
            results.append({
                'Feature': col,
                'Pearson_with_Mean': corr_mean,
                'Pearson_with_Rating1': corr_r1,
                'Pearson_with_Rating2': corr_r2
            })

    # 轉換為 DataFrame 並依照與平均值的相關係數絕對值排序
    results_df = pd.DataFrame(results)
    results_df['Abs_Corr_Mean'] = results_df['Pearson_with_Mean'].abs()
    results_df = results_df.sort_values(by='Abs_Corr_Mean', ascending=False).drop(columns=['Abs_Corr_Mean'])
    
    # 顯示前 10 名並儲存
    print("\n與平均評分相關性最高的前 10 個特徵：")
    print(results_df.head(10))
    
    results_df.to_csv(output_path, index=False)
    print(f"\n完整分析結果已儲存至: {output_path}")

    # --- 新增：繪製散佈圖 ---
    # 建立存放圖表的資料夾
    plot_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 設定繪圖風格
    sns.set(style="whitegrid")
    
    # 取前 6 個相關性最高的特徵
    top_n = 20
    print(f"\n正在繪製前 {top_n} 個特徵的散佈圖...")

    for _, row in results_df.head(top_n).iterrows():
        feature = row['Feature']
        corr = row['Pearson_with_Mean']
        
        plt.figure(figsize=(8, 6))
        # 繪製散佈圖與迴歸線 (紅色線條代表線性趨勢)
        sns.regplot(x=feature, y='Rating_Mean', data=df, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
        
        plt.title(f'Feature: {feature}\nPearson r = {corr:.4f}')
        plt.xlabel(feature)
        plt.ylabel('Average Rating')
        
        # 儲存圖片
        safe_name = feature.replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(plot_dir, f'scatter_{safe_name}.png'))
        plt.close()
        
    print(f"圖表已儲存至資料夾: {plot_dir}")

if __name__ == "__main__":
    main()