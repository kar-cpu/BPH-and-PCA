import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shap
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. 数据集成与清洗
# ==========================================
def load_combined_data():
    # 检查必要文件
    files = ['PCA_inci_data.csv', 'BPH_inci_cleaned_full.csv', 'exposure2021_GBD_BPH_combined.csv']
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}")

    pca_df = pd.read_csv('PCA_inci_data.csv')
    bph_df = pd.read_csv('BPH_inci_cleaned_full.csv')
    exposure_df = pd.read_csv('exposure2021_GBD_BPH_combined.csv')

    # 清洗 PCA 和 BPH
    pca_clean = pca_df[pca_df['Population'] == 'All Population'][['Location', 'Value']].rename(columns={'Value': 'PCA_Inc'})
    bph_clean = bph_df[['location_name', 'val']].rename(columns={'location_name': 'Location', 'val': 'BPH_Inc'})
    
    # 暴露因素透视 (31个因素)
    exp_pivot = exposure_df.pivot_table(index='Location', columns='Risk factor', values='Value').reset_index()
    
    # 整合主表
    df = pd.merge(exp_pivot, pca_clean, on='Location', how='inner')
    df = pd.merge(df, bph_clean, on='Location', how='inner')
    
    features = [c for c in exp_pivot.columns if c != 'Location']
    return df, features

# ==========================================
# 2. 执行分析与绘图 (修改了画布高度和显示数量)
# ==========================================
def plot_combined_shap_full():
    try:
        df, features = load_combined_data()
        X = df[features]
        
        # 增加画布高度到 22，以容纳 31 个因素而不拥挤
        fig = plt.figure(figsize=(26, 22)) 
        
        # --- 子图 1: PCA ---
        print("分析 PCA 模型 (显示全部 31 种因素)...")
        model_pca = RandomForestRegressor(n_estimators=500, random_state=42)
        model_pca.fit(X, df['PCA_Inc'])
        explainer_pca = shap.TreeExplainer(model_pca)
        shap_values_pca = explainer_pca.shap_values(X)
        
        ax1 = fig.add_subplot(1, 2, 1)
        plt.sca(ax1)
        # 修改点：max_display 设置为 31
        shap.summary_plot(shap_values_pca, X, max_display=31, show=False, plot_size=None)
        plt.title('(A) Impact on PCA Incidence', fontsize=24, pad=30, fontweight='bold')
        plt.xlabel('SHAP Value (Impact on PCA)', fontsize=16)

        # --- 子图 2: BPH ---
        print("分析 BPH 模型 (显示全部 31 种因素)...")
        model_bph = RandomForestRegressor(n_estimators=500, random_state=42)
        model_bph.fit(X, df['BPH_Inc'])
        explainer_bph = shap.TreeExplainer(model_bph)
        shap_values_bph = explainer_bph.shap_values(X)
        
        ax2 = fig.add_subplot(1, 2, 2)
        plt.sca(ax2)
        # 修改点：max_display 设置为 31
        shap.summary_plot(shap_values_bph, X, max_display=31, show=False, plot_size=None)
        plt.title('(B) Impact on BPH Incidence', fontsize=24, pad=30, fontweight='bold')
        plt.xlabel('SHAP Value (Impact on BPH)', fontsize=16)

        # 调整布局，增加横向间距 w_pad
        plt.tight_layout(w_pad=12) 
        
        output_name = 'Combined_SHAP_PCA_BPH_Full_31.png'
        plt.savefig(output_name, dpi=300, bbox_inches='tight')
        print(f"\n[成功] 包含全部 31 种因素的组合图已保存为: {os.path.abspath(output_name)}")
        plt.show()

    except Exception as e:
        print(f"运行失败: {e}")

if __name__ == "__main__":
    plot_combined_shap_full()
