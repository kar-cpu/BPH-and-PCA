import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings

# 忽略干扰性的收敛警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径与数据准备 (精准适配 Upper/Lower bound)
# ==========================================
def load_data():
    files = {
        'pca': 'PCA_inci_data.csv',
        'bph': 'BPH_inci_cleaned_full.csv',
        'exp': 'exposure2021_GBD_BPH_combined.csv'
    }

    # 检查文件
    for k, v in files.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"找不到文件: {v}")

    print("✅ 正在读取并处理数据...")
    pca_df = pd.read_csv(files['pca'])
    bph_df = pd.read_csv(files['bph'])
    exp_df = pd.read_csv(files['exp'])

    # --- 强力列名识别逻辑 ---
    clean_columns = {c.strip().lower(): c for c in exp_df.columns}
    col_map = {}
    
    # 查找 Value
    if 'value' in clean_columns: col_map['val'] = clean_columns['value']
    elif 'val' in clean_columns: col_map['val'] = clean_columns['val']
    
    # 查找 Upper bound 或 Upper
    if 'upper bound' in clean_columns: col_map['upper'] = clean_columns['upper bound']
    elif 'upper' in clean_columns: col_map['upper'] = clean_columns['upper']
    
    # 查找 Lower bound 或 Lower
    if 'lower bound' in clean_columns: col_map['lower'] = clean_columns['lower bound']
    elif 'lower' in clean_columns: col_map['lower'] = clean_columns['lower']

    # 检查是否找齐了三列
    if len(col_map) < 3:
        actual_cols = list(exp_df.columns)
        raise KeyError(f"在暴露文件中找不齐 Value/Upper/Lower 列。实际检测到的列名为: {actual_cols}")

    print(f"📊 已成功匹配列名: Mean -> [{col_map['val']}], Upper -> [{col_map['upper']}], Lower -> [{col_map['lower']}]")

    # 提取发病率
    pca_clean = pca_df[pca_df['Population'] == 'All Population'][['Location', 'Value']].rename(columns={'Value': 'PCA_Inc'})
    bph_clean = bph_df[['location_name', 'val']].rename(columns={'location_name': 'Location', 'val': 'BPH_Inc'})

    # 透视暴露数据
    exp_val = exp_df.pivot_table(index='Location', columns='Risk factor', values=col_map['val'])
    exp_up = exp_df.pivot_table(index='Location', columns='Risk factor', values=col_map['upper'])
    exp_low = exp_df.pivot_table(index='Location', columns='Risk factor', values=col_map['lower'])
    
    # 合并
    merged = pd.merge(pca_clean, bph_clean, on='Location')
    merged = pd.merge(merged, exp_val.add_suffix('_val'), on='Location')
    merged = pd.merge(merged, exp_up.add_suffix('_up'), on='Location')
    merged = pd.merge(merged, exp_low.add_suffix('_low'), on='Location')
    
    return merged

# ==========================================
# 2. 鲁棒性负二项回归函数 (防止 BPH 卡死)
# ==========================================
def get_robust_beta(y, x):
    X = sm.add_constant(x)
    # 使用 bfgs 求解器，这是处理大规模或高离散数据最稳定的算法
    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(
            method='bfgs', maxiter=300, skip_hessian=True
        )
        beta = model.params.iloc[1]
        if np.isfinite(beta):
            return beta
    except:
        pass
    
    # 如果 bfgs 失败，尝试弹性求解
    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit(method='cg', maxiter=200)
        return model.params.iloc[1]
    except:
        return None

# ==========================================
# 3. 核心归因分析函数 (TMREL 修正版)
# ==========================================
def perform_paf_analysis(df, target_col, factors):
    tmrels = {
        'Kidney dysfunction': 0,
        'High alcohol use': 0,
        'High body-mass index': 22.5,
        'Diet high in sodium': 3.0,
        'Smoking': 0
    }

    results = []
    country_paf_matrix = [] 

    print(f"\n🚀 开始分析 {target_col} 的归因负担...")

    for factor in factors:
        val_col = f"{factor}_val"
        up_col = f"{factor}_up"
        low_col = f"{factor}_low"

        if val_col not in df.columns:
            print(f"  ⚠️ 跳过 {factor}: 找不到暴露数据")
            continue

        print(f"  > 正在计算: {factor} ...")
        
        beta = get_robust_beta(df[target_col], df[val_col])
        if beta is None:
            print(f"  ❌ {factor} 的负二项回归无法收敛。")
            continue

        tmrel = tmrels.get(factor, 0)

        # 核心 PAF 公式 (1 - exp(beta * (tmrel - X)))
        paf_val_arr = np.clip(1 - np.exp(beta * (tmrel - df[val_col])), 0, 1)
        paf_low_arr = np.clip(1 - np.exp(beta * (tmrel - df[low_col])), 0, 1)
        paf_up_arr = np.clip(1 - np.exp(beta * (tmrel - df[up_col])), 0, 1)

        country_paf_matrix.append(paf_val_arr)

        results.append({
            'Risk_Factor': factor,
            'PAF_Mean (%)': paf_val_arr.mean() * 100,
            'PAF_Lower (%)': paf_low_arr.mean() * 100,
            'PAF_Upper (%)': paf_up_arr.mean() * 100
        })

    # 计算联合 PAF (Combined)
    if country_paf_matrix:
        combined_arr = 1 - np.prod([1 - p for p in country_paf_matrix], axis=0)
        total_paf = combined_arr.mean() * 100
    else:
        total_paf = 0

    return pd.DataFrame(results), total_paf

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    try:
        data = load_data()
        
        # PCA
        pca_factors = ['Kidney dysfunction', 'High alcohol use', 'High body-mass index']
        pca_res, pca_total = perform_paf_analysis(data, 'PCA_Inc', pca_factors)
        
        # BPH
        bph_factors = ['Kidney dysfunction', 'Diet high in sodium', 'Smoking']
        bph_res, bph_total = perform_paf_analysis(data, 'BPH_Inc', bph_factors)

        print("\n" + "="*50)
        print("📊 最终归因总结报告 (95% UI)")
        print("="*50)
        print(f"PCA 总体联合归因: {pca_total:.2f}%")
        for _, r in pca_res.iterrows():
            print(f"  - {r['Risk_Factor']}: {r['PAF_Mean (%)']:.2f}% ({r['PAF_Lower (%)']:.2f}% - {r['PAF_Upper (%)']:.2f}%)")
        
        print(f"\nBPH 总体联合归因: {bph_total:.2f}%")
        for _, r in bph_res.iterrows():
            print(f"  - {r['Risk_Factor']}: {r['PAF_Mean (%)']:.2f}% ({r['PAF_Lower (%)']:.2f}% - {r['PAF_Upper (%)']:.2f}%)")
        print("="*50)

        pca_res.to_csv('PCA_PAF_Final_UI.csv', index=False)
        bph_res.to_csv('BPH_PAF_Final_UI.csv', index=False)
        print("✅ 详细数据已导出。")

    except Exception as e:
        print(f"\n❌ 程序运行失败: {str(e)}")
