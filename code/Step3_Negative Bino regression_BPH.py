import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==========================================
# 1. 数据加载与集成
# ==========================================
def prepare_bph_data():
    files = ['BPH_inci_cleaned_full.csv', 'exposure2021_GBD_BPH_combined.csv']
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}，请确保它在当前脚本所在的文件夹。")

    # 读取数据
    bph_df = pd.read_csv('BPH_inci_cleaned_full.csv')
    exp_df = pd.read_csv('exposure2021_GBD_BPH_combined.csv')

    # 清洗：提取 BPH 发病率
    bph_clean = bph_df[['location_name', 'val']].rename(columns={'location_name': 'Location', 'val': 'BPH_Inc'})
    
    # 暴露因素透视表 (将长表转为宽表)
    exp_pivot = exp_df.pivot_table(index='Location', columns='Risk factor', values='Value').reset_index()
    
    # 合并
    df = pd.merge(exp_pivot, bph_clean, on='Location', how='inner')
    return df

# ==========================================
# 2. BPH 负二项回归 (1% 增量)
# ==========================================
def run_bph_regression():
    try:
        df = prepare_bph_data()
        
        # 锁定您指定的 10 个 BPH 暴露因素
        bph_exposure_list = [
            'Kidney dysfunction',
            'Diet high in sodium',
            'High alcohol use',
            'Smoking',
            'High LDL cholesterol',
            'High fasting plasma glucose',
            'Diet low in whole grains',
            'Diet low in fiber',
            'Diet low in milk',
            'High body-mass index'
        ]

        # 检查特征是否都在数据中
        missing = [f for f in bph_exposure_list if f not in df.columns]
        if missing:
            print(f"警告：数据中缺失以下因素: {missing}")
            bph_exposure_list = [f for f in bph_exposure_list if f in df.columns]

        # 准备变量
        y = df['BPH_Inc']
        X = df[bph_exposure_list]
        X = sm.add_constant(X)

        print(f"正在对 BPH 的 {len(bph_exposure_list)} 个因素运行负二项回归分析...")

        # 步骤 A: 拟合 Poisson 模型以估算 alpha (离散参数)
        p_mod = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        y_hat = p_mod.predict(X)
        alpha_est = max(0.01, (((y - y_hat)**2 - y) / (y_hat**2)).mean())
        
        # 步骤 B: 拟合负二项回归模型
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_est)).fit()

        # 提取结果并计算 RR 值
        results = pd.DataFrame({
            'Exposure_Factor': nb_model.params.index,
            'RR': np.exp(nb_model.params),
            '95%_CI_Low': np.exp(nb_model.conf_int()[0]),
            '95%_CI_High': np.exp(nb_model.conf_int()[1]),
            'P_value': nb_model.pvalues
        })

        # 剔除常数项 (Intercept)
        results = results[results['Exposure_Factor'] != 'const'].reset_index(drop=True)
        
        # 打印结果概览
        print("\n--- BPH 负二项回归结果概览 (1% 增量) ---")
        print(results[['Exposure_Factor', 'RR', 'P_value']])

        # 保存结果到 CSV
        output_name = 'BPH_Negative_Binomial_Results_1Percent.csv'
        results.to_csv(output_name, index=False)
        print(f"\n详细统计结果已保存至: {output_name}")

    except Exception as e:
        print(f"运行失败: {e}")

# ==========================================
# 3. 执行
# ==========================================
if __name__ == "__main__":
    # 请确保文件名不叫 statsmodels.py 或 pandas.py
    run_bph_regression()
