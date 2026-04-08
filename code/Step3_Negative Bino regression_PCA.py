import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==========================================
# 1. 数据加载与集成
# ==========================================
def prepare_pca_data():
    files = ['PCA_inci_data.csv', 'exposure2021_GBD_BPH_combined.csv']
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"找不到文件: {f}，请确保它在当前文件夹。")

    # 读取数据
    pca_df = pd.read_csv('PCA_inci_data.csv')
    exp_df = pd.read_csv('exposure2021_GBD_BPH_combined.csv')

    # 清洗：提取 PCA 发病率
    pca_clean = pca_df[pca_df['Population'] == 'All Population'][['Location', 'Value']].rename(columns={'Value': 'PCA_Inc'})
    
    # 暴露因素透视表
    exp_pivot = exp_df.pivot_table(index='Location', columns='Risk factor', values='Value').reset_index()
    
    # 合并
    df = pd.merge(exp_pivot, pca_clean, on='Location', how='inner')
    return df

# ==========================================
# 2. 负二项回归 (1% 增量)
# ==========================================
def run_pca_regression():
    try:
        df = prepare_pca_data()
        
        # 锁定您指定的 11 个暴露因素
        pca_exposure_list = [
            'Kidney dysfunction',
            'High alcohol use',
            'Diet high in processed meat',
            'High body-mass index',
            'Diet low in legumes',
            'Diet low in whole grains',
            'High LDL cholesterol',
            'Diet high in sugar-sweetened beverages',
            'Diet high in red meat',
            'Diet low in nuts and seeds',
            'Diet low in fiber'
        ]

        # 检查特征是否都在数据中
        missing = [f for f in pca_exposure_list if f not in df.columns]
        if missing:
            print(f"警告：数据中缺失以下因素: {missing}")
            # 移除缺失的因素继续运行
            pca_exposure_list = [f for f in pca_exposure_list if f in df.columns]

        # 准备变量 (1% 增量，不除以 10)
        y = df['PCA_Inc']
        X = df[pca_exposure_list]
        X = sm.add_constant(X)

        print(f"正在对 PCA 的 {len(pca_exposure_list)} 个因素运行负二项回归...")

        # 步骤 A: 拟合 Poisson 模型以估算 alpha (离散参数)
        p_mod = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        y_hat = p_mod.predict(X)
        alpha_est = max(0.01, (((y - y_hat)**2 - y) / (y_hat**2)).mean())
        
        # 步骤 B: 拟合负二项回归模型
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_est)).fit()

        # 提取结果
        results = pd.DataFrame({
            'Exposure_Factor': nb_model.params.index,
            'RR': np.exp(nb_model.params),
            '95%_CI_Low': np.exp(nb_model.conf_int()[0]),
            '95%_CI_High': np.exp(nb_model.conf_int()[1]),
            'P_value': nb_model.pvalues
        })

        # 剔除常数项并格式化
        results = results[results['Exposure_Factor'] != 'const'].reset_index(drop=True)
        
        # 打印简要结果
        print("\n--- PCA 负二项回归结果 (1% 增量) ---")
        print(results[['Exposure_Factor', 'RR', 'P_value']])

        # 保存结果
        results.to_csv('PCA_Negative_Binomial_Results_1Percent.csv', index=False)
        print("\n详细结果已保存至: PCA_Negative_Binomial_Results_1Percent.csv")

    except Exception as e:
        print(f"运行失败: {e}")

# ==========================================
# 3. 执行
# ==========================================
if __name__ == "__main__":
    # 提醒：文件名请勿命名为 shap.py 或 statsmodels.py
    run_pca_regression()
