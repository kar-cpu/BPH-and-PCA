import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# ==========================================
# 1. 核心分析逻辑：负二项回归 (1% 增量)
# ==========================================
def run_nb_regression(df, target, factors):
    data = df[[target] + factors].dropna()
    y, X = data[target], sm.add_constant(data[factors])
    # 拟合并估算离散参数 alpha
    p_mod = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    alpha = max(0.01, (((y - p_mod.predict(X))**2 - y) / (p_mod.predict(X))**2).mean())
    nb_mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    return pd.DataFrame({
        'Risk_factor': nb_mod.params.index,
        'RR': np.exp(nb_mod.params),
        'lower': np.exp(nb_mod.conf_int()[0]),
        'upper': np.exp(nb_mod.conf_int()[1]),
        'p': nb_mod.pvalues
    }).iloc[1:]

# ==========================================
# 2. 斑马纹森林图绘制模块
# ==========================================
def draw_forest_zebra(pca_results, bph_results):
    common_name = 'Kidney dysfunction'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    plt.subplots_adjust(wspace=0.3)

    def draw_panel(ax, title, df, label, rr_range):
        XS, XE = 0.35, 0.65
        scale = lambda v: XS + (max(min(v, rr_range[1]), rr_range[0]) - rr_range[0])/(rr_range[1]-rr_range[0]) * (XE - XS)
        y_pos = np.linspace(0.95, 0.2, 15)
        
        # 基础装饰
        ax.text(-0.02, 0.98, label, weight='bold', size=24, transform=ax.transAxes)
        ax.text(0.5, 0.98, title, weight='bold', size=18, ha='center', transform=ax.transAxes)
        ax.plot([0, 1], [0.94, 0.94], color='black', lw=2.5, transform=ax.transAxes)
        
        # 表头
        ax.text(0.01, y_pos[1], "Exposure (Per 1% Increase)", weight='bold', size=12, transform=ax.transAxes)
        ax.text(0.8, y_pos[1], "RR (95% CI)", weight='bold', size=12, ha='center', transform=ax.transAxes)
        ax.text(0.95, y_pos[1], "P value", weight='bold', size=12, ha='center', transform=ax.transAxes)

        # 区分 Common 和 Specific
        c_df = df[df['Risk_factor'] == common_name]
        s_df = df[df['Risk_factor'] != common_name]

        def fill_section(data, start_idx, subtitle):
            ptr = start_idx
            if not data.empty:
                ax.text(0.01, y_pos[ptr], subtitle, weight='bold', size=12, color='#1f77b4', transform=ax.transAxes)
                ptr += 1
                for i, (_, row) in enumerate(data.iterrows()):
                    y = y_pos[ptr]
                    # 斑马纹效果
                    ax.axhspan(y-0.025, y+0.025, xmin=0, xmax=1, color='#f5f5f5' if i%2==0 else 'white', transform=ax.transAxes, zorder=0)
                    # 文字信息
                    ax.text(0.01, y, row['Risk_factor'], transform=ax.transAxes, va='center', fontsize=11)
                    ax.text(0.8, y, f"{row['RR']:.2f} ({row['lower']:.2f}-{row['upper']:.2f})", ha='center', transform=ax.transAxes, va='center', fontsize=11)
                    p_txt = "<0.001" if row['p'] < 0.001 else f"{row['p']:.3f}"
                    ax.text(0.95, y, p_txt, ha='center', transform=ax.transAxes, va='center', fontsize=11)
                    # 森林图标记
                    sx, slo, sup = scale(row['RR']), scale(row['lower']), scale(row['upper'])
                    ax.plot([slo, sup], [y, y], color='black', lw=1.8, transform=ax.transAxes, zorder=2)
                    ax.plot(sx, y, marker='s', color='#d62728', markersize=10, markeredgecolor='black', transform=ax.transAxes, zorder=3)
                    ptr += 1
            return ptr

        nxt = fill_section(c_df, 3, "Common exposure")
        fill_section(s_df, nxt + 1, "Specific exposure")

        # 无效线 (RR=1.0)
        v1 = scale(1.0)
        ax.plot([v1, v1], [0.15, y_pos[2]], color='black', ls='--', lw=1.2, transform=ax.transAxes, zorder=1)
        # 刻度
        for t in np.linspace(rr_range[0], rr_range[1], 5):
            tx = scale(t)
            ax.text(tx, 0.11, f"{t:.2f}", ha='center', transform=ax.transAxes, fontsize=10)
            ax.plot([tx, tx], [0.15, 0.13], color='black', lw=1.5, transform=ax.transAxes)
        ax.text((XS+XE)/2, 0.04, "Relative Risk (RR)", ha='center', weight='bold', size=13, transform=ax.transAxes)
        ax.set_axis_off()

    # 绘制
    draw_panel(ax1, "Prostate cancer (PCA)", pca_results, "A", (0.95, 1.35))
    draw_panel(ax2, "Benign prostatic hyperplasia (BPH)", bph_results, "B", (0.98, 1.15))
    
    plt.savefig('ForestPlot_Final_Selected_Zebra.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 3. 执行流程 (数据加载与筛选)
# ==========================================
if __name__ == "__main__":
    # 模拟从回归中提取数据 (您可以直接使用之前生成的 csv)
    # 此处假设文件存在并加载
    try:
        # 加载基础数据并进行回归获取最新值
        pca_raw = pd.read_csv('PCA_inci_data.csv')
        bph_raw = pd.read_csv('BPH_inci_cleaned_full.csv')
        exp_raw = pd.read_csv('exposure2021_GBD_BPH_combined.csv')

        pca_c = pca_raw[pca_raw['Population'] == 'All Population'][['Location', 'Value']].rename(columns={'Value': 'PCA_Inc'})
        bph_c = bph_raw[['location_name', 'val']].rename(columns={'location_name': 'Location', 'val': 'BPH_Inc'})
        exp_p = exp_raw.pivot_table(index='Location', columns='Risk factor', values='Value').reset_index()
        df_all = pd.merge(exp_p, pca_c, on='Location').merge(bph_c, on='Location')

        # 提取指定因素
        pca_res = run_nb_regression(df_all, 'PCA_Inc', ['Kidney dysfunction', 'High alcohol use', 'High body-mass index'])
        bph_res = run_nb_regression(df_all, 'BPH_Inc', ['Kidney dysfunction', 'Diet high in sodium', 'Smoking'])

        # 绘图
        draw_forest_zebra(pca_res, bph_res)
        print("森林图已生成：ForestPlot_Final_Selected_Zebra.png")
    except Exception as e:
        print(f"运行失败: {e}. 请确保 CSV 文件在同一目录下。")
