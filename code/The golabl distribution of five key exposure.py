import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np

# ================= 1. 配置区 =================
CSV_PATH = 'exposure2021_GBD_BPH_combined.csv'
SHP_PATH = 'ne_50m_admin_0_countries.shp' 

# 定义 5 个关键风险因子
RISK_FACTORS = [
    'Kidney dysfunction', 
    'High alcohol use', 
    'High body-mass index', 
    'Smoking', 
    'Diet high in sodium'
]

# 国家名称映射表 (请补充完整)
NAME_MAP = {
    'United States': 'United States of America', 'Türkiye': 'Turkey',
    'Russian Federation': 'Russia', 'China (mainland)': 'China',
    'Democratic Republic of the Congo': 'Dem. Rep. Congo',
    'Republic of Korea': 'South Korea', 'Viet Nam': 'Vietnam'
}

# ================= 2. 数据处理：计算综合得分 (CRI) =================
print("正在读取数据并计算综合风险指数...")
df = pd.read_csv(CSV_PATH)
world = gpd.read_file(SHP_PATH)

combined_scores = None

for risk in RISK_FACTORS:
    # 筛选数据
    target = df.loc[
        (df['Risk factor'] == risk) & (df['Year'] == 2021) & 
        (df['Sex'] == 'Male') & (df['Age'] == 'Age-standardized')
    ].copy()
    
    # 计算四分位数 (0, 1, 2, 3 分)
    q = target['Value'].quantile([0.25, 0.5, 0.75]).values
    target['Score'] = 0
    target.loc[target['Value'] > q[0], 'Score'] = 1
    target.loc[target['Value'] > q[1], 'Score'] = 2
    target.loc[target['Value'] > q[2], 'Score'] = 3
    
    # 提取得分表
    risk_score = target[['Location', 'Score', 'Value']].rename(columns={'Score': f'S_{risk}', 'Value': f'V_{risk}'})
    
    if combined_scores is None:
        combined_scores = risk_score
    else:
        combined_scores = pd.merge(combined_scores, risk_score, on='Location', how='outer')

# 计算总分 (CRI): 5个因子每个最高3分，总分最高15分
score_cols = [f'S_{r}' for r in RISK_FACTORS]
combined_scores['CRI'] = combined_scores[score_cols].sum(axis=1)
combined_scores['Location_Map'] = combined_scores['Location'].replace(NAME_MAP)

# 合并到地图
merged = world.merge(combined_scores, left_on='NAME', right_on='Location_Map', how='left')

# ================= 3. 绘图：模仿 Figure 4 (5图组合) =================
print("正在生成 Figure 4...")
fig4, axes = plt.subplots(3, 2, figsize=(22, 18))
plt.subplots_adjust(top=0.92, hspace=0.1, wspace=0.05)
axes = axes.flatten()

# 不同因子的颜色系
COLOR_SCHEMES = ['Blues', 'Greens', 'Reds', 'Purples', 'Oranges']

for i, risk in enumerate(RISK_FACTORS):
    ax = axes[i]
    world.plot(ax=ax, color='#f5f5f5', edgecolor='#dcdcdc', linewidth=0.1)
    
    # 使用分位数着色
    merged.plot(column=f'S_{risk}', ax=ax, cmap=COLOR_SCHEMES[i], 
                edgecolor='white', linewidth=0.2, legend=False)
    
    ax.set_axis_off()
    ax.set_title(f'({chr(65+i)}) {risk}', fontsize=18, fontweight='bold')

axes[5].set_axis_off() # 隐藏多余的格子
fig4.suptitle('Figure 4: Global Exposure of 5 Key Risk Factors (2021)', fontsize=26, fontweight='bold', y=0.97)
fig4.savefig('Figure_4_Combined_Factors.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

# ================= 4. 绘图：模仿 Figure 5 (综合指数) =================
print("正在生成 Figure 5...")
fig5, ax5 = plt.subplots(figsize=(15, 10))

# 绘制综合指数图 (CRI)
world.plot(ax=ax5, color='#f5f5f5', edgecolor='#dcdcdc', linewidth=0.2)
merged.plot(column='CRI', ax=ax5, cmap='YlOrRd', legend=True,
            legend_kwds={'label': "Composite Risk Index (Score 0-15)", 'orientation': "horizontal", 'pad': 0.05, 'shrink': 0.6},
            edgecolor='white', linewidth=0.3)

ax5.set_axis_off()
ax5.set_title('Figure 5: Global Distribution of Composite Risk Index', fontsize=22, fontweight='bold', pad=30)
fig5.savefig('Figure_5_Composite_Risk_Index.png', dpi=300, bbox_inches='tight', pad_inches=0.5)

print("所有图片已生成并保存到本地。")
plt.show()
