import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import country_converter as coco
import numpy as np
import matplotlib.colors as mcolors

# =====================================
# 辅助函数：截断色带（让 Q1 颜色更显眼）
# =====================================
def truncate_colormap(cmap, minval=0.18, maxval=0.9, n=100):
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

# =========================
# 1. 读取地图与数据
# =========================
world = gpd.read_file("ne_50m_admin_0_countries.shp")
world = world[world['ADMIN'] != 'Antarctica']

# 匹配键优化
world['merge_key'] = world['ISO_A3']
world.loc[world['merge_key'].isin(['-99', None]), 'merge_key'] = world['ADM0_A3']

df = pd.read_csv("BPH_PCA_cooccurrence_2021.csv")
df["ISO3"] = coco.convert(names=df["Location"], to="ISO3")

world_data = world.merge(df, left_on="merge_key", right_on="ISO3", how="left")

# =========================
# 2. 四分位分类：统一圆括号 ( )
# =========================
def get_formatted_quartiles(series):
    if series.dropna().empty: return series
    cat = pd.qcut(series, 4, duplicates="drop")
    new_labels = [f"({i.left:.1f}, {i.right:.1f})" for i in cat.cat.categories]
    return cat.cat.rename_categories(new_labels)

world_data["BPH_cat"] = get_formatted_quartiles(world_data["BPH_incidence"])
world_data["PCA_cat"] = get_formatted_quartiles(world_data["PCA_incidence"])

# =========================
# 3. 绘图配置 (Figure 1 终极布局)
# =========================
fig, axes = plt.subplots(3, 1, figsize=(12, 22), gridspec_kw={'height_ratios': [1, 1, 1]})
plt.subplots_adjust(hspace=0.2) 

custom_cmap = truncate_colormap(plt.get_cmap("RdYlGn_r"), minval=0.15, maxval=0.85)

# --- A 图：BPH ---
world.plot(ax=axes[0], color="#eeeeee", edgecolor="black", linewidth=0.1) 
world_data.dropna(subset=["BPH_cat"]).plot(
    column="BPH_cat", cmap=custom_cmap, linewidth=0.3, edgecolor="black",
    legend=True, ax=axes[0],
    legend_kwds={
        "loc": "lower left", "bbox_to_anchor": (0.02, 0.05), "frameon": False, 
        "title": "BPH (per 100,000)", "title_fontsize": 10, "fontsize": 8, "handlelength": 1.2
    }
)
axes[0].set_title("A  Global distribution of BPH incidence", fontsize=15, loc="left", pad=15, fontweight='bold')
axes[0].axis("off")

# --- B 图：PCA ---
world.plot(ax=axes[1], color="#eeeeee", edgecolor="black", linewidth=0.1)
world_data.dropna(subset=["PCA_cat"]).plot(
    column="PCA_cat", cmap=custom_cmap, linewidth=0.3, edgecolor="black",
    legend=True, ax=axes[1],
    legend_kwds={
        "loc": "lower left", "bbox_to_anchor": (0.02, 0.05), "frameon": False, 
        "title": "PCA (per 100,000)", "title_fontsize": 10, "fontsize": 8, "handlelength": 1.2
    }
)
axes[1].set_title("B  Global distribution of PCA incidence", fontsize=15, loc="left", pad=15, fontweight='bold')
axes[1].axis("off")

# --- C 图：共病模式 ---
pattern_colors = {"consistent": "#d73027", "BPH-dominant": "#fee08b", "PCA-dominant": "#a6d96a"}
world_data["color"] = world_data["pattern"].map(pattern_colors).fillna("#eeeeee")

world_data.plot(color=world_data["color"], edgecolor="black", linewidth=0.3, ax=axes[2])
axes[2].set_title("C  Co-occurrence pattern of BPH and PCA incidence", fontsize=15, loc="left", pad=15, fontweight='bold')
axes[2].axis("off")

legend_elements = [
    mpatches.Patch(color="#d73027", label="Consistent"),
    mpatches.Patch(color="#a6d96a", label="PCA dominant"),
    mpatches.Patch(color="#fee08b", label="BPH dominant")
]
axes[2].legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.02, 0.05), frameon=False, fontsize=10)

# =========================
# 4. Inset Matrix (严格对齐版)
# =========================
# 使用 df_clean 确保计数准确
df_clean = df.dropna(subset=["BPH_incidence", "PCA_incidence"])
df_clean["BPH_pct"] = pd.qcut(df_clean["BPH_incidence"], 4, labels=[0, 1, 2, 3])
df_clean["PCA_pct"] = pd.qcut(df_clean["PCA_incidence"], 4, labels=[0, 1, 2, 3])
count_matrix = pd.crosstab(df_clean["PCA_pct"], df_clean["BPH_pct"])

# --- 重新构建 4x4 颜色矩阵：仅对角线为红色 ---
pattern_matrix = np.zeros((4, 4))
for i in range(4): # PCA 秩次 (y)
    for j in range(4): # BPH 秩次 (x)
        if i == j:
            pattern_matrix[i, j] = 0 # 对角线：Consistent (Red)
        elif i < j:
            pattern_matrix[i, j] = 1 # 右下半：BPH dominant (Yellow)
        else:
            pattern_matrix[i, j] = 2 # 左上半：PCA dominant (Green)

matrix_cmap = mcolors.ListedColormap([pattern_colors["consistent"], 
                                      pattern_colors["BPH-dominant"], 
                                      pattern_colors["PCA-dominant"]])

# 将矩阵放置在大西洋中心 (0.55 为最佳位置)
inset = axes[2].inset_axes([0.55, 0.05, 0.28, 0.28]) 
inset.imshow(pattern_matrix, cmap=matrix_cmap, interpolation='nearest', origin='lower')

# 填充各单元格对应的实际国家数量
for i in range(4):
    for j in range(4):
        val = count_matrix.loc[i, j] if (i in count_matrix.index and j in count_matrix.columns) else 0
        inset.text(j, i, int(val), ha="center", va="center", fontsize=8, fontweight="bold")

# 格式化刻度（不添加外部文字）
inset.set_xticks(range(4)); inset.set_yticks(range(4))
inset.set_xticklabels(["Q1","Q2","Q3","Q4"], fontsize=7)
inset.set_yticklabels(["Q1","Q2","Q3","Q4"], fontsize=7)
inset.set_title("", pad=0)
inset.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

# =========================
# 5. 保存为 PNG 格式
# =========================
plt.savefig("Figure1_Final_Strict.png", dpi=600, bbox_inches="tight")
print("Figure 1 已更新！方格图颜色已修正为严格对齐模式，已保存为 PNG。")
plt.show()
