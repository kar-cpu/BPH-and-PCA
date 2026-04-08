import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ================= 1. 配置区 =================
CSV_PATH = 'Global_CRI_and_Classification_Final.csv'
OUTPUT_NAME = 'Figure_3_Global_CRI_Distribution_Stacked.png'

# 大洲配色方案
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
CONTINENT_ORDER = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']

# 终极版全量映射表 (已补充全部 GBD 官方长名称与微型岛国)
CONTINENT_MAPPING = {
    'Africa': [
        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 
        'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo', "Cote d'Ivoire", 
        "Côte d'Ivoire", 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 
        'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 
        'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 
        'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 
        'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 
        'South Sudan', 'Sudan', 'Tanzania', 'United Republic of Tanzania', 'Togo', 'Tunisia', 
        'Uganda', 'Zambia', 'Zimbabwe'
    ],
    'Americas': [
        'Antigua and Barbuda', 'Argentina', 'Bahamas', 'Barbados', 'Belize', 'Bolivia', 
        'Bolivia (Plurinational State of)', 'Brazil', 'Canada', 'Chile', 'Colombia', 'Costa Rica', 
        'Cuba', 'Dominica', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Greenland', 'Grenada', 
        'Guatemala', 'Guyana', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 
        'Paraguay', 'Peru', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 
        'Suriname', 'Trinidad and Tobago', 'United States of America', 'Uruguay', 'Venezuela', 
        'Venezuela (Bolivarian Republic of)', 'Puerto Rico', 'Bermuda', 'United States Virgin Islands'
    ],
    'Asia': [
        'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 
        'Brunei Darussalam', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran', 
        'Iran (Islamic Republic of)', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 
        'Kyrgyzstan', 'Laos', "Lao People's Democratic Republic", 'Lebanon', 'Malaysia', 'Maldives', 
        'Mongolia', 'Myanmar', 'Nepal', 'North Korea', "Democratic People's Republic of Korea", 
        'Oman', 'Pakistan', 'Palestine', 'Philippines', 'Qatar', 'Saudi Arabia', 'Singapore', 
        'South Korea', 'Republic of Korea', 'Sri Lanka', 'Syria', 'Syrian Arab Republic', 'Taiwan', 
        'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Türkiye', 'Turkmenistan', 
        'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Viet Nam', 'Yemen'
    ],
    'Europe': [
        'Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 
        'Croatia', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 
        'Iceland', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 
        'Republic of Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 
        'Poland', 'Portugal', 'Romania', 'Russia', 'Russian Federation', 'San Marino', 'Serbia', 
        'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom'
    ],
    'Oceania': [
        'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia', 'Micronesia (Federated States of)', 
        'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 
        'Tuvalu', 'Vanuatu', 'Northern Mariana Islands', 'Tokelau', 'Guam', 'Cook Islands', 'Niue', 
        'American Samoa'
    ]
}

# ================= 2. 数据处理 =================
print("正在处理 CRI 分布数据...")
df = pd.read_csv(CSV_PATH)

# 将映射表反转为 {国家: 大洲} 格式
country_to_continent = {}
for continent, countries in CONTINENT_MAPPING.items():
    for country in countries:
        country_to_continent[country] = continent

# 添加大洲列
df['Continent'] = df['Location'].map(country_to_continent).fillna('Other')

# 检查是否还有漏网之鱼
unmapped = df[df['Continent'] == 'Other']['Location'].unique()
if len(unmapped) > 0:
    print(f"\n提示: 仍有 {len(unmapped)} 个地点未匹配到大洲 (归入 Other):\n{unmapped}\n")
else:
    print("\n太棒了！所有 204 个国家/地区已全部完美匹配！\n")

# 统计每个 CRI 分数下，各个大洲的国家数量
max_cri = int(df['CRI'].max()) if pd.notna(df['CRI'].max()) else 15
grouped = df.groupby(['CRI', 'Continent']).size().reset_index(name='Count')

# 数据透视
pivot_df = grouped.pivot(index='CRI', columns='Continent', values='Count').fillna(0)
pivot_df = pivot_df.reindex(index=range(0, max_cri + 1), fill_value=0)

# 确保列顺序
plot_columns = [c for c in CONTINENT_ORDER if c in pivot_df.columns]
if 'Other' in pivot_df.columns:
    plot_columns.append('Other')
    COLORS.append('#cccccc')

# ================= 3. 绘图 =================
fig, ax = plt.subplots(figsize=(14, 8))

bottom = np.zeros(len(pivot_df))

for i, continent in enumerate(plot_columns):
    values = pivot_df[continent].values
    ax.bar(pivot_df.index, values, bottom=bottom, 
           color=COLORS[i], label=continent, 
           edgecolor='white', linewidth=1.0, width=0.75)
    bottom += values

# ================= 4. 美化与标注 =================
for idx, total in zip(pivot_df.index, bottom):
    if total > 0:
        ax.text(idx, total + 0.5, str(int(total)), 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')

ax.set_xlabel('Composite Risk Index (Score 0-15)', fontsize=16, fontweight='bold', labelpad=15)
ax.set_ylabel('Number of Countries', fontsize=16, fontweight='bold', labelpad=15)
ax.set_title('Global Distribution of Countries by CRI Score', fontsize=22, fontweight='bold', pad=30)

plt.xticks(range(0, max_cri + 1), fontsize=12)
plt.yticks(fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(title='Continent', fontsize=12, title_fontsize=14, frameon=False, 
          loc='upper left', bbox_to_anchor=(1.02, 1))

plt.tight_layout()
plt.savefig(OUTPUT_NAME, dpi=300, bbox_inches='tight', facecolor='white')
print(f"堆叠柱状图已保存至: {os.path.abspath(OUTPUT_NAME)}")

plt.show()
