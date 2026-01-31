import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===================== 1. 路径配置 =====================
BASE_DIR = "/root/autodl-tmp/2026MCM"
CSV_PATH = os.path.join(BASE_DIR, "results/all_batteries_metrics.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

import matplotlib
matplotlib.use('Agg')

# ===================== 2. 独立归一化绘图逻辑 =====================
def plot_radar_matrix_individual(df):
    labels = ['RMSE', 'Stability', 'MaxErr', 'PhysFid', 'MAE']
    num_vars = len(labels)
    
    # 获取电池列表
    battery_ids = df['ID'].tolist()
    num_bats = len(battery_ids)
    cols = 4
    rows = (num_bats + cols - 1) // cols
    
    fig = plt.figure(figsize=(20, 5 * rows))
    cmap = mpl.colormaps['tab10'] # 更有区分度的色板

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for i, bid in enumerate(battery_ids):
        ax = fig.add_subplot(rows, cols, i+1, polar=True)
        
        # 提取该电池的数据行
        row_data = df[df['ID'] == bid].iloc[0]
        
        # 🔥 关键：此处不再进行跨电池归一化，而是展示指标在“阈值范围”内的相对表现
        # 假设我们定义一个“理想范围”作为参考（也可根据该电池的最大偏差动态调整）
        raw_values = [row_data[l] for l in labels]
        
        # 为了让图形可见，我们进行局部 log 处理或简单的逆向归一化显示
        # 这里使用局部 Max-Min（仅用于形态观察）
        norm_values = []
        for l in labels:
            # 这里的逻辑是：值越小，分数越高（越靠近外圈）
            # 我们用该列全局的最大值作为分母，确保形状可对比
            val = row_data[l]
            global_max = df[l].max()
            global_min = df[l].min()
            score = 1 - (val - global_min) / (global_max - global_min + 1e-6)
            norm_values.append(score)
        
        norm_values += [norm_values[0]]
        
        color = cmap(i % 10)
        ax.plot(angles, norm_values, color=color, linewidth=2)
        ax.fill(angles, norm_values, color=color, alpha=0.3)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_title(f"Battery: {bid}\n(Relative Score)", size=14, color=color, pad=25)
        
        # 即使是个体展示，我们也保留统一的 0-1 轴感，但形状会被拉伸
        ax.set_ylim(0, 1.1) 

    plt.tight_layout(pad=5.0)
    save_path = os.path.join(OUTPUT_DIR, "radar_matrix_individual_look.png")
    plt.savefig(save_path, dpi=300)
    print(f"🌈 独立观察版雷达图已保存至: {save_path}")

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        df_metrics = pd.read_csv(CSV_PATH)
        plot_radar_matrix_individual(df_metrics)
    else:
        print(f"❌ 找不到数据文件: {CSV_PATH}，请先运行分析脚本。")