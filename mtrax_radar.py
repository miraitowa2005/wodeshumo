import torch
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib as mpl
from model import PhysicsAttentionModel
from torch.utils.data import DataLoader, Dataset

# ===================== 1. 环境配置 =====================
BASE_DIR = "/root/autodl-tmp/2026MCM"
DATA_PATH = os.path.join(BASE_DIR, "processed_data/test_gamer.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "best_physics_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data/scaler.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')

DEVICE = torch.device("cpu")
SEQ_LENGTH = 60
DOWNSAMPLE_STEP = 200 # 7组数据量大，进一步加大采样步长以保证速度

# ===================== 2. 核心分析引擎 =====================
def analyze_all_batteries():
    print("🧠 正在对全量电池组进行多维物理感知评估...")
    model = PhysicsAttentionModel(4, 256, 8, 6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    mean_curr, scale_curr = scaler.mean_[0], scaler.scale_[0]

    full_df = pd.read_parquet(DATA_PATH)
    battery_ids = sorted(full_df['battery_id'].unique())
    
    results = []
    for bid in battery_ids:
        print(f"📊 正在处理: {bid}")
        df_bat = full_df[full_df['battery_id'] == bid].iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)
        if len(df_bat) < 100: continue

        raw_data = df_bat[['current', 'temperature', 'SOC', 'dI']].values.astype(np.float32)
        true_v = df_bat['voltage'].values[SEQ_LENGTH:]
        i_unnorm = raw_data[SEQ_LENGTH:, 0] * scale_curr + mean_curr

        # 推理
        with torch.no_grad():
            inputs = torch.from_numpy(raw_data)
            preds = []
            # 简化推理流以加速
            for t in range(0, len(raw_data) - SEQ_LENGTH):
                x = inputs[t : t + SEQ_LENGTH].unsqueeze(0)
                cur_i = torch.tensor([[i_unnorm[t]]])
                pv, _, _, _ = model(x, cur_i)
                preds.append(pv.item())
        
        pv_all = np.array(preds)
        
        # 指标计算 (已针对论文视角优化)
        err = (true_v - pv_all) * 1000
        results.append({
            "ID": bid,
            "RMSE": np.sqrt(np.mean(err**2)),
            "Stability": np.std(err),
            "MaxErr": np.max(np.abs(err)),
            "PhysFid": np.mean(np.abs(np.diff(pv_all) - np.diff(true_v))) * 1000,
            "MAE": np.mean(np.abs(err))
        })

    return pd.DataFrame(results)

# ===================== 3. 小倍数雷达图矩阵绘制 =====================
def plot_radar_matrix(df):
    labels = ['RMSE', 'Stability', 'MaxErr', 'PhysFid', 'MAE']
    num_vars = len(labels)
    
    # 归一化：用于绘图展示，值越大代表性能越好（误差越小）
    df_norm = df.copy()
    for col in labels:
        mi, ma = df[col].min(), df[col].max()
        df_norm[col] = 1 - (df[col] - mi) / (ma - mi + 1e-6)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 设置网格布局 (例如 2行4列)
    num_bats = len(df)
    cols = 4
    rows = (num_bats + cols - 1) // cols
    
    fig = plt.figure(figsize=(20, 5 * rows))
    cmap = mpl.colormaps['viridis']

    for i, (idx, row) in enumerate(df_norm.iterrows()):
        ax = fig.add_subplot(rows, cols, i+1, polar=True)
        
        values = [row[l] for l in labels]
        values += [values[0]]
        
        color = cmap(i / num_bats)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"Battery: {df.iloc[i]['ID']}", size=14, color=color, pad=20)
        ax.set_ylim(0, 1) # 统一刻度才有对比意义

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "radar_matrix.png"), dpi=300)
    print(f"🌈 雷达矩阵图已保存至: {OUTPUT_DIR}/radar_matrix.png")

if __name__ == "__main__":
    metrics_df = analyze_all_batteries()
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "all_batteries_metrics.csv"), index=False)
    plot_radar_matrix(metrics_df)