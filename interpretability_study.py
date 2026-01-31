import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from model import PhysicsAttentionModel

# ===================== 1. 基础配置 (相对路径) =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "test_gamer.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_physics_model_final.pth")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.pkl")
RESULT_DIR = os.path.join(BASE_DIR, "interpretability_results")
os.makedirs(RESULT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial'] # 确保学术字体显示

# ===================== 2. 核心分析逻辑 =====================
def analyze_physics_meaning(target_battery="RW17"):
    print(f"🔍 正在深入分析电池 {target_battery} 的物理参数含义...")
    
    # 加载模型 (必须与 train.py 中的配置一致!)
    # RTX 5090 Config: d_model=256, nhead=8, num_layers=6
    model = PhysicsAttentionModel(feature_dim=4, d_model=256, nhead=8, num_layers=6)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        print("💡 提示: 请先运行 'python train.py' 完成训练！")
        return

    # 加载权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=True)
    model.eval()

    # 读取并准备数据
    df_all = pd.read_parquet(DATA_PATH)
    df_bat = df_all[df_all['battery_id'] == target_battery].iloc[:5000].reset_index(drop=True)
    
    raw_data = df_bat[['current', 'temperature', 'SOC', 'dI']].values.astype(np.float32)
    true_v = df_bat['voltage'].values[60:]
    
    inputs = torch.from_numpy(raw_data)
    
    # 存储提取出的物理参数
    param_records = []

    with torch.no_grad():
        for t in range(len(raw_data) - 60):
            x = inputs[t : t + 60].unsqueeze(0)
            # 💡 假设你的模型 forward 返回: (voltage, r0, ocv, up, r0_seq)
            # 请根据你 model.py 中真实的 return 顺序调整
            pred_v, hat_r0, hat_ocv, hat_up, _ = model(x, torch.tensor([[0.0]])) 
            
            param_records.append({
                "Time": t,
                "True_V": true_v[t],
                "Pred_V": pred_v.item(),
                "Estimated_OCV": hat_ocv.item(),
                "Estimated_R0": hat_r0.item(),
                "Estimated_Up": hat_up.item(),
                "Current": raw_data[t+60, 0] # 实时电流
            })

    res_df = pd.DataFrame(param_records)

    # ===================== 3. 物理意义可视化 =====================
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # 图 1: 电压拟合对齐
    axes[0].plot(res_df['Time'], res_df['True_V'], label='Measured Voltage', color='black', alpha=0.6)
    axes[0].plot(res_df['Time'], res_df['Pred_V'], label='PIRNN Predicted', color='red', linestyle='--')
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend()
    axes[0].set_title(f"Interpretability Analysis: {target_battery}")

    # 图 2: OCV 的物理趋势 (应该随放电平滑下降)
    axes[1].plot(res_df['Time'], res_df['Estimated_OCV'], label='Estimated OCV (Internal State)', color='blue')
    axes[1].set_ylabel("OCV (V)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 图 3: R0 与电流的关联 (验证欧姆定律)
    ax3_2 = axes[2].twinx()
    axes[2].plot(res_df['Time'], res_df['Estimated_R0'], label='Estimated R0 (Ohmic)', color='green')
    ax3_2.plot(res_df['Time'], res_df['Current'], label='Current Load', color='gray', alpha=0.3)
    axes[2].set_ylabel("Internal Resistance (Ω)")
    ax3_2.set_ylabel("Current (A)")
    axes[2].legend(loc='upper left')

    # 图 4: Up 极化电压 (验证动态滞后)
    axes[3].plot(res_df['Time'], res_df['Estimated_Up'], label='Estimated Up (Polarization)', color='purple')
    axes[3].set_ylabel("Polarization Voltage (V)")
    axes[3].set_xlabel("Time Steps (Decimated)")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/physics_verification_{target_battery}.png", dpi=300)
    print(f"✅ 物理意义分析图已保存至 {RESULT_DIR} 文件夹。")

if __name__ == "__main__":
    analyze_physics_meaning("RW20")