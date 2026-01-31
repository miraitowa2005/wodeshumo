import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from model import PhysicsAttentionModel

# ===================== 1. 服务器路径配置 =====================
BASE_DIR = "/root/autodl-tmp/2026MCM"
MODEL_PATH = os.path.join(BASE_DIR, "best_physics_model.pth") 
SCALER_PATH = os.path.join(BASE_DIR, "processed_data/scaler.pkl")
TEST_FILES = {
    "Gamer": os.path.join(BASE_DIR, "processed_data/test_gamer.parquet"),
    "Reader": os.path.join(BASE_DIR, "processed_data/test_reader.parquet")
}
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 2. 模型与物理配置 =====================
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 6
SEQ_LEN = 60
Q_MAX = 2.1  # 电池额定容量 Ah

# 物理校准参数
CALIB_OCV_BIAS = 1.49729
CALIB_OCV_SCALE = 0.61104
CALIB_R0_SCALE = 1.14432

def load_model():
    print(f"正在加载模型: {MODEL_PATH}")
    model = PhysicsAttentionModel(feature_dim=4, d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

# ===================== 3. 深度分析函数 (包含 pred_R0_seq) =====================
def analyze_scenario(model, name, file_path, scaler):
    """
    不仅预测电压，还深入分析内阻序列的平滑性和物理一致性
    """
    print(f"🔍 正在执行深度物理分析: {name} ...")
    df = pd.read_parquet(file_path)
    battery_id = df['battery_id'].unique()[0]
    # 取一段 400 步的连续区间用于时序演变展示
    df_segment = df[df['battery_id'] == battery_id].iloc[2000:2400].reset_index(drop=True)
    
    feature_cols = ['current', 'temperature', 'SOC', 'dI']
    features = torch.tensor(df_segment[feature_cols].values, dtype=torch.float32).to(DEVICE)
    true_voltages = df_segment['voltage'].values[SEQ_LEN:]
    
    mean_curr, scale_curr = scaler.mean_[0], scaler.scale_[0]
    
    results_vol = []
    results_r0_last = []      # 存储每步最终确定的 R0
    r0_evolution_matrix = []   # 存储 pred_R0_seq 用于分析平滑性

    with torch.no_grad():
        for i in range(len(true_voltages)):
            seq = features[i : i+SEQ_LEN].unsqueeze(0) 
            curr_I_raw = (features[i+SEQ_LEN, 0] * scale_curr + mean_curr).view(1, 1)
            
            # ✅ 调用模型，完整接收 4 个返回值
            p_vol, p_r0, p_ocv, p_r0_seq = model(seq, curr_I_raw)
            
            # 应用物理校准
            vol_c = p_vol.item() + CALIB_OCV_BIAS - curr_I_raw.item() * p_r0.item() * (CALIB_R0_SCALE - 1.0)
            
            results_vol.append(vol_c)
            results_r0_last.append(p_r0.item() * CALIB_R0_SCALE)
            r0_evolution_matrix.append(p_r0_seq.squeeze().cpu().numpy() * CALIB_R0_SCALE)

    # 4. 绘制内阻序列演变热力图 (展示 Transformer 内部的物理一致性)
    r0_matrix = np.array(r0_evolution_matrix) # [Steps, 60]
    plt.figure(figsize=(12, 5))
    sns.heatmap(r0_matrix.T, cmap='magma', cbar_kws={'label': 'R0 (Ohms)'})
    plt.title(f"Dynamic R0 Evolution within 60s Window - {name}")
    plt.xlabel("Prediction Step (Time)")
    plt.ylabel("Historical Context (0-60s)")
    heatmap_path = os.path.join(OUTPUT_DIR, f"r0_heatmap_{name}.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    rmse = np.sqrt(np.mean((np.array(results_vol) - true_voltages)**2))
    print(f"   ✅ {name} 分析完毕. RMSE: {rmse:.4f}V. 热力图已保存.")
    
    return results_vol, results_r0_last, true_voltages

# ===================== 4. 高速并行 TTE 预测 =====================
def predict_tte_parallel(model, name, file_path, scaler, num_simulations=100):
    """
    使用并行张量运算进行蒙特卡洛寿命预测
    """
    print(f"🚀 [并行蒙特卡洛] 场景: {name}, 样本数: {num_simulations}...")
    df = pd.read_parquet(file_path)
    load_currents = torch.from_numpy(df['current'].values).float().to(DEVICE)
    load_temps = torch.from_numpy(df['temperature'].values).float().to(DEVICE)
    
    mean_curr, scale_curr = scaler.mean_[0], scaler.scale_[0]
    soc_mean, soc_scale = scaler.mean_[2], scaler.scale_[2]
    dI_mean, dI_scale = scaler.mean_[3], scaler.scale_[3]
    
    # 初始化
    current_soc = (0.8 + torch.randn(num_simulations) * 0.05).clamp(0.5, 0.9).to(DEVICE)
    seq_tensor = torch.zeros((num_simulations, SEQ_LEN, 4)).to(DEVICE)
    
    for s in range(num_simulations):
        idx = np.random.randint(0, len(load_currents) - SEQ_LEN - 10000)
        seq_tensor[s, :, 0] = load_currents[idx : idx+SEQ_LEN]
        seq_tensor[s, :, 1] = (load_temps[idx : idx+SEQ_LEN] - scaler.mean_[1]) / scaler.scale_[1]
        seq_tensor[s, :, 2] = (current_soc[s] - soc_mean) / soc_scale

    alive = torch.ones(num_simulations, dtype=torch.bool).to(DEVICE)
    tte_steps = torch.zeros(num_simulations).to(DEVICE)
    time_step, cutoff_v = 0, 2.7

    while alive.any() and time_step < 15000:
        with torch.no_grad():
            curr_I_norm = load_currents[(time_step + SEQ_LEN) % len(load_currents)] + torch.randn(num_simulations).to(DEVICE)*0.05
            curr_I_raw = (curr_I_norm * scale_curr + mean_curr).unsqueeze(1)
            
            # ✅ 同步模型返回值
            p_vol, p_r0, p_ocv, _ = model(seq_tensor, curr_I_raw)
            
            # 物理层计算
            voltages = (p_ocv.squeeze() * CALIB_OCV_SCALE + CALIB_OCV_BIAS) - \
                       curr_I_raw.squeeze() * (p_r0.squeeze() * CALIB_R0_SCALE)
            
            current_soc -= (curr_I_raw.squeeze() * (1.0/3600.0)) / Q_MAX
            alive = alive & (voltages > cutoff_v) & (current_soc > 0)
            tte_steps[alive] += 1
            
            # 更新滑动窗口
            prev_I_raw = seq_tensor[:, -1, 0] * scale_curr + mean_curr
            dI_norm = ((curr_I_raw.squeeze() - prev_I_raw) - dI_mean) / dI_scale
            new_frame = torch.stack([curr_I_norm, 
                                    torch.ones_like(curr_I_norm) * 0.5, # 归一化温度约 25C
                                    (current_soc - soc_mean) / soc_scale,
                                    dI_norm], dim=1).unsqueeze(1)
            seq_tensor = torch.cat([seq_tensor[:, 1:, :], new_frame], dim=1)
        
        time_step += 1
    return (tte_steps / 3600.0).cpu().numpy().tolist()

# ===================== 5. 主程序 =====================
def main():
    try:
        model = load_model()
        scaler = joblib.load(SCALER_PATH)
        
        plt.figure(figsize=(16, 12))
        colors = {'Gamer': '#e74c3c', 'Reader': '#3498db'}
        tte_results = {}

        # 场景循环
        for i, (name, path) in enumerate(TEST_FILES.items()):
            vols, r0s, true_vols = analyze_scenario(model, name, path, scaler)
            
            # 子图1: R0 稳定性
            plt.subplot(3, 2, 1)
            plt.plot(r0s, label=f'{name} $R_0$', color=colors[name], alpha=0.7)
            plt.title("Physical Parameter: Internal Resistance")
            plt.ylabel("Resistance ($\Omega$)"); plt.legend()

            # 子图2: 电压预测
            plt.subplot(3, 2, 2)
            plt.plot(true_vols, color=colors[name], alpha=0.3, lw=3, label=f"{name} Actual")
            plt.plot(vols, color='black', ls='--', lw=1, label=f"{name} Physics-Pred")
            plt.title("Voltage Prediction Accuracy")
            plt.ylabel("Voltage (V)"); plt.legend()

            # TTE 模拟
            ttes = predict_tte_parallel(model, name, path, scaler, num_simulations=200)
            tte_results[name] = ttes

        # 子图3: TTE 分布图 (论文核心)
        
        plt.subplot(3, 1, 3)
        for name, ttes in tte_results.items():
            sns.kdeplot(ttes, fill=True, color=colors[name], label=f'{name} Scenario', bw_adjust=1.2)
            mean_v = np.mean(ttes)
            plt.axvline(mean_v, color=colors[name], ls='--', lw=2)
            print(f"📊 [{name}] Mean TTE: {mean_v:.2f}h")

        plt.title("Monte Carlo Reliability Analysis (Battery Life Distribution)")
        plt.xlabel("Remaining Run Time (Hours)"); plt.ylabel("Density"); plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, "comprehensive_battery_report.png")
        plt.savefig(save_path, dpi=300)
        print(f"\n✨ 所有分析已完成！报告保存在: {save_path}")

    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()