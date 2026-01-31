import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from model import PhysicsAttentionModel

# --- 配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "test_gamer.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_physics_model_final.pth")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LENGTH = 60
BATTERY_ID = 'RW17' # 选择一个测试电池进行可视化

# 模型参数 (需与 train.py 保持一致)
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 6
DROPOUT = 0.1

def load_data_and_model():
    print("1. 加载数据与模型...")
    # 加载 Scaler
    scaler = joblib.load(SCALER_PATH)
    mean_curr = scaler.mean_[0]
    scale_curr = scaler.scale_[0]

    # 加载数据
    df = pd.read_parquet(DATA_PATH)
    df_bat = df[df['battery_id'] == BATTERY_ID].reset_index(drop=True)
    # 取前 2000 个点进行详细展示 (约 30分钟)
    df_bat = df_bat.iloc[:2000] 
    
    print(f"   已加载电池 {BATTERY_ID} 数据: {len(df_bat)} 行")

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsAttentionModel(
        feature_dim=4,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS
    ).to(device)
    # 允许加载部分匹配的权重 (以防模型微调过)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()
    
    return df_bat, model, device, (mean_curr, scale_curr)

def prepare_input(df, seq_length=60):
    # 构建序列
    data_array = df[['current', 'temperature', 'SOC', 'dI']].values
    
    xs, currents_curr = [], []
    valid_indices = []
    
    for i in range(len(df) - seq_length):
        xs.append(data_array[i : i+seq_length])
        # 这里需要注意：currents_curr 应该是原始值还是归一化值？
        # 模型 forward 里用 current_I 来计算 V = OCV - I*R
        # 如果模型训练时用的是真实值 I，这里也要还原。
        # 查看 train.py，我们还原了 raw_current。
        # 这里为了简单，我们先存归一化的，后面在传入模型前统一处理
        currents_curr.append(data_array[i+seq_length, 0]) 
        valid_indices.append(i + seq_length)
        
    return np.array(xs), np.array(currents_curr), valid_indices

# --- Hook 用于提取 Attention ---
attention_weights = []
def attention_hook(module, input, output):
    # input[0] 是 query, input[1] 是 key, input[2] 是 value
    # nn.MultiheadAttention 的 forward 返回 (attn_output, attn_output_weights)
    # 但我们 hook 的是 self_attn 模块，它返回的就是这两个
    # output[1] 是 attention weights [Batch, Num_Heads, Seq_Len, Seq_Len] (如果是 batch_first=True)
    # 或者 [Batch, Seq_Len, Seq_Len] (如果 average_attn_weights=True)
    
    # 检查 output 类型
    if isinstance(output, tuple):
        # output[1] 是 weights
        # 注意：nn.MultiheadAttention 默认 need_weights=True 会返回 weights
        # 但 TransformerEncoderLayer 内部调用 self_attn 时，
        # PyTorch 默认实现可能不会把 weights 传出来，除非我们修改源码或使用 eager 模式
        # 实际上 TransformerEncoderLayer 的 self_attn 调用是：
        # x, _ = self.self_attn(x, x, x, key_padding_mask=..., need_weights=False)
        # 默认是 False！所以 Hook 可能拿不到 weights。
        
        # *** 解决方案 ***
        # 由于 PyTorch 标准层的封装性，直接 Hook 拿不到 weights (因为 need_weights=False)。
        # 我们只能：
        # 1. 相信模型已经训练好了。
        # 2. 手动运行一遍 self_attn 来获取 weights 用于可视化。
        pass
    
# 替代方案：手动计算 Attention
def get_attention(model, x_tensor):
    # 获取 Transformer Encoder 的第一层
    encoder_layer = model.transformer.layers[0]
    self_attn = encoder_layer.self_attn
    
    # 嵌入
    x = model.input_proj(x_tensor)
    x = model.pos_encoder(x)
    
    # 手动调用 attention
    # query=x, key=x, value=x
    # need_weights=True
    _, weights = self_attn(x, x, x, need_weights=True)
    return weights.detach().cpu().numpy()

def main():
    df, model, device, (mean_curr, scale_curr) = load_data_and_model()
    
    # 准备数据
    print("2. 准备推理数据...")
    X, I_norm, valid_idx = prepare_input(df, SEQ_LENGTH)
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    # 还原真实电流 (用于物理公式)
    I_raw = I_norm * scale_curr + mean_curr
    I_tensor = torch.FloatTensor(I_raw).unsqueeze(1).to(device)
    
    # 推理
    print("3. 执行模型推理...")
    with torch.no_grad():
        # 模型返回 5 个值 (V, R0, OCV, Up, R0_seq)
        pred_V, pred_R0, pred_OCV, _, _ = model(X_tensor, I_tensor)
        
    # 提取结果
    pred_V = pred_V.cpu().numpy().flatten()
    pred_R0 = pred_R0.cpu().numpy().flatten()
    pred_OCV = pred_OCV.cpu().numpy().flatten()
    
    true_V = df.iloc[valid_idx]['voltage'].values
    
    # 计算误差
    mse = np.mean((true_V - pred_V)**2)
    print(f"   MSE Loss: {mse:.6f}")
    
    # --- 绘图 1: 电压预测对比 ---
    print("4. 生成图表...")
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(true_V, 'k-', label='True Voltage', linewidth=1.5)
    plt.plot(pred_V, 'r--', label='Predicted Voltage', linewidth=1.5)
    plt.title(f'Voltage Prediction (Battery: {BATTERY_ID}) - Physics-Informed Transformer')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- 绘图 2: 内部物理参数 (可解释性) ---
    plt.subplot(3, 1, 2)
    # 双轴
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.plot(pred_R0, 'b-', label='Predicted R0 (Internal Resistance)', alpha=0.7)
    ax2.plot(I_raw, 'g-', label='Load Current (A)', alpha=0.3) # 背景显示电流
    
    ax1.set_ylabel('Internal Resistance (Ohms)', color='b')
    ax2.set_ylabel('Current (A)', color='g')
    plt.title('Predicted Internal Physics Parameters vs Load')
    ax1.grid(True, alpha=0.3)
    
    # --- 绘图 3: Attention Heatmap ---
    # 选取一个样本 (比如第 1000 个时刻)
    sample_idx = 1000
    if sample_idx >= len(X_tensor): sample_idx = 0
    
    sample_tensor = X_tensor[sample_idx:sample_idx+1] # [1, 60, 4]
    
    # 获取注意力权重
    attn_weights = get_attention(model, sample_tensor) 
    print(f"   Attention Weights Shape: {attn_weights.shape}")

    # 处理不同维度的返回值
    if attn_weights.ndim == 4:
        # [Batch, Num_Heads, Seq_Len, Seq_Len]
        attn_map = np.mean(attn_weights[0], axis=0)
    elif attn_weights.ndim == 3:
        # [Batch, Seq_Len, Seq_Len] (已经平均了)
        attn_map = attn_weights[0]
    else:
        raise ValueError(f"Unexpected attention shape: {attn_weights.shape}")
    
    # 只要最后一行：代表 "当前时刻" 关注 "过去哪些时刻"
    final_step_attn = attn_map[-1, :] # [60]
    
    plt.subplot(3, 1, 3)
    # 画成热力条
    sns.heatmap(final_step_attn.reshape(1, -1), cmap='viridis', cbar=True, 
                xticklabels=5, yticklabels=False)
    plt.title(f'Attention Weights (Importance of Past 60s for Prediction at t={sample_idx})')
    plt.xlabel('Time Lag (Seconds ago: 0=oldest, 60=now)')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'final_result_viz.png')
    plt.savefig(save_path, dpi=300)
    print(f"图表已保存: {save_path}")

if __name__ == "__main__":
    main()
