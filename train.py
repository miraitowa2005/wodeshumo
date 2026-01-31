import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os
from model import PhysicsAttentionModel
import time

# --- 旗舰级配置 (RTX 5090 32GB Optimized) ---
# 使用相对路径，兼容本地和服务器环境
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed_data", "train_main.parquet")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "best_physics_model_final.pth")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.pkl")

# 确保模型保存目录存在
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

SEQ_LENGTH = 128
BATCH_SIZE = 4096      # 32GB 显存允许极大 Batch Size，加速训练
EPOCHS = 50            # 算力充足，增加轮数以获得更优收敛
LEARNING_RATE = 0.0005 # 大 Batch Size 配合稍低的 LR 或 Warmup (这里简单调低)
SAMPLE_ROWS = None     # 使用全量数据 (No Limit)
STRIDE = 1             # 步长为1，不跳过任何数据点 (Maximum Precision)

# 模型架构参数 (S-Level: Super) - 提升模型容量以利用算力
D_MODEL = 256          # 增加特征维度 (原 64)
N_HEAD = 8             # 注意力头数
NUM_LAYERS = 6         # 加深网络 (原 2)
DROPOUT = 0.1

def create_sequences(df, seq_length=60):
    """
    高效构建序列数据
    """
    # 转换为 numpy 提高速度
    data_array = df[['current', 'temperature', 'SOC', 'dI']].values
    target_voltage = df['voltage'].values
    
    # 获取未归一化的 current 用于物理公式
    scaler = joblib.load(SCALER_PATH)
    mean_curr = scaler.mean_[0]
    scale_curr = scaler.scale_[0]
    
    # I_raw = I_norm * scale + mean
    # 预先计算好 raw_current 数组
    raw_current_array = data_array[:, 0] * scale_curr + mean_curr
    
    xs, ys, currents_curr = [], [], []
    
    grouped = df.groupby('battery_id')
    
    print(f"开始处理数据切片，Stride={STRIDE}...")
    
    for bid, group in grouped:
        # 获取该组的索引范围
        indices = group.index.values
        # 既然已经 groupby，直接用 numpy values 即可，不需要 iloc
        group_data = data_array[group.index] # 注意：前提是 df index 未重置或与 iloc 一致。
        # 安全起见，直接取 group 的 values
        # 但 group 是 dataframe，重新提取 values 较慢。
        # 优化：df 已经按 battery_id 排序了吗？preprocess_final.py 里是排了的。
        # 我们可以直接利用 group 的 values
        
        g_data = group[['current', 'temperature', 'SOC', 'dI']].values
        g_voltage = group['voltage'].values
        g_raw_curr = g_data[:, 0] * scale_curr + mean_curr
        
        L = len(g_data)
        if L <= seq_length:
            continue
            
        # 向量化切片 (Vectorized Slicing) - 极速模式
        # 创建索引矩阵
        # shape: (num_samples, seq_length)
        num_samples = (L - seq_length) // STRIDE
        if num_samples <= 0:
            continue
            
        # 构建起始索引
        start_indices = np.arange(0, num_samples * STRIDE, STRIDE)
        
        # 这种方式在内存中构建巨大的 3D 数组可能会爆内存 (即使是 64GB RAM)
        # 1000万行 * 60 * 4 * 4bytes ≈ 9.6GB，应该没问题
        # 但为了安全，我们还是用列表，或者分块
        # 既然追求性能，我们直接 append 到 list，最后转换
        
        # 为了避免内存溢出，我们还是用循环 append，但 stride=1 数据量巨大
        # 如果内存不够，建议改为 PyTorch Dataset 的 __getitem__ 懒加载模式
        # 鉴于用户可能有大内存，尝试直接构建
        
        for i in start_indices:
            xs.append(g_data[i : i+seq_length])
            ys.append(g_voltage[i+seq_length])
            currents_curr.append(g_raw_curr[i+seq_length])
            
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32), np.array(currents_curr, dtype=np.float32)

class LazyDataset(torch.utils.data.Dataset):
    """
    内存优化型 Dataset：只存储原始大数组，getitem 时实时切片
    """
    def __init__(self, data_array, voltage_array, raw_current_array, battery_ids, seq_length=60):
        self.data = torch.FloatTensor(data_array)
        self.voltage = torch.FloatTensor(voltage_array)
        self.raw_current = torch.FloatTensor(raw_current_array)
        self.seq_length = seq_length
        
        # 预计算所有合法的 (start_idx)
        # 需要根据 battery_id 区分边界
        self.valid_indices = []
        
        # 找出每个 battery_id 的起止位置
        # 假设 battery_ids 是排好序的
        # 使用 pandas 或 numpy 找边界
        # 这里传入的 battery_ids 最好是 Series
        
        # 这种方式比较复杂，为了稳健，我们采用“记录每个样本的 (start_idx)” 的方式
        # 预处理阶段已经把不同 battery 的数据拼在一起了，我们需要知道哪些区间是连续的
        
        # 简单实现：传入 valid_start_indices 列表
        pass

# 定义 Lazy Dataset
class BatteryDataset(torch.utils.data.Dataset):
    def __init__(self, data, voltage, raw_current, indices, seq_len):
        self.data = torch.from_numpy(data) # Share memory
        self.voltage = torch.from_numpy(voltage)
        self.raw_current = torch.from_numpy(raw_current)
        self.indices = indices
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        # 切片
        x = self.data[start_idx:end_idx]
        y = self.voltage[end_idx] # 预测序列末尾的电压
        curr = self.raw_current[end_idx]
        
        return x, y, curr

def main():
    # 开启 CuDNN 自动调优
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device} (RTX 5090 Mode On)")

    # 1. 加载数据
    print(f"正在读取数据: {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    
    if SAMPLE_ROWS:
        df = df.head(SAMPLE_ROWS)
        print(f"⚠️ 警告: 仅使用前 {SAMPLE_ROWS} 行数据 (测试模式)")
    
    print(f"数据加载完成，Shape: {df.shape}")
    
    # 2. 构建序列 (使用内存优化策略或全量加载)
    # 由于数据量极大 (5000万行)，直接生成 3D 数组 (5000w, 60, 4) 会占用 5000w*60*4*4 bytes ≈ 48GB 内存 -> 可能会炸
    # 因此我们必须重写 Dataset，使用 Lazy Loading 模式
    
    # 提取原始数组
    print("准备 Lazy Loading 数据集...")
    data_values = df[['current', 'temperature', 'SOC', 'dI']].values.astype(np.float32)
    voltage_values = df['voltage'].values.astype(np.float32)
    battery_ids = df['battery_id'].values
    
    # 计算 raw_current
    scaler = joblib.load(SCALER_PATH)
    mean_curr = scaler.mean_[0]
    scale_curr = scaler.scale_[0]
    raw_current_values = data_values[:, 0] * scale_curr + mean_curr
    
    # 构建合法的索引列表
    valid_indices = []
    # 找到每个 battery_id 的切换点
    # 这种方法比 groupby 快
    # df['battery_id'] 已经是 categorical 或 string
    # 使用 numpy 判断边界
    
    # 简单做法：遍历一次，记录合法起点
    # 为了加速，利用 pandas groupby
    print("正在计算合法索引 (Index Map)...")
    grouped = df.groupby('battery_id')
    for bid, group in grouped:
        indices = group.index.values
        # 每个 group 内部，从 0 到 len-seq_len
        # indices 是全局索引
        if len(indices) > SEQ_LENGTH:
            # 这里的 indices 是连续的吗？是的，因为 sort_values 过
            # 能够作为起点的最大索引是 indices[-1] - SEQ_LENGTH
            # 起点: indices[0] ... indices[len - 1 - SEQ_LENGTH]
            # 对应的切片是 [i : i+SEQ_LENGTH]
            
            # 使用 STRIDE
            valid_starts = indices[:-(SEQ_LENGTH)]
            if STRIDE > 1:
                valid_starts = valid_starts[::STRIDE]
            valid_indices.extend(valid_starts)
            
    valid_indices = np.array(valid_indices, dtype=np.int64)
    print(f"总样本数: {len(valid_indices)}")
    
    dataset = BatteryDataset(data_values, voltage_values, raw_current_values, valid_indices, SEQ_LENGTH)
    
    # 4090 显存大，num_workers 可以多开，但 windows 下要注意多进程开销
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    
    # 4. 初始化模型 (Pro Config)
    model = PhysicsAttentionModel(
        feature_dim=4, 
        d_model=D_MODEL, 
        nhead=N_HEAD, 
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 混合精度训练 Scaler
    scaler_amp = torch.cuda.amp.GradScaler()
    
    # 5. 训练
    print("🔥 开始全量训练 (Mixed Precision Enabled)...")
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_mse = 0
        
        # 进度条效果 (每 100 batch 打印一次)
        steps = len(dataloader)
        
        for i, (batch_X, batch_y, batch_I) in enumerate(dataloader):
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True).unsqueeze(1)
            batch_I = batch_I.to(device, non_blocking=True).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast():
                pred_voltage, pred_R0, pred_OCV, _, pred_R0_seq = model(batch_X, batch_I)
                
                loss_mse = criterion(pred_voltage, batch_y)
                loss_reg_r = torch.mean(torch.relu(pred_R0 - 0.5)) 
                
                # Smoothness Loss: 惩罚序列内部 R0 的剧烈变化
                # pred_R0_seq: [Batch, Seq, 1]
                loss_smooth_r = torch.mean((pred_R0_seq[:, 1:, :] - pred_R0_seq[:, :-1, :])**2)
                
                # 总损失 (增加平滑项权重)
                loss = loss_mse + 0.1 * loss_reg_r + 0.5 * loss_smooth_r
            
            # 混合精度反向传播
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch+1} [{i+1}/{steps}] Loss: {loss.item():.6f} | MSE: {loss_mse.item():.6f} | Smooth: {loss_smooth_r.item():.8f}", end='\r')
        
        avg_loss = total_loss / steps
        avg_mse = total_mse / steps
        epoch_time = time.time() - start_time
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s | Loss: {avg_loss:.6f} | MSE: {avg_mse:.6f} | LR: {current_lr:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  --> 模型已保存 (New Best Loss: {best_loss:.6f})")
            
    print(f"🏆 训练完成！最佳模型已保存至 {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()