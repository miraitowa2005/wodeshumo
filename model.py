import torch
import torch.nn as nn
import math

class PhysicsAttentionModel(nn.Module):
    def __init__(self, feature_dim=4, d_model=64, nhead=4, num_layers=2):
        """
        feature_dim: 输入特征数量 (Current, Temp, SOC, dI)
        """
        super().__init__()
        
        # --- 1. 嵌入层 ---
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # --- 2. 核心: Multi-Head Attention (Transformer) ---
        # 它的作用是回顾过去 60s 的电流历史，提取“负载模式”
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- 3. 物理参数预测头 (Parameter Heads) ---
        # 我们不预测 V，我们预测 R0 和 OCV_correction
        # R0 必须大于 0，所以用 Softplus 激活
        self.head_r0 = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Softplus() 
        )
        # OCV 通常是 SOC 的函数，这里我们要预测“偏差值”或者直接预测 OCV
        self.head_ocv = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        # 预测极化电压 Up (简化处理，或者通过 RNN 递归计算)
        self.head_up = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, current_I):
        """
        x_seq: [Batch, Seq_Len, Features] -> 历史窗口数据
        current_I: [Batch, 1] -> 当前时刻的电流 (用于物理公式计算)
        """
        # A. Transformer 提取特征
        x = self.input_proj(x_seq)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        
        # 取最后一个时间步的隐状态
        final_state = out[:, -1, :] 
        
        # B. 神经网络预测物理参数
        # 1. 为了计算平滑 Loss，我们需要 R0 的整个序列，而不仅仅是最后一步
        # out shape: [Batch, Seq_Len, d_model]
        pred_R0_seq = self.head_r0(out)  # [Batch, Seq_Len, 1]
        pred_R0 = pred_R0_seq[:, -1, :]  # 取最后一个时间步 [Batch, 1]
        
        # 其他参数目前只需要最后一步 (也可以改为序列输出以增强约束)
        pred_OCV = self.head_ocv(final_state)  # [Batch, 1]
        pred_Up = self.head_up(final_state)    # [Batch, 1]
        
        # C. 物理融合层 (Physics Layer)
        # 公式: V = OCV - I * R0 - Up
        # 这就是 Explicit Continuous Model 的体现！
        # 注意：这里假设 current_I 正值为放电，负值为充电。
        # V_terminal = OCV - I * R (放电时电压降低，充电时电压升高)
        pred_Voltage = pred_OCV - (current_I * pred_R0) - pred_Up
        
        return pred_Voltage, pred_R0, pred_OCV, pred_Up, pred_R0_seq

# 位置编码 (Transformer 标配)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
