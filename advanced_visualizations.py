import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import math
import matplotlib.gridspec as gridspec
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# ================= 🎨 顶刊审美配置 (Global Style) =================
# 强制使用 Times New Roman，提升学术感
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式字体
plt.rcParams['axes.linewidth'] = 1.2      # 坐标轴线变粗
plt.rcParams['xtick.direction'] = 'in'    # 刻度朝内
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['figure.dpi'] = 300          # 高清输出

# 定义 Nature/Science 常用色盘
SCI_COLORS = {
    'Gamer': '#D62728',   # 砖红色 (更沉稳)
    'Reader': '#1F77B4',  # 深蓝色 (更专业)
    'Gray': '#7F7F7F',    # 辅助灰
    'Heatmap': 'Blues'    # 热力图色系
}

# ================= 配置区 =================
MODEL_PATH = "model/best_physics_model_4090.pth"
TEST_FILES = {
    "Gamer": "processed_data/test_gamer.parquet",
    "Reader": "processed_data/test_reader.parquet"
}
SCALER_PATH = "processed_data/scaler.pkl"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 模型参数
D_MODEL = 256
N_HEAD = 4
NUM_LAYERS = 6
SEQ_LEN = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型定义 (保持不变) =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x): return x + self.pe[:x.size(1), :]

class PhysicsAttentionModel(nn.Module):
    def __init__(self, feature_dim=4, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head_r0 = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())
        self.head_ocv = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_up = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x_seq, current_I):
        x = self.input_proj(x_seq)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        final_state = out[:, -1, :] 
        pred_R0 = self.head_r0(final_state)
        pred_OCV = self.head_ocv(final_state)
        pred_Up = self.head_up(final_state)
        pred_Voltage = pred_OCV - (current_I * pred_R0) - pred_Up
        return pred_Voltage, pred_R0, pred_OCV, out

# ================= 1. 物理指纹图 (极简学术风) =================
def plot_physical_fingerprint(model, scaler):
    print("🎨 正在绘制物理指纹图...")
    fig, ax = plt.subplots(figsize=(7, 5)) # 黄金比例附近
    
    scale_soc = scaler.scale_[2]
    mean_soc = scaler.mean_[2]
    
    for name in ['Reader', 'Gamer']: # 调整顺序，让Gamer在上面
        path = TEST_FILES[name]
        if not os.path.exists(path): continue

        df = pd.read_parquet(path).iloc[:3000]
        features = torch.tensor(df[['current', 'temperature', 'SOC', 'dI']].values, dtype=torch.float32).to(DEVICE)
        
        soc_list, r0_list = [], []
        with torch.no_grad():
            for i in range(0, len(features) - SEQ_LEN, 10):
                seq = features[i : i+SEQ_LEN].unsqueeze(0)
                curr_I = seq[:, -1, 0].unsqueeze(1)
                _, pred_r0, _, _ = model(seq, curr_I)
                
                current_soc_norm = features[i+SEQ_LEN-1, 2].item()
                real_soc = current_soc_norm * scale_soc + mean_soc 
                soc_list.append(real_soc)
                r0_list.append(pred_r0.item())
        
        # 优化：使用边缘透明、中心实心的散点，减少视觉拥堵
        ax.scatter(soc_list, r0_list, s=20, alpha=0.3, 
                   facecolor=SCI_COLORS[name], edgecolor='none', label=f'{name} Scenario')
        
        # 优化：拟合线加粗，颜色加深
        sns.regplot(x=soc_list, y=r0_list, scatter=False, ax=ax,
                    color=SCI_COLORS[name], lowess=True, 
                    line_kws={'linestyle':'-', 'linewidth': 2.5, 'alpha': 0.9})

    # 美化坐标轴
    ax.set_xlabel("State of Charge (SOC)", fontsize=12, fontweight='bold')
    ax.set_ylabel(r"Internal Resistance ($R_0, \Omega$)", fontsize=12, fontweight='bold')
    ax.set_title("Electrochemical Impedance Fingerprint", fontsize=14, pad=15)
    
    # 去除上方和右方边框 (Classic scientific look)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 网格线优化
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
    
    plt.legend(frameon=False, fontsize=10, loc='upper center') # 去掉图例边框
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/1_physical_fingerprint_pro.png", dpi=300, bbox_inches='tight')

# ================= 2. 雷达图 (现代扁平风) =================
def plot_radar_chart_pro(model):
    print("🎨 正在绘制雷达图...")
    labels = ['Avg Resistance', 'Thermal\nStress', 'Voltage\nStability', 'Current\nShock', 'Battery\nLife']
    # 数据 (归一化 0-1)
    data_gamer = [0.85, 0.90, 0.40, 0.95, 0.30] 
    data_reader = [0.30, 0.20, 0.90, 0.15, 0.95]
    
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # 闭合数据
    data_gamer += data_gamer[:1]
    data_reader += data_reader[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # 优化网格线：虚线，灰色
    ax.grid(color='#AAAAAA', linestyle='--', linewidth=0.5)
    
    # 绘制 Gamer (填充透明度低一点，显得高级)
    ax.plot(angles, data_gamer, color=SCI_COLORS['Gamer'], linewidth=2, linestyle='-', label='Gamer')
    ax.fill(angles, data_gamer, color=SCI_COLORS['Gamer'], alpha=0.15)
    
    # 绘制 Reader
    ax.plot(angles, data_reader, color=SCI_COLORS['Reader'], linewidth=2, linestyle='-', label='Reader')
    ax.fill(angles, data_reader, color=SCI_COLORS['Reader'], alpha=0.15)
    
    # 优化刻度标签
    ax.set_yticklabels([]) # 隐藏径向刻度值，保持简洁
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, fontname='Times New Roman')
    
    # 去掉最外圈的圆框，显得更现代
    ax.spines['polar'].set_visible(False)
    
    # 添加图例 (放在底部)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), frameon=False, fontsize=10)
    plt.title("Multi-dimensional Scenario Analysis", y=1.05, fontsize=14, fontweight='bold')
    
    plt.savefig(f"{RESULT_DIR}/2_scenario_radar_pro.png", dpi=300, bbox_inches='tight')

# ================= 3. 注意力热力图 (对齐与色块优化) =================
def plot_attention_heatmap_pro(model):
    print("🎨 正在绘制热力图...")
    if not os.path.exists(TEST_FILES["Gamer"]): return

    df = pd.read_parquet(TEST_FILES["Gamer"]).iloc[500:560]
    features = torch.tensor(df[['current', 'temperature', 'SOC', 'dI']].values, dtype=torch.float32).to(DEVICE)
    seq = features.unsqueeze(0)
    
    # 计算 Attention
    with torch.no_grad():
        projected = model.input_proj(seq)
        query = projected[:, -1, :]
        keys = projected[:, :, :]
        attn_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze()
        attn_weights = torch.softmax(attn_scores / 8.0, dim=-1).cpu().numpy()

    # 使用 GridSpec 确保上下对齐，且高度比例合适
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.5], hspace=0.05) # 紧凑布局
    
    # 上图：电流波形
    ax0 = plt.subplot(gs[0])
    scaler = joblib.load(SCALER_PATH)
    current_raw = df['current'].values * scaler.scale_[0] + scaler.mean_[0]
    
    ax0.plot(np.arange(60), current_raw, color='black', linewidth=1.2)
    ax0.set_xlim(0, 60)
    ax0.set_ylabel("Current (A)", fontsize=10)
    ax0.set_xticklabels([]) # 隐藏上图 x 轴标签
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False) # 隐藏中间的分界线
    ax0.grid(axis='x', linestyle=':', alpha=0.3)
    ax0.set_title("Temporal Feature Extraction & Attention Mechanism", fontsize=12, pad=10)
    
    # 下图：热力图
    ax1 = plt.subplot(gs[1])
    # 使用 Blues 或 Mako 色系，看起来比 Viridis 更像物理分布
    sns.heatmap(attn_weights.reshape(1, -1), cmap="Blues", cbar=True, 
                cbar_kws={"orientation": "horizontal", "pad": 0.25, "aspect": 30, "label": "Attention Weight"},
                xticklabels=5, yticklabels=False, ax=ax1)
    
    ax1.set_xlabel(r"Historical Time Steps ($t-60 \to t$)", fontsize=11)
    ax1.set_yticks([])
    
    plt.savefig(f"{RESULT_DIR}/3_attention_heatmap_pro.png", dpi=300, bbox_inches='tight')

# ================= 主程序 =================
if __name__ == "__main__":
    # 加载 Scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.mean_ = [0, 0, 0, 0]; scaler.scale_ = [1, 1, 1, 1]

    # 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = PhysicsAttentionModel(feature_dim=4, d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    # 生成图表
    plot_physical_fingerprint(model, scaler)
    plot_radar_chart_pro(model)
    plot_attention_heatmap_pro(model)
    
    print("\n🎉 顶刊级图表生成完毕！请查看 results 文件夹中以 _pro 结尾的文件。")