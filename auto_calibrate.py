import torch
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize
from model import PhysicsAttentionModel 
import os

# 根据之前的 ls 结果，模型文件名为 best_physics_model.pth
MODEL_PATH = r"d:\CatalogForProj\PythonProj\2026MCM\model\best_physics_model_4090.pth" 
TEST_FILES = {
    "Gamer": r"d:\CatalogForProj\PythonProj\2026MCM\processed_data\test_gamer.parquet",
    "Reader": r"d:\CatalogForProj\PythonProj\2026MCM\processed_data\test_reader.parquet"
}
SCALER_PATH = r"d:\CatalogForProj\PythonProj\2026MCM\processed_data\scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型参数 (必须与 train.py 一致)
D_MODEL = 256
N_HEAD = 8
NUM_LAYERS = 6
SEQ_LEN = 60

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        exit()
        
    print(f"📂 Loading calibration data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH).iloc[:5000] 
    
    features = df[['current', 'temperature', 'SOC', 'dI']].values
    targets = df['voltage'].values
    
    # 加载 Scaler
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Error: Scaler not found at {SCALER_PATH}")
        exit()
        
    scaler = joblib.load(SCALER_PATH)
    features_norm = scaler.transform(features)
    
    X_tensor = torch.tensor(features_norm, dtype=torch.float32).to(DEVICE)
    I_raw = torch.tensor(df['current'].values, dtype=torch.float32).to(DEVICE).unsqueeze(1)
    y_true = targets[SEQ_LEN:] 
    
    return X_tensor, I_raw, y_true

def get_raw_predictions(model, X_tensor, I_raw):
    print("⚡ Running inference to extract raw physics parameters...")
    model.eval()
    ocv_list, r0_list, up_list = [], [], []
    
    with torch.no_grad():
        for i in range(len(X_tensor) - SEQ_LEN):
            seq = X_tensor[i : i+SEQ_LEN].unsqueeze(0)
            curr_I = I_raw[i+SEQ_LEN-1].unsqueeze(0)
            
            # Forward pass
            _, pred_r0, pred_ocv = model(seq, curr_I)
            
            ocv_list.append(pred_ocv.item())
            r0_list.append(pred_r0.item())
            up_list.append(0.0) # 如果你的模型输出只有3个值，这里补0
            
    return np.array(ocv_list), np.array(r0_list), np.array(up_list)

def objective_function(params, ocv, r0, up, i_true, v_true):
    ocv_bias, r0_scale, ocv_scale = params
    # 物理校准公式 (修复符号问题：I < 0 为放电，应导致电压下降)
    # V = OCV + I * R - UP
    # 放电时 I 为负 -> V = OCV - |I|*R (符合物理规律)
    v_pred = (ocv * ocv_scale + ocv_bias) + i_true * (r0 * r0_scale) - up
    return np.mean((v_true - v_pred)**2)

def run_calibration():
    print(f"🔧 Initializing Model (d_model={D_MODEL}, layers={NUM_LAYERS})...")
    model = PhysicsAttentionModel(feature_dim=4, d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS).to(DEVICE)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        # 移除 DDP 可能产生的 module. 前缀
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Weight loading failed: {e}")
        return

    # 获取数据与预测
    DATA_PATH = TEST_FILES["Gamer"] # Fix undefined DATA_PATH
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return

    print(f"📂 Loading calibration data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH).iloc[:5000] 
    
    features = df[['current', 'temperature', 'SOC', 'dI']].values
    targets = df['voltage'].values
    
    # 加载 Scaler
    if not os.path.exists(SCALER_PATH):
        print(f"❌ Error: Scaler not found at {SCALER_PATH}")
        return
        
    scaler = joblib.load(SCALER_PATH)
    features_norm = scaler.transform(features)
    
    X_tensor = torch.tensor(features_norm, dtype=torch.float32).to(DEVICE)
    
    # Correct handling of current:
    # We need I_raw for the physics formula.
    # The 'current' column in df is RAW (before normalization) because we read from parquet 
    # BUT wait, the parquet files in processed_data ARE normalized!
    # preprocess_final.py saves NORMALIZED data to parquet.
    # So df['current'] IS normalized.
    # We must inverse transform it to get I_raw.
    
    mean_curr = scaler.mean_[0]
    scale_curr = scaler.scale_[0]
    I_norm = df['current'].values
    I_raw_values = I_norm * scale_curr + mean_curr
    
    I_raw = torch.tensor(I_raw_values, dtype=torch.float32).to(DEVICE).unsqueeze(1)
    y_true = targets[SEQ_LEN:] 
    
    # raw_ocv, raw_r0, raw_up = get_raw_predictions(model, X_tensor, I_raw) # I_raw needed? 
    # get_raw_predictions passes I_raw to model. forward(seq, current_I).
    # current_I should be raw? Yes, model logic expects raw for physics.
    
    raw_ocv, raw_r0, raw_up = get_raw_predictions(model, X_tensor, I_raw)
    
    i_true_segment = I_raw_values[SEQ_LEN:]
    
    # 初始误差
    initial_v = raw_ocv - i_true_segment * raw_r0
    initial_mse = np.mean((y_true - initial_v)**2)
    print(f"\n📉 [Before Calibration] MSE: {initial_mse:.6f} | RMSE: {np.sqrt(initial_mse):.4f} V")
    
    print("🎯 Optimizing calibration parameters...")
    # 初始猜测: bias=0, r0_scale=1, ocv_scale=1
    res = minimize(objective_function, [0.0, 1.0, 1.0], args=(raw_ocv, raw_r0, raw_up, i_true_segment, y_true), method='Nelder-Mead')
    
    best_bias, best_r0_scale, best_ocv_scale = res.x
    
    final_mse = res.fun
    print(f"📈 [After Calibration]  MSE: {final_mse:.6f} | RMSE: {np.sqrt(final_mse):.4f} V")
    
    print(f"\n✨ Optimal Calibration Parameters:")
    print(f"   👉 OCV Bias (Offset) : {best_bias:+.5f} V")
    print(f"   👉 OCV Scale (Gain)  : {best_ocv_scale:.5f} x")
    print(f"   👉 R0 Scale (Gain)   : {best_r0_scale:.5f} x")
    
    print("\n💡 Action Item:")
    print("Update your prediction formula in model.py or predict.py:")
    print(f"pred_Voltage = (pred_OCV * {best_ocv_scale:.5f} + {best_bias:+.5f}) - (current_I * pred_R0 * {best_r0_scale:.5f})")

if __name__ == "__main__":
    run_calibration()