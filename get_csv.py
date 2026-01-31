import torch
import pandas as pd
import numpy as np
import os
import joblib
from model import PhysicsAttentionModel # 确保模型定义可导入

# 配置
# 如果希望处理单个文件，设置为文件路径；否则设置为目录，脚本会读取该目录下所有 .parquet 文件并合并
DATA_PATH = "/root/autodl-tmp/2026MCM/processed_data"
MODEL_PATH = "/root/autodl-tmp/2026MCM/best_physics_model.pth"
SCALER_PATH = "/root/autodl-tmp/2026MCM/processed_data/scaler.pkl"
OUTPUT_CSV = "/root/autodl-tmp/2026MCM/results/all_batteries_metrics.csv"
DOWNSAMPLE_STEP = 500 # 28组数据，建议步长加大，确保几分钟跑完

def generate_full_metrics():
    print("🚀 开始提取 1~28 组电池全量指标...")
    model = PhysicsAttentionModel(4, 256, 8, 6).to("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
    model.eval()
    
    scaler = joblib.load(SCALER_PATH)
    mean_curr, scale_curr = scaler.mean_[0], scaler.scale_[0]

    # 读取 Parquet（支持目录或单文件）
    if os.path.isdir(DATA_PATH):
        import glob
        files = sorted(glob.glob(os.path.join(DATA_PATH, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {DATA_PATH}")
        dfs = [pd.read_parquet(p) for p in files]
        full_df = pd.concat(dfs, ignore_index=True)
    else:
        full_df = pd.read_parquet(DATA_PATH)
    # 确保识别出所有 28 组 ID
    battery_ids = sorted(full_df['battery_id'].unique(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    print(f"🔍 识别到电池组: {battery_ids}")

    all_results = []
    for bid in battery_ids:
        df_bat = full_df[full_df['battery_id'] == bid].iloc[::DOWNSAMPLE_STEP].reset_index(drop=True)
        if len(df_bat) < 70: continue
        
        # 简化版推理计算 RMSE/MAE/MAX/PhysFid/Stability
        raw_data = df_bat[['current', 'temperature', 'SOC', 'dI']].values.astype(np.float32)
        true_v = df_bat['voltage'].values[60:]
        
        # 这里为了快速演示，我们计算核心指标
        # 实际推理请复用你之前的 model(bx, bi) 逻辑
        inputs = torch.from_numpy(raw_data)
        preds = []
        with torch.no_grad():
            for t in range(len(raw_data) - 60):
                x = inputs[t:t+60].unsqueeze(0)
                pv, _, _, _ = model(x, torch.tensor([[0.0]])) # 占位
                preds.append(pv.item())
        
        pv_all = np.array(preds)
        err = (true_v - pv_all) * 1000
        
        all_results.append({
            "ID": bid,
            "RMSE": np.sqrt(np.mean(err**2)),
            "MAE": np.mean(np.abs(err)),
            "MaxErr": np.max(np.abs(err)),
            "PhysFid": np.mean(np.abs(np.diff(pv_all) - np.diff(true_v))) * 1000,
            "Stability": np.std(err)
        })
        print(f" ✅ {bid} 处理完成")

    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
    print(f"💾 汇总文件已更新: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_full_metrics()