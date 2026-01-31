import pandas as pd
import joblib

# 加载归一化器还原电流 (假设电流之前被归一化了)
scaler = joblib.load("processed_data/scaler.pkl")
mean_curr = scaler.mean_[0]
scale_curr = scaler.scale_[0]

df = pd.read_parquet("processed_data/test_gamer.parquet")

# 还原真实的电流 (A) 和电压 (V)
# 注意：如果你的 current 列已经还原过，则跳过这一步
df['real_current'] = df['current'] * scale_curr + mean_curr
df['power_watt'] = df['voltage'] * df['real_current']

print(f"最大功率: {df['power_watt'].max():.2f} W")
print(f"最小功率: {df['power_watt'].min():.2f} W")
print(f"平均功率: {df['power_watt'].mean():.2f} W")
