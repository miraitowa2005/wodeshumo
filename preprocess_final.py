import os
import glob
import scipy.io
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
import joblib

# ================= 配置区 =================
# 数据根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "database")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 物理约束
VOLTAGE_RANGE = (2.0, 4.5)  # 伏特
TEMP_RANGE = (0, 60)        # 摄氏度

# 数据集分组策略 (根据 NASA README)
DATA_GROUPS = {
    "train_main": [f"RW{i}" for i in range(1, 13)] + [f"RW{i}" for i in range(21, 29)], # 训练集 (1-12, 21-28)
    "test_reader": [f"RW{i}" for i in range(13, 17)], # 测试集: 轻负载 (13-16)
    "test_gamer": [f"RW{i}" for i in range(17, 21)]   # 测试集: 重负载 (17-20)
}
# =========================================

def find_mat_file(rw_name):
    """递归查找文件"""
    search_pattern = os.path.join(BASE_DIR, "**", f"{rw_name}.mat")
    files = glob.glob(search_pattern, recursive=True)
    if files:
        return files[0]
    return None

def get_capacity_curve(rw_name, time_stamp):
    """
    SOH 衰减模型 (简化版)
    实际比赛中，你应该先提取 Reference Discharge 数据拟合出真实的 Q_max(t) 函数
    这里用线性衰减模拟: 2.1Ah -> 1.5Ah
    """
    decay_rate = 0.3 / 1e7 # 假设跑很久才衰减
    return 2.1 - (decay_rate * time_stamp)

def process_single_file(filepath, rw_name):
    print(f"正在处理: {rw_name} ...")
    try:
        mat = scipy.io.loadmat(filepath)
        steps = mat['data']['step'][0,0]
    except Exception as e:
        print(f"无法读取 {filepath}: {e}")
        return None

    data_list = []
    for i in range(steps.shape[1]):
        step = steps[0, i]
        try:
            comment = step['comment'][0]
        except:
            continue
            
        # 提取随机充放电过程
        if 'random' in comment.lower():
            try:
                # 检查字段是否存在
                time = step['time'].flatten()
                rel_time = step['relativeTime'].flatten()
                voltage = step['voltage'].flatten()
                current = step['current'].flatten()
                temp = step['temperature'].flatten()
                
                df = pd.DataFrame({
                    'time': time,
                    'relative_time': rel_time,
                    'voltage': voltage,
                    'current': current,
                    'temperature': temp
                })
                data_list.append(df)
            except Exception as e:
                # print(f"  Skipping step {i}: {e}")
                pass
    
    if not data_list: return None
    
    # 1. 合并与排序
    df_full = pd.concat(data_list, ignore_index=True)
    df_full = df_full.sort_values('time').reset_index(drop=True)
    
    # 2. 物理清洗 (Outlier Removal)
    # 将不合理的值设为 NaN，然后插值
    mask_bad = (df_full['voltage'] < VOLTAGE_RANGE[0]) | (df_full['voltage'] > VOLTAGE_RANGE[1])
    df_full.loc[mask_bad, 'voltage'] = np.nan
    df_full = df_full.interpolate(method='linear', limit_direction='both')
    
    # 3. SOC 计算 (安时积分)
    df_full['dt'] = df_full['time'].diff().fillna(0) / 3600.0 # hour
    df_full['Q_max'] = df_full['time'].apply(lambda t: get_capacity_curve(rw_name, t))
    df_full['delta_ah'] = df_full['current'] * df_full['dt']
    # 假设初始 SOC=0.8 (可根据第一帧电压查 OCV 表反推)
    df_full['consumed_ah'] = df_full['delta_ah'].cumsum()
    df_full['SOC'] = 0.8 - (df_full['consumed_ah'] / df_full['Q_max'])
    df_full['SOC'] = df_full['SOC'].clip(0, 1) # 限制在 0-1 之间
    
    # 4. 降采样 (Downsampling) -> 1Hz
    # 这是关键降噪步骤，也是减小数据量的核心
    # 使用 drop_duplicates 确保 time 唯一，以防原始数据有重复时间戳
    df_full = df_full.drop_duplicates(subset=['time'])
    df_full.index = pd.to_timedelta(df_full['time'], unit='s')
    
    # resample 需要 numeric_only=True 否则会报错如果有些列非数字
    df_resampled = df_full.resample('1s').mean().dropna().reset_index(drop=True)
    
    # 5. 特征工程 (Feature Engineering)
    # 增加一阶差分特征，帮助模型捕捉突变
    df_resampled['dI'] = df_resampled['current'].diff().fillna(0)
    df_resampled['dV'] = df_resampled['voltage'].diff().fillna(0)
    
    # 添加 Battery ID 标识 (方便后续按电池分组)
    df_resampled['battery_id'] = rw_name
    
    return df_resampled

def main_pipeline():
    # 1. 处理所有文件并暂存
    all_data = {} # key: group_name, value: list of dfs
    
    for group, rw_list in DATA_GROUPS.items():
        print(f"\n=== 开始组: {group} ===")
        group_dfs = []
        for rw_name in rw_list:
            path = find_mat_file(rw_name)
            if path:
                df = process_single_file(path, rw_name)
                if df is not None:
                    group_dfs.append(df)
            else:
                print(f"未找到文件: {rw_name}")
        
        if group_dfs:
            all_data[group] = pd.concat(group_dfs, ignore_index=True)
    
    if 'train_main' not in all_data:
        print("错误: 没有生成训练数据！")
        return

    # 2. 全局归一化 (Global Normalization)
    # 只根据训练集计算均值和方差，防止数据泄露
    print("\n=== 正在计算归一化参数 ===")
    train_df = all_data['train_main']
    
    # 我们只归一化输入特征，不归一化 Target(Voltage)
    # 输入特征: Current, Temperature, SOC, dI (还可以加上历史特征)
    features = ['current', 'temperature', 'SOC', 'dI']
    
    scaler = StandardScaler()
    scaler.fit(train_df[features])
    
    # 保存 Scaler，后面预测要用
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    
    # 3. 应用归一化并保存
    for group, df in all_data.items():
        # 归一化
        df[features] = scaler.transform(df[features])
        
        # 保存为 Parquet
        save_path = os.path.join(OUTPUT_DIR, f"{group}.parquet")
        df.to_parquet(save_path, index=False)
        print(f"已保存: {save_path}, Shape: {df.shape}")

if __name__ == '__main__':
    main_pipeline()
