import torch
import numpy as np
import joblib
import pandas as pd
from model import PhysicsAttentionModel

import os

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_physics_model_final.pth")
SCALER_PATH = os.path.join(BASE_DIR, "processed_data", "scaler.pkl")

# 电池参数
BATTERY_CAPACITY_AH = 2.1  # 标称容量
CUTOFF_VOLTAGE = 2.5       # 截止电压
SEQ_LENGTH = 128           # 必须与训练时保持一致 (train.py: SEQ_LENGTH=128)
D_MODEL = 256              # 模型维度 (train.py: D_MODEL=256)
NUM_LAYERS = 6             # 模型层数 (train.py: NUM_LAYERS=6)

class RuntimePredictor:
    def __init__(self):
        print("正在初始化预测引擎...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载 Scaler
        try:
            self.scaler = joblib.load(SCALER_PATH)
            # scaler.mean_ 和 scaler.scale_ 用于手动归一化
            # features: ['current', 'temperature', 'SOC', 'dI']
            self.mean = self.scaler.mean_
            self.scale = self.scaler.scale_
            print("Scaler 加载成功")
        except FileNotFoundError:
            print(f"错误: 找不到 Scaler 文件 {SCALER_PATH}")
            raise

        # 2. 加载模型
        self.model = PhysicsAttentionModel(
            feature_dim=4,
            d_model=D_MODEL,
            nhead=8,
            num_layers=NUM_LAYERS
        ).to(self.device)
        
        try:
            # 加载权重
            state_dict = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("模型加载成功")
        except FileNotFoundError:
            print(f"错误: 找不到模型文件 {MODEL_PATH}")
            raise

    def predict_remaining_time(self, initial_soc, temperature, power_watts, dt=1.0):
        """
        预测剩余使用时间
        :param initial_soc: 初始 SOC (0.0 - 1.0)
        :param temperature: 环境温度 (摄氏度)
        :param power_watts: 恒定功耗 (瓦特)，正值表示放电
        :param dt: 模拟步长 (秒)
        :return: 剩余时间 (分钟), 模拟数据历史
        """
        print(f"\n--- 开始预测 (SOC={initial_soc:.1%}, Temp={temperature}°C, Power={power_watts}W) ---")
        
        # 初始化状态
        current_soc = initial_soc
        current_voltage = 4.2 if initial_soc > 0.9 else 3.7 # 初始电压猜测值
        time_elapsed = 0
        
        # 历史记录 (用于绘图或分析)
        history = {
            'time': [],
            'voltage': [],
            'soc': [],
            'current': []
        }
        
        # 初始化输入序列 Buffer (长度 SEQ_LENGTH)
        # 假设开始前电池处于稳态：电流为 Power/InitV，dI=0
        init_current = power_watts / current_voltage
        
        # 构建初始 Buffer [Current, Temp, SOC, dI]
        # 我们用初始状态填充整个窗口，模拟"已经在这个状态运行了一段时间"
        buffer = np.zeros((SEQ_LENGTH, 4))
        for i in range(SEQ_LENGTH):
            buffer[i, 0] = init_current
            buffer[i, 1] = temperature
            buffer[i, 2] = initial_soc # SOC 略微变化也可以，这里简化为常数
            buffer[i, 3] = 0.0 # dI = 0
            
        # 模拟循环
        max_steps = 3600 * 5 # 最大模拟 5 小时，防止死循环
        
        for step in range(max_steps):
            # 1. 计算当前电流 (保持恒功率 P = V * I -> I = P / V)
            # 使用上一步的电压来估算当前电流
            # 如果电压过低，电流会变得非常大
            if current_voltage < 0.1: 
                break
                
            current_i = power_watts / current_voltage
            
            # 2. 更新 SOC (安时积分)
            # dSOC = - I * dt / Capacity / 3600
            # 注意：如果 current_i 是正值代表放电，则 SOC 减少
            delta_soc = (current_i * dt) / (BATTERY_CAPACITY_AH * 3600)
            current_soc -= delta_soc
            
            # 截止条件检查
            if current_soc <= 0:
                print("预测结束: SOC 耗尽")
                break
                
            # 3. 准备模型输入
            # 滚动 Buffer
            buffer[:-1] = buffer[1:] # 左移
            buffer[-1] = [current_i, temperature, current_soc, 0.0] # dI 简化为 0 (稳态放电)
            # 如果想更精确，可以计算 dI = current_i - last_i
            
            # 归一化
            # (Input - Mean) / Scale
            input_norm = (buffer - self.mean) / self.scale
            
            # 转 Tensor
            input_tensor = torch.tensor(input_norm, dtype=torch.float32).unsqueeze(0).to(self.device) # [1, Seq, 4]
            # 还需要 dummy currents_curr (用于物理公式，这里已经包含了 raw current 的逻辑，但模型内部可能还需要)
            # 在 train.py 中，我们传给模型的是 raw_current 用于物理约束
            # model forward: x, current_I
            # current_I 应该是 shape [Batch, 1] 的当前时刻真实电流
            current_I_tensor = torch.tensor([[current_i]], dtype=torch.float32).to(self.device)
            
            # 4. 模型预测
            with torch.no_grad():
                # model 返回: pred_Voltage, pred_R0, pred_OCV, pred_Up, pred_R0_seq
                # 注意: 根据之前的错误提示，返回值的数量可能不匹配
                # 让我们检查 model.py 的 forward 返回
                # 假设它现在返回 5 个值 (包含 pred_Up)
                outputs = self.model(input_tensor, current_I_tensor)
                if len(outputs) == 5:
                     pred_v, _, _, _, _ = outputs
                elif len(outputs) == 4:
                     pred_v, _, _, _ = outputs
                else:
                     # Fallback usually the first one is voltage
                     pred_v = outputs[0]
                
            current_voltage = pred_v.item()
            
            # 5. 记录与检查
            time_elapsed += dt
            
            # 每 60 秒记录一次 (减少内存占用)
            if step % 60 == 0:
                history['time'].append(time_elapsed / 60.0) # 分钟
                history['voltage'].append(current_voltage)
                history['soc'].append(current_soc)
                history['current'].append(current_i)
            
            if current_voltage <= CUTOFF_VOLTAGE:
                print(f"预测结束: 达到截止电压 {CUTOFF_VOLTAGE}V")
                break
        
        minutes = time_elapsed / 60.0
        return minutes, history

if __name__ == "__main__":
    # 示例用法
    predictor = RuntimePredictor()
    
    print("\n请根据提示输入参数:")
    try:
        t_input = input("请输入环境温度 (°C) [默认 25]: ")
        temp = float(t_input) if t_input else 25.0
        
        s_input = input("请输入当前电量 SOC (0.0-1.0) [默认 1.0]: ")
        soc = float(s_input) if s_input else 1.0
        
        p_input = input("请输入设备功耗 (W) [默认 5.0]: ")
        power = float(p_input) if p_input else 5.0
        
        mins, hist = predictor.predict_remaining_time(soc, temp, power)
        
        print(f"\n============================")
        print(f"预测结果: 还能使用 {mins:.2f} 分钟 ({mins/60:.2f} 小时)")
        print(f"终止时电压: {hist['voltage'][-1]:.3f} V")
        print(f"终止时电量: {hist['soc'][-1]:.1%}")
        print(f"============================")
        
    except ValueError:
        print("输入格式错误，请输入数字。")
    except Exception as e:
        print(f"发生错误: {e}")
