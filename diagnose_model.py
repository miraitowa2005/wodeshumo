import torch
import torch.nn as nn
import os
import sys

# === 配置 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_physics_model_final.pth")  # 你的模型路径

def analyze_weights():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到文件 {MODEL_PATH}")
        return

    print(f"🔍 正在加载模型: {MODEL_PATH} ...")
    try:
        # 加载权重 (map_location='cpu' 确保在任何环境都能跑)
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    print("✅ 模型加载成功！开始分析关键层权重...\n")

    # 1. 检查物理头 (Physical Heads) 的活跃度
    # 如果权重接近 0，说明模型“放弃”了该物理参数的预测，直接输出常数
    heads = ['head_r0.2.weight', 'head_ocv.2.weight', 'head_up.2.weight'] # 假设是第2层为输出层
    
    # 自动查找实际的键名 (防止层数不同)
    keys = list(state_dict.keys())
    
    for head_name in ['head_r0', 'head_ocv', 'head_up']:
        # 找到该头部的最后一层权重
        relevant_keys = [k for k in keys if head_name in k and 'weight' in k]
        if not relevant_keys:
            print(f"⚠️ 警告: 未找到 {head_name} 的权重，模型结构可能不匹配。")
            continue
        
        last_layer_key = relevant_keys[-1] # 取最后一层
        bias_key = last_layer_key.replace('weight', 'bias')
        
        w = state_dict[last_layer_key]
        b = state_dict.get(bias_key, None)
        
        print(f"📊 [{head_name.upper()} 分析] (Layer: {last_layer_key})")
        print(f"   - 权重均值 (Mean): {w.mean().item():.6f}")
        print(f"   - 权重标准差 (Std): {w.std().item():.6f}")
        if b is not None:
            print(f"   - 偏置值 (Bias): {b.mean().item():.6f}")
        
        # 诊断逻辑
        if w.std() < 1e-4:
            print(f"   🔴 警报: 权重几乎为 0！模型可能发生了'坍缩'，只输出常数。")
        elif head_name == 'head_r0' and b is not None and b.item() < 0:
            print(f"   🟠 注意: 内阻 R0 偏置为负，需依靠 Softplus 激活函数修正。")
        else:
            print(f"   🟢 状态正常: 权重分布合理。")
        print("-" * 40)

if __name__ == "__main__":
    analyze_weights()