# MCM 2026 Problem A - Physics-Informed Transformer Solution

本项目提供了一套完整的 **Physics-Informed Transformer (物理感知 Transformer)** 解决方案，专为 **MCM 2026 Problem A (Randomized Battery Usage)** 打造。项目涵盖了从原始 `.mat` 数据清洗、特征工程、到高性能模型训练及最终论文图表生成的全流程。

特别针对 **RTX 4090** 等高性能服务器进行了深度优化，支持混合精度训练 (AMP) 和全量数据加载。

---

## 🚀 1. 环境配置 (Environment Setup)

在开始之前，请确保你的服务器安装了 Python 3.8+。

### 1.1 安装依赖
我们提供了一键安装脚本。请在终端运行：

```bash
pip install -r requirements.txt
```

**核心依赖库：**
*   `torch`: 深度学习框架 (建议安装 CUDA 版本以发挥 RTX 4090 性能)
*   `pandas` / `numpy`: 数据处理
*   `pyarrow` / `fastparquet`: 高性能 Parquet 文件读写
*   `scikit-learn`: 数据归一化 (StandardScaler)
*   `matplotlib` / `seaborn`: 论文绘图

---

## 🛠️ 2. 数据处理流程 (Data Pipeline)

原始数据为 `.mat` 格式，我们需要将其清洗、降采样并转换为高效的 Parquet 格式。

### 运行预处理脚本
```bash
python preprocess_final.py
```

### 该脚本执行的操作：
1.  **数据映射**: 自动扫描 `database/` 目录下的 7 个子文件夹，识别电池 ID (如 RW9, RW17 等)。
2.  **SOH/SOC 计算**:
    *   通过库仑计数法 (Coulomb Counting) 估算实时 SOC。
    *   通过参考放电周期 (Reference Discharge) 计算电池容量衰减 (SOH)。
3.  **降采样 (Downsampling)**: 将原始高频数据降采样至 **1Hz**，减少噪声并对齐时间步。
4.  **物理特征工程**:
    *   计算微分特征 `dI` (电流变化率) 和 `dV` (电压变化率)。
    *   自动去除异常值 (Outliers)。
5.  **归一化 (Normalization)**: 使用 `StandardScaler` 对输入特征进行标准化，保存 `scaler.pkl` 以备推理使用。
6.  **格式转换**: 生成 `train_main.parquet` (训练集) 和 `test_gamer.parquet` (测试集)，大幅提升 I/O 速度。

**输出文件：**
*   `processed_data/train_main.parquet`: 全量训练数据 (约 5000万行)
*   `processed_data/test_gamer.parquet`: 测试/验证数据
*   `processed_data/scaler.pkl`: 归一化参数文件

---

## 🔥 3. 模型训练 (Model Training)

训练脚本已针对 **RTX 4090** 进行了旗舰级优化，支持 4096 Batch Size 和混合精度训练。

### 运行训练
```bash
python train.py
```

### 关键配置 (可在 `train.py` 顶部修改)：
*   `BATCH_SIZE = 4096`: 榨干 24GB 显存，极大加速训练。
*   `SEQ_LENGTH = 60`: 使用过去 60 秒的历史数据预测当前状态。
*   `STRIDE = 1`: **滑动窗口步长为 1**，不跳过任何数据点，实现真正的全量训练。
*   `EPOCHS = 100`: 配合 `CosineAnnealingLR` 调度器，确保 Loss 收敛至极限。
*   **Physics Regularization**: Loss 函数包含物理约束项 `loss_reg_r`，强制模型预测的内阻 $R_0$ 符合物理规律 ($R_0 > 0$)。

### 模型架构 (PhysicsAttentionModel):
*   **Transformer Encoder**: 6 层，8 头注意力，256 隐藏层维度。
*   **Physical Heads**: 独立的神经网络头分别预测 $R_0$ (内阻), $OCV$ (开路电压), $U_p$ (极化电压)。
*   **Physical Fusion Layer**: 显式物理层 $V = OCV - I \times R_0 - U_p$，将深度学习与欧姆定律完美融合。

**输出文件：**
*   `best_physics_model_4090.pth`: 训练好的最佳模型权重。

---

## 📊 4. 预测与可视化 (Prediction & Visualization)

模型训练完成后，使用此脚本生成论文所需的关键图表。

### 运行推理
```bash
python predict_and_visualize.py
```

### 生成的图表 (`results/final_result_viz.png`) 包含三部分：
1.  **Prediction vs Ground Truth**: 红线(预测)与黑线(真实)的重合度展示模型精度。
2.  **Physical Parameters ($R_0$)**: 展示模型“反演”出的电池内阻随电流变化的曲线 (这是物理模型的核心优势)。
3.  **Attention Heatmap**: 解释性热力图，展示模型关注过去 60 秒中的哪些关键时刻 (Interpretability)。

---

## 📂 5. 项目文件结构说明

```text
2026MCM/
├── database/                   # 原始 NASA 数据集
├── processed_data/             # 清洗后的 Parquet 数据
│   ├── train_main.parquet
│   ├── test_gamer.parquet
│   └── scaler.pkl
├── preprocess_final.py         # [Step 1] 数据预处理脚本
├── train.py                    # [Step 2] 模型训练脚本 (RTX 4090 Optimized)
├── predict_and_visualize.py    # [Step 3] 推理与绘图脚本
├── model.py                    # 模型定义文件 (PhysicsAttentionModel)
├── requirements.txt            # 依赖包列表
└── README.md                   # 本说明文件
```

---

## 💡 6. 常见问题 (FAQ)

**Q: 显存不足 (OOM) 怎么办？**
A: 请在 `train.py` 中减小 `BATCH_SIZE` (例如从 4096 改为 1024) 或减小 `D_MODEL` (从 256 改为 128)。

**Q: 如何查看训练进度？**
A: 脚本内置了进度条，会实时显示当前的 Loss 和 MSE。

**Q: 为什么生成的 $R_0$ 是动态变化的？**
A: 锂电池的内阻并非常数，它随 SOC、温度和电流倍率变化。我们的模型成功捕捉到了这种非线性物理特性，这正是 O 奖论文的加分项。
"# wodeshumo" 
