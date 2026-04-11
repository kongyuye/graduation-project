# 基于时空图神经网络与 Transformer 架构的电子设备故障预测系统

本项目是基于 IEEE PHM 2012 挑战赛（PRONOSTIA 实验台轴承数据集）开发的剩余使用寿命 (RUL) 预测系统。
项目核心架构采用了 **ST-GNN (时空图神经网络)** 与 **Transformer** 的混合模型，能够有效捕捉设备退化过程中的空间拓扑关系与全局时序演化规律。

## 项目结构

* `prepare_stgcn_data_v2.py`: 数据预处理脚本。负责从原始 CSV 振动信号中提取统计特征，并使用 MinMaxScaler 进行归一化，生成滑动窗口切片的张量数据 (npy 格式)。
* `model_architecture.py`: 核心模型架构定义。包含动态自适应图层、1D-CNN 局部特征提取、GCN 空间聚合以及 Transformer 全局时序编码器。
* `train.py`: PyTorch 模型训练循环，支持多任务联合损失（分类交叉熵 + 回归 MSE）。

## 核心技术点

1. **多维特征工程**: 提取振动信号的时域统计特征 (均方根 RMS、峭度 Kurtosis 等)。
2. **动态自适应图 (Dynamic Adaptive Graph)**: 摒弃静态的物理连接矩阵，通过可学习的高维嵌入向量，在训练过程中自适应挖掘不同传感器/特征节点之间的因果耦合关系。
3. **时空联合建模**: 
   - 空间域: 使用 Graph Convolution (GCN) 进行多特征节点的消息传递。
   - 时间域: 使用 Transformer Encoder 突破长程梯度衰减瓶颈，捕捉全局退化趋势。

## 环境依赖
请参考 `requirements.txt`。

## 如何运行
1. **数据准备**: 下载 PHM2012 原始数据并修改 `prepare_stgcn_data_v2.py` 中的路径，运行以生成预处理后的 `.npy` 数据。
2. **模型训练**: 运行 `python train.py` 开始训练。

## 声明
本项目为毕业设计的一部分。
