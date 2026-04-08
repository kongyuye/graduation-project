import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 引入已有的核心模型模块
from train_pipeline import PHM_STGNN_Model

class PHMRealDataset(Dataset):
    """
    针对 IEEE PHM 2012 预处理数据的 PyTorch Dataset。
    核心机制：从特征时间序列中执行「滑动窗口」截取，生成 [T, N, C_in] 的三维张量。
    """
    def __init__(self, data_dir: str, window_size: int = 50, stride: int = 5):
        self.window_size = window_size
        self.windows_x = []
        self.windows_cond = []
        self.windows_y_cls = []
        self.windows_y_rul = []
        
        # 匹配之前处理好的 train_features.csv 文件
        csv_files = glob.glob(os.path.join(data_dir, "*_train_features.csv"))
        print(f"找到 {len(csv_files)} 个训练集轴承特征文件。")
        
        for file in csv_files:
            df = pd.read_csv(file)
            total_len = len(df)
            if total_len < window_size:
                continue
            
            # --- 构建目标标签 ---
            # 1. 真实 RUL (剩余寿命百分比: 从 1.0 线性递减到 0.0)
            rul_array = (total_len - np.arange(total_len) - 1) / total_len
            
            # 2. 分类标签 (0: 正常 >0.6, 1: 轻微退化 0.2~0.6, 2: 严重退化 <=0.2)
            cls_array = np.zeros(total_len, dtype=np.int64)
            cls_array[(rul_array <= 0.6) & (rul_array > 0.2)] = 1
            cls_array[rul_array <= 0.2] = 2
            
            # --- 提取输入特征 ---
            # 我们选取水平振动的 8 个小波包能量比例作为图神经网络的 8 个空间节点 (N=8, C_in=1)
            node_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
            x_data = df[node_cols].values # 形状: (L, 8)
            x_data = np.expand_dims(x_data, axis=-1) # 增加通道维度 -> (L, 8, 1)
            
            # 我们选取 RMS 和 温度 作为设备的外部全局工况提示 (cond_dim=2)
            cond_cols = ['h_rms', 'temperature']
            cond_data = df[cond_cols].values # 形状: (L, 2)
            
            # --- 滑动窗口切片 ---
            for i in range(0, total_len - window_size + 1, stride):
                # 截取长度为 window_size 的连续时间步
                self.windows_x.append(x_data[i : i + window_size])
                
                # 取窗口最后一个时刻的工况作为全局 Prompt
                self.windows_cond.append(cond_data[i + window_size - 1])
                
                # 取窗口最后一个时刻的真实状态作为预测目标
                self.windows_y_cls.append(cls_array[i + window_size - 1])
                self.windows_y_rul.append(rul_array[i + window_size - 1])
                
        # 转换为内存连续的 Numpy 数组，提升 PyTorch 加载效率
        self.windows_x = np.array(self.windows_x, dtype=np.float32)
        self.windows_cond = np.array(self.windows_cond, dtype=np.float32)
        self.windows_y_cls = np.array(self.windows_y_cls, dtype=np.int64)
        self.windows_y_rul = np.array(self.windows_y_rul, dtype=np.float32)
        
        print(f"成功生成 {len(self.windows_x)} 个时间窗样本 (窗口大小: {window_size}, 步长: {stride})")

    def __len__(self):
        return len(self.windows_x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.windows_x[idx]),
            torch.tensor(self.windows_cond[idx]),
            torch.tensor(self.windows_y_cls[idx]),
            torch.tensor(self.windows_y_rul[idx])
        )

def main():
    print("=== 初始化 ST-GNN 真实数据训练流程 ===")
    
    # 1. 准备数据
    data_dir = "/root/autodl-tmp/processed_data/train"
    dataset = PHMRealDataset(data_dir, window_size=50, stride=5)
    # 使用 DataLoader 批量加载数据，启用 shuffle 打乱时序防止过拟合
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 2. 设备与模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练加速设备: {device}")
    
    # 实例化我们的多任务模型
    model = PHM_STGNN_Model(num_nodes=8, c_in=1, d_model=256, cond_dim=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    num_epochs = 50
    alpha = 0.5  # 分类与回归损失的联合优化权重
    
    print("\n=== 开始模型训练 ===")
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cls_loss, total_reg_loss = 0.0, 0.0, 0.0
        
        for batch_idx, (x, cond, y_cls, y_rul) in enumerate(dataloader):
            # 将张量挂载到 GPU/MPS
            x, cond = x.to(device), cond.to(device)
            y_cls = y_cls.to(device)
            y_rul = y_rul.to(device).unsqueeze(-1) # 形状对齐为 (B, 1)
            
            optimizer.zero_grad()
            
            # 前向传播
            preds = model(x, cond)
            
            # 多任务损失计算
            loss_cls = criterion_cls(preds['class_logits'], y_cls)
            loss_reg = criterion_reg(preds['rul_pred'], y_rul)
            loss = alpha * loss_cls + (1 - alpha) * loss_reg
            
            # 反向传播与参数更新
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg.item()
            
            # 打印日志 (每 20 个 Batch)
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1:3d}/{len(dataloader)}] | "
                      f"Total Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, Reg: {loss_reg.item():.4f})")
                
        # 打印 Epoch 级统计 (这能真实反映模型是否在收敛)
        avg_loss = total_loss / len(dataloader)
        avg_cls_loss = total_cls_loss / len(dataloader)
        avg_reg_loss = total_reg_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} 结束 | 平均 Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f})\n")
        
    # 3. 固化模型并保存权重
    save_path = "/root/autodl-tmp/graduation-project/phm_stgnn_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 训练大循环圆满完成！模型已固化并成功保存至:\n {save_path}")

if __name__ == "__main__":
    main()
