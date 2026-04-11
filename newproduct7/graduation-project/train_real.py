import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

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
        
        # 引入数据增强机制（路线 A：宏观截取 + 微观滑动窗口）
        num_augments = 3  # 每个轴承文件增强/截取 3 次
        
        for file in csv_files:
            df = pd.read_csv(file)
            total_len = len(df)
            if total_len < window_size:
                continue
            
            # 对同一个轴承数据进行多次随机截取（Data Augmentation）
            for aug in range(num_augments):
                # --- 路线 A：宏观随机截取 ---
                # 1. 确定截取的起止点 (起点 0~50%, 终点 75~100%)
                # 如果是第一次循环 (aug==0)，我们保留完整的全生命周期，保证基线数据完整
                if aug == 0:
                    start_idx = 0
                    end_idx = total_len
                else:
                    start_ratio = np.random.uniform(0.0, 0.5)
                    end_ratio = np.random.uniform(0.75, 1.0)
                    start_idx = int(start_ratio * total_len)
                    end_idx = int(end_ratio * total_len)
                    
                    # 容错处理：确保截取长度大于滑动窗口大小
                    if end_idx - start_idx < window_size:
                        start_idx = 0
                        end_idx = total_len

                # 2. 截取对应片段的数据
                truncated_df = df.iloc[start_idx:end_idx].copy()
                truncated_len = len(truncated_df)
                
                # --- 构建目标标签 (注意：真实 RUL 的计算必须基于真实的 total_len) ---
                # 即使被截取，该时刻的绝对剩余寿命百分比不能变，必须用原 total_len 计算
                # rul_array = (total_len - 当前绝对步数 - 1) / total_len
                absolute_steps = np.arange(start_idx, end_idx)
                rul_array = (total_len - absolute_steps - 1) / total_len
                
                # 分类标签 (0: 正常 >0.6, 1: 轻微退化 0.2~0.6, 2: 严重退化 <=0.2)
                cls_array = np.zeros(truncated_len, dtype=np.int64)
                cls_array[(rul_array <= 0.6) & (rul_array > 0.2)] = 1
                cls_array[rul_array <= 0.2] = 2
                
                # --- 提取输入特征 ---
                # 时域特征 (4*2 = 8)
                h_time_cols = ['h_skewness', 'h_kurtosis', 'h_p2p', 'h_shape_factor']
                v_time_cols = ['v_skewness', 'v_kurtosis', 'v_p2p', 'v_shape_factor']
                # 频域特征 (9*2 = 18)
                h_freq_cols = ['h_spectral_centroid'] + [f'h_fft_band_energy_ratio_{i}' for i in range(8)]
                v_freq_cols = ['v_spectral_centroid'] + [f'v_fft_band_energy_ratio_{i}' for i in range(8)]
                # 时频域特征 (8*2 = 16)
                h_wpt_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
                v_wpt_cols = [f'v_wpt_energy_ratio_{i}' for i in range(8)]
                
                node_cols = h_time_cols + v_time_cols + h_freq_cols + v_freq_cols + h_wpt_cols + v_wpt_cols
                
                x_data = truncated_df[node_cols].values # 形状: (L, 42)
                x_data = np.expand_dims(x_data, axis=-1) # 增加通道维度 -> (L, 42, 1)
                
                cond_cols = ['h_rms', 'temperature']
                cond_data = truncated_df[cond_cols].values # 形状: (L, 2)
                
                # --- 滑动窗口切片 ---
                for i in range(0, truncated_len - window_size + 1, stride):
                    self.windows_x.append(x_data[i : i + window_size])
                    self.windows_cond.append(cond_data[i + window_size - 1])
                    self.windows_y_cls.append(cls_array[i + window_size - 1])
                    self.windows_y_rul.append(rul_array[i + window_size - 1])
                
        # 转换为内存连续的 Numpy 数组，提升 PyTorch 加载效率
        # - 第 1 个维度：样本索引（第几个滑动窗口）。
        # - 第 2 个维度：时间步长（50 个时刻）。
        # - 第 3 个维度：空间节点（42 个特征）。
        # - 第 4 个维度：通道数（1）。
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
    data_dir = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct7/processed_data/train"
    dataset = PHMRealDataset(data_dir, window_size=50, stride=5)
    # 使用 DataLoader 批量加载数据，启用 shuffle 打乱时序防止过拟合
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 2. 设备与模型初始化
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"训练加速设备: {device}")
    
    # 实例化我们的多任务模型
    model = PHM_STGNN_Model(num_nodes=42, c_in=1, d_model=256, cond_dim=2, num_classes=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    # 定义损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    num_epochs = 100
    alpha = 0.5  # 分类与回归损失的联合优化权重
    
    # === 添加学习率调度器：Warmup (前20%) + 余弦退火 (Cosine Annealing) ===
    warmup_epochs = int(num_epochs * 0.2)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性 Warmup：从 0.1 倍逐渐增加到 1.0
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # 余弦退火：从 1.0 衰减到 0.001
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.001 + 0.5 * (1 - 0.001) * (1 + np.cos(np.pi * progress))
            
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 用于记录 Loss 曲线的数据
    history_total_loss = []
    history_cls_loss = []
    history_reg_loss = []
    history_lr = []
    
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
        
        # 记录并更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        history_lr.append(current_lr)
        scheduler.step()
        
        # 记录到历史列表中
        history_total_loss.append(avg_loss)
        history_cls_loss.append(avg_cls_loss)
        history_reg_loss.append(avg_reg_loss)
        
        print(f"==> Epoch {epoch+1} 结束 | LR: {current_lr:.6f} | 平均 Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Reg: {avg_reg_loss:.4f})\n")
        
    # 3. 固化模型并保存权重
    save_path = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct7/phm_stgnn_model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"🎉 训练大循环圆满完成！模型已固化并成功保存至:\n {save_path}")
    
    # 4. 绘制并保存 Loss 曲线
    print("\n=== 绘制 Loss 曲线 ===")
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, num_epochs + 1)
    
    plt.plot(epochs_range, history_total_loss, label='Total Loss', color='black', linewidth=2)
    plt.plot(epochs_range, history_cls_loss, label='Classification Loss (CrossEntropy)', color='blue', linestyle='--')
    plt.plot(epochs_range, history_reg_loss, label='Regression Loss (MSE)', color='red', linestyle='-.')
    
    plt.title('Training Loss Curve (Multi-Task ST-GNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 保存图片
    plot_save_path = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct7/graduation-project/loss_curve.png"
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Loss 曲线已保存至: {plot_save_path}")
    
    # 尝试直接显示图片（如果在支持 GUI 的环境下）
    try:
        plt.show()
    except Exception as e:
        print("当前环境无法直接显示弹窗，请直接查看保存的图片文件。")

if __name__ == "__main__":
    main()