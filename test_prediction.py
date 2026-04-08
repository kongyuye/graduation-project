import os
import glob
import pandas as pd
import numpy as np
import torch
import sys

# 添加当前目录到系统路径，以确保能正确导入模型文件
sys.path.append("/root/autodl-tmp/graduation-project")
from train_pipeline import PHM_STGNN_Model

# ---------------------------------------------------------
# 1. 真实 RUL (Remaining Useful Life) - 单位: 秒
# 来源于 IEEE PHM 2012 Challenge 官方测试集真值
# ---------------------------------------------------------
actual_ruls_sec = {
    "Bearing1_3": 5730,
    "Bearing1_4": 339,
    "Bearing1_5": 1610,
    "Bearing1_6": 1460,
    "Bearing1_7": 7570,
    "Bearing2_3": 7530,
    "Bearing2_4": 1390,
    "Bearing2_5": 3090,
    "Bearing2_6": 1290,
    "Bearing2_7": 580,
    "Bearing3_3": 820
}

def predict_test_set():
    print("=== 初始化 ST-GNN 测试集 RUL 预测流程 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 推理加速设备: {device}")
    
    # ---------------------------------------------------------
    # 2. 实例化模型并加载权重
    # ---------------------------------------------------------
    model = PHM_STGNN_Model(num_nodes=8, c_in=1, d_model=256, cond_dim=2, num_classes=3).to(device)
    
    model_path = "/root/autodl-tmp/graduation-project/phm_stgnn_model_weights.pth"
    if not os.path.exists(model_path):
        print(f"[!] Error: 模型权重文件不存在 {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("[*] 模型权重加载成功。")

    # ---------------------------------------------------------
    # 3. 读取测试集特征数据
    # ---------------------------------------------------------
    test_data_dir = "/root/autodl-tmp/processed_data/test"
    csv_files = glob.glob(os.path.join(test_data_dir, "*_test_features.csv"))
    print(f"[*] 找到 {len(csv_files)} 个测试集特征文件。")
    
    window_size = 50
    results = []
    
    # ---------------------------------------------------------
    # 4. 遍历所有测试轴承进行预测
    # ---------------------------------------------------------
    for file in sorted(csv_files):
        filename = os.path.basename(file)
        bearing_name = filename.replace("_test_features.csv", "")
        if bearing_name not in actual_ruls_sec:
            print(f"[!] Warning: {bearing_name} 不在真实 RUL 列表中。")
            continue
            
        df = pd.read_csv(file)
        truncated_len = len(df)
        
        if truncated_len < window_size:
            print(f"[!] Warning: {bearing_name} 的长度 ({truncated_len}) 小于滑动窗口大小 ({window_size})，跳过。")
            continue
            
        # 提取图神经网络输入特征 (8个频带的小波包能量比例)
        node_cols = [f'h_wpt_energy_ratio_{i}' for i in range(8)]
        x_data = df[node_cols].values # (L, 8)
        x_data = np.expand_dims(x_data, axis=-1) # (L, 8, 1)
        
        # 提取全局工况提示特征 (RMS 和 温度)
        cond_cols = ['h_rms', 'temperature']
        cond_data = df[cond_cols].values # (L, 2)
        
        # 截取时间序列最后一个窗口，用于预测最新的剩余寿命
        last_x = x_data[-window_size:] # (50, 8, 1)
        last_cond = cond_data[-1] # (2,)
        
        # 转换为 PyTorch 张量并增加 Batch 维度
        x_tensor = torch.tensor(last_x, dtype=torch.float32).unsqueeze(0).to(device) # (1, 50, 8, 1)
        cond_tensor = torch.tensor(last_cond, dtype=torch.float32).unsqueeze(0).to(device) # (1, 2)
        
        with torch.no_grad():
            preds = model(x_tensor, cond_tensor)
            predicted_p = preds['rul_pred'].item() # 预测的寿命百分比 p (剩余寿命占总寿命比例)
            predicted_class = torch.argmax(preds['class_logits'], dim=1).item()
            
        # 限制 p 的范围防止除以 0 或负数出现
        p_clipped = max(0.01, min(0.99, predicted_p))
        
        # ---------------------------------------------------------
        # 5. 反归一化计算：由百分比转换为具体物理时间
        # ---------------------------------------------------------
        # 根据模型训练时的归一化公式: p = RUL_rows / (Truncated_len + RUL_rows)
        # 推导得出: RUL_rows = p * Truncated_len / (1 - p)
        predicted_rul_rows = p_clipped * truncated_len / (1 - p_clipped)
        
        # PHM2012 数据集采样规则：每 10 秒记录一次 0.1s 信号。所以每行特征代表 10 秒钟。
        predicted_rul_sec = predicted_rul_rows * 10
        actual_rul_sec = actual_ruls_sec[bearing_name]
        
        # 计算误差
        error_pct = (predicted_rul_sec - actual_rul_sec) / actual_rul_sec * 100
        
        results.append({
            "Bearing": bearing_name,
            "截断特征长度(行)": truncated_len,
            "预测剩余寿命比例(p)": f"{predicted_p:.4f}",
            "预测健康状态(0-2)": predicted_class,
            "实际RUL(秒)": actual_rul_sec,
            "预测RUL(秒)": f"{predicted_rul_sec:.1f}",
            "误差Error(%)": f"{error_pct:.2f}%"
        })
        
    # ---------------------------------------------------------
    # 6. 打印与保存结果
    # ---------------------------------------------------------
    print("\n=== 预测结果与真实值对比分析 ===")
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    output_csv = "/root/autodl-tmp/graduation-project/test_predictions.csv"
    df_results.to_csv(output_csv, index=False)
    print(f"\n[*] 所有测试集预测结果已成功保存至: {output_csv}")

if __name__ == "__main__":
    predict_test_set()
