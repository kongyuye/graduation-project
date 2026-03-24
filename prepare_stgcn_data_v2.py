import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

def extract_node_features(train_dir):
    """
    不再计算静态邻接矩阵，而是收集所有特征，仅仅为缩放做准备
    """
    all_data = []
    files = glob.glob(os.path.join(train_dir, '*_features.csv'))
    
    for f in files:
        df = pd.read_csv(f)
        df = df.drop(columns=['time_idx'])
        all_data.append(df)
        
    combined_df = pd.concat(all_data, axis=0)
    return combined_df.columns.tolist()

def create_stgcn_dataset(input_dir, output_dir, window_size=30, scaler=None, is_train=True):
    """
    读取清洗后的特征，进行归一化和滑动窗口切片
    PDF要求:
    - 归一化策略：必须采用最大最小值归一化(Min-Max Normalization)将特征统一至[0,1]区间。
    - 不再需要计算静态物理图(皮尔逊相关系数)，因为模型将采用"动态自适应图"。
    """
    files = glob.glob(os.path.join(input_dir, '*_features.csv'))
    
    X_all = []
    Y_all = []
    
    # 收集所有数据以适配 scaler (仅限训练集)
    if is_train and scaler is None:
        all_df = []
        for f in files:
            df = pd.read_csv(f).drop(columns=['time_idx'])
            all_df.append(df)
        scaler = MinMaxScaler(feature_range=(0, 1)) # PDF 明确要求的归一化
        scaler.fit(pd.concat(all_df, axis=0))
        
    for f in files:
        bearing_name = os.path.basename(f).replace('_features.csv', '')
        df = pd.read_csv(f)
        
        # 1. 丢弃 time_idx
        df_features = df.drop(columns=['time_idx'])
        
        # 2. 归一化 (直接使用特征，去除了之前擅自加的EMA平滑)
        data_scaled = scaler.transform(df_features)
        
        # 3. 滑动窗口切片生成 (Batch, T, V, C)
        # 这里 C=1, V=14, T=window_size
        num_samples = len(data_scaled) - window_size + 1
        
        if num_samples <= 0:
            continue
            
        X_bearing = np.zeros((num_samples, window_size, data_scaled.shape[1], 1))
        # RUL 标签定义
        Y_bearing = np.zeros(num_samples) 
        
        for i in range(num_samples):
            window_data = data_scaled[i : i + window_size, :]
            X_bearing[i] = np.expand_dims(window_data, axis=-1)
            Y_bearing[i] = len(data_scaled) - (i + window_size)
            
        X_all.append(X_bearing)
        Y_all.append(Y_bearing)
        
    if not X_all:
        print(f"警告: 在 {input_dir} 中没有找到数据或样本数太少。")
        return scaler
        
    # 合并所有轴承的数据
    X_final = np.concatenate(X_all, axis=0)
    Y_final = np.concatenate(Y_all, axis=0)
    
    # 转换为 STGCN 标准格式: (Batch, Channels, Spatial_Nodes, Time_Steps)
    # 当前 X_final 形状: (Batch, T, V, C) -> transpose to (Batch, C, V, T)
    X_stgcn = np.transpose(X_final, (0, 3, 2, 1))
    
    print(f"[{'Train' if is_train else 'Test'}] 最终输入数据形状: {X_stgcn.shape}")
    print(f"[{'Train' if is_train else 'Test'}] 最终标签形状: {Y_final.shape}")
    
    # 保存数据
    os.makedirs(output_dir, exist_ok=True)
    prefix = 'train' if is_train else 'test'
    np.save(os.path.join(output_dir, f'X_{prefix}.npy'), X_stgcn)
    np.save(os.path.join(output_dir, f'Y_{prefix}.npy'), Y_final)
    
    return scaler

def main():
    train_in_dir = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct5/processed_train_data"
    test_in_dir = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct5/processed_test_data"
    output_dir = "/Users/wanglixiao/Desktop/大学/大四上/毕设/newproduct5/stgcn_dataset_v2"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("PDF 算法要求: 使用 MinMaxScaler [0,1]，采用动态自适应图(不再预计算静态物理图)。")
    print("-" * 50)
    
    node_names = extract_node_features(train_in_dir)
    print(f"节点(特征)列表: {node_names}")
    print("-" * 50)
    
    print("1. 处理训练集...")
    scaler = create_stgcn_dataset(train_in_dir, output_dir, window_size=30, scaler=None, is_train=True)
    
    # 保存 scaler 以备后用
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("-" * 50)
    print("2. 处理测试集...")
    create_stgcn_dataset(test_in_dir, output_dir, window_size=30, scaler=scaler, is_train=False)
    
    print("-" * 50)
    print(f"符合 PDF 要求的新时空图数据集已保存至: {output_dir}")

if __name__ == "__main__":
    main()
