import torch
import torch.nn as pd
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphConstruction(nn.Module):
    """
    动态自适应图构建层 (Dynamic Adaptive Graph)
    根据 PDF 指南:
    通过为每一个节点引入一个可学习的高维嵌入向量 E，
    通过计算 A = Softmax(ReLU(E * E^T)) 动态生成节点间的注意力权重矩阵。
    """
    def __init__(self, num_nodes, embedding_dim=64):
        super(DynamicGraphConstruction, self).__init__()
        # E in R^{N x d}
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        nn.init.xavier_uniform_(self.node_embeddings)

    def forward(self):
        # E * E^T
        e_e_t = torch.matmul(self.node_embeddings, self.node_embeddings.transpose(0, 1))
        # A = Softmax(ReLU(E * E^T))
        adj = F.softmax(F.relu(e_e_t), dim=-1)
        return adj

class GraphConvolution(nn.Module):
    """
    基于谱图理论或空间消息传递的基础图卷积层
    这里实现一个基于动态邻接矩阵的标准 GCN 层
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x shape: (Batch, Nodes, in_channels)
        # adj shape: (Nodes, Nodes)
        support = torch.matmul(x, self.weight) # (Batch, Nodes, out_channels)
        # 节点间消息传递
        output = torch.einsum('ij, bjc -> bic', adj, support) + self.bias
        return F.relu(output)

class STGNN_Block(nn.Module):
    """
    时空图卷积模块
    PDF 指南中提到：局部特征提取层(1D-CNN) + 空间拓扑聚合层(GCN)
    """
    def __init__(self, in_channels, spatial_channels, num_nodes):
        super(STGNN_Block, self).__init__()
        # 局部特征提取层 (沿时间轴 1D-CNN) - PDF 中推荐多尺度，这里先用基础 1D CNN 示例
        self.temporal_conv = nn.Conv2d(in_channels, spatial_channels, kernel_size=(1, 3), padding=(0, 1))
        # 图空间聚合层
        self.dynamic_graph = DynamicGraphConstruction(num_nodes)
        self.gcn = GraphConvolution(spatial_channels, spatial_channels)
        
    def forward(self, x):
        # x shape: (Batch, in_channels, Nodes, Time_Steps)
        
        # 1. 局部时序特征提取
        x = self.temporal_conv(x) # (B, spatial_channels, N, T)
        
        B, C, N, T = x.shape
        # 调整形状以适应 GCN: 把时间步视为 Batch 的一部分 -> (B*T, N, C)
        x_gcn = x.permute(0, 3, 2, 1).contiguous().view(B * T, N, C)
        
        # 2. 动态生成邻接矩阵
        adj = self.dynamic_graph() # (N, N)
        
        # 3. 空间特征聚合
        x_gcn = self.gcn(x_gcn, adj) # (B*T, N, C)
        
        # 恢复形状: (B, C, N, T)
        out = x_gcn.view(B, T, N, C).permute(0, 3, 2, 1)
        return out

class TransformerTemporalModule(nn.Module):
    """
    全局时序演化层 (Transformer Encoder)
    PDF 指南要求：引入正弦位置编码；堆叠 3 层 Encoder Block；隐藏层维度 256；8头注意力。
    """
    def __init__(self, feature_dim, d_model=256, nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1):
        super(TransformerTemporalModule, self).__init__()
        # 将图卷积输出的特征维度映射到 d_model
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # PyTorch 1.9+ 支持 batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.d_model = d_model

    def forward(self, x):
        # x shape 预期为: (Batch, Time_Steps, Feature_Dim)
        x = self.input_projection(x) # (B, T, d_model)
        
        # TODO: 可以加入正弦位置编码 (Positional Encoding)
        # 这里简化处理，直接传入 Transformer
        out = self.transformer_encoder(x) # (B, T, d_model)
        return out

class PredictionHead(nn.Module):
    """
    多任务预测输出层 (GAP + 残差 MLP)
    PDF 指南要求: 全局平均池化 (GAP) + 两层全连接层 (128 -> 64) -> 分支A(分类) / 分支B(回归)
    """
    def __init__(self, d_model, num_classes=3):
        super(PredictionHead, self).__init__()
        # 将 d_model 降维到 128
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # 分支 A: 分类头 (例如: 正常、轻微退化、严重退化)
        self.classifier = nn.Linear(64, num_classes)
        # 分支 B: 回归头 (预测 RUL 百分比或绝对值)
        self.regressor = nn.Linear(64, 1)
        
    def forward(self, x):
        # x shape: (Batch, Time_Steps, d_model)
        
        # 全局平均池化 (GAP) - 沿着时间轴平均
        x = torch.mean(x, dim=1) # (Batch, d_model)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        logits = self.classifier(x)
        rul_pred = self.regressor(x)
        
        return logits, rul_pred

class STGNN_Transformer_Model(nn.Module):
    """
    端到端混合架构模型 (End-to-End Model)
    整合: ST-GNN (提取时空局部特征) -> Flatten -> Transformer (全局时序) -> Prediction Head
    """
    def __init__(self, in_channels=1, num_nodes=14, gcn_out_channels=64, 
                 d_model=256, nhead=8, num_layers=3, num_classes=3):
        super(STGNN_Transformer_Model, self).__init__()
        
        # 1. 空间拓扑聚合 (含局部特征提取)
        self.stgnn = STGNN_Block(in_channels, gcn_out_channels, num_nodes)
        
        # 将 GCN 输出的多节点特征展平，准备输入 Transformer
        # 每个时间步的特征维度 = num_nodes * gcn_out_channels
        transformer_input_dim = num_nodes * gcn_out_channels
        
        # 2. 全局时序演化
        self.transformer = TransformerTemporalModule(
            feature_dim=transformer_input_dim, 
            d_model=d_model, 
            nhead=nhead, 
            num_layers=num_layers
        )
        
        # 3. 预测头
        self.prediction_head = PredictionHead(d_model, num_classes)

    def forward(self, x):
        # x shape: (Batch, Channels=1, Nodes=14, Time_Steps=30)
        
        # --- ST-GNN 阶段 ---
        # 提取时空特征并进行节点间信息交互
        x = self.stgnn(x) # (B, gcn_out_channels, N, T)
        
        # --- 维度重塑阶段 ---
        B, C, N, T = x.shape
        # 将节点维度和通道维度合并，作为 Transformer 每个时间步的特征
        x = x.permute(0, 3, 2, 1).contiguous() # (B, T, N, C)
        x = x.view(B, T, N * C) # (B, T, N*C)
        
        # --- Transformer 阶段 ---
        x = self.transformer(x) # (B, T, d_model)
        
        # --- 预测阶段 ---
        logits, rul = self.prediction_head(x)
        
        return logits, rul

# 测试模型维度的简单代码
if __name__ == "__main__":
    # 模拟我们预处理好的数据形状: (Batch, C, V, T) -> (32, 1, 14, 30)
    batch_size = 32
    channels = 1
    nodes = 14
    time_steps = 30
    
    dummy_input = torch.randn(batch_size, channels, nodes, time_steps)
    
    print(f"输入数据形状: {dummy_input.shape}")
    
    # 初始化模型
    model = STGNN_Transformer_Model(in_channels=channels, num_nodes=nodes)
    
    # 前向传播
    logits, rul_pred = model(dummy_input)
    
    print(f"分类输出形状: {logits.shape}")
    print(f"回归输出形状: {rul_pred.shape}")
