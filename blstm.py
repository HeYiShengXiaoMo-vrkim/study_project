import torch
import numpy as np
import torch.utils.data as tchdata
import torch.nn.functional as F
from torch import nn
from datetime import datetime
import math

class MultiScaleAttention(nn.Module):
    """多尺度注意力机制，捕捉不同时间尺度特征"""
    def __init__(self, hidden_size):
        super(MultiScaleAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = math.sqrt(hidden_size)
        
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        q = self.query(x)  # 短期关注
        k = self.key(x)    # 中期关注
        v = self.value(x)  # 长期关注
        
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        return context
        
class MA_BLSTM(nn.Module):
    """增强版BLSTM模型 - 针对TEP故障检测优化"""
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.3):
        super(MA_BLSTM, self).__init__()
        
        # 确保hidden_size为偶数
        if hidden_size % 2 != 0:
            hidden_size += 1
        
        # 输入预处理
        self.batch_norm_input = nn.BatchNorm1d(input_size)
        
        # 多尺度特征提取 - 捕捉不同时间尺度的故障模式
        self.conv_small = nn.Conv1d(input_size, hidden_size//3, kernel_size=3, padding=1)  # 短期模式
        self.conv_medium = nn.Conv1d(input_size, hidden_size//3, kernel_size=5, padding=2)  # 中期模式
        self.conv_large = nn.Conv1d(input_size, hidden_size//3, kernel_size=7, padding=3)  # 长期模式
        
        # 批归一化
        self.bn_small = nn.BatchNorm1d(hidden_size//3)
        self.bn_medium = nn.BatchNorm1d(hidden_size//3)
        self.bn_large = nn.BatchNorm1d(hidden_size//3)
        
        # 融合层 - 结合多个时间尺度特征
        concat_size = hidden_size
        self.fusion = nn.Linear(concat_size, hidden_size)
        self.bn_fusion = nn.BatchNorm1d(hidden_size)
        
        # 双层BLSTM - 更好地捕捉时序依赖
        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            batch_first=True,
            bidirectional=True
        )
        
        # 残差连接 - 帮助捕捉缓慢漂移
        self.residual_proj = nn.Linear(hidden_size, hidden_size)
        
        # 多尺度注意力 - 关注不同类型故障的关键模式
        self.attention = MultiScaleAttention(hidden_size)
        
        # 全局信息提取
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 批归一化层
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 分类头 - 分层设计，提高区分能力
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  # GELU激活更适合复杂模式
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # 输入形状: [batch_size, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # 输入归一化
        x_flat = x.reshape(-1, features)
        x_flat = self.batch_norm_input(x_flat)
        x = x_flat.reshape(batch_size, seq_len, features)
        
        # 多尺度特征提取
        x_t = x.transpose(1, 2)  # [batch, features, seq_len]
        
        # 不同感受野的卷积层提取多尺度特征
        x_small = F.gelu(self.bn_small(self.conv_small(x_t)))
        x_medium = F.gelu(self.bn_medium(self.conv_medium(x_t)))
        x_large = F.gelu(self.bn_large(self.conv_large(x_t)))
        
        # 特征融合
        x_concat = torch.cat([x_small, x_medium, x_large], dim=1)  # 通道维度拼接
        x_t = x_concat
        
        # 转回序列格式
        x = x_t.transpose(1, 2)  # [batch, seq_len, hidden_size]
        x = F.gelu(self.fusion(x))
        
        # LSTM层级处理 - 捕捉不同时间尺度的依赖关系
        residual = x
        x, _ = self.lstm1(x)
        x = x + residual  # 残差连接1 - 帮助捕捉缓慢漂移
        
        residual = x
        x, _ = self.lstm2(x)
        x = x + residual  # 残差连接2
        
        # 注意力机制 - 关注重要的时序模式
        x_attn = self.attention(x)
        
        # 多视角特征提取
        x_t = x.transpose(1, 2)
        x_max = self.global_max_pool(x_t).squeeze(-1)  # 最大池化 - 捕捉突变特征
        x_avg = self.global_avg_pool(x_t).squeeze(-1)  # 平均池化 - 捕捉趋势特征
        
        # 特征融合
        x = torch.cat([x_max, x_avg], dim=1)
        x = self.batch_norm(x)
        
        # 分类
        x = self.classifier(x)
        
        return x


def MA_BLSTM_TE(n_samples, n_hidden, target, train_data, train_labels, test_data, test_labels):
    """TEP数据集优化版BLSTM训练与评估"""
    from utils.metrics import validate
    import math
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 数据准备
    if len(train_data.shape) == 2:
        train_data = np.expand_dims(train_data, axis=1)
    if len(test_data.shape) == 2:
        test_data = np.expand_dims(test_data, axis=1)
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    # 数据标准化 - 使用更健壮的方法
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0) + 1e-8
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    # 检查可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 转换为PyTorch张量
    train_data = torch.from_numpy(train_data).float().to(device)
    train_labels = torch.from_numpy(train_labels).long().to(device)
    test_data = torch.from_numpy(test_data).float().to(device)
    test_labels = torch.from_numpy(test_labels).long().to(device)
    
    # 创建数据集和加载器
    train_dataset = tchdata.TensorDataset(train_data, train_labels)
    test_dataset = tchdata.TensorDataset(test_data, test_labels)
    
    # 计算类别权重以处理数据不平衡
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)
    
    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False)
    
    # 初始化TEP优化模型
    model = MA_BLSTM(
        input_size=train_data.size(2),
        hidden_size=n_hidden,
        output_size=len(target),
        dropout_rate=0.3,
    ).to(device)
    
    # 优化设置 - 使用余弦退火学习率
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.001, 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度 - 余弦退火适合捕获复杂模式
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,                # 初始周期长度
        T_mult=2,              # 每次重启后周期长度增加的倍数
        eta_min=1e-6           # 最小学习率
    )
    
    # 启用混合精度训练
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 训练记录
    train_accs = []
    test_accs = []
    train_losses = []
    
    # 最佳模型保存
    best_acc = 0
    best_pred = None
    best_targets = None
    best_model_state = None
    
    print("开始训练MA优化版BLSTM模型...")
    
    # 训练循环
    for epoch in range(60):
        # 训练模式
        model.train()
        train_loss = 0
        
        for inputs, labels in train_loader:
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:  # 使用混合精度
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    # 使用类别权重处理不平衡问题
                    loss = F.cross_entropy(outputs, labels, weight=class_weights)
                
                # 使用缩放器处理梯度
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 标准训练
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, labels, weight=class_weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
        
        # 学习率更新
        scheduler.step()
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            train_acc, _, _ = validate(model, train_loader)
            train_accs.append(train_acc)
            
            test_acc, test_preds, test_targets_current = validate(model, test_loader)
            test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_pred = test_preds
            best_targets = test_targets_current
            no_improve = 0
        else:
            no_improve += 1
        
        # 记录本轮损失
        epoch_loss = train_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 打印训练信息
        print(f"{datetime.now()}\tepoch = {epoch}\tloss: {epoch_loss:.4f}\ttrain: {train_acc:.3f}\ttest: {test_acc:.3f}\tbest: {best_acc:.3f}\tlr: {scheduler.get_last_lr()[0]:.6f}")
    
    # 加载最佳模型状态
    if best_model_state is not None:
        for k, v in best_model_state.items():
            model.state_dict()[k].copy_(v.to(device))
    
    # 详细评估最佳模型
    model.eval()
    with torch.no_grad():
        final_acc, final_preds, final_targets = validate(model, test_loader)
        
        # 计算故障检测指标
        # 1. 计算类别准确率
        test_outputs = []
        all_test_labels = []
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_outputs.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
        
        # 转为numpy数组
        test_outputs = np.array(test_outputs)
        all_test_labels = np.array(all_test_labels)
        
        # 打印详细分类报告
        print("\n===== TEP故障检测详细评估 =====")
        print("\n分类报告:")
        print(classification_report(all_test_labels, test_outputs, target_names=[f"Fault {i}" for i in range(len(target))]))
        
        # 计算混淆矩阵
        cm = confusion_matrix(all_test_labels, test_outputs)
        
        # 计算每个故障类别的检出率
        print("\n各故障类别检出率:")
        detection_rates = []
        for i in range(len(cm)):
            tp = cm[i, i]  # 对角线元素为真正例
            total = np.sum(cm[i, :])  # 行和为该类别的总样本
            if total > 0:
                rate = tp / total
                detection_rates.append(rate)
                print(f"故障{i}: {rate:.4f}")
            else:
                detection_rates.append(0)
                print(f"故障{i}: 无样本")
        
        # 计算平均检出率
        avg_od = np.mean(detection_rates)
        print(f"\n平均检出率 (OD) = {avg_od:.4f}")
    
    print(f"\n训练完成! 最终测试精度: {final_acc:.4f}, 最佳测试精度: {best_acc:.4f}")
    
    return train_accs, test_accs, best_targets, best_pred, train_losses