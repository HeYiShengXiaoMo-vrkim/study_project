import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常

# 1. 加载和清洗数据
def load_and_clean_data(file_path):
    # 加载数据
    df = pd.read_csv(
        file_path,
        dtype={'subCategoryA': 'str'},  # 明确指定可能混合类型的列为字符串
        low_memory=False,  # 禁用分块模式读取
        parse_dates=['starttime', 'lastupdated']  # 直接在读取时解析日期
    )

    # 将时间列转换为datetime格式
    df['starttime'] = pd.to_datetime(df['starttime'], format='%d-%m-%Y %H:%M')
    df['lastupdated'] = pd.to_datetime(df['lastupdated'], format='%d-%m-%Y %H:%M')

    # 提取时间特征
    df['hour_of_day'] = df['starttime'].dt.hour
    df['day_of_week'] = df['starttime'].dt.dayofweek
    df['month'] = df['starttime'].dt.month

    # 处理缺失值
    df = df.fillna({
        'trafficvolume': 'Unknown',
        'lanes': 0,
        'lanesaffected': 0
    })

    # 将车道数转换为数值型
    df['lanes'] = pd.to_numeric(df['lanes'], errors='coerce').fillna(0).astype(int)
    df['lanesaffected'] = pd.to_numeric(df['lanesaffected'], errors='coerce').fillna(0).astype(int)

    # 计算车道关闭百分比（严重性指标之一）
    df['lane_closure_pct'] = df.apply(
        lambda x: (x['lanesaffected'] / x['lanes']) if x['lanes'] > 0 else 0,
        axis=1
    )

    # 提取是否涉及多辆车或重型车辆
    df['multiple_vehicles'] = df['subCategoryA'].str.contains('cars|vehicles', case=False, na=False)
    df['heavy_vehicle'] = df['subCategoryA'].str.contains('truck|bus|lorry', case=False, na=False)

    # 定义严重性指标
    # 这里我们创建一个基于多个特征的复合指标
    df['severity'] = 0  # 初始化为低严重性

    # 中等严重性：较长持续时间或多个应急服务或多辆车辆
    medium_severity_mask = (
            (df['duration'] > 20) |
            (df['attendinggroups'].astype(str).str.len() > 20) |
            (df['multiple_vehicles']) |
            (df['heavy_vehicle']) |
            (df['lane_closure_pct'] >= 0.3)
    )
    df.loc[medium_severity_mask, 'severity'] = 1

    # 高严重性：长持续时间或涉及重型车辆和多车道关闭
    high_severity_mask = (
            (df['duration'] > 40) |
            ((df['heavy_vehicle']) & (df['lane_closure_pct'] > 0.5)) |
            (df['isMajor'] == 1)
    )
    df.loc[high_severity_mask, 'severity'] = 2

    return df

# 2. 特征工程
def feature_engineering(df):
    # 新增数据验证
    required_columns = ['maincategory', 'direction', 'closuretype', 'suburb', 'trafficvolume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if (missing_cols):
        raise ValueError(f"关键列缺失: {missing_cols}")

    # 选择用于预测的特征
    features = [
        'maincategory', 'hour_of_day', 'day_of_week', 'month',
        'lane_closure_pct', 'multiple_vehicles', 'heavy_vehicle',
        'direction', 'closuretype', 'suburb', 'trafficvolume'
    ]

    # 如果数据中有这些列，添加到特征中
    additional_features = ['adviceA', 'adviceB', 'otherAdvice']
    for feature in additional_features:
        if (feature in df.columns):
            features.append(feature)

    # 添加时间差特征
    df['incident_duration'] = (df['lastupdated'] - df['starttime']).dt.total_seconds() / 60
    
    # 高峰时段标记 (早7-9点，晚5-7点)
    df['is_rush_hour'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 9)) | \
                         ((df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 19))
    
    # 周末标记
    df['is_weekend'] = df['day_of_week'] >= 5  # 5和6分别对应星期六和星期日
    
    # 添加这些新特征到选择的特征列表中
    features.extend(['incident_duration', 'is_rush_hour', 'is_weekend'])

    # 丢弃有大量缺失值的行
    df_model = df[features + ['severity']].dropna(thresh=len(features) - 2)
    # 新增空数据检查
    if (df_model.empty):
        raise ValueError("特征工程后数据为空，请检查输入数据或缺失值处理逻辑")
    return df_model, features

# 3. 创建序列数据
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(target[i + sequence_length])
    return np.array(X), np.array(y)

# 4. 构建CNN-LSTM模型
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(CNNLSTMModel, self).__init__()
        # CNN部分
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 计算卷积后序列长度
        self.seq_len_after_pool = sequence_length // 2
        
        # LSTM部分
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, 
                            bidirectional=True, dropout=0.2)
        
        # 全连接层
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64*2, 32)  # bidirectional LSTM有2倍的输出维度
        self.fc2 = nn.Linear(32, 3)     # 3个严重程度类别

    def forward(self, x):
        # 输入x的预期形状: [batch_size, sequence_length, features]
        # 需要调整为CNN的输入格式: [batch_size, features, sequence_length]
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        # CNN层
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 准备LSTM输入: [batch_size, seq_len, features]
        x = x.permute(0, 2, 1)
        
        # LSTM层
        x, _ = self.lstm(x)
        
        # 只使用最后一个时间步的输出
        x = x[:, -1, :]
        x = self.dropout(x)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# 5. 训练和评估模型
def train_and_evaluate(df_model, features, sequence_length, device):
    # 分离特征和目标变量
    X = df_model[features]
    y = df_model['severity']

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # One-hot encode categorical features and standardize numerical features
    categorical_features = ['maincategory', 'direction', 'closuretype', 'suburb', 'trafficvolume']
    numeric_features = ['hour_of_day', 'day_of_week', 'month', 'lane_closure_pct', 'multiple_vehicles', 'heavy_vehicle']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ])

    X_preprocessed = preprocessor.fit_transform(X).toarray().astype(np.float32)  # Convert to dense array

    # 在创建序列前先检查类别不平衡
    print(f"原始类别分布: {Counter(y)}")
    
    # 对预处理后的数据应用SMOTE过采样
    if len(np.unique(y)) > 1:  # 确保有多于一个类别
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)
        print(f"SMOTE采样后类别分布: {Counter(y_resampled)}")
        
        # 使用重采样后的数据创建序列
        X_seq, y_seq = create_sequences(X_resampled, y_resampled, sequence_length)
    else:
        # 使用原始数据创建序列
        X_seq, y_seq = create_sequences(X_preprocessed, y, sequence_length)

    # 创建序列数据
    X_seq, y_seq = create_sequences(X_preprocessed, y, sequence_length)

    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)

    # 转换为张量
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 构建模型
    model = CNNLSTMModel(X_train_tensor.shape[1:]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 早停设置
    best_loss = float('inf')
    patience = 5
    counter = 0

    # 训练模型
    num_epochs = 50
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # 评估模型
        model.eval()
        test_loss = 0.0
        y_pred_list = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                _, y_pred = torch.max(outputs, 1)
                y_pred_list.extend(y_pred.cpu().numpy())
        test_losses.append(test_loss / len(test_loader))

        # 计算召回率和准确率
        y_true = y_test_tensor.cpu().numpy()
        y_pred = np.array(y_pred_list)
        recall = recall_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        # 早停检查
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1

        # 更新学习率
        scheduler.step(test_loss)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        
        # 检查是否早停
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # 恢复最佳模型
        model.load_state_dict(best_model_state)
    # 保存最佳模型
    torch.save({
        'model_state_dict': best_model_state,
        'preprocessor': preprocessor
    }, 'cnn_lstm_model.pth')
    
    # 绘制训练和测试损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和测试损失')
    plt.savefig('loss_curve.png')
    plt.show()

    # 绘制混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    plt.show()

    return model, preprocessor

# 调用函数进行交叉验证
from sklearn.model_selection import KFold

def cross_validate_model(df_model, features, sequence_length, device, n_splits=5):
    # 准备数据
    X = df_model[features]
    y = df_model['severity']
    
    # 设置K折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n训练第 {fold+1}/{n_splits} 折...")
        
        # 获取当前折的训练和验证数据
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # 预处理和模型训练代码...
        
        # 保存每折的结果
        fold_results.append({
            'fold': fold+1,
            'recall': recall,
            'precision': precision,
            'f1_score': f1
        })
    
    # 打印平均结果
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_f1 = np.mean([r['f1_score'] for r in fold_results])
    
    print(f"\n交叉验证平均结果: Recall={avg_recall:.4f}, Precision={avg_precision:.4f}, F1={avg_f1:.4f}")
    
    return fold_results
def predict_severity(model, preprocessor, new_data_path, sequence_length, device):
    """完整的预测流水线"""
    # 加载并预处理新数据
    new_df = load_and_clean_data(new_data_path)
    processed_data, _ = feature_engineering(new_df)

    # 确保特征完全一致
    required_features = list(preprocessor.transformers_[0][1].get_feature_names_out()) + preprocessor.transformers_[1][2]

    # 特征对齐
    missing_features = set(required_features) - set(processed_data.columns)
    for f in missing_features:
        processed_data[f] = 0  # 用0填充模型需要但新数据缺失的特征

    # 将数据标准化和编码
    X_preprocessed = preprocessor.transform(processed_data[required_features]).toarray().astype(np.float32)  # Convert to dense array

    # 创建序列数据
    X_seq, _ = create_sequences(X_preprocessed, processed_data['severity'], sequence_length)

    # 转换为张量
    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(X_seq_tensor)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().numpy(), processed_data

# 7. 主函数
def main(file_path):
    print("开始加载和清洗数据...")
    df = load_and_clean_data(file_path)

    print("数据加载完成，开始特征工程...")
    df_model, features = feature_engineering(df)

    print("特征工程完成，开始构建模型...")
    sequence_length = 10  # 根据需要设置序列长度
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocessor = train_and_evaluate(df_model, features, sequence_length, device)

    print("模型构建完成！")

    # 可视化数据分布
    plt.figure(figsize=(15, 10))

    # 事故类型分布
    plt.subplot(2, 3, 1)
    df['maincategory'].value_counts().plot(kind='bar')
    plt.title('事故类型分布')

    # 一天中的小时分布
    plt.subplot(2, 3, 2)
    df['hour_of_day'].hist(bins=24)
    plt.title('事故发生时间（小时）分布')

    # 周几分布
    plt.subplot(2, 3, 3)
    df['day_of_week'].hist(bins=7)
    plt.title('事故发生时间（星期）分布')

    # 严重性分布
    plt.subplot(2, 3, 4)
    df['severity'].value_counts().plot(kind='bar')
    plt.title('事故严重性分布')

    # 持续时间分布
    plt.subplot(2, 3, 5)
    df[df['duration'] < 100]['duration'].hist(bins=20)
    plt.title('事故持续时间分布 (<100分钟)')

    plt.tight_layout()
    plt.savefig('data_distribution.png')
    plt.show()

    return df, model, preprocessor

if __name__ == "__main__":
    # 训练流程
    train_path = "E:\\python代码\\开放性课题项目\\train.csv"
    df, model, preprocessor = main(train_path)

    # 新数据预测
    new_data_path = "E:\\python代码\\开放性课题项目\\predict.csv"
    sequence_length = 10  # 与训练时的序列长度保持一致
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions, processed_new_data = predict_severity(model, preprocessor, new_data_path, sequence_length, device)

    # 添加预测结果到数据框
    processed_new_data['预测严重程度'] = predictions
    print("预测结果样例：\n", processed_new_data[['maincategory', '预测严重程度']].head())