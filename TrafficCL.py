import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import warnings
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')
# Haversine公式计算地理距离（公里）
def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的地理距离（公里）"""
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # 地球平均半径（公里）
    km = 6371.393 * c
    return km
# 自定义的collate_fn函数，确保数据类型正确
# 修改collate_fn函数
def collate_fn(batch):
    """确保数据类型转换，支持有地理距离和无地理距离两种情形"""
    # 检查批次中的第一个样本，看是否有4个元素（包含地理距离）
    if len(batch[0]) == 4:
        lefts, rights, labels, geo_dists = zip(*batch)
    else:
        lefts, rights, labels = zip(*batch)
        geo_dists = None

    # 转换为PyTorch张量并指定类型
    lefts = torch.tensor(np.array(lefts), dtype=torch.float32)
    rights = torch.tensor(np.array(rights), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.int64)

    if geo_dists is not None:
        geo_dists = torch.tensor(np.array(geo_dists), dtype=torch.float32)
        return lefts, rights, labels, geo_dists
    else:
        return lefts, rights, labels
class ContrastiveDataset(Dataset):
    """对比学习数据集类，包含归一化处理和地理距离计算"""

    def __init__(self, left_data, right_data, labels, geo_dists=None, scaler=None):
        """
        Args:
            left_data (np.ndarray): 左特征数据
            right_data (np.ndarray): 右特征数据
            labels (np.ndarray): 标签
            geo_dists (np.ndarray, optional): 地理距离
            scaler (StandardScaler, optional): 归一化器对象
        """
        # 数据校验
        assert len(left_data) == len(right_data) == len(labels), "输入数据长度不一致"

        # 如果提供了scaler，使用它来归一化数据
        if scaler is not None:
            self.left_data = scaler.transform(left_data).astype(np.float32)
            self.right_data = scaler.transform(right_data).astype(np.float32)
        else:
            self.left_data = left_data.astype(np.float32)
            self.right_data = right_data.astype(np.float32)

        # 确保标签是int64类型
        self.labels = labels.astype(np.int64)

        # 地理距离（训练时使用，测试时可省略）
        self.geo_dists = geo_dists
        if self.geo_dists is not None:
            self.geo_dists = self.geo_dists.astype(np.float32)
            self.has_geo = True
        else:
            self.has_geo = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.has_geo:
            return (
                self.left_data[idx].copy(),
                self.right_data[idx].copy(),
                self.labels[idx],
                self.geo_dists[idx]
            )
        else:
            return (
                self.left_data[idx].copy(),
                self.right_data[idx].copy(),
                self.labels[idx]
            )
def smote_with_geo_features(df, label):
    """
    改进的SMOTE实现，对整个数据集的0/1类别进行样本平衡
    参数:
        df: 包含左右特征和地理坐标的DataFrame
        label: 目标列名
    返回:
        增加合成样本后的DataFrame
    """
    # 确保finger_left和finger_right是列表类型
    df['finger_left'] = df['finger_left'].apply(ast.literal_eval)
    df['finger_right'] = df['finger_right'].apply(ast.literal_eval)

    # 全局样本统计（正样本1，负样本0）
    n_minority = sum(df[label] == 1)  # 正样本数量
    n_majority = sum(df[label] == 0)  # 负样本数量
    total_original = len(df)

    # 确定平衡目标：以数量较多的类别为基准，实现1:1平衡
    target_count = max(n_minority, n_majority)

    print(f"\n全局样本统计:")
    print(f"原始总样本数: {total_original}")
    print(f"正样本数(1): {n_minority}, 负样本数(0): {n_majority}")
    print(f"平衡目标: 各类别样本数均达到 {target_count}")

    # 计算需要生成的正/负样本数量
    need_minority = max(0, target_count - n_minority)
    need_majority = max(0, target_count - n_majority)

    print(f"\n需要生成的样本数:")
    print(f"正样本: {need_minority} 个, 负样本: {need_majority} 个")

    # 1. 生成正样本（使用自定义SMOTE逻辑）
    minority_new_rows = []
    if need_minority > 0:
        minority_new_rows = _generate_smote_samples(
            df, label, need_minority, 1
        )

    # 2. 生成负样本（使用相同的SMOTE逻辑：指纹+经纬度偏移）
    majority_new_rows = []
    if need_majority > 0:
        majority_new_rows = _generate_smote_samples(
            df, label, need_majority, 0
        )

    # 合并原有数据和新生成的样本
    all_new_rows = minority_new_rows + majority_new_rows
    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        df_res = pd.concat([df, new_df], ignore_index=True)
    else:
        df_res = df.copy()

    # 最终验证全局样本平衡情况
    final_n_minority = sum(df_res[label] == 1)
    final_n_majority = sum(df_res[label] == 0)
    final_total = len(df_res)

    print(f"\n最终样本统计:")
    print(f"总样本数: {final_total}")
    print(f"正样本数(1): {final_n_minority}, 负样本数(0): {final_n_majority}")
    print(f"类别比例(1:0): {final_n_minority / final_n_majority:.2f}:1")

    return df_res
def _generate_smote_samples(df, label, n_samples, class_type):
    """
    生成SMOTE样本的实现：通过随机偏移原始值（指纹特征+经纬度）
    参数:
        df: 原始数据框
        label: 目标列名
        n_samples: 要生成的样本数
        class_type: 要生成的类别（0或1）
    返回:
        生成的新样本列表
    """
    # 只处理指定类型的样本
    df_filtered = df[df[label] == class_type].copy()
    if len(df_filtered) == 0:
        print(f"警告：没有类型为 {class_type} 的样本")
        return []

    new_rows = []
    generated = 0

    # 经纬度偏移范围（度）：约10-15米的地理偏移
    lat_offset = 0.0001  # 纬度每0.0001度约10米
    lon_offset = 0.00015  # 经度每0.00015度约15米

    while generated < n_samples:
        # 随机选择一个基础样本
        base_idx = np.random.randint(0, len(df_filtered))
        base_row = df_filtered.iloc[base_idx].copy()

        # 创建新行（深拷贝避免修改原数据）
        new_row = base_row.copy()

        # 1. 处理指纹特征：对每个值添加0.1%-0.2%的随机偏移
        for feature in ['finger_left', 'finger_right']:
            original = np.array(base_row[feature])
            # 计算基于原始值绝对值的偏移量（0.1%-0.2%）
            offset = np.abs(original) * np.random.uniform(0.001, 0.002)
            # 随机决定偏移方向
            sign = np.random.choice([-1, 1])
            adjusted = original + sign * offset
            # 保留列表类型
            new_row[feature] = list(adjusted)

        # 2. 处理经纬度特征：左右位置偏移方向相反，模拟真实地理变化
        sign = np.random.choice([-1, 1])
        # 左位置经纬度偏移
        new_row['lat_left'] = base_row['lat_left'] + sign * lat_offset * np.random.rand()
        new_row['lng_left'] = base_row['lng_left'] + sign * lon_offset * np.random.rand()

        # 右位置经纬度偏移（与左位置方向相反）
        sign = np.random.choice([-1, 1])
        new_row['lat_right'] = base_row['lat_right'] - sign * lat_offset * np.random.rand()
        new_row['lng_right'] = base_row['lng_right'] - sign * lon_offset * np.random.rand()

        # 添加到新样本列表
        new_rows.append(new_row)
        generated += 1

    print(f"成功生成 {generated}/{n_samples} 个类型 {class_type} 的样本")
    return new_rows
class TransformerEncoder(nn.Module):
    """Transformer特征提取器"""

    def __init__(self, input_dim, hidden_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        x = self.embedding(x)  # (batch_size, 1, hidden_dim)
        x = x.transpose(0, 1)  # (1, batch_size, hidden_dim)
        x = self.transformer(x)  # (1, batch_size, hidden_dim)
        x = x.transpose(0, 1)  # (batch_size, 1, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, 1)
        x = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        return x
class EnhancedContrastiveModel(nn.Module):
    """增强的对比学习模型，结合分类和地理对齐"""

    def __init__(self, input_dim):
        super().__init__()
        # 共享权重的Transformer编码器
        self.encoder = TransformerEncoder(input_dim)

        # 投影头（用于对比学习）
        self.projection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # 分类头 (输出2维logits)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),  # 拼接左右特征
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 输出2个类别的logits
        )

        # 距离缩放参数（可学习）
        self.distance_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, left_input, right_input):
        # 确保输入是float32类型
        if left_input.dtype != torch.float32:
            left_input = left_input.float()
        if right_input.dtype != torch.float32:
            right_input = right_input.float()

        # 编码左右特征
        left_features = self.encoder(left_input)
        right_features = self.encoder(right_input)

        # 投影到嵌入空间
        left_emb = self.projection_head(left_features)
        right_emb = self.projection_head(right_features)

        # 计算嵌入距离（归一化余弦距离）
        norm_left = F.normalize(left_emb, p=2, dim=1)
        norm_right = F.normalize(right_emb, p=2, dim=1)
        emb_distance = (1 - (norm_left * norm_right).sum(dim=1)) * self.distance_scale

        # 拼接特征用于分类
        combined = torch.cat([left_features, right_features], dim=1)

        # 分类
        logits = self.classifier(combined)

        return logits, emb_distance  # 返回分类logits和嵌入距离
def prepare_data(df, scaler_path, label):
    """数据预处理：特征提取、划分、归一化、地理距离计算"""
    left_cols = "finger_left"
    right_cols = "finger_right"

    X_left = np.array(df[left_cols].tolist())
    X_right = np.array(df[right_cols].tolist())
    y = df[label].values

    # 计算地理距离
    geo_dists = []
    for _, row in df.iterrows():
        dist = haversine_distance(
            row['lat_left'], row['lng_left'],
            row['lat_right'], row['lng_right']
        )
        geo_dists.append(dist)
    geo_dists = np.array(geo_dists)

    # 归一化
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_left, X_right]))
    joblib.dump(scaler, scaler_path)

    return X_left, X_right, y, geo_dists, scaler
def prepare_test_data(df, label):
    """数据预处理：特征提取、划分、归一化（测试不需要地理距离）"""
    left_cols = "finger_left"
    right_cols = "finger_right"

    # 确保数据是列表形式
    df['finger_left'] = df['finger_left'].apply(ast.literal_eval)
    df['finger_right'] = df['finger_right'].apply(ast.literal_eval)

    X_left = np.array(df[left_cols].tolist())
    X_right = np.array(df[right_cols].tolist())
    y = df[label].values

    return X_left, X_right, y
def evaluate(model, test_loader, device):
    """评估模型在测试集上的整体性能（移除ISP分组逻辑）"""
    results = {}

    # 评估测试集的整体性能
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for left, right, labels in test_loader:
            left, right, labels = left.to(device), right.to(device), labels.to(device)
            logits, _ = model(left, right)  # 只使用分类结果
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算整体评估指标
    results['all'] = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'support': len(all_labels),
        'report': classification_report(all_labels, all_preds)
    }

    return results
def train_model(model, train_loader, val_loader, out_path, epochs=50, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    # 计算类别权重
    all_labels = []
    for batch in train_loader:
        if len(batch) == 4:  # 包含地理距离
            _, _, labels, _ = batch
        else:  # 不包含地理距离
            _, _, labels = batch
        all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)

    pos_count = all_labels.sum()
    neg_count = len(all_labels) - pos_count
    pos_weight = neg_count / (pos_count + 1e-5)

    # 确保类别权重是float32类型
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)

    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    alignment_criterion = nn.MSELoss()  # 地理对齐损失

    # 损失权重
    alignment_weight = 0.01  # 平衡分类和对齐损失

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, verbose=True
    )

    best_val_acc = 0.0
    best_model_weights = None
    best_model_info = None
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_align_loss = 0.0
        train_class_loss = 0.0

        # 使用tqdm显示进度条
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch in train_iter:
            # 检查批次结构并解包
            if len(batch) == 4:  # 包含地理距离
                left, right, labels, geo_dists = batch
                geo_dists = geo_dists.to(device)
            else:  # 不包含地理距离
                left, right, labels = batch
                geo_dists = None

            # 移动数据到设备并确保正确类型
            left = left.to(device)
            right = right.to(device)
            labels = labels.to(device).long()  # 确保标签是long类型

            optimizer.zero_grad()
            logits, emb_dist = model(left, right)  # 获取logits和嵌入距离

            # 计算分类损失
            class_loss = classification_criterion(logits, labels)

            # 计算对齐损失（如果提供地理距离）

            #normalized_geo = geo_dists/ geo_dists.max()
            #normalized_emb = emb_dist / emb_dist.max()

            align_loss = alignment_criterion(emb_dist, geo_dists)

            # 总损失
            total_loss = class_loss + alignment_weight * align_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 计算准确率
            with torch.no_grad():
                probs = F.softmax(logits, dim=1)
                _, predicted = torch.max(probs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                train_loss += total_loss.item() * labels.size(0)
                train_class_loss += class_loss.item() * labels.size(0)
                if isinstance(align_loss, torch.Tensor):
                    train_align_loss += align_loss.item() * labels.size(0)
                else:
                    train_align_loss += align_loss * labels.size(0)

            # 更新进度条
            train_iter.set_postfix(loss=total_loss.item(),
                                   class_loss=class_loss.item(),
                                   align_loss=align_loss if isinstance(align_loss, float) else align_loss.item(),
                                   acc=train_correct / train_total)

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels_val = []

        with torch.no_grad():
            val_iter = tqdm(val_loader, desc="Validating", leave=False)
            for batch in val_iter:
                # 检查批次结构并解包
                if len(batch) == 4:  # 包含地理距离
                    left, right, labels, _ = batch
                else:  # 不包含地理距离
                    left, right, labels = batch

                left = left.to(device)
                right = right.to(device)
                labels = labels.to(device).long()

                logits, _ = model(left, right)  # 验证时只使用分类结果
                loss = classification_criterion(logits, labels)

                # 计算预测结果
                probs = F.softmax(logits, dim=1)
                _, predicted = torch.max(probs, 1)

                val_loss += loss.item() * labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        # 计算平均损失和准确率
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_class_loss /= len(train_loader.dataset)
        train_align_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        # 更新学习率
        scheduler.step(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(
            f"  Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Align: {train_align_loss:.4f}), Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc
            }, out_path)
            print("  Saved best model")
    # 加载最佳模型并生成最终报告
    if best_model_info is None:  # 如果没有保存过模型，加载最后保存的
        best_model_info = torch.load(out_path)

    # 加载最佳模型状态
    model.load_state_dict(best_model_info['model_state_dict'])
    model.eval()
    print("\nFinal Validation Report:")
    print(classification_report(all_labels_val, all_preds))
    print(f"Accuracy: {accuracy_score(all_labels_val, all_preds):.4f}")

    return model
def main():
    train_path = "feature_data/cross_district_train.csv"
    test_path = "feature_data/cross_district_test.csv"
    out_model = "model/cross_district_model.pth"
    scaler_path = "model/cross_district_scaler.pkl"
    label = "cross_district"
    df_train = pd.read_csv(train_path, low_memory=False)
    df_train = smote_with_geo_features(
        df_train,
        label=label
    )

    df_test = pd.read_csv(test_path, low_memory=False)
    print(f"训练数据形状: {df_train.shape}, 测试数据形状: {df_test.shape}")

    # 准备训练数据（包括地理距离）
    print("准备训练数据集...")
    X_left_train, X_right_train, y_train, geo_dists_train, scaler = prepare_data(df_train, scaler_path, label)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    idx = np.arange(len(y_train))
    X_left_train_full, X_left_val, X_right_train_full, X_right_val, y_train_full, y_val, idx_train, idx_val = train_test_split(
        X_left_train, X_right_train, y_train, idx,
        test_size=0.2, random_state=42, stratify=y_train
    )

    # 提取对应的地理距离
    geo_dists_val = geo_dists_train[idx_val]
    geo_dists_train = geo_dists_train[idx_train]

    print(f"训练特征形状: 左={X_left_train_full.shape}, 右={X_right_train_full.shape}, 标签={y_train_full.shape}")
    print(f"验证特征形状: 左={X_left_val.shape}, 右={X_right_val.shape}, 标签={y_val.shape}")

    # 创建数据集（包含地理距离）
    train_dataset = ContrastiveDataset(X_left_train_full, X_right_train_full, y_train_full, geo_dists_train, scaler)
    val_dataset = ContrastiveDataset(X_left_val, X_right_val, y_val, geo_dists_val, scaler)

    # 创建数据加载器
    batch_size = 1024
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # 准备测试数据（不包含地理距离）
    print("\n准备测试数据集...")
    X_left_test, X_right_test, y_test= prepare_test_data(df_test, label)
    print(f"测试数据: 左特征形状: {X_left_test.shape}, 右特征形状: {X_right_test.shape}, 标签形状: {y_test.shape}")

    # 创建测试数据集 (使用训练集的scaler进行归一化)
    test_dataset = ContrastiveDataset(X_left_test, X_right_test, y_test, scaler=scaler)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 批大小为1用于逐样本评估
        collate_fn=collate_fn
    )


    # 创建增强模型
    input_dim = X_left_train.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = EnhancedContrastiveModel(input_dim)

    # 训练模型
    print("\n开始训练模型...")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        out_model,
        epochs=1,
        lr=0.001
    )

    # 按ISP类型评估（测试逻辑不变）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    isp_results = evaluate(trained_model, test_loader, device)

    # 打印评估结果
    print("\n===== 按ISP类型评估模型性能 =====")
    for isp, metrics in isp_results.items():
        if isp == 'all':
            print("\n整体性能:")
        else:
            print(f"\n{isp} ISP 性能 (样本数: {metrics.get('support', len(test_dataset))}):")

        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1值: {metrics['f1']:.4f}")
        print("\n详细分类报告:")
        print(metrics['report'])


def test():
    """重新训练流程 - 从已有模型参数继续训练"""
    print("====== 模型继续训练流程 ======")

    # 加载数据
    print("正在加载数据...")
    test_path = "feature_data/cross_district_test.csv"
    checkpoint_path = "model/cross_district_model.pth"  # 已有模型文件
    scaler_path = "model/cross_district_scaler.pkl"
    label = "cross_district"
    # 加载数据（与原始训练相同）
    scaler = joblib.load(scaler_path)
    df_test = pd.read_csv(test_path, low_memory=False)

    # 准备测试数据（不包含地理距离）
    print("\n准备测试数据集...")
    X_left_test, X_right_test, y_test = prepare_test_data(df_test, label)
    print(f"测试数据: 左特征形状: {X_left_test.shape}, 右特征形状: {X_right_test.shape}, 标签形状: {y_test.shape}")

    # 创建测试数据集 (使用训练集的scaler进行归一化)
    test_dataset = ContrastiveDataset(X_left_test, X_right_test, y_test, scaler=scaler)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 批大小为1用于逐样本评估
        collate_fn=collate_fn
    )
    input_dim = X_left_test.shape[1]
    print(f"\n输入特征维度: {input_dim}")
    model = EnhancedContrastiveModel(input_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    # 加载已有的模型参数
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检查检查点类型并正确加载模型参数
    if 'model_state_dict' in checkpoint:  # 如果是完整检查点
        print(
            f"加载完整检查点 (epoch {checkpoint.get('epoch', 'unknown')}, acc={checkpoint.get('best_val_acc', 'unknown'):.4f})")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # 如果是纯模型参数
        print("加载纯模型参数")
        model.load_state_dict(checkpoint)

    print(f"已加载模型参数: {checkpoint_path}")
    # 检查ISP分布

    # 按ISP类型评估（测试逻辑不变）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    isp_results = evaluate(model, test_loader, device)

    # 打印评估结果
    print("\n===== 重新训练后按ISP类型评估模型性能 =====")
    for isp, metrics in isp_results.items():
        if isp == 'all':
            print("\n整体性能:")
        else:
            print(f"\n{isp} ISP 性能 (样本数: {metrics.get('support', len(test_dataset))}):")

        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1值: {metrics['f1']:.4f}")
        print("\n详细分类报告:")
        print(metrics['report'])



if __name__ == "__main__":
    main()
    test()