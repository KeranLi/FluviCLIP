import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from utils.numeric import inverse_normalize

def train_model(model, train_loader, criterion, optimizer, device, save_path="train_losses.csv"):
    """
    训练模型并保存每个 batch 的损失到 CSV 文件。

    Args:
        model (torch.nn.Module): 模型。
        train_loader (DataLoader): 训练集数据加载器。
        criterion (torch.nn.Module): 损失函数。
        optimizer (torch.optim.Optimizer): 优化器。
        device (str): 设备（如 'cuda' 或 'cpu'）。
        save_path (str): 保存每个 batch 损失的文件路径。

    Returns:
        float: 训练集平均损失。
    """
    model.train()
    total_loss = 0
    batch_losses = []

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # 数据迁移到设备
        inputs, labels = inputs.to(device), labels.to(device)

        # 确保标签形状与模型输出一致
        labels = labels.view(-1, 1)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累积损失
        total_loss += loss.item()
        batch_losses.append({"Batch": batch_idx + 1, "Loss": loss.item()})

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)

    # 保存 batch 损失到 CSV
    pd.DataFrame(batch_losses).to_csv(save_path, index=False)

    return avg_loss

def validate_model(model, val_loader, criterion, device, label_mean, label_std, save_path="validation_results.csv", epoch=None):
    """
    验证模型性能并保存验证结果到 CSV 文件。

    Args:
        model (torch.nn.Module): 模型。
        val_loader (DataLoader): 验证集数据加载器。
        criterion (torch.nn.Module): 损失函数。
        device (str): 设备（如 'cuda' 或 'cpu'）。
        label_mean (float): 标签的均值（用于反归一化）。
        label_std (float): 标签的标准差（用于反归一化）。
        save_path (str): 保存验证结果的文件路径。
        epoch (int): 当前 epoch，用于记录结果。

    Returns:
        float: 验证集平均损失。
        list: 验证集的实际值。
        list: 验证集的预测值。
    """
    model.eval()
    total_loss = 0
    actuals = []
    predictions = []
    results = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 确保标签形状与模型输出一致
            labels = labels.view(-1, 1)

            # 模型预测
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 收集实际值和预测值
            prediction = inverse_normalize(outputs.cpu().numpy().flatten(), label_mean, label_std)
            actual = inverse_normalize(labels.cpu().numpy().flatten(), label_mean, label_std)
            actuals.extend(actual)
            predictions.extend(prediction)

            # 保存到结果列表
            for act, pred in zip(actual, prediction):
                results.append({"Epoch": epoch, "Actual": act, "Predicted": pred})

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)

    # 将结果保存到 CSV 文件
    pd.DataFrame(results).to_csv(save_path, index=False, mode="a", header=not os.path.exists(save_path))

    return avg_loss, actuals, predictions