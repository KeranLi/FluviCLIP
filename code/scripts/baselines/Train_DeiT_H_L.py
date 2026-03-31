import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.train_utils import train_model, validate_model
from utils.seed import set_seed
from utils.plot import plot_predictions, plot_loss_curves, plot_actual_vs_pred
from utils.numeric import calculate_mean_std
from utils.earlystop import EarlyStopping
#---------------------------Fixed designed modules--------------------------#
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#---------------------------Fixed modules--------------------------#
from models.DeiT import DeiTModel
from configs.DeiT import Config
#---------------------------Changeable modules-----------------------------#

# 固定随机种子
set_seed(42)

if __name__ == "__main__":
    config = Config()
    device = config.device
    print(f"Using device: {device}")
    
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载数据
    image_paths, labels = load_excel_data(
        os.path.join(config.data_path, config.excel_file),
        os.path.join(config.data_path, config.image_dir),
        config.sheet_name
    )
    means, stds = calculate_mean_std(image_paths)
    
    # 计算标签值的中位数以划分高和低标签值
    median_label = np.median(labels)
    high_label_indices = np.where(labels >= median_label)[0]
    low_label_indices = np.where(labels < median_label)[0]

    # 创建两个数据集
    high_dataset = Sentinel2Dataset(
        [image_paths[i] for i in high_label_indices],
        [labels[i] for i in high_label_indices],
        means=means,
        stds=stds
    )
    low_dataset = Sentinel2Dataset(
        [image_paths[i] for i in low_label_indices],
        [labels[i] for i in low_label_indices],
        means=means,
        stds=stds
    )
    
    # 初始化两个数据集的统计信息
    high_dataset.label_mean = np.mean(high_dataset.labels)
    high_dataset.label_std = np.std(high_dataset.labels)
    low_dataset.label_mean = np.mean(low_dataset.labels)
    low_dataset.label_std = np.std(low_dataset.labels)
    
    # 数据划分
    train_size_high = int(0.6 * len(high_dataset))
    val_size_high = int(0.2 * len(high_dataset))
    test_size_high = len(high_dataset) - train_size_high - val_size_high
    high_train_dataset, high_val_dataset, high_test_dataset = random_split(high_dataset, [train_size_high, val_size_high, test_size_high])

    train_size_low = int(0.6 * len(low_dataset))
    val_size_low = int(0.2 * len(low_dataset))
    test_size_low = len(low_dataset) - train_size_low - val_size_low
    low_train_dataset, low_val_dataset, low_test_dataset = random_split(low_dataset, [train_size_low, val_size_low, test_size_low])

    # 数据加载器
    high_train_loader = DataLoader(high_train_dataset, batch_size=config.batch_size, shuffle=True)
    high_val_loader = DataLoader(high_val_dataset, batch_size=config.batch_size, shuffle=False)
    high_test_loader = DataLoader(high_test_dataset, batch_size=config.batch_size, shuffle=False)
    
    low_train_loader = DataLoader(low_train_dataset, batch_size=config.batch_size, shuffle=True)
    low_val_loader = DataLoader(low_val_dataset, batch_size=config.batch_size, shuffle=False)
    low_test_loader = DataLoader(low_test_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型、优化器和早停机制
    high_model = DeiTModel().to(config.device)
    high_optimizer = Adam(
        high_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    high_criterion = torch.nn.MSELoss()
    high_early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        delta=config.early_stopping_delta,
        verbose=True,
        path=os.path.join(config.output_dir, "high_checkpoint.pth")
    )
    
    low_model = DeiTModel().to(config.device)
    low_optimizer = Adam(
        low_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    low_criterion = torch.nn.MSELoss()
    low_early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        delta=config.early_stopping_delta,
        verbose=True,
        path=os.path.join(config.output_dir, "low_checkpoint.pth")
    )

    # 初始化损失记录
    high_train_losses = []
    high_val_losses = []
    high_test_losses = []
    
    low_train_losses = []
    low_val_losses = []
    low_test_losses = []
    
    # 训练与验证循环
    writer = SummaryWriter(config.log_dir)
    for epoch in range(config.num_epochs):
        # 高标签值模型训练
        high_train_loss = train_model(
            high_model,
            high_train_loader,
            high_criterion,
            high_optimizer,
            device,
            save_path=os.path.join(config.output_dir, "high_train_losses.csv")
        )
        high_train_losses.append(high_train_loss)
        
        high_val_loss, high_val_actuals, high_val_predictions = validate_model(
            high_model,
            high_val_loader,
            high_criterion,
            device,
            high_dataset.label_mean,
            high_dataset.label_std,
            save_path=os.path.join(config.output_dir, "high_val_results.csv"),
            epoch=epoch + 1
        )
        high_val_losses.append(high_val_loss)
        
        high_test_loss, high_test_actuals, high_test_predictions = validate_model(
            high_model,
            high_test_loader,
            high_criterion,
            device,
            high_dataset.label_mean,
            high_dataset.label_std,
            save_path=os.path.join(config.output_dir, "high_test_results.csv"),
            epoch=epoch + 1
        )
        high_test_losses.append(high_test_loss)
        
        # 低标签值模型训练
        low_train_loss = train_model(
            low_model,
            low_train_loader,
            low_criterion,
            low_optimizer,
            device,
            save_path=os.path.join(config.output_dir, "low_train_losses.csv")
        )
        low_train_losses.append(low_train_loss)
        
        low_val_loss, low_val_actuals, low_val_predictions = validate_model(
            low_model,
            low_val_loader,
            low_criterion,
            device,
            low_dataset.label_mean,
            low_dataset.label_std,
            save_path=os.path.join(config.output_dir, "low_val_results.csv"),
            epoch=epoch + 1
        )
        low_val_losses.append(low_val_loss)
        
        low_test_loss, low_test_actuals, low_test_predictions = validate_model(
            low_model,
            low_test_loader,
            low_criterion,
            device,
            low_dataset.label_mean,
            low_dataset.label_std,
            save_path=os.path.join(config.output_dir, "low_test_results.csv"),
            epoch=epoch + 1
        )
        low_test_losses.append(low_test_loss)
        
        # 打印日志
        print(f"Epoch {epoch + 1}/{config.num_epochs}:"
              f"\nHigh Model - Train Loss={high_train_loss:.4f}, Validation Loss={high_val_loss:.4f}, Test Loss={high_test_loss:.4f}"
              f"\nLow Model - Train Loss={low_train_loss:.4f}, Validation Loss={low_val_loss:.4f}, Test Loss={low_test_loss:.4f}"
        )
        
        # 绘图
        # 高标签值模型
        high_val_plot_file_1 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_high_val_predictions_1.png")
        high_val_plot_file_2 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_high_val_predictions_2.png")
        plot_predictions(high_val_actuals, high_val_predictions, f"Epoch {epoch + 1}: High Value Model - Validation Actual vs Predicted", high_val_plot_file_1)
        plot_actual_vs_pred(high_val_actuals, high_val_predictions, f"Epoch {epoch + 1}: High Value Model - Validation Actual vs Predicted", high_val_plot_file_2)
        
        high_test_plot_file_1 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_high_test_predictions_1.png")
        high_test_plot_file_2 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_high_test_predictions_2.png")
        plot_predictions(high_test_actuals, high_test_predictions, f"Epoch {epoch + 1}: High Value Model - Test Actual vs Predicted", high_test_plot_file_1)
        plot_actual_vs_pred(high_test_actuals, high_test_predictions, f"Epoch {epoch + 1}: High Value Model - Test Actual vs Predicted", high_test_plot_file_2)
        
        # 低标签值模型
        low_val_plot_file_1 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_low_val_predictions_1.png")
        low_val_plot_file_2 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_low_val_predictions_2.png")
        plot_predictions(low_val_actuals, low_val_predictions, f"Epoch {epoch + 1}: Low Value Model - Validation Actual vs Predicted", low_val_plot_file_1)
        plot_actual_vs_pred(low_val_actuals, low_val_predictions, f"Epoch {epoch + 1}: Low Value Model - Validation Actual vs Predicted", low_val_plot_file_2)
        
        low_test_plot_file_1 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_low_test_predictions_1.png")
        low_test_plot_file_2 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_low_test_predictions_2.png")
        plot_predictions(low_test_actuals, low_test_predictions, f"Epoch {epoch + 1}: Low Value Model - Test Actual vs Predicted", low_test_plot_file_1)
        plot_actual_vs_pred(low_test_actuals, low_test_predictions, f"Epoch {epoch + 1}: Low Value Model - Test Actual vs Predicted", low_test_plot_file_2)
        
        # TensorBoard记录
        writer.add_scalars("High Model Loss", {"Train": high_train_loss, "Validation": high_val_loss, "Test": high_test_loss}, epoch)
        writer.add_scalars("Low Model Loss", {"Train": low_train_loss, "Validation": low_val_loss, "Test": low_test_loss}, epoch)
        
        # 调用早停机制
        # 高标签值模型
        high_early_stopping(high_val_loss, high_model)
        if high_early_stopping.early_stop:
            print("Early stopping triggered for High Value Model.")
            break
        
        # 低标签值模型
        low_early_stopping(low_val_loss, low_model)
        if low_early_stopping.early_stop:
            print("Early stopping triggered for Low Value Model.")
            break
        
        # 加载最佳模型
        high_model.load_state_dict(torch.load(os.path.join(config.output_dir, "high_checkpoint.pth")))
        low_model.load_state_dict(torch.load(os.path.join(config.output_dir, "low_checkpoint.pth")))
    
    # 绘制训练、验证和测试损失曲线
    #plot_loss_curves(high_train_losses, high_val_losses, high_test_losses, os.path.join(config.output_dir, "high_model_loss_curves.png"))
    #plot_loss_curves(low_train_losses, low_val_losses, low_test_losses, os.path.join(config.output_dir, "low_model_loss_curves.png"))

    # 绘制高模型损失曲线
    high_loss_curve_path = os.path.join(config.output_dir, "high_model_loss_curves.png")
    plot_loss_curves(high_train_losses, high_val_losses, high_test_losses, high_loss_curve_path)

    # 绘制低模型损失曲线
    low_loss_curve_path = os.path.join(config.output_dir, "low_model_loss_curves.png")
    plot_loss_curves(low_train_losses, low_val_losses, low_test_losses, low_loss_curve_path)
    
    writer.close()