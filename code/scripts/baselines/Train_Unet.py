import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import os
import torch
from utils.data_utils import load_excel_data, calculate_mean_std, Sentinel2Dataset
from utils.train_utils import train_model, validate_model, plot_predictions
from utils.seed import set_seed
#---------------------------Fixed desined modules----------------------------------#
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
#---------------------------Fixed pytorch modules-----------------------------------#
from models.Unet_FC import UNetWithFC
from configs.Unet import Config
#---------------------------Changeable modules--------------------------------------#

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
    dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)

    # 数据划分
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 初始化模型
    model = UNetWithFC(num_features=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # 训练与验证
    writer = SummaryWriter(config.log_dir)
    for epoch in range(config.num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, "cuda")
        val_loss = validate_model(model, val_loader, criterion, "cuda")
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        print(f"Epoch {epoch + 1}/{config.num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

    writer.close()
