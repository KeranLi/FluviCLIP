"""
Training script for ResNet50 baseline.
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.train_utils import train_model, validate_model
from utils.seed import set_seed
from utils.plot import plot_predictions, plot_loss_curves, plot_actual_vs_pred
from utils.numeric import calculate_mean_std
from utils.earlystop import EarlyStopping
from models.resnet import ResNet50Regressor
from configs.ResNet50 import Config

set_seed(42)


if __name__ == "__main__":
    config = Config()
    device = config.device
    print(f"Using device: {device}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load data
    image_paths, labels = load_excel_data(
        os.path.join(config.data_path, config.excel_file),
        os.path.join(config.data_path, config.image_dir),
        config.sheet_name
    )
    means, stds = calculate_mean_std(image_paths)
    
    dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)
    dataset.label_mean = np.mean(labels)
    dataset.label_std = np.std(labels)
    
    # Split: 60-20-20
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = ResNet50Regressor(in_channels=26, num_classes=1, pretrained=False).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        delta=config.early_stopping_delta,
        verbose=True,
        path=os.path.join(config.output_dir, config.checkpoint_path)
    )
    
    writer = SummaryWriter(config.log_dir)
    
    train_losses = []
    val_losses = []
    test_losses = []
    
    for epoch in range(config.num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        val_loss, val_actuals, val_predictions = validate_model(
            model, val_loader, criterion, device, dataset.label_mean, dataset.label_std,
            save_path=os.path.join(config.output_dir, "validation_results.csv"),
            epoch=epoch + 1
        )
        val_losses.append(val_loss)
        
        test_loss, test_actuals, test_predictions = validate_model(
            model, test_loader, criterion, device, dataset.label_mean, dataset.label_std,
            save_path=os.path.join(config.output_dir, "test_results.csv"),
            epoch=epoch + 1
        )
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch + 1}/{config.num_epochs}: Train Loss={train_loss:.4f}, "
              f"Validation Loss={val_loss:.4f}, Test Loss={test_loss:.4f}")
        
        # Plotting
        val_plot_file_1 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_val_predictions_1.png")
        val_plot_file_2 = os.path.join(config.output_dir, f"epoch_{epoch + 1}_val_predictions_2.png")
        plot_predictions(val_actuals, val_predictions, f"Epoch {epoch + 1}: Validation", val_plot_file_1)
        plot_actual_vs_pred(val_actuals, val_predictions, f"Epoch {epoch + 1}: Validation", val_plot_file_2)
        
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss, "Test": test_loss}, epoch)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        model.load_state_dict(torch.load(os.path.join(config.output_dir, config.checkpoint_path)))
    
    plot_loss_curves(train_losses, val_losses, test_losses, os.path.join(config.output_dir, "loss_curves.png"))
    writer.close()
