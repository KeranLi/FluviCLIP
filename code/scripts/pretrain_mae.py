import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.seed import set_seed
from utils.plot import plot_loss_curves
from utils.numeric import calculate_mean_std
from models.mae import MaskedAutoencoder
from configs.FluviCLIP import Config


set_seed(42)


def pretrain_mae_epoch(model, train_loader, optimizer, device):
    """
    Pre-train Masked Autoencoder for one epoch.
    """
    model.train()
    total_loss = 0.0
    
    for images, _ in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        loss, pred, mask = model(images)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate_mae_epoch(model, val_loader, device):
    """
    Validate MAE reconstruction loss.
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            loss, pred, mask = model(images)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss


if __name__ == "__main__":
    config = Config()
    device = config.device
    print(f"Using device: {device}")
    
    output_dir = "output/MAE_Pretrain"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load multi-source pre-training data
    image_paths, labels = load_excel_data(
        os.path.join(config.data_path, config.excel_file),
        os.path.join(config.data_path, config.image_dir),
        config.sheet_name
    )
    means, stds = calculate_mean_std(image_paths)
    
    dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)
    
    # Split: 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.pretrain_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.pretrain_batch_size, shuffle=False, num_workers=0)
    
    # Initialize MAE model
    model = MaskedAutoencoder(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_size=config.window_size,
        mask_ratio=config.pretrain_mask_ratio,
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=config.pretrain_weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.pretrain_epochs, eta_min=1e-6)
    
    writer = SummaryWriter(os.path.join(config.log_dir, "MAE_Pretrain"))
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(config.pretrain_epochs):
        train_loss = pretrain_mae_epoch(model, train_loader, optimizer, device)
        val_loss = validate_mae_epoch(model, val_loader, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{config.pretrain_epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
        
        writer.add_scalars("MAE_Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.encoder.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  -> Saved best model with val loss {val_loss:.6f}")
    
    # Plot curves
    plot_loss_curves(train_losses, val_losses, val_losses, os.path.join(output_dir, "pretrain_loss_curves.png"))
    writer.close()
    print("MAE pre-training completed.")
