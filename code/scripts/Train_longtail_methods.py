"""
Training script for various long-tail handling methods.
Implements:
- Focal Loss + Focal MSE
- Inverse Frequency + Weighted MSE
- GHMC + Gradient Harmonized
- LDAM (adapted) + Margin-adjusted
- L1 Loss + MAE-based
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.seed import set_seed
from utils.numeric import calculate_mean_std, inverse_normalize
from models.fluviformer import FluviFormer

set_seed(42)


class FocalMSELoss(nn.Module):
    """Focal MSE Loss for addressing class imbalance."""
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # Focus on hard examples
        focal_weight = torch.abs(pred - target) ** self.gamma
        loss = focal_weight * mse
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class InverseFrequencyWeightedMSELoss(nn.Module):
    """Inverse frequency weighted MSE."""
    def __init__(self, labels, n_bins=10):
        super().__init__()
        # Compute inverse frequency weights based on label distribution
        bins = np.linspace(min(labels), max(labels), n_bins + 1)
        bin_counts = np.histogram(labels, bins=bins)[0]
        bin_weights = 1.0 / (bin_counts + 1e-6)
        bin_weights = bin_weights / bin_weights.sum() * n_bins
        
        self.bins = torch.tensor(bins, dtype=torch.float32)
        self.bin_weights = torch.tensor(bin_weights, dtype=torch.float32)

    def forward(self, pred, target):
        # Assign each target to a bin
        target_np = target.cpu().numpy()
        bin_indices = np.digitize(target_np.flatten(), self.bins.numpy()) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_weights) - 1)
        
        weights = self.bin_weights[bin_indices].to(target.device)
        mse = (pred - target) ** 2
        loss = weights.view(-1, 1) * mse
        return loss.mean()


class GHMCLoss(nn.Module):
    """
    Gradient Harmonized Mechanism for continuous values.
    Adapted from GHMC (Gradient Harmonized Mechanism) for classification.
    """
    def __init__(self, bins=10, alpha=0.75):
        super().__init__()
        self.bins = bins
        self.alpha = alpha
        self.register_buffer('gradient_hist', torch.zeros(bins))
        self.register_buffer('grad_density', torch.ones(bins))

    def forward(self, pred, target):
        # Compute gradient (error)
        gradient = torch.abs(pred - target).detach()
        
        # Assign to bins
        bin_indices = (gradient / gradient.max() * (self.bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, 0, self.bins - 1)
        
        # Compute weights (gradient density)
        weights = torch.ones_like(pred)
        for i in range(len(pred)):
            bin_idx = bin_indices[i].item()
            weights[i] = 1.0 / (self.grad_density[bin_idx] + 1e-6)
        
        # Update histogram
        for i in range(self.bins):
            self.gradient_hist[i] += (bin_indices == i).float().sum()
        self.grad_density = self.gradient_hist / self.gradient_hist.sum()
        
        mse = (pred - target) ** 2
        loss = weights * mse
        return loss.mean()


class LDAMRegLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss adapted for regression.
    """
    def __init__(self, labels, max_margin=1.0, n_bins=10):
        super().__init__()
        # Compute margins based on frequency
        bins = np.linspace(min(labels), max(labels), n_bins + 1)
        bin_counts = np.histogram(labels, bins=bins)[0]
        # More frequent bins get larger margins (harder to predict)
        margins = max_margin * bin_counts / bin_counts.sum()
        
        self.bins = torch.tensor(bins, dtype=torch.float32)
        self.margins = torch.tensor(margins, dtype=torch.float32)

    def forward(self, pred, target):
        # Assign to bins
        target_np = target.cpu().numpy()
        bin_indices = np.digitize(target_np.flatten(), self.bins.numpy()) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.margins) - 1)
        
        margins = self.margins[bin_indices].to(target.device).view(-1, 1)
        
        # Apply margin
        adjusted_target = target + margins * torch.sign(pred - target)
        loss = F.mse_loss(pred, adjusted_target)
        return loss


class L1MAELoss(nn.Module):
    """L1 Loss (MAE) based loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.l1_loss(pred, target)


def train_with_longtail_loss(loss_type, train_loader, val_loader, test_loader,
                             label_mean, label_std, head_threshold, device, num_epochs=50):
    """Train FluviFormer with specified long-tail loss."""
    print(f"\n{'='*60}")
    print(f"Training with {loss_type}")
    print(f"{'='*60}")
    
    # Get labels for computing statistics
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    # Initialize model
    model = FluviFormer(
        img_size=224, patch_size=4, in_chans=26, embed_dim=96,
        depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
        num_classes=1
    ).to(device)
    
    # Select loss function
    if loss_type == 'Focal Loss + Focal MSE':
        criterion = FocalMSELoss(gamma=2.0)
    elif loss_type == 'Inverse Frequency + Weighted MSE':
        criterion = InverseFrequencyWeightedMSELoss(all_labels)
    elif loss_type == 'GHMC + Gradient Harmonized':
        criterion = GHMCLoss(bins=10)
    elif loss_type == 'LDAM (adapted) + Margin-adjusted':
        criterion = LDAMRegLoss(all_labels, max_margin=1.0)
    elif loss_type == 'L1 Loss + MAE-based':
        criterion = L1MAELoss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).view(-1, 1)
                outputs = model(images)
                loss = F.mse_loss(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # Test evaluation
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            pred_denorm = inverse_normalize(outputs.cpu().numpy().flatten(), label_mean, label_std)
            actual_denorm = inverse_normalize(labels.numpy().flatten(), label_mean, label_std)
            
            predictions.extend(pred_denorm)
            targets.extend(actual_denorm)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Compute metrics
    def compute_metrics(pred, tgt, mask):
        p = pred[mask]
        t = tgt[mask]
        if len(t) == 0:
            return {'MAE': 0, 'RMSE': 0, 'MSE': 0, 'R2': 0}
        mae = np.mean(np.abs(p - t))
        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((p - t) ** 2) / (np.sum((t - np.mean(t)) ** 2) + 1e-8)
        return {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'R2': r2 * 100}
    
    head_mask = targets <= head_threshold
    tail_mask = targets > head_threshold
    
    overall = compute_metrics(predictions, targets, np.ones_like(targets, dtype=bool))
    head = compute_metrics(predictions, targets, head_mask)
    tail = compute_metrics(predictions, targets, tail_mask)
    
    print(f"\nResults for {loss_type}:")
    print(f"  Overall: R2={overall['R2']:.2f}%, MAE={overall['MAE']:.2f}")
    print(f"  Head:    R2={head['R2']:.2f}%, MAE={head['MAE']:.2f}")
    print(f"  Tail:    R2={tail['R2']:.2f}%, MAE={tail['MAE']:.2f}")
    
    return {
        'loss_type': loss_type,
        'overall': overall,
        'head': head,
        'tail': tail
    }


def main():
    from torch.utils.data import random_split
    import pandas as pd
    
    config = type('Config', (), {
        'data_path': 'datasets/train',
        'excel_file': 'data.xlsx',
        'image_dir': 'images/',
        'sheet_name': 'Sheet1',
        'batch_size': 16,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'num_epochs': 50
    })()
    
    # Load data
    image_paths, labels = load_excel_data(
        os.path.join(config.data_path, config.excel_file),
        os.path.join(config.data_path, config.image_dir),
        config.sheet_name
    )
    means, stds = calculate_mean_std(image_paths)
    head_threshold = np.percentile(labels, 75)
    
    dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)
    label_mean, label_std = dataset.label_mean, dataset.label_std
    
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Test all long-tail methods
    methods = [
        'Focal Loss + Focal MSE',
        'Inverse Frequency + Weighted MSE',
        'GHMC + Gradient Harmonized',
        'LDAM (adapted) + Margin-adjusted',
        'L1 Loss + MAE-based',
    ]
    
    results = []
    for method in methods:
        result = train_with_longtail_loss(
            method, train_loader, val_loader, test_loader,
            label_mean, label_std, head_threshold, config.device, config.num_epochs
        )
        results.append(result)
    
    # Save results
    df = pd.DataFrame([
        {
            'Method': r['loss_type'],
            'Head_MAE': r['head']['MAE'],
            'Tail_MAE': r['tail']['MAE'],
            'Head_RMSE': r['head']['RMSE'],
            'Tail_RMSE': r['tail']['RMSE'],
            'Head_MSE': r['head']['MSE'],
            'Tail_MSE': r['tail']['MSE'],
            'Head_R2': r['head']['R2'],
            'Tail_R2': r['tail']['R2'],
        }
        for r in results
    ])
    
    os.makedirs('output/longtail_comparison', exist_ok=True)
    df.to_csv('output/longtail_comparison/results.csv', index=False)
    print(f"\nResults saved to output/longtail_comparison/results.csv")
    print("\n" + "="*80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
