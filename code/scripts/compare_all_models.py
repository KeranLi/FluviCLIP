"""
Comprehensive comparison script for all baseline models.
Trains and evaluates all models mentioned in the paper.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from collections import defaultdict
import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.seed import set_seed
from utils.numeric import calculate_mean_std, inverse_normalize
from utils.train_utils import train_model, validate_model

# Import all models
from models.resnet import ResNet50Regressor, Res2Net, ResNeXt50, MultimodalResNet50
from models.SwinT import SwinTransformerWithReducer
from models.ViT import VisionTransformerWithReducer
from models.Unet2D import RemoteSensingRegressionModel
from models.fluviclip import FluviCLIP
from models.foundation_models import RemoteCLIPWrapper, MultimodalVariant
from models.sequence_models import PureFF

set_seed(42)


class Config:
    """Configuration for comparison experiments."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = "datasets/train"
        self.excel_file = "data.xlsx"
        self.image_dir = "images/"
        self.sheet_name = "Sheet1"
        self.output_dir = "output/comparison"
        self.num_epochs = 50  # Reduced for faster comparison
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4


def compute_metrics(predictions, targets, head_threshold):
    """
    Compute metrics for head and tail distributions.
    
    Returns:
        dict with overall, head, and tail metrics
    """
    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()
    
    # Overall metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((predictions - targets) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    # Head/Tail split
    head_mask = targets <= head_threshold
    tail_mask = targets > head_threshold
    
    # Head metrics
    head_pred = predictions[head_mask]
    head_tgt = targets[head_mask]
    head_mae = np.mean(np.abs(head_pred - head_tgt)) if len(head_tgt) > 0 else 0
    head_mse = np.mean((head_pred - head_tgt) ** 2) if len(head_tgt) > 0 else 0
    head_rmse = np.sqrt(head_mse)
    head_ss_res = np.sum((head_pred - head_tgt) ** 2) if len(head_tgt) > 0 else 0
    head_ss_tot = np.sum((head_tgt - np.mean(head_tgt)) ** 2) if len(head_tgt) > 0 else 1e-8
    head_r2 = 1 - head_ss_res / head_ss_tot
    
    # Tail metrics
    tail_pred = predictions[tail_mask]
    tail_tgt = targets[tail_mask]
    tail_mae = np.mean(np.abs(tail_pred - tail_tgt)) if len(tail_tgt) > 0 else 0
    tail_mse = np.mean((tail_pred - tail_tgt) ** 2) if len(tail_tgt) > 0 else 0
    tail_rmse = np.sqrt(tail_mse)
    tail_ss_res = np.sum((tail_pred - tail_tgt) ** 2) if len(tail_tgt) > 0 else 0
    tail_ss_tot = np.sum((tail_tgt - np.mean(tail_tgt)) ** 2) if len(tail_tgt) > 0 else 1e-8
    tail_r2 = 1 - tail_ss_res / tail_ss_tot
    
    return {
        'overall': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2},
        'head': {'MAE': head_mae, 'MSE': head_mse, 'RMSE': head_rmse, 'R2': head_r2, 'count': len(head_tgt)},
        'tail': {'MAE': tail_mae, 'MSE': tail_mse, 'RMSE': tail_rmse, 'R2': tail_r2, 'count': len(tail_tgt)}
    }


def train_and_evaluate_model(model_name, model, train_loader, val_loader, test_loader, 
                             label_mean, label_std, head_threshold, device, num_epochs=50):
    """
    Train and evaluate a single model.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            
            # Handle models that take texts
            try:
                outputs = model(images)
            except TypeError:
                outputs = model(images, texts=None)
            
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
                try:
                    outputs = model(images)
                except TypeError:
                    outputs = model(images, texts=None)
                loss = criterion(outputs, labels)
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
            try:
                outputs = model(images)
            except TypeError:
                outputs = model(images, texts=None)
            
            pred_denorm = inverse_normalize(outputs.cpu().numpy().flatten(), label_mean, label_std)
            actual_denorm = inverse_normalize(labels.numpy().flatten(), label_mean, label_std)
            
            predictions.extend(pred_denorm)
            targets.extend(actual_denorm)
    
    metrics = compute_metrics(predictions, targets, head_threshold)
    
    print(f"\n{model_name} Results:")
    print(f"  Overall: R2={metrics['overall']['R2']:.4f}, MAE={metrics['overall']['MAE']:.2f}")
    print(f"  Head:    R2={metrics['head']['R2']:.4f}, MAE={metrics['head']['MAE']:.2f}")
    print(f"  Tail:    R2={metrics['tail']['R2']:.4f}, MAE={metrics['tail']['MAE']:.2f}")
    
    return metrics


def main():
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
    head_threshold = np.percentile(labels, 75)
    
    dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)
    label_mean, label_std = dataset.label_mean, dataset.label_std
    
    # Split
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Define all models
    models_dict = {
        # End-to-end baselines
        'ResNet50': ResNet50Regressor(in_channels=26, num_classes=1, pretrained=False),
        'Res2Net': Res2Net(in_channels=26, layers=[3, 4, 6, 3], num_classes=1, scale=4),
        'ResNeXt50': ResNeXt50(in_channels=26, layers=[3, 4, 6, 3], num_classes=1, cardinality=32),
        'U-Net': RemoteSensingRegressionModel(img_size=224, in_channels=26, out_channels=3, num_classes=1),
        'ViT': VisionTransformerWithReducer(img_size=224, in_channels=26, out_channels=3, num_classes=1),
        'Swin-T': SwinTransformerWithReducer(img_size=224, in_channels=26, out_channels=3, num_classes=1),
        
        # Multimodal variants (using ResNet50 backbone + CLIP text encoder)
        'ResNet50+CLIP': MultimodalResNet50(in_channels=26, num_classes=1, 
                                            text_encoder_name="openai/clip-vit-base-patch32"),
        
        # Ours
        'FluviCLIP (Ours)': FluviCLIP(
            img_size=224, patch_size=4, in_chans=26, embed_dim=96,
            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
            text_encoder_name="openai/clip-vit-base-patch32"
        ),
        
        # Pure-FF baseline (FluviFormer without multimodal pretraining)
        'Pure-FF': PureFF(
            img_size=224, patch_size=4, in_chans=26, embed_dim=96,
            depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], num_classes=1
        ),
    }
    
    # Store results
    all_results = {}
    
    # Train and evaluate each model
    for model_name, model in models_dict.items():
        try:
            metrics = train_and_evaluate_model(
                model_name, model, train_loader, val_loader, test_loader,
                label_mean, label_std, head_threshold, device, config.num_epochs
            )
            all_results[model_name] = metrics
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Overall_R2': m['overall']['R2'],
            'Overall_MAE': m['overall']['MAE'],
            'Head_R2': m['head']['R2'],
            'Head_MAE': m['head']['MAE'],
            'Tail_R2': m['tail']['R2'],
            'Tail_MAE': m['tail']['MAE'],
        }
        for name, m in all_results.items()
    ])
    
    results_df.to_csv(os.path.join(config.output_dir, 'comparison_results.csv'), index=False)
    print(f"\nResults saved to {os.path.join(config.output_dir, 'comparison_results.csv')}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save detailed JSON
    with open(os.path.join(config.output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
