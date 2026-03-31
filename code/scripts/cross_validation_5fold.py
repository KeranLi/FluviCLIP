"""
5-Fold Cross-Validation with 6:2:2 split for robust evaluation.
Each fold maintains consistent Head/Tail distribution through stratified sampling.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.seed import set_seed
from utils.numeric import calculate_mean_std, inverse_normalize
from models.fluviclip import FluviCLIP
from configs.FluviCLIP import Config

set_seed(42)


def create_stratified_folds(labels, n_splits=5):
    """
    Create stratified folds based on SSC values.
    Uses quantile-based binning for stratification.
    """
    # Create bins for stratification (10 quantiles)
    n_bins = min(10, len(labels) // n_splits)
    labels_array = np.array(labels)
    
    # Use percentile-based bins
    bins = np.percentile(labels_array, np.linspace(0, 100, n_bins + 1))
    bins[-1] += 1  # Ensure last value is included
    
    # Assign each sample to a bin
    stratify_labels = np.digitize(labels_array, bins[:-1]) - 1
    stratify_labels = np.clip(stratify_labels, 0, n_bins - 1)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), stratify_labels)):
        # Further split train_val into train (60%) and val (20%) - approx 75:25 split
        train_labels = stratify_labels[train_val_idx]
        
        # Create inner splitter for train/val
        inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  # 3:1 split = 45:15 from original 60
        
        # Get first split as train/val division
        inner_splits = list(inner_skf.split(np.zeros(len(train_val_idx)), train_labels))
        train_rel_idx, val_rel_idx = inner_splits[0]
        
        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]
        
        folds.append({
            'fold': fold_idx + 1,
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        })
    
    return folds


def train_model_fold(model, train_loader, val_loader, device, num_epochs=50, patience=15):
    """Train model for one fold with early stopping."""
    criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            
            try:
                outputs = model(images)
            except TypeError:
                outputs = model(images, texts=None)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    model.load_state_dict(best_state)
    return model


def evaluate_fold(model, test_loader, label_mean, label_std, head_threshold, device):
    """Evaluate model on test set with head/tail metrics."""
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
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    def compute_metrics(pred, tgt, mask):
        p = pred[mask]
        t = tgt[mask]
        if len(t) == 0:
            return {'MAE': 0, 'RMSE': 0, 'MSE': 0, 'R2': 0, 'count': 0}
        mae = np.mean(np.abs(p - t))
        mse = np.mean((p - t) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return {'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'R2': r2 * 100, 'count': len(t)}
    
    head_mask = targets <= head_threshold
    tail_mask = targets > head_threshold
    
    overall = compute_metrics(predictions, targets, np.ones_like(targets, dtype=bool))
    head = compute_metrics(predictions, targets, head_mask)
    tail = compute_metrics(predictions, targets, tail_mask)
    
    return {'overall': overall, 'head': head, 'tail': tail}


def run_5fold_cv(model_name, model_class, model_kwargs, config, num_epochs=50):
    """
    Run 5-fold cross-validation for a given model.
    
    Returns:
        dict with average metrics across 5 folds
    """
    print(f"\n{'='*70}")
    print(f"5-Fold Cross-Validation: {model_name}")
    print(f"{'='*70}")
    
    device = config.device
    
    # Load data
    image_paths, labels = load_excel_data(
        os.path.join(config.data_path, config.excel_file),
        os.path.join(config.data_path, config.image_dir),
        config.sheet_name
    )
    means, stds = calculate_mean_std(image_paths)
    
    # Create full dataset
    full_dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)
    label_mean, label_std = full_dataset.label_mean, full_dataset.label_std
    head_threshold = np.percentile(labels, 75)
    
    # Create stratified folds
    folds = create_stratified_folds(labels, n_splits=5)
    
    # Store results for each fold
    fold_results = []
    
    for fold_info in folds:
        fold_idx = fold_info['fold']
        print(f"\n--- Fold {fold_idx}/5 ---")
        
        # Create data loaders for this fold
        train_dataset = Subset(full_dataset, fold_info['train'])
        val_dataset = Subset(full_dataset, fold_info['val'])
        test_dataset = Subset(full_dataset, fold_info['test'])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Initialize fresh model
        model = model_class(**model_kwargs).to(device)
        
        # Train
        print(f"  Training...")
        model = train_model_fold(model, train_loader, val_loader, device, num_epochs)
        
        # Evaluate
        print(f"  Evaluating...")
        metrics = evaluate_fold(model, test_loader, label_mean, label_std, head_threshold, device)
        
        print(f"  Test R2: Overall={metrics['overall']['R2']:.2f}%, Head={metrics['head']['R2']:.2f}%, Tail={metrics['tail']['R2']:.2f}%")
        
        fold_results.append({
            'fold': fold_idx,
            'metrics': metrics
        })
    
    # Compute average across folds
    avg_results = {
        'model': model_name,
        'overall': {
            'R2': np.mean([r['metrics']['overall']['R2'] for r in fold_results]),
            'R2_std': np.std([r['metrics']['overall']['R2'] for r in fold_results]),
            'MAE': np.mean([r['metrics']['overall']['MAE'] for r in fold_results]),
            'MAE_std': np.std([r['metrics']['overall']['MAE'] for r in fold_results]),
            'RMSE': np.mean([r['metrics']['overall']['RMSE'] for r in fold_results]),
            'MSE': np.mean([r['metrics']['overall']['MSE'] for r in fold_results]),
        },
        'head': {
            'R2': np.mean([r['metrics']['head']['R2'] for r in fold_results]),
            'R2_std': np.std([r['metrics']['head']['R2'] for r in fold_results]),
            'MAE': np.mean([r['metrics']['head']['MAE'] for r in fold_results]),
            'MAE_std': np.std([r['metrics']['head']['MAE'] for r in fold_results]),
            'RMSE': np.mean([r['metrics']['head']['RMSE'] for r in fold_results]),
            'MSE': np.mean([r['metrics']['head']['MSE'] for r in fold_results]),
        },
        'tail': {
            'R2': np.mean([r['metrics']['tail']['R2'] for r in fold_results]),
            'R2_std': np.std([r['metrics']['tail']['R2'] for r in fold_results]),
            'MAE': np.mean([r['metrics']['tail']['MAE'] for r in fold_results]),
            'MAE_std': np.std([r['metrics']['tail']['MAE'] for r in fold_results]),
            'RMSE': np.mean([r['metrics']['tail']['RMSE'] for r in fold_results]),
            'MSE': np.mean([r['metrics']['tail']['MSE'] for r in fold_results]),
        },
        'fold_details': fold_results
    }
    
    return avg_results


def main():
    """Run 5-fold CV for FluviCLIP and baselines."""
    config = Config()
    
    # Define models to evaluate
    models_to_test = {
        'FluviCLIP (Ours)': (
            FluviCLIP,
            {
                'img_size': 224, 'patch_size': 4, 'in_chans': 26, 'embed_dim': 96,
                'depths': [2, 2, 6, 2], 'num_heads': [3, 6, 12, 24], 'window_size': 7,
                'text_encoder_name': "openai/clip-vit-base-patch32"
            }
        ),
    }
    
    all_results = {}
    
    for model_name, (model_class, model_kwargs) in models_to_test.items():
        results = run_5fold_cv(model_name, model_class, model_kwargs, config, num_epochs=50)
        all_results[model_name] = results
    
    # Print summary
    print(f"\n{'='*70}")
    print("5-Fold Cross-Validation Summary (Mean ± Std)")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Head R2':<20} {'Tail R2':<20} {'Overall R2':<20}")
    print(f"{'-'*70}")
    
    for model_name, results in all_results.items():
        head_r2 = results['head']['R2']
        head_std = results['head']['R2_std']
        tail_r2 = results['tail']['R2']
        tail_std = results['tail']['R2_std']
        overall_r2 = results['overall']['R2']
        overall_std = results['overall']['R2_std']
        
        print(f"{model_name:<20} {head_r2:>6.2f}±{head_std:<5.2f}    {tail_r2:>6.2f}±{tail_std:<5.2f}    {overall_r2:>6.2f}±{overall_std:<5.2f}")
    
    # Save results
    output_dir = 'output/5fold_cv'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create DataFrame for CSV
    rows = []
    for model_name, results in all_results.items():
        rows.append({
            'Model': model_name,
            'Head_R2_Mean': results['head']['R2'],
            'Head_R2_Std': results['head']['R2_std'],
            'Head_MAE_Mean': results['head']['MAE'],
            'Head_MAE_Std': results['head']['MAE_std'],
            'Tail_R2_Mean': results['tail']['R2'],
            'Tail_R2_Std': results['tail']['R2_std'],
            'Tail_MAE_Mean': results['tail']['MAE'],
            'Tail_MAE_Std': results['tail']['MAE_std'],
            'Overall_R2_Mean': results['overall']['R2'],
            'Overall_R2_Std': results['overall']['R2_std'],
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(f'{output_dir}/results.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
