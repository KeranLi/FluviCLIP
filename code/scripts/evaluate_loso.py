"""
Leave-One-Station-Out (LOSO) Cross-Validation for FluviCLIP.
Evaluates spatial generalization to ungauged basins.
"""
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils_4D import Sentinel2Dataset
from utils.seed import set_seed
from utils.numeric import calculate_mean_std, inverse_normalize
from models.fluviclip import FluviCLIP
from configs.FluviCLIP import Config


set_seed(42)


class StationDataset(Dataset):
    """Dataset that includes station information for LOSO validation."""
    def __init__(self, image_paths, labels, stations, means=None, stds=None):
        self.image_paths = image_paths
        self.labels = labels
        self.stations = stations
        self.means = means
        self.stds = stds
        self.label_mean = np.mean(labels)
        self.label_std = np.std(labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from osgeo import gdal
        import torch.nn.functional as F
        
        # Load image
        image_path = self.image_paths[idx]
        image_ds = gdal.Open(image_path)
        if image_ds is None:
            raise FileNotFoundError(f"Unable to open file: {image_path}")
        
        image = []
        for b in range(1, image_ds.RasterCount + 1):
            band = image_ds.GetRasterBand(b)
            image.append(band.ReadAsArray())
        image = np.stack(image, axis=-1)
        
        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), 
                             mode='bilinear', align_corners=False).squeeze(0)
        
        # Normalize
        if self.means is not None and self.stds is not None:
            for i in range(image.shape[0]):
                if self.stds[i] == 0:
                    self.stds[i] = 1e-6
                image[i] = (image[i] - self.means[i]) / self.stds[i]
        
        label = self.labels[idx]
        label = (label - self.label_mean) / self.label_std
        label = torch.tensor(label, dtype=torch.float32)
        
        station = self.stations[idx]
        
        return image, label, station


def load_data_with_stations(file_path, image_dir, sheet_name):
    """Load data with station information for LOSO."""
    excel_data = pd.read_excel(file_path, sheet_name=sheet_name)
    
    image_paths, labels, stations = [], [], []
    k = 0
    for i in range(len(excel_data["水文站编号"])):
        if excel_data.at[i, "可行"] == 1:
            station_id = int(excel_data.at[i, "水文站编号"])
            sample_id = str(excel_data.at[i, "样本序号"])
            
            image_paths.append(os.path.join(image_dir, f"{station_id}_{sample_id}_{k+1}.tif"))
            labels.append(float(excel_data.at[i, "含沙量（g/m3）"]))
            stations.append(station_id)
            k += 1
    
    return image_paths, labels, stations


def train_on_stations(model, train_loader, optimizer, criterion, device, epochs=50):
    """Train model on training stations."""
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            
            # For LOSO training, we use standard regression (no text for simplicity)
            ssc_pred, _ = model(images, texts=None)
            loss = criterion(ssc_pred, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model


def evaluate_on_station(model, test_loader, device, label_mean, label_std, head_threshold):
    """Evaluate model on held-out station with head/tail metrics."""
    model.eval()
    head_actuals, head_preds = [], []
    tail_actuals, tail_preds = [], []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)
            
            ssc_pred, _ = model(images, texts=None)
            
            # Denormalize
            pred_denorm = inverse_normalize(ssc_pred.cpu().numpy().flatten(), label_mean, label_std)
            actual_denorm = inverse_normalize(labels.cpu().numpy().flatten(), label_mean, label_std)
            
            for p, a in zip(pred_denorm, actual_denorm):
                if a <= head_threshold:
                    head_actuals.append(a)
                    head_preds.append(p)
                else:
                    tail_actuals.append(a)
                    tail_preds.append(p)
    
    def compute_metrics(actuals, preds):
        actuals = np.array(actuals)
        preds = np.array(preds)
        if len(actuals) == 0:
            return {"MAE": 0, "RMSE": 0, "MSE": 0, "R2": 0, "count": 0}
        mae = np.mean(np.abs(actuals - preds))
        mse = np.mean((actuals - preds) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return {"MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2 * 100, "count": len(actuals)}
    
    head_metrics = compute_metrics(head_actuals, head_preds)
    tail_metrics = compute_metrics(tail_actuals, tail_preds)
    
    return head_metrics, tail_metrics


def run_loso_validation():
    """Run full LOSO cross-validation."""
    config = Config()
    device = config.device
    
    # Load all data with station info
    file_path = os.path.join(config.data_path, config.excel_file)
    image_dir = os.path.join(config.data_path, config.image_dir)
    image_paths, labels, stations = load_data_with_stations(file_path, image_dir, config.sheet_name)
    
    # Get unique stations
    unique_stations = list(set(stations))
    print(f"Found {len(unique_stations)} stations: {unique_stations}")
    
    # Compute statistics
    means, stds = calculate_mean_std(image_paths)
    head_threshold = np.percentile(labels, 75)
    
    # Store results
    results = []
    
    # LOSO: Leave one station out each time
    for held_out_station in unique_stations:
        print(f"\n{'='*50}")
        print(f"LOSO: Holding out Station {held_out_station}")
        print(f"{'='*50}")
        
        # Split data
        train_paths, train_labels = [], []
        test_paths, test_labels = []
        
        for path, label, station in zip(image_paths, labels, stations):
            if station == held_out_station:
                test_paths.append(path)
                test_labels.append(label)
            else:
                train_paths.append(path)
                train_labels.append(label)
        
        print(f"Train samples: {len(train_paths)}, Test samples: {len(test_paths)}")
        
        # Create datasets
        train_dataset = StationDataset(train_paths, train_labels, 
                                      [s for s in stations if s != held_out_station],
                                      means=means, stds=stds)
        test_dataset = StationDataset(test_paths, test_labels, 
                                     [held_out_station] * len(test_paths),
                                     means=means, stds=stds)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Initialize fresh model
        model = FluviCLIP(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
            depths=config.depths,
            num_heads=config.num_heads,
            window_size=config.window_size,
            text_encoder_name=config.text_encoder_name,
            prompt_length=config.prompt_length,
        ).to(device)
        
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.MSELoss()
        
        # Train
        print("Training...")
        model = train_on_stations(model, train_loader, optimizer, criterion, device, epochs=50)
        
        # Evaluate
        print("Evaluating...")
        head_metrics, tail_metrics = evaluate_on_station(
            model, test_loader, device,
            test_dataset.label_mean, test_dataset.label_std,
            head_threshold
        )
        
        print(f"\nResults for Station {held_out_station}:")
        print(f"  Head (<= {head_threshold:.1f}): R2={head_metrics['R2']:.2f}%, MAE={head_metrics['MAE']:.2f}, n={head_metrics['count']}")
        print(f"  Tail (> {head_threshold:.1f}):  R2={tail_metrics['R2']:.2f}%, MAE={tail_metrics['MAE']:.2f}, n={tail_metrics['count']}")
        
        results.append({
            "held_out_station": held_out_station,
            "head_r2": head_metrics['R2'],
            "head_mae": head_metrics['MAE'],
            "head_count": head_metrics['count'],
            "tail_r2": tail_metrics['R2'],
            "tail_mae": tail_metrics['MAE'],
            "tail_count": tail_metrics['count'],
        })
    
    # Summary
    print(f"\n{'='*50}")
    print("LOSO Validation Summary")
    print(f"{'='*50}")
    
    df_results = pd.DataFrame(results)
    print(f"\nAverage Head R2: {df_results['head_r2'].mean():.2f}% ± {df_results['head_r2'].std():.2f}%")
    print(f"Average Tail R2: {df_results['tail_r2'].mean():.2f}% ± {df_results['tail_r2'].std():.2f}%")
    
    # Save results
    output_file = "output/LOSO_results.csv"
    os.makedirs("output", exist_ok=True)
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return df_results


if __name__ == "__main__":
    results = run_loso_validation()
