"""
SHAP (SHapley Additive exPlanations) analysis for FluviCLIP interpretability.
Generates spatial attribution maps to verify physically meaningful predictions.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import shap

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.seed import set_seed
from utils.numeric import calculate_mean_std
from models.fluviclip import FluviCLIP
from configs.FluviCLIP import Config


set_seed(42)


class SHAPAnalyzer:
    """
    SHAP analyzer for generating spatial attribution maps.
    """
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def compute_shap_values(self, images, background_samples=10):
        """
        Compute SHAP values for the given images.
        
        Args:
            images: (N, C, H, W) input images
            background_samples: Number of background samples for DeepExplainer
        
        Returns:
            shap_values: SHAP values of shape (N, C, H, W)
        """
        # Select background samples
        background = images[:background_samples].to(self.device)
        
        # Create a wrapper function for prediction
        def model_wrapper(x):
            # x is numpy array (N, C, H, W)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                pred, _ = self.model(x, texts=None)
            return pred.cpu().numpy()
        
        # Use DeepExplainer
        explainer = shap.DeepExplainer(self.model.visual_encoder, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(images.to(self.device))
        
        return shap_values
    
    def visualize_shap_maps(self, images, shap_values, titles=None, save_dir="output/shap"):
        """
        Visualize SHAP attribution maps.
        
        Args:
            images: (N, C, H, W) input images
            shap_values: (N, C, H, W) SHAP values
            titles: List of titles for each sample
            save_dir: Directory to save visualizations
        """
        os.makedirs(save_dir, exist_ok=True)
        
        n_samples = min(images.shape[0], len(shap_values))
        
        for i in range(n_samples):
            img = images[i].cpu().numpy()
            shap_val = shap_values[i] if isinstance(shap_values, list) else shap_values[i]
            
            # Sum absolute SHAP values across channels for visualization
            shap_map = np.abs(shap_val).sum(axis=0)  # (H, W)
            
            # Normalize
            shap_map = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
            
            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image (RGB composite)
            rgb = np.stack([img[2], img[1], img[0]], axis=-1)  # Use first 3 bands as RGB
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[0].imshow(rgb)
            axes[0].set_title("Input Image (RGB)")
            axes[0].axis('off')
            
            # SHAP heatmap
            im = axes[1].imshow(shap_map, cmap='hot')
            axes[1].set_title("SHAP Attribution Map")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
            
            # Overlay
            axes[2].imshow(rgb)
            axes[2].imshow(shap_map, cmap='hot', alpha=0.5)
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            
            if titles:
                fig.suptitle(titles[i])
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"shap_sample_{i}.png"), dpi=150)
            plt.close()
        
        print(f"SHAP visualizations saved to {save_dir}")
    
    def analyze_feature_importance(self, images, band_names=None):
        """
        Analyze feature importance across spectral bands.
        
        Args:
            images: (N, C, H, W) input images
            band_names: List of band names
        
        Returns:
            importance: (C,) mean absolute SHAP value per band
        """
        shap_values = self.compute_shap_values(images)
        
        # Mean absolute SHAP value per band
        importance = np.abs(shap_values).mean(axis=(0, 2, 3))
        
        if band_names is None:
            band_names = [f"Band {i+1}" for i in range(len(importance))]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(band_names, importance)
        plt.xlabel("Spectral Band")
        plt.ylabel("Mean |SHAP Value|")
        plt.title("Feature Importance by Spectral Band")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/shap_band_importance.png", dpi=150)
        plt.close()
        
        return importance


def analyze_samples():
    """
    Analyze specific samples mentioned in the paper:
    - High-SSC events (concentrated at channel confluences)
    - Low-SSC conditions (uniform across water surface)
    """
    config = Config()
    device = config.device
    
    # Load data
    image_paths, labels = load_excel_data(
        os.path.join(config.data_path, config.excel_file),
        os.path.join(config.data_path, config.image_dir),
        config.sheet_name
    )
    means, stds = calculate_mean_std(image_paths)
    
    # Load model
    model = FluviCLIP(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_size=config.window_size,
    ).to(device)
    
    # Load trained weights
    checkpoint_path = os.path.join(config.output_dir, config.checkpoint_path)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using random weights")
    
    # Create dataset
    dataset = Sentinel2Dataset(image_paths, labels, means=means, stds=stds)
    
    # Select representative samples
    # High-SSC (tail)
    high_indices = np.where(np.array(labels) > np.percentile(labels, 90))[0]
    # Low-SSC (head)
    low_indices = np.where(np.array(labels) < np.percentile(labels, 25))[0]
    
    print(f"Selected {len(high_indices)} high-SSC samples and {len(low_indices)} low-SSC samples")
    
    # Sample a few for analysis
    selected_high = high_indices[:3]
    selected_low = low_indices[:3]
    selected = list(selected_high) + list(selected_low)
    
    # Load images
    images = []
    titles = []
    for idx in selected:
        img, _ = dataset[idx]
        images.append(img)
        titles.append(f"SSC: {labels[idx]:.1f} g/m³")
    
    images = torch.stack(images)
    
    # Analyze with SHAP
    analyzer = SHAPAnalyzer(model, device)
    
    print("Computing SHAP values...")
    shap_values = analyzer.compute_shap_values(images, background_samples=5)
    
    print("Generating visualizations...")
    analyzer.visualize_shap_maps(images, shap_values, titles=titles)
    
    # Band importance analysis
    print("Analyzing spectral band importance...")
    band_names = [f"B{i+1}" for i in range(config.in_chans)]
    importance = analyzer.analyze_feature_importance(images, band_names)
    
    print("\nTop 5 most important bands:")
    top_indices = np.argsort(importance)[::-1][:5]
    for idx in top_indices:
        print(f"  {band_names[idx]}: {importance[idx]:.4f}")
    
    print("\nSHAP analysis completed!")


if __name__ == "__main__":
    analyze_samples()
