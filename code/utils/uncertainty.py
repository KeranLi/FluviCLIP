"""
Uncertainty quantification utilities using Monte Carlo Dropout.
Provides pixel-wise prediction uncertainty for operational reliability assessment.
"""
import torch
import numpy as np
import torch.nn.functional as F


def enable_mc_dropout(model):
    """
    Enable dropout layers for Monte Carlo sampling during inference.
    """
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()  # Keep dropout active during eval


def mc_dropout_predict(model, image, n_samples=30, device='cuda'):
    """
    Generate prediction with uncertainty estimate using Monte Carlo Dropout.
    
    Args:
        model: Trained model
        image: Input image tensor (C, H, W) or (1, C, H, W)
        n_samples: Number of MC samples
        device: Device for computation
    
    Returns:
        mean_pred: Mean prediction
        std_pred: Standard deviation (uncertainty)
        all_preds: All individual predictions
    """
    model.eval()
    enable_mc_dropout(model)
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            if hasattr(model, 'forward'):
                pred, _ = model(image, texts=None)
            else:
                pred = model(image)
            preds.append(pred.cpu().numpy())
    
    preds = np.array(preds).squeeze()
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    
    return mean_pred, std_pred, preds


def batch_uncertainty_estimation(model, dataloader, n_samples=30, device='cuda'):
    """
    Estimate uncertainty for a batch of samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        n_samples: Number of MC samples per image
        device: Device for computation
    
    Returns:
        results: Dictionary with predictions, uncertainties, and ground truths
    """
    model.eval()
    enable_mc_dropout(model)
    
    all_preds = []
    all_uncertainties = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            batch_preds = []
            for _ in range(n_samples):
                if hasattr(model, 'forward'):
                    pred, _ = model(images, texts=None)
                else:
                    pred = model(images)
                batch_preds.append(pred.cpu().numpy())
            
            batch_preds = np.array(batch_preds)  # (n_samples, B, 1)
            mean_pred = batch_preds.mean(axis=0).flatten()
            std_pred = batch_preds.std(axis=0).flatten()
            
            all_preds.extend(mean_pred)
            all_uncertainties.extend(std_pred)
            all_targets.extend(labels.cpu().numpy().flatten())
    
    return {
        'predictions': np.array(all_preds),
        'uncertainties': np.array(all_uncertainties),
        'targets': np.array(all_targets)
    }


def calibration_curve(predictions, uncertainties, targets, n_bins=10):
    """
    Compute calibration curve to evaluate uncertainty quality.
    Well-calibrated uncertainties should correlate with prediction errors.
    
    Args:
        predictions: Predicted values
        uncertainties: Predicted uncertainties (std)
        targets: Ground truth values
        n_bins: Number of bins for calibration
    
    Returns:
        bin_centers: Centers of uncertainty bins
        bin_accuracies: Mean accuracy in each bin
        bin_confidences: Mean predicted uncertainty in each bin
    """
    errors = np.abs(predictions - targets)
    
    # Create bins based on uncertainty
    bin_edges = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(errors[mask].mean())
            bin_confidences.append(uncertainties[mask].mean())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
    
    return bin_centers, np.array(bin_accuracies), np.array(bin_confidences)


def plot_uncertainty_map(prediction_map, uncertainty_map, save_path=None):
    """
    Plot prediction map with uncertainty overlay.
    
    Args:
        prediction_map: 2D array of predictions
        uncertainty_map: 2D array of uncertainties
        save_path: Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prediction map
    im1 = axes[0].imshow(prediction_map, cmap='viridis')
    axes[0].set_title('SSC Prediction')
    plt.colorbar(im1, ax=axes[0])
    
    # Uncertainty map
    im2 = axes[1].imshow(uncertainty_map, cmap='hot')
    axes[1].set_title('Prediction Uncertainty')
    plt.colorbar(im2, ax=axes[1])
    
    # Overlay
    im3 = axes[2].imshow(prediction_map, cmap='viridis')
    axes[2].imshow(uncertainty_map, cmap='hot', alpha=0.5)
    axes[2].set_title('Prediction + Uncertainty')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
