import os
import torch
import numpy as np
from osgeo import gdal
import cv2
import matplotlib.pyplot as plt
import argparse

from models.fluviclip import FluviCLIP
from models.Unet2D import RemoteSensingRegressionModel
from utils.numeric import inverse_normalize
from utils.uncertainty import mc_dropout_predict, plot_uncertainty_map
from configs.FluviCLIP import Config


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for visualizing model attention.
    """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations.append(output.detach())

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())

    def __call__(self, input_tensor):
        self.gradients = []
        self.activations = []
        output = self.model(input_tensor)
        return output

    def release(self):
        for handle in self.handles:
            handle.remove()

    def compute_cam(self):
        """Compute class activation map."""
        if len(self.gradients) == 0 or len(self.activations) == 0:
            return None
        pooled_grads = torch.mean(self.gradients[0], dim=[2, 3])
        for i in range(self.activations[0].size(1)):
            self.activations[0][:, i, :, :] *= pooled_grads[:, i].unsqueeze(-1).unsqueeze(-1)
        heatmap = torch.sum(self.activations[0], dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap.cpu().numpy()
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / (np.max(heatmap) + 1e-6)
        return heatmap


def load_image(image_path, target_size=(224, 224)):
    """
    Load a multi-spectral image from a GeoTIFF file.
    
    Args:
        image_path: Path to the GeoTIFF file
        target_size: Target size (H, W) for resizing
    
    Returns:
        image: Tensor of shape (C, H, W)
    """
    image_ds = gdal.Open(image_path)
    if image_ds is None:
        raise FileNotFoundError(f"Unable to open file: {image_path}")
    
    # Read all bands
    image = []
    for b in range(1, image_ds.RasterCount + 1):
        band = image_ds.GetRasterBand(b)
        image.append(band.ReadAsArray())
    image = np.stack(image, axis=0)  # shape: (C, H, W)
    
    # Convert to float32
    image = image.astype(np.float32)
    
    # Resize to target size
    image = torch.from_numpy(image).float()
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0), 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    return image


def inference_fluviclip(image_paths, model_path=None, use_uncertainty=False, n_mc_samples=30):
    """
    Run inference with FluviCLIP model.
    
    Args:
        image_paths: List of image file paths
        model_path: Path to model checkpoint (optional)
        use_uncertainty: Whether to use Monte Carlo Dropout for uncertainty
        n_mc_samples: Number of MC samples for uncertainty estimation
    
    Returns:
        results: Dictionary with predictions and optionally uncertainties
    """
    config = Config()
    device = config.device
    print(f"Using device: {device}")
    
    # Initialize model
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
    
    # Load weights
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    elif os.path.exists(os.path.join(config.output_dir, config.checkpoint_path)):
        checkpoint = os.path.join(config.output_dir, config.checkpoint_path)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Loaded model from {checkpoint}")
    else:
        print("Warning: No checkpoint found, using random weights")
    
    model.eval()
    
    # Load images
    input_tensors = []
    for image_path in image_paths:
        image = load_image(image_path)
        input_tensors.append(image)
    
    input_tensor = torch.stack(input_tensors).to(device)
    
    results = {
        'predictions': [],
        'gating_weights': [],
        'uncertainties': [] if use_uncertainty else None
    }
    
    with torch.no_grad():
        if use_uncertainty:
            # Monte Carlo Dropout for uncertainty
            for i in range(len(image_paths)):
                mean_pred, std_pred, _ = mc_dropout_predict(
                    model, input_tensors[i], n_samples=n_mc_samples, device=device
                )
                results['predictions'].append(mean_pred)
                results['uncertainties'].append(std_pred)
        else:
            # Standard inference
            ssc_preds, gates = model(input_tensor, texts=None)
            results['predictions'] = ssc_preds.cpu().numpy().flatten().tolist()
            results['gating_weights'] = gates.cpu().numpy().flatten().tolist()
    
    return results


def inference_with_gradcam(image_paths, model_path=None, target_layer_name='layers'):
    """
    Run inference with Grad-CAM visualization.
    
    Args:
        image_paths: List of image file paths
        model_path: Path to model checkpoint
        target_layer_name: Name of the layer to visualize
    
    Returns:
        predictions: List of predicted SSC values
        heatmaps: List of Grad-CAM heatmaps
    """
    config = Config()
    device = config.device
    
    # Initialize model
    model = FluviCLIP(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_size=config.window_size,
    ).to(device)
    
    # Load weights
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    
    # Load images
    input_tensors = []
    for image_path in image_paths:
        image = load_image(image_path)
        input_tensors.append(image)
    
    input_tensor = torch.stack(input_tensors).to(device)
    
    # Initialize Grad-CAM
    # Access the last layer of the visual encoder
    target_layers = [model.visual_encoder.layers[-1].blocks[-1].norm2]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Forward pass
    outputs = cam(input_tensor)
    predictions = outputs[0].detach().cpu().numpy().flatten()
    
    # Backward pass (manual loss computation)
    loss = outputs[0].sum()
    loss.backward()
    
    # Compute CAM
    heatmaps = []
    for i in range(len(image_paths)):
        heatmap = cam.compute_cam()
        if heatmap is not None:
            # Resize to original image size if needed
            if isinstance(heatmap, np.ndarray):
                heatmap = cv2.resize(heatmap, (224, 224))
            heatmaps.append(heatmap)
    
    cam.release()
    
    return predictions, heatmaps


def save_visualization(image_path, prediction, save_dir="output/inference", 
                      uncertainty=None, heatmap=None):
    """
    Save inference results with optional uncertainty and attention maps.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load image for visualization
    image = load_image(image_path)
    img_np = image[:3].cpu().numpy().transpose(1, 2, 0)  # Use first 3 bands as RGB
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 3 if heatmap is not None else 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Input Image\nPred: {prediction:.2f} g/m³")
    axes[0].axis('off')
    
    # Uncertainty
    if uncertainty is not None:
        im = axes[1].imshow(np.full_like(img_np[:, :, 0], uncertainty), cmap='hot')
        axes[1].set_title(f"Uncertainty\nσ = {uncertainty:.2f}")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    
    # Grad-CAM
    if heatmap is not None:
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        axes[-1].imshow(img_np)
        axes[-1].imshow(heatmap_img, alpha=0.5)
        axes[-1].set_title("Attention Map")
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    basename = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(os.path.join(save_dir, f"{basename}_result.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="FluviCLIP Inference")
    parser.add_argument("--input", "-i", nargs="+", help="Input image file(s)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--uncertainty", "-u", action="store_true", help="Enable uncertainty estimation")
    parser.add_argument("--gradcam", "-g", action="store_true", help="Enable Grad-CAM visualization")
    parser.add_argument("--output", "-o", type=str, default="output/inference", help="Output directory")
    args = parser.parse_args()
    
    if args.input is None:
        # Use default inference images
        input_dir = "datasets/inference"
        args.input = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    print(f"Running inference on {len(args.input)} images...")
    
    if args.gradcam:
        predictions, heatmaps = inference_with_gradcam(args.input, args.model)
        for i, (img_path, pred) in enumerate(zip(args.input, predictions)):
            save_visualization(img_path, pred, args.output, heatmap=heatmaps[i] if i < len(heatmaps) else None)
    else:
        results = inference_fluviclip(args.input, args.model, use_uncertainty=args.uncertainty)
        predictions = results['predictions']
        uncertainties = results['uncertainties']
        
        print("\nResults:")
        for i, img_path in enumerate(args.input):
            pred = predictions[i]
            unc = uncertainties[i] if uncertainties else None
            print(f"  {os.path.basename(img_path)}: {pred:.2f} g/m³" + 
                  (f" ± {unc:.2f}" if unc else ""))
            save_visualization(img_path, pred, args.output, uncertainty=unc)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
