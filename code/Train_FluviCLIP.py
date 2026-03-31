import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils.data_utils_4D import load_excel_data, Sentinel2Dataset
from utils.train_utils import validate_model
from utils.seed import set_seed
from utils.plot import plot_predictions, plot_loss_curves, plot_actual_vs_pred
from utils.numeric import calculate_mean_std, inverse_normalize
from utils.earlystop import EarlyStopping
from utils.contrastive_utils import generate_text_descriptions, split_head_tail

from models.fluviclip import FluviCLIP
from configs.FluviCLIP import Config


set_seed(42)


def train_fluviclip_epoch(model, train_loader, optimizer, device, lambda_contrastive=0.3):
    """
    Train FluviCLIP for one epoch with joint contrastive and regression objectives.
    """
    model.train()
    total_loss = 0.0
    total_ssc_loss = 0.0
    total_contrastive_loss = 0.0
    batch_losses = []
    
    for batch_idx, (images, labels, texts) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        ssc_pred, gate, head_out, tail_out, visual_features, text_features = model(images, texts)
        
        # Compute joint loss
        loss_dict = model.compute_loss(
            ssc_pred, gate, head_out, tail_out,
            visual_features, text_features, labels
        )
        loss = loss_dict["total_loss"]
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_ssc_loss += loss_dict["ssc_loss"].item()
        total_contrastive_loss += loss_dict["contrastive_loss"].item()
        batch_losses.append({
            "Batch": batch_idx + 1,
            "TotalLoss": loss.item(),
            "SSCLoss": loss_dict["ssc_loss"].item(),
            "ContrastiveLoss": loss_dict["contrastive_loss"].item(),
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_ssc_loss = total_ssc_loss / len(train_loader)
    avg_contrastive_loss = total_contrastive_loss / len(train_loader)
    
    return avg_loss, avg_ssc_loss, avg_contrastive_loss


def evaluate_head_tail(model, data_loader, device, label_mean, label_std, head_threshold):
    """
    Evaluate model separately on head and tail distributions.
    Returns MAE, RMSE, MSE, R2 for both head and tail.
    """
    model.eval()
    head_actuals, head_preds = [], []
    tail_actuals, tail_preds = [], []
    
    with torch.no_grad():
        for images, labels in data_loader:
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
            return 0.0, 0.0, 0.0, 0.0
        mae = np.mean(np.abs(actuals - preds))
        mse = np.mean((actuals - preds) ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        return mae, rmse, mse, r2
    
    head_metrics = compute_metrics(head_actuals, head_preds)
    tail_metrics = compute_metrics(tail_actuals, tail_preds)
    
    return head_metrics, tail_metrics


class TextDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset that attaches text descriptions to each sample.
    """
    def __init__(self, base_dataset, texts):
        self.base_dataset = base_dataset
        self.texts = texts
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return image, label, self.texts[idx]


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
    
    # Generate text descriptions for contrastive learning
    texts = generate_text_descriptions(labels, num_variants=4)
    
    # Stratified split to maintain SSC distribution (6:2:2)
    train_paths, temp_paths, train_labels, temp_labels, train_texts, temp_texts = train_test_split(
        image_paths, labels, texts, test_size=0.4, random_state=42, stratify=pd.qcut(labels, q=10, duplicates='drop')
    )
    val_paths, test_paths, val_labels, test_labels, val_texts, test_texts = train_test_split(
        temp_paths, temp_labels, temp_texts, test_size=0.5, random_state=42,
        stratify=pd.qcut(temp_labels, q=10, duplicates='drop')
    )
    
    # Build datasets
    train_dataset = Sentinel2Dataset(train_paths, train_labels, means=means, stds=stds)
    val_dataset = Sentinel2Dataset(val_paths, val_labels, means=means, stds=stds)
    test_dataset = Sentinel2Dataset(test_paths, test_labels, means=means, stds=stds)
    
    # Attach text descriptions only for training set
    train_dataset_with_text = TextDataset(train_dataset, train_texts)
    # Validation and test sets do not need text descriptions for inference
    
    # Compute head/tail threshold (75th percentile)
    head_threshold = np.percentile(labels, 75)
    
    # Data loaders
    train_loader = DataLoader(train_dataset_with_text, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    model = FluviCLIP(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        embed_dim=config.embed_dim,
        depths=config.depths,
        num_heads=config.num_heads,
        window_size=config.window_size,
        mlp_ratio=config.mlp_ratio,
        text_encoder_name=config.text_encoder_name,
        prompt_length=config.prompt_length,
        projection_dim=config.projection_dim,
        temperature=config.temperature,
        lambda_contrastive=config.lambda_contrastive,
    ).to(device)
    
    # Load MAE pre-trained weights if available
    if os.path.exists(config.pretrain_checkpoint):
        print(f"Loading pre-trained visual encoder from {config.pretrain_checkpoint}")
        pretrain_state = torch.load(config.pretrain_checkpoint, map_location=device)
        # Load only visual encoder weights
        model.visual_encoder.load_state_dict(pretrain_state, strict=False)
    
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
        # Training
        train_loss, train_ssc_loss, train_contrastive_loss = train_fluviclip_epoch(
            model, train_loader, optimizer, device, config.lambda_contrastive
        )
        train_losses.append(train_loss)
        
        # Validation (overall)
        val_loss, val_actuals, val_predictions = validate_model(
            model, val_loader, torch.nn.MSELoss(), device,
            val_dataset.label_mean, val_dataset.label_std,
            save_path=os.path.join(config.output_dir, "val_results.csv"),
            epoch=epoch + 1
        )
        val_losses.append(val_loss)
        
        # Test (overall)
        test_loss, test_actuals, test_predictions = validate_model(
            model, test_loader, torch.nn.MSELoss(), device,
            test_dataset.label_mean, test_dataset.label_std,
            save_path=os.path.join(config.output_dir, "test_results.csv"),
            epoch=epoch + 1
        )
        test_losses.append(test_loss)
        
        # Head / Tail evaluation
        val_head, val_tail = evaluate_head_tail(
            model, val_loader, device,
            val_dataset.label_mean, val_dataset.label_std,
            head_threshold
        )
        test_head, test_tail = evaluate_head_tail(
            model, test_loader, device,
            test_dataset.label_mean, test_dataset.label_std,
            head_threshold
        )
        
        print(
            f"Epoch {epoch + 1}/{config.num_epochs}: "
            f"Train Loss={train_loss:.4f} (SSC={train_ssc_loss:.4f}, Cont={train_contrastive_loss:.4f}), "
            f"Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}\n"
            f"  Val  Head -> MAE={val_head[0]:.2f}, RMSE={val_head[1]:.2f}, MSE={val_head[2]:.2f}, R2={val_head[3]:.4f}\n"
            f"  Val  Tail -> MAE={val_tail[0]:.2f}, RMSE={val_tail[1]:.2f}, MSE={val_tail[2]:.2f}, R2={val_tail[3]:.4f}\n"
            f"  Test Head -> MAE={test_head[0]:.2f}, RMSE={test_head[1]:.2f}, MSE={test_head[2]:.2f}, R2={test_head[3]:.4f}\n"
            f"  Test Tail -> MAE={test_tail[0]:.2f}, RMSE={test_tail[1]:.2f}, MSE={test_tail[2]:.2f}, R2={test_tail[3]:.4f}"
        )
        
        # TensorBoard logging
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss, "Test": test_loss}, epoch)
        writer.add_scalars("DetailLoss", {"SSC": train_ssc_loss, "Contrastive": train_contrastive_loss}, epoch)
        writer.add_scalars("Val_Metrics", {"Head_R2": val_head[3], "Tail_R2": val_tail[3]}, epoch)
        writer.add_scalars("Test_Metrics", {"Head_R2": test_head[3], "Tail_R2": test_tail[3]}, epoch)
        
        # Plotting
        plot_predictions(
            val_actuals, val_predictions,
            f"Epoch {epoch + 1}: Validation Actual vs Predicted",
            os.path.join(config.output_dir, f"epoch_{epoch + 1}_val_predictions_1.png")
        )
        plot_actual_vs_pred(
            val_actuals, val_predictions,
            f"Epoch {epoch + 1}: Validation Actual vs Predicted",
            os.path.join(config.output_dir, f"epoch_{epoch + 1}_val_predictions_2.png")
        )
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(config.output_dir, config.checkpoint_path)))
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, test_losses, os.path.join(config.output_dir, "loss_curves.png"))
    writer.close()
