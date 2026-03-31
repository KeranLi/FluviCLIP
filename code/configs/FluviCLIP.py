import torch


class Config:
    """
    Configuration class for FluviCLIP training and inference.
    """
    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Data paths
        self.data_path = "datasets/train"
        self.excel_file = "data.xlsx"
        self.image_dir = "images/"
        self.sheet_name = "Sheet1"
        
        # Output paths
        self.output_dir = "output/FluviCLIP"
        self.log_dir = "runs/FluviCLIP"
        self.checkpoint_path = "best_model.pth"
        self.pretrain_checkpoint = "output/MAE_Pretrain/best_model.pth"
        
        # Model architecture
        self.img_size = 224
        self.patch_size = 4
        self.in_chans = 26
        self.embed_dim = 96
        self.depths = [2, 2, 6, 2]
        self.num_heads = [3, 6, 12, 24]
        self.window_size = 7
        self.mlp_ratio = 4.0
        
        # Text encoder
        self.text_encoder_name = "openai/clip-vit-base-patch32"
        self.prompt_length = 20
        self.projection_dim = 512
        self.temperature = 0.05
        
        # Training hyperparameters
        self.num_epochs = 100
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.lambda_contrastive = 0.3
        
        # Early stopping
        self.early_stopping_patience = 15
        self.early_stopping_delta = 1e-4
        
        # Pre-training
        self.pretrain_epochs = 2000
        self.pretrain_mask_ratio = 0.75
        self.pretrain_lr = 1.5e-4
        self.pretrain_batch_size = 64
        self.pretrain_weight_decay = 0.05
