import torch


class Config:
    """Configuration for ResNeXt-50 (32x4d) baseline training."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = "datasets/train"
        self.excel_file = "data.xlsx"
        self.image_dir = "images/"
        self.sheet_name = "Sheet1"
        self.output_dir = "output/ResNeXt50"
        self.log_dir = "runs/ResNeXt50"
        self.checkpoint_path = "best_model.pth"
        
        # Model architecture
        self.img_size = 224
        self.in_channels = 26
        self.num_classes = 1
        self.layers = [3, 4, 6, 3]  # ResNet-50 layers
        self.cardinality = 32  # ResNeXt cardinality
        
        # Training hyperparameters
        self.num_epochs = 100
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        
        # Early stopping
        self.early_stopping_patience = 15
        self.early_stopping_delta = 1e-4
