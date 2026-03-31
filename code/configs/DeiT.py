import torch


class Config:
    """Configuration for DeiT training."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = "datasets/train"
        self.excel_file = "data.xlsx"
        self.image_dir = "images/"
        self.sheet_name = "Sheet1"
        self.output_dir = "output/DeiT"
        self.log_dir = "runs/DeiT"
        self.checkpoint_path = "best_model.pth"
        
        self.img_size = 224
        self.in_channels = 26
        self.out_channels = 3
        self.num_classes = 1
        
        self.num_epochs = 100
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.early_stopping_patience = 15
        self.early_stopping_delta = 1e-4
