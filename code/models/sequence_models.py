"""
Sequence models (LSTM, GRU) for SSC prediction using temporal sequences.
These models require auxiliary hydrometeorological data (rainfall-runoff).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMRegressor(nn.Module):
    """
    LSTM-based sequence model for SSC prediction.
    Takes temporal sequences of hydrometeorological features.
    
    Note: This requires auxiliary temporal data (e.g., rainfall, discharge)
    which may not be available in remote or data-scarce regions.
    """
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, 
                 num_classes=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size) - temporal sequences
        Returns:
            output: (batch_size, num_classes) - SSC predictions
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        output = self.fc(hidden)
        return output


class GRURegressor(nn.Module):
    """
    GRU-based sequence model for SSC prediction.
    More efficient than LSTM with fewer parameters.
    """
    def __init__(self, input_size=10, hidden_size=128, num_layers=2,
                 num_classes=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_size) - temporal sequences
        Returns:
            output: (batch_size, num_classes) - SSC predictions
        """
        # GRU forward
        gru_out, hidden = self.gru(x)
        
        # Use last hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        output = self.fc(hidden)
        return output


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model.
    Extracts spatial features with CNN and temporal dependencies with LSTM.
    Suitable for satellite image time series.
    """
    def __init__(self, in_channels=26, hidden_size=128, num_layers=2, num_classes=1):
        super().__init__()
        
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, C, H, W) - image time series
        Returns:
            output: (batch_size, num_classes) - SSC predictions
        """
        batch_size, seq_len, C, H, W = x.shape
        
        # Process each frame with CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            feat = self.cnn(frame)
            cnn_features.append(feat.view(batch_size, -1))
        
        # Stack CNN features: (batch_size, seq_len, 256)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # Use last hidden state
        output = self.fc(hidden[-1])
        return output


class PureFF(nn.Module):
    """
    Pure-FF: FluviFormer backbone without multimodal pretraining.
    Used as a temporal-agnostic baseline in LOSO validation.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=26, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                 num_classes=1):
        super().__init__()
        from models.fluviformer import FluviFormer
        
        self.backbone = FluviFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            num_classes=num_classes,
            use_ndwi_mask=True
        )
        
        # Replace the pooling with a regression head
        visual_dim = int(embed_dim * 2 ** (len(depths) - 1))
        self.regression_head = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.regression_head(features)


class TemporalDataset(torch.utils.data.Dataset):
    """
    Dataset for temporal sequence models.
    Creates sequences from consecutive time steps.
    """
    def __init__(self, image_paths, labels, seq_length=7, means=None, stds=None):
        self.image_paths = image_paths
        self.labels = labels
        self.seq_length = seq_length
        self.means = means
        self.stds = stds
        self.label_mean = np.mean(labels)
        self.label_std = np.std(labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        from osgeo import gdal
        import numpy as np
        
        # Load current and previous frames to form a sequence
        # For simplicity, we use the same image replicated
        # In practice, this should load consecutive time steps
        image_path = self.image_paths[idx]
        image_ds = gdal.Open(image_path)
        
        image = []
        for b in range(1, image_ds.RasterCount + 1):
            band = image_ds.GetRasterBand(b)
            image.append(band.ReadAsArray())
        image = np.stack(image, axis=0).astype(np.float32)
        
        # Normalize
        if self.means is not None and self.stds is not None:
            for i in range(image.shape[0]):
                if self.stds[i] == 0:
                    self.stds[i] = 1e-6
                image[i] = (image[i] - self.means[i]) / self.stds[i]
        
        # Convert to tensor and resize
        image = torch.from_numpy(image).float()
        image = F.interpolate(image.unsqueeze(0), size=(224, 224), 
                             mode='bilinear', align_corners=False).squeeze(0)
        
        # Create sequence by replicating (placeholder for actual temporal data)
        sequence = image.unsqueeze(0).repeat(self.seq_length, 1, 1, 1)
        
        label = self.labels[idx]
        label = (label - self.label_mean) / self.label_std
        label = torch.tensor(label, dtype=torch.float32)
        
        return sequence, label
