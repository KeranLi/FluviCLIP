import torch
import torch.nn as nn


class WeightedChannelReducer(nn.Module):
    """
    Weighted channel reduction module that reduces input channels from in_channels to out_channels.
    Used for adapting multi-spectral remote sensing data (26 channels) to standard vision backbones (3 channels).
    """
    def __init__(self, in_channels=26, out_channels=3):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        Returns:
            Tensor: Reduced tensor of shape [batch_size, out_channels, height, width].
        """
        batch_size, in_channels, height, width = x.shape
        # Flatten spatial dimensions, process channels only
        x = x.view(batch_size, in_channels, -1)
        # Matrix multiplication: [batch_size, out_channels, height * width]
        x = torch.matmul(self.weights, x)
        # Add bias
        x = x + self.bias.view(-1, 1)
        # Restore spatial dimensions
        x = x.view(batch_size, -1, height, width)
        return x
