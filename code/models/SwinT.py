import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from models.reducer import WeightedChannelReducer

class SwinTransformerWithReducer(nn.Module):
    def __init__(self, img_size=224, in_channels=26, out_channels=3, num_classes=1, embed_dim=768, patch_size=4, window_size=7):
        super().__init__()
        # 加权降维模块
        self.reducer = WeightedChannelReducer(in_channels=in_channels, out_channels=out_channels)

        # Swin Transformer
        self.swin = SwinTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=out_channels,  # 降维后通道数
            embed_dim=embed_dim,
            window_size=window_size,
            num_classes=embed_dim,  # 输出维度设置为 embed_dim，用于接入 MLP
        )

        # MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 降维
        x = self.reducer(x)  # [batch_size, out_channels, height, width]
        # Swin Transformer
        x = self.swin(x)  # [batch_size, embed_dim]
        # MLP
        x = self.mlp(x)  # [batch_size, num_classes]
        return x
