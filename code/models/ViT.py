import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from models.reducer import WeightedChannelReducer

class VisionTransformerWithReducer(nn.Module):
    def __init__(self, img_size=224, in_channels=26, out_channels=3, num_classes=1, embed_dim=768, patch_size=16):
        super().__init__()
        # 加权降维模块
        self.reducer = WeightedChannelReducer(in_channels=in_channels, out_channels=out_channels)
        # 标准 ViT
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=out_channels,  # 降维后为 3 通道
            num_classes=embed_dim,  # 输出维度设置为 embed_dim，用于接入 MLP
            embed_dim=embed_dim,
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
        #print(f"Input shape: {x.shape}")  # [batch_size, 26, height, width]
        x = self.reducer(x)
        #print(f"Reduced shape: {x.shape}")  # [batch_size, 3, height, width]
        x = self.vit(x)  # [batch_size, embed_dim]
        #print(f"Transformer output shape: {x.shape}")
        x = self.mlp(x)  # [batch_size, num_classes]
        return x
