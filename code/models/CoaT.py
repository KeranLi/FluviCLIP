import torch
import torch.nn as nn
from timm.models.coat import coat_mini  # 使用 CoaT 标准模型
from models.reducer import WeightedChannelReducer


class CoaTWithReducer(nn.Module):
    def __init__(self, img_size=224, in_channels=26, out_channels=3, num_classes=1, embed_dim=768):
        super().__init__()
        # 加权降维模块
        self.reducer = WeightedChannelReducer(in_channels=in_channels, out_channels=out_channels)

        # CoaT 标准模型
        self.coat = coat_mini(  # 替换为标准 CoaT 模型，例如 coat_small, coat_base 等
            img_size=img_size,
            in_chans=out_channels,  # 降维后的通道数
            num_classes=embed_dim  # 输出维度设置为 embed_dim，用于接入 MLP
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
        # 通道降维
        x = self.reducer(x)  # [batch_size, out_channels, height, width]
        # CoaT 特征提取
        x = self.coat(x)  # [batch_size, embed_dim]
        # MLP
        x = self.mlp(x)  # [batch_size, num_classes]
        return x
