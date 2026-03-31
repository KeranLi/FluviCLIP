import torch
import torch.nn as nn
from timm.models.deit import deit_small_patch16_224
from models.reducer import WeightedChannelReducer

class DeiTModel(nn.Module):
    def __init__(self, img_size=224, in_channels=26, out_channels=3, num_classes=1):
        super().__init__()
        # 加权降维模块
        self.reducer = WeightedChannelReducer(in_channels=in_channels, out_channels=out_channels)

        # 使用 DeiT 模型
        self.deit = deit_small_patch16_224(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        # 降维
        x = self.reducer(x)
        # 使用 DeiT 模型进行前向传播
        x = self.deit(x)
        return x