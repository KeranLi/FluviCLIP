import torch
import torch.nn as nn

class WeightedChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reducer(x)

class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # 只对高度和宽度进行池化
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                          diffY // 2, diffY - diffY // 2,
                                          diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv3D(in_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        self.down4 = Down3D(512, 1024)
        self.up1 = Up3D(1024, 512)
        self.up2 = Up3D(512, 256)
        self.up3 = Up3D(256, 128)
        self.up4 = Up3D(128, 64)
        self.outc = OutConv3D(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class RemoteSensingRegressionModel3D(nn.Module):
    def __init__(self, img_size=224, in_channels=26, out_channels=3, num_classes=1):  # 修改 in_channels 为 26
        super().__init__()
        # 加权降维模块
        self.reducer = WeightedChannelReducer(in_channels=in_channels, out_channels=out_channels)

        # 3D U-Net 部分
        self.unet3d = UNet3D(in_channels=out_channels, out_channels=out_channels)

        # 回归头
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(8, num_classes)
        )

    def forward(self, x):
        # 添加深度维度，将形状从 (batch_size, channels, height, width) 转换为 (batch_size, channels, 1, height, width)
        x = x.unsqueeze(2)
        #print("Input shape after adding depth dimension:", x.shape)  # 打印添加维度后的形状

        # 降维
        x = self.reducer(x)
        #print("After reducer shape:", x.shape)  # 打印降维后形状

        # 3D U-Net 部分
        x = self.unet3d(x)
        #print("After UNet3D shape:", x.shape)  # 打印 UNet3D 后形状

        # 回归头
        x = self.regression_head(x)
        #print("After regression head shape:", x.shape)  # 打印回归头后形状

        return x