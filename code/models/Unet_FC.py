import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithFC(nn.Module):
    def __init__(self, num_features):
        super(UNetWithFC, self).__init__()
        self.encoder1 = self.conv_block(26, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.decoder4 = self.upconv_block(512, 256)
        self.decoder3 = self.upconv_block(256, 128)
        self.decoder2 = self.upconv_block(128, 64)
        self.decoder1 = self.upconv_block(64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_features)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(dec4 + enc3)
        dec2 = self.decoder2(dec3 + enc2)
        dec1 = self.decoder1(dec2 + enc1)
        out = self.final_conv(dec1)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
