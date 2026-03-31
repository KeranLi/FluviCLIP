"""
ResNet-based models for SSC regression.
Includes ResNet50, Res2Net, and ResNeXt variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class WeightedChannelReducer(nn.Module):
    """Channel reduction from multi-spectral to 3 channels."""
    def __init__(self, in_channels=26, out_channels=3):
        super().__init__()
        self.reducer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reducer(x)


class ResNet50Regressor(nn.Module):
    """
    ResNet50-based regressor for SSC estimation.
    Used as a baseline in the comparison experiments.
    """
    def __init__(self, in_channels=26, num_classes=1, pretrained=False):
        super().__init__()
        self.reducer = WeightedChannelReducer(in_channels, 3)
        self.backbone = resnet50(pretrained=pretrained)
        
        # Replace final FC layer for regression
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.reducer(x)
        x = self.backbone(x)
        return x


class Res2NetBottleneck(nn.Module):
    """
    Res2Net bottleneck block with scale dimension.
    Splits features into multiple scales for richer representation.
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, scale=4):
        super().__init__()
        self.scale = scale
        self.width = channels // scale
        
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 3x3 convs for each scale
        self.convs = nn.ModuleList([
            nn.Conv2d(self.width, self.width, kernel_size=3, stride=stride if i == 0 else 1,
                     padding=1, bias=False)
            for i in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in range(scale - 1)])
        
        # 1x1 conv
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 1x1 conv
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Split into scales
        spx = torch.split(out, self.width, dim=1)
        
        # Process each scale
        for i in range(self.scale - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), dim=1)
        
        # Append last scale
        out = torch.cat((out, spx[self.scale - 1]), dim=1)

        # 1x1 conv
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    """
    Res2Net backbone for SSC estimation.
    Multi-scale features improve representation of sediment variations.
    """
    def __init__(self, in_channels=26, layers=[3, 4, 6, 3], num_classes=1, scale=4):
        super().__init__()
        self.in_channels = 64
        self.scale = scale
        
        self.reducer = WeightedChannelReducer(in_channels, 3)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * Res2NetBottleneck.expansion, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * Res2NetBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * Res2NetBottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * Res2NetBottleneck.expansion),
            )

        layers = []
        layers.append(Res2NetBottleneck(self.in_channels, channels, stride, downsample, self.scale))
        self.in_channels = channels * Res2NetBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Res2NetBottleneck(self.in_channels, channels, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.reducer(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNeXtBottleneck(nn.Module):
    """
    ResNeXt bottleneck with cardinality (grouped convolutions).
    Aggregates residual transformations for better feature learning.
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None, cardinality=32):
        super().__init__()
        self.cardinality = cardinality
        self.width = channels // cardinality
        
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride,
                              padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNeXt50(nn.Module):
    """
    ResNeXt-50 (32x4d) for SSC estimation.
    Uses grouped convolutions for improved accuracy.
    """
    def __init__(self, in_channels=26, layers=[3, 4, 6, 3], num_classes=1, cardinality=32):
        super().__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        
        self.reducer = WeightedChannelReducer(in_channels, 3)
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(128, layers[0])
        self.layer2 = self._make_layer(256, layers[1], stride=2)
        self.layer3 = self._make_layer(512, layers[2], stride=2)
        self.layer4 = self._make_layer(1024, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024 * ResNeXtBottleneck.expansion, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * ResNeXtBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * ResNeXtBottleneck.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * ResNeXtBottleneck.expansion),
            )

        layers = []
        layers.append(ResNeXtBottleneck(self.in_channels, channels, stride, downsample, self.cardinality))
        self.in_channels = channels * ResNeXtBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(ResNeXtBottleneck(self.in_channels, channels, cardinality=self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.reducer(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class MultimodalResNet50(nn.Module):
    """
    ResNet50 with multimodal contrastive learning.
    Combines visual features with text embeddings from CLIP.
    """
    def __init__(self, in_channels=26, num_classes=1, text_encoder_name="openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        
        self.reducer = WeightedChannelReducer(in_channels, 3)
        self.backbone = resnet50(pretrained=False)
        
        # Visual feature dimension
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Projection head for visual features
        self.visual_projection = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LayerNorm(512),
        )
        
        # Text encoder (frozen)
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        except Exception as e:
            print(f"Warning: Could not load text encoder: {e}")
            self.tokenizer = None
            self.text_encoder = None
        
        self.text_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
        
        self.temperature = 0.05

    def forward(self, x, texts=None):
        x = self.reducer(x)
        visual_features = self.backbone(x)
        visual_embeds = F.normalize(self.visual_projection(visual_features), p=2, dim=-1)
        
        ssc_pred = self.regression_head(visual_features)
        
        if texts is not None and self.text_encoder is not None:
            # Encode text
            encoded = self.tokenizer(texts, padding=True, truncation=True, 
                                    max_length=77, return_tensors="pt")
            input_ids = encoded["input_ids"].to(x.device)
            attention_mask = encoded["attention_mask"].to(x.device)
            
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_features = text_outputs.pooler_output
            text_embeds = F.normalize(self.text_projection(text_features), p=2, dim=-1)
            
            return ssc_pred, visual_embeds, text_embeds
        
        return ssc_pred
