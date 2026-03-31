"""
Foundation model wrappers for remote sensing.
Includes RemoteCLIP, HyperFree, SkySense, HyperSigma, CMID, SpectralGPT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import warnings


class RemoteCLIPWrapper(nn.Module):
    """
    RemoteCLIP wrapper for SSC regression.
    RemoteCLIP is a CLIP variant pre-trained on remote sensing imagery.
    Reference: Liu et al., "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing"
    """
    def __init__(self, model_name="microsoft/remoteclip-base", num_classes=1, freeze_encoder=True):
        super().__init__()
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.available = True
        except Exception as e:
            warnings.warn(f"Could not load RemoteCLIP: {e}. Using standard CLIP instead.")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.available = False
        
        if freeze_encoder:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            for param in self.model.text_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = self.model.config.projection_dim
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images, texts=None):
        """
        Args:
            images: (B, C, H, W) - expects 3-channel RGB images
            texts: Optional text descriptions
        Returns:
            predictions: (B, num_classes)
            image_embeds: (B, embed_dim) if texts provided
            text_embeds: (B, embed_dim) if texts provided
        """
        # Ensure 3 channels
        if images.shape[1] > 3:
            images = images[:, :3, :, :]
        elif images.shape[1] < 3:
            # Pad with zeros
            padding = torch.zeros(images.shape[0], 3 - images.shape[1], 
                                 images.shape[2], images.shape[3], 
                                 device=images.device)
            images = torch.cat([images, padding], dim=1)
        
        # Get image embeddings
        vision_outputs = self.model.vision_model(pixel_values=images)
        image_embeds = vision_outputs.pooler_output
        image_features = F.normalize(self.model.visual_projection(image_embeds), dim=-1)
        
        # Regression
        predictions = self.regression_head(image_embeds)
        
        if texts is not None:
            # Get text embeddings
            text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(images.device) for k, v in text_inputs.items()}
            text_outputs = self.model.text_model(**text_inputs)
            text_embeds = text_outputs.pooler_output
            text_features = F.normalize(self.model.text_projection(text_embeds), dim=-1)
            return predictions, image_features, text_features
        
        return predictions


class FoundationModelBaseline(nn.Module):
    """
    Generic wrapper for foundation model backbones.
    Supports various architectures like SkySense, HyperSigma, etc.
    """
    def __init__(self, backbone_name, in_channels=26, num_classes=1, embed_dim=768):
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels = in_channels
        
        # Channel adapter for foundation models expecting 3 channels
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.ReLU()
        )
        
        # Placeholder for foundation model backbone
        # In practice, this would load actual pretrained weights
        self.backbone = self._create_backbone(backbone_name, embed_dim)
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    
    def _create_backbone(self, name, embed_dim):
        """
        Create backbone based on name.
        This is a placeholder - actual implementation would load pretrained weights.
        """
        # Placeholder ViT-based backbone
        from timm.models.vision_transformer import VisionTransformer
        return VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=embed_dim,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12
        )
    
    def forward(self, x):
        """Forward pass."""
        # Adapt channels
        x = self.channel_adapter(x)
        
        # Extract features
        features = self.backbone(x)
        
        # Regression
        predictions = self.regression_head(features)
        return predictions


class HyperFreeBaseline(FoundationModelBaseline):
    """
    HyperFree baseline.
    Reference: Li et al., "HyperFree: A Hyperspectral Remote Sensing Foundation Model"
    """
    def __init__(self, in_channels=26, num_classes=1):
        super().__init__("hyperfree", in_channels, num_classes, embed_dim=768)


class SkySenseBaseline(FoundationModelBaseline):
    """
    SkySense baseline.
    Reference: Guo et al., "SkySense: A Multi-Modal Foundation Model for Remote Sensing"
    """
    def __init__(self, in_channels=26, num_classes=1):
        super().__init__("skysense", in_channels, num_classes, embed_dim=1024)


class HyperSigmaBaseline(FoundationModelBaseline):
    """
    HyperSigma baseline.
    Reference: Wang et al., "HyperSigma: Foundation Model for Hyperspectral Remote Sensing"
    """
    def __init__(self, in_channels=26, num_classes=1):
        super().__init__("hypersigma", in_channels, num_classes, embed_dim=768)


class CMIDBaseline(FoundationModelBaseline):
    """
    CMID (Contrastive Masked Image Distillation) baseline.
    Reference: Muhtar et al., "CMID: A Unified Self-Supervised Learning Framework for Remote Sensing"
    """
    def __init__(self, in_channels=26, num_classes=1):
        super().__init__("cmid", in_channels, num_classes, embed_dim=768)


class SpectralGPTBaseline(FoundationModelBaseline):
    """
    SpectralGPT baseline.
    Reference: Hong et al., "SpectralGPT: Spectral Remote Sensing Foundation Model"
    """
    def __init__(self, in_channels=26, num_classes=1):
        super().__init__("spectralgpt", in_channels, num_classes, embed_dim=768)


class MultimodalVariant(nn.Module):
    """
    Generic multimodal variant combining any vision backbone with text encoder.
    Used for ablation studies on different backbone + text encoder combinations.
    """
    def __init__(self, vision_backbone, text_encoder_name="openai/clip-vit-base-patch32",
                 num_classes=1, projection_dim=512, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        
        # Vision backbone
        self.vision_backbone = vision_backbone
        
        # Determine vision output dimension
        if hasattr(vision_backbone, 'num_features'):
            vision_dim = vision_backbone.num_features
        elif hasattr(vision_backbone, 'embed_dim'):
            vision_dim = vision_backbone.embed_dim
        else:
            vision_dim = 768  # Default
        
        # Vision projection
        self.visual_projection = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        # Text encoder (frozen)
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            text_dim = self.text_encoder.config.hidden_size
        except Exception as e:
            warnings.warn(f"Could not load text encoder: {e}")
            self.tokenizer = None
            self.text_encoder = None
            text_dim = projection_dim
        
        # Text projection
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    
    def encode_image(self, images):
        """Encode images."""
        if hasattr(self.vision_backbone, 'forward_features'):
            features = self.vision_backbone.forward_features(images)
        else:
            features = self.vision_backbone(images)
        
        projected = self.visual_projection(features)
        return F.normalize(projected, p=2, dim=-1), features
    
    def encode_text(self, texts):
        """Encode texts."""
        if self.text_encoder is None:
            return None
        
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=77, return_tensors="pt")
        input_ids = encoded["input_ids"].to(next(self.parameters()).device)
        attention_mask = encoded["attention_mask"].to(next(self.parameters()).device)
        
        outputs = self.text_encoder(input_ids, attention_mask)
        text_features = outputs.pooler_output
        projected = self.text_projection(text_features)
        return F.normalize(projected, p=2, dim=-1)
    
    def forward(self, images, texts=None):
        """Forward pass."""
        # Get vision features
        if hasattr(self.vision_backbone, 'forward_features'):
            vision_features = self.vision_backbone.forward_features(images)
        else:
            vision_features = self.vision_backbone(images)
        
        # Regression
        predictions = self.regression_head(vision_features)
        
        if texts is not None:
            visual_embeds = self.encode_image(images)[0]
            text_embeds = self.encode_text(texts)
            return predictions, visual_embeds, text_embeds
        
        return predictions
