import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from models.fluviformer import FluviFormer
from models.gated_head import GatedDualBranchHead


class SoftPromptEncoder(nn.Module):
    """
    Soft prompt tuning module that prepends learnable tokens to the frozen text encoder.
    """
    def __init__(self, text_encoder, prompt_length=20, embed_dim=512):
        super().__init__()
        self.text_encoder = text_encoder
        # Freeze the text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Learnable soft prompt tokens
        self.soft_prompt = nn.Parameter(torch.randn(1, prompt_length, embed_dim) * 0.02)
        self.prompt_length = prompt_length
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (B, L) token ids from tokenizer
            attention_mask: (B, L) attention mask
        Returns:
            text_features: (B, embed_dim) pooled text embeddings
        """
        B = input_ids.shape[0]
        
        # Get text embeddings from the frozen encoder's embedding layer
        inputs_embeds = self.text_encoder.text_model.embeddings.token_embedding(input_ids)
        
        # Prepend soft prompts
        soft_prompt = self.soft_prompt.expand(B, -1, -1)
        inputs_embeds = torch.cat([soft_prompt, inputs_embeds], dim=1)
        
        # Adjust attention mask for soft prompts
        if attention_mask is not None:
            prompt_mask = torch.ones((B, self.prompt_length), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # Forward through the frozen text encoder
        outputs = self.text_encoder.text_model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use the pooled output (similar to CLIP's default behavior)
        last_hidden_state = outputs.last_hidden_state
        # CLIP takes the output at the position of the EOS token for pooling
        # For simplicity, we use mean pooling over the sequence
        text_features = last_hidden_state.mean(dim=1)
        
        # L2 normalize for contrastive learning
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features


class FluviCLIP(nn.Module):
    """
    FluviCLIP: Multimodal contrastive learning framework for SSC estimation.
    
    Architecture:
        - FluviFormer as the visual backbone
        - Frozen RemoteCLIP/CLIP text encoder with soft prompt tuning
        - Gated Dual-Branch Regression Head for decoupled head/tail optimization
        - Joint contrastive + regression objective
    """
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=26,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 text_encoder_name="openai/clip-vit-base-patch32",
                 prompt_length=20,
                 text_embed_dim=512,
                 projection_dim=512,
                 temperature=0.05,
                 lambda_contrastive=0.3,
                 use_ndwi_mask=True):
        super().__init__()
        
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.text_embed_dim = text_embed_dim
        self.projection_dim = projection_dim
        
        # Visual backbone: FluviFormer
        self.visual_encoder = FluviFormer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_ndwi_mask=use_ndwi_mask
        )
        
        visual_dim = int(embed_dim * 2 ** (len(depths) - 1))
        
        # Visual projection head for contrastive alignment
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        # Text encoder (frozen) with soft prompt tuning
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_name)
            text_encoder = CLIPTextModel.from_pretrained(text_encoder_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load text encoder '{text_encoder_name}'. "
                f"Please ensure the model name is correct and transformers library is installed. Error: {e}"
            )
        
        # Adjust text_embed_dim if the loaded model has a different dimension
        actual_text_dim = text_encoder.config.hidden_size
        if actual_text_dim != text_embed_dim:
            text_embed_dim = actual_text_dim
            self.text_embed_dim = actual_text_dim
        
        self.text_encoder = SoftPromptEncoder(text_encoder, prompt_length=prompt_length, embed_dim=text_embed_dim)
        
        # Text projection head for contrastive alignment
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        
        # Gated Dual-Branch Regression Head
        self.regression_head = GatedDualBranchHead(in_dim=visual_dim, hidden_dim=512, num_classes=1, dropout=0.25)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def encode_image(self, images):
        """
        Encode images through the visual backbone and projection head.
        Args:
            images: (B, C, H, W)
        Returns:
            visual_features: (B, projection_dim) L2-normalized visual embeddings
            visual_pooled: (B, visual_dim) raw pooled visual features for regression
        """
        visual_pooled = self.visual_encoder(images)  # (B, visual_dim)
        visual_features = self.visual_projection(visual_pooled)
        visual_features = F.normalize(visual_features, p=2, dim=-1)
        return visual_features, visual_pooled
    
    def encode_text(self, texts):
        """
        Encode text descriptions through the soft-prompt text encoder and projection head.
        Args:
            texts: list of str, batch of text descriptions
        Returns:
            text_features: (B, projection_dim) L2-normalized text embeddings
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"].to(next(self.parameters()).device)
        attention_mask = encoded["attention_mask"].to(next(self.parameters()).device)
        
        text_features = self.text_encoder(input_ids, attention_mask)  # (B, text_embed_dim)
        text_features = self.text_projection(text_features)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features
    
    def forward(self, images, texts=None):
        """
        Args:
            images: (B, C, H, W)
            texts: list of str or None
        Returns:
            If texts is provided (training mode):
                ssc_pred: (B, 1) predicted SSC values
                gate: (B, 1) gating weight
                head_out: (B, 1) head expert output
                tail_out: (B, 1) tail expert output
                visual_features: (B, projection_dim)
                text_features: (B, projection_dim)
            If texts is None (inference mode):
                ssc_pred: (B, 1)
                gate: (B, 1)
        """
        visual_features, visual_pooled = self.encode_image(images)
        ssc_pred, gate, head_out, tail_out = self.regression_head(visual_pooled)
        
        if texts is not None:
            text_features = self.encode_text(texts)
            return ssc_pred, gate, head_out, tail_out, visual_features, text_features
        
        return ssc_pred, gate
    
    def compute_loss(self, ssc_pred, gate, head_out, tail_out, visual_features, text_features, targets):
        """
        Compute the joint training objective:
            L_total = lambda * L_SSC + (1 - lambda) * L_Contrastive
        
        Args:
            ssc_pred: (B, 1) final SSC predictions
            gate: (B, 1) gating weights
            head_out: (B, 1) head expert predictions
            tail_out: (B, 1) tail expert predictions
            visual_features: (B, projection_dim)
            text_features: (B, projection_dim)
            targets: (B, 1) ground-truth SSC values
        Returns:
            dict of losses
        """
        # SSC regression loss (MSE on final prediction + auxiliary losses on experts)
        mse_loss = F.mse_loss(ssc_pred, targets)
        head_loss = F.mse_loss(head_out, targets)
        tail_loss = F.mse_loss(tail_out, targets)
        ssc_loss = mse_loss + 0.1 * head_loss + 0.1 * tail_loss
        
        # Contrastive loss (InfoNCE)
        logits = torch.matmul(visual_features, text_features.t()) / self.temperature  # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        contrastive_loss = (loss_i2t + loss_t2i) / 2.0
        
        # Total loss
        total_loss = self.lambda_contrastive * ssc_loss + (1.0 - self.lambda_contrastive) * contrastive_loss
        
        return {
            "total_loss": total_loss,
            "ssc_loss": ssc_loss,
            "contrastive_loss": contrastive_loss,
            "mse_loss": mse_loss,
            "head_loss": head_loss,
            "tail_loss": tail_loss,
        }
