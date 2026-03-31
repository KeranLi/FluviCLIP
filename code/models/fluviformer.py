import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """
    Patch embedding layer that splits an image into patches and projects them into embeddings.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=26, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN block with interleaved standard and dilated convolutions.
    Captures fine-grained local spatial-spectral contexts.
    """
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        
        # Interleaved standard and dilated convolutions with 3x3 kernels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv(x)


class FluvialAttention(nn.Module):
    """
    Fluvial Attention module with relative position bias tailored for river networks.
    """
    def __init__(self, dim, num_heads=8, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position bias table for fluvial attention
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B_, N, C) where N = window_size * window_size
            mask: (B_, N, N) or None, attention mask for fluvial regions
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, num_heads, N, head_dim)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )  # (Wh*Ww, Wh*Ww, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        # Apply fluvial mask if provided
        if mask is not None:
            attn = attn + mask.unsqueeze(1)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (B * num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition to reconstruct feature map.
    Args:
        windows: (B * num_windows, window_size, window_size, C)
        window_size: int
        H, W: int
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def compute_ndwi_mask(x, green_band=2, nir_band=7, threshold=0.3):
    """
    Compute NDWI-based fluvial mask from multi-spectral input.
    Args:
        x: (B, C, H, W) input image
        green_band: index of green band (default 2 for Sentinel-2-like data)
        nir_band: index of near-infrared band (default 7)
        threshold: NDWI threshold for water detection
    Returns:
        mask: (B, 1, H, W) binary mask
    """
    green = x[:, green_band:green_band+1, :, :]
    nir = x[:, nir_band:nir_band+1, :, :]
    # Avoid division by zero
    denom = green + nir
    denom = torch.where(denom == 0, torch.ones_like(denom) * 1e-6, denom)
    ndwi = (green - nir) / denom
    mask = (ndwi > threshold).float()
    return mask


class FluvialSwinShift(nn.Module):
    """
    Fluvial Swing Shift (FSS) module.
    Implements anisotropic directional search (East, South, Southeast) guided by NDWI mask.
    """
    def __init__(self, window_size=7, shift_direction='se'):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.shift_direction = shift_direction  # 'e', 's', or 'se'
    
    def forward(self, x, ndwi_mask=None):
        """
        Args:
            x: (B, H, W, C)
            ndwi_mask: (B, 1, H, W) or None
        Returns:
            shifted_x: (B, H, W, C)
            attn_mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, H, W, C = x.shape
        
        # Determine shift based on direction
        if self.shift_direction == 'e':
            shift_h, shift_w = 0, self.shift_size
        elif self.shift_direction == 's':
            shift_h, shift_w = self.shift_size, 0
        else:  # 'se' default
            shift_h, shift_w = self.shift_size, self.shift_size
        
        # Cyclic shift (only positive shifts: East, South, Southeast)
        shifted_x = torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))
        
        # Generate attention mask for fluvial regions
        attn_mask = None
        if ndwi_mask is not None:
            # Downsample mask to match feature resolution
            mask_h, mask_w = ndwi_mask.shape[2], ndwi_mask.shape[3]
            if mask_h != H or mask_w != W:
                ndwi_mask = F.interpolate(ndwi_mask, size=(H, W), mode='nearest')
            ndwi_mask = ndwi_mask.squeeze(1)  # (B, H, W)
            
            # Apply same cyclic shift to mask
            shifted_mask = torch.roll(ndwi_mask, shifts=(shift_h, shift_w), dims=(1, 2))
            
            # Partition mask into windows
            mask_windows = window_partition(shifted_mask.unsqueeze(-1), self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # (B*num_windows, Wh*Ww)
            
            # Compute attention mask: non-fluvial patches get large negative values
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (B*num_windows, Wh*Ww, Wh*Ww)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return shifted_x, attn_mask


class FluviFormerBlock(nn.Module):
    """
    FluviFormer block combining Fluvial Attention, MLP, and Multi-scale CNN.
    Equation:
        H = F + A(LN(F))
        F_att = H + M(LN(H))
        F_out = F_att + C(F_att)
    """
    def __init__(self, dim, num_heads=8, window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., shift_direction='se'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_direction = shift_direction
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FluvialAttention(dim, num_heads, window_size, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cnn = MultiScaleCNN(dim)
        
        self.fss = FluvialSwinShift(window_size, shift_direction)
    
    def forward(self, x, H, W, ndwi_mask=None):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
            ndwi_mask: (B, 1, H_orig, W_orig) or None
        Returns:
            x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, "Input feature length must match H * W"
        
        # Fluvial Attention with residual
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Apply FSS
        shifted_x, attn_mask = self.fss(x, ndwi_mask)
        
        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*num_windows, Wh, Ww, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (B*num_windows, Wh*Ww, C)
        
        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Reverse windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
        
        # Reverse cyclic shift
        if self.shift_direction == 'e':
            shift_h, shift_w = 0, -self.window_size // 2
        elif self.shift_direction == 's':
            shift_h, shift_w = -self.window_size // 2, 0
        else:
            shift_h, shift_w = -self.window_size // 2, -self.window_size // 2
        x = torch.roll(shifted_x, shifts=(shift_h, shift_w), dims=(1, 2))
        
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        # Multi-scale CNN with residual
        shortcut2 = x
        x = self.norm3(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.cnn(x)
        x = x.flatten(2).transpose(1, 2)
        x = shortcut2 + x
        
        return x


class FluviFormerStage(nn.Module):
    """
    A FluviFormer stage containing multiple FluviFormer blocks.
    """
    def __init__(self, dim, depth, num_heads=8, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., downsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            FluviFormerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                shift_direction='se' if (i % 2 == 1) else None
            )
            for i in range(depth)
        ])
        self.downsample = downsample
    
    def forward(self, x, H, W, ndwi_mask=None):
        for blk in self.blocks:
            x = blk(x, H, W, ndwi_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x, H, W


class PatchMerging(nn.Module):
    """
    Patch merging layer for downsampling in hierarchical transformers.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "Input feature length must match H * W"
        
        x = x.view(B, H, W, C)
        
        # Pad if H or W is odd
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # (B, H/2, W/2, 4*C)
        
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        
        return x


class FluviFormer(nn.Module):
    """
    FluviFormer: A hierarchical Vision Transformer with fluvial morphological priors.
    Integrates Fluvial Swing Shift (FSS), Fluvial Attention, and Multi-scale CNN blocks.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=26, embed_dim=96,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 num_classes=1, use_ndwi_mask=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.use_ndwi_mask = use_ndwi_mask
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Positional embedding
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            depth = depths[i_layer]
            heads = num_heads[i_layer]
            
            downsample = PatchMerging(dim=dim) if (i_layer < self.num_layers - 1) else None
            stage = FluviFormerStage(
                dim=dim,
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                downsample=downsample
            )
            self.layers.append(stage)
        
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_ndwi_mask=False):
        """
        Args:
            x: (B, C, H, W) input multi-spectral image
            return_ndwi_mask: whether to return the computed NDWI mask
        Returns:
            x: (B, embed_dim_final) pooled feature vector
            ndwi_mask: (B, 1, H, W) if return_ndwi_mask=True
        """
        # Compute NDWI mask from input if enabled
        ndwi_mask = None
        if self.use_ndwi_mask:
            ndwi_mask = compute_ndwi_mask(x, green_band=2, nir_band=7, threshold=0.3)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        x = self.pos_drop(x)
        
        H = W = self.img_size // self.patch_size
        
        # Hierarchical stages
        for layer in self.layers:
            x, H, W = layer(x, H, W, ndwi_mask)
        
        x = self.norm(x)  # (B, H*W, C)
        x = x.transpose(1, 2)  # (B, C, H*W)
        x = self.avgpool(x)  # (B, C, 1)
        x = x.flatten(1)  # (B, C)
        
        if return_ndwi_mask:
            return x, ndwi_mask
        return x
