# Configuration files for different models
from .FluviCLIP import Config as FluviCLIPConfig
from .SwinT import Config as SwinTConfig
from .ViT import Config as ViTConfig
from .CoaT import Config as CoaTConfig
from .Unet2D import Config as Unet2DConfig
from .Unet3D import Config as Unet3DConfig
from .DeiT import Config as DeiTConfig
from .ResNet50 import Config as ResNet50Config
from .Res2Net import Config as Res2NetConfig
from .ResNeXt import Config as ResNeXtConfig

__all__ = [
    "FluviCLIPConfig",
    "SwinTConfig",
    "ViTConfig",
    "CoaTConfig",
    "Unet2DConfig",
    "Unet3DConfig",
    "DeiTConfig",
    "ResNet50Config",
    "Res2NetConfig",
    "ResNeXtConfig",
]
