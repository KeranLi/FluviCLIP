# FluviCLIP models package
from .reducer import WeightedChannelReducer
from .fluviformer import FluviFormer
from .gated_head import GatedDualBranchHead
from .fluviclip import FluviCLIP, SoftPromptEncoder
from .mae import MaskedAutoencoder

# Baseline CNN models
from .resnet import (
    ResNet50Regressor, 
    Res2Net, 
    ResNeXt50, 
    MultimodalResNet50
)

# UNet models
from .Unet2D import UNet2D, RemoteSensingRegressionModel
from .Unet3D import UNet3D, RemoteSensingRegressionModel3D
from .Unet_FC import UNetWithFC

# Transformer models
from .SwinT import SwinTransformerWithReducer
from .ViT import VisionTransformerWithReducer
from .CoaT import CoaTWithReducer
from .DeiT import DeiTModel

# Sequence models
from .sequence_models import (
    LSTMRegressor,
    GRURegressor,
    CNNLSTM,
    PureFF
)

# Machine learning baselines
from .ml_baselines import (
    SVMBaseline,
    LightGBMBaseline,
    XGBoostBaseline,
    NDWIEmpirical,
    NIRRedRatio,
    MLModelWrapper
)

# Foundation models
from .foundation_models import (
    RemoteCLIPWrapper,
    HyperFreeBaseline,
    SkySenseBaseline,
    HyperSigmaBaseline,
    CMIDBaseline,
    SpectralGPTBaseline,
    MultimodalVariant
)

# Distillation
from .distillation import (
    LightweightStudent,
    KnowledgeDistillationLoss,
    DistillationTrainer
)

__all__ = [
    # Core models
    "WeightedChannelReducer",
    "FluviFormer",
    "GatedDualBranchHead",
    "FluviCLIP",
    "SoftPromptEncoder",
    "MaskedAutoencoder",
    
    # CNN baselines
    "ResNet50Regressor",
    "Res2Net",
    "ResNeXt50",
    "MultimodalResNet50",
    
    # UNet
    "UNet2D",
    "UNet3D",
    "RemoteSensingRegressionModel",
    "RemoteSensingRegressionModel3D",
    "UNetWithFC",
    
    # Transformers
    "SwinTransformerWithReducer",
    "VisionTransformerWithReducer",
    "CoaTWithReducer",
    "DeiTModel",
    
    # Sequence models
    "LSTMRegressor",
    "GRURegressor",
    "CNNLSTM",
    "PureFF",
    
    # ML baselines
    "SVMBaseline",
    "LightGBMBaseline",
    "XGBoostBaseline",
    "NDWIEmpirical",
    "NIRRedRatio",
    "MLModelWrapper",
    
    # Foundation models
    "RemoteCLIPWrapper",
    "HyperFreeBaseline",
    "SkySenseBaseline",
    "HyperSigmaBaseline",
    "CMIDBaseline",
    "SpectralGPTBaseline",
    "MultimodalVariant",
    
    # Distillation
    "LightweightStudent",
    "KnowledgeDistillationLoss",
    "DistillationTrainer",
]
