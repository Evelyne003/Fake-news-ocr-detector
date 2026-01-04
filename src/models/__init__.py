# Modelos para detección de noticias falsas

from .text_branch import (
    TFIDFTextEncoder,
    TextBranchMLP,
    TextOnlyClassifier,
    BaselineTextClassifier,
    create_text_model
)

from .visual_branch import (
    VGG16FeatureExtractor,
    VisualBranchMLP,
    VisualOnlyClassifier,
    VGG16EndToEnd,
    create_visual_model
)

from .fusion_model import (
    MultimodalFusionModel,
    EarlyFusionModel,
    CrossModalAttentionFusion,
    create_multimodal_model
)

__all__ = [
    # Text modelos
    'TFIDFTextEncoder',
    'TextBranchMLP',
    'TextOnlyClassifier',
    'BaselineTextClassifier',
    'create_text_model',
    
    # Visual modelos
    'VGG16FeatureExtractor',
    'VisualBranchMLP',
    'VisualOnlyClassifier',
    'VGG16EndToEnd',
    'create_visual_model',
    
    # Multimodal modelos
    'MultimodalFusionModel',
    'EarlyFusionModel',
    'CrossModalAttentionFusion',
    'create_multimodal_model'
]