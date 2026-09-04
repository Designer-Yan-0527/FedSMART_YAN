"""
模型注册模块

该模块用于注册 Vision Transformer 模型，支持 L2P (Learn to Prompt) 框架的实现。
"""

from timm.models.registry import register_model

from vision_transformer import _create_vision_transformer

__all__ = [
    'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224',
]


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Tiny 模型 (ViT-Ti/16)
    
    Args:
        pretrained: 是否加载预训练权重
        **kwargs: 额外的模型参数
    
    Returns:
        VisionTransformer: ViT-Tiny 模型实例
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Small 模型 (ViT-S/16)
    
    Args:
        pretrained: 是否加载预训练权重
        **kwargs: 额外的模型参数
    
    Returns:
        VisionTransformer: ViT-Small 模型实例
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base 模型 (ViT-B/16)
    
    来自原始论文 https://arxiv.org/abs/2010.11929
    ImageNet-1k 权重从 in21k 微调而来，源地址: https://github.com/google-research/vision_transformer
    
    Args:
        pretrained: 是否加载预训练权重
        **kwargs: 额外的模型参数
    
    Returns:
        VisionTransformer: ViT-Base 模型实例
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
