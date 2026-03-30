"""
PyTorch network architectures (ResNet-18, ViT-Small, CCT-7)
"""
import torch
import torch.nn as nn
from typing import Literal, Optional
from pathlib import Path

NetworkType = Literal[
    "resnet18",
    "resnet34",
    "resnet50",
    "vgg16",
    "vgg19",
    "vit_base",
    "vit_small",
    "efficientnet_b0",
    "efficientnet_b1",
    "densenet121",
    "densenet169",
    "densenet201",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "convnext_tiny",
    "convnext_small",
    "inception_v3",
    "swin_tiny",
    "swin_small",
    "maxvit_t",
    "cct_7",
    "unet",
]


def create_network(
    network_type: NetworkType,
    num_classes: int = 10,
    pretrained: bool = False,
    pretrained_weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Create a neural network model by architecture type.

    Args:
        network_type: Type of network architecture
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (ImageNet)
        pretrained_weights_path: Path to experiment-specific pretrained weights.
            If provided, loads from this path instead of torchvision cache.

    Returns:
        PyTorch model
    """
    import torchvision.models as models

    if network_type == "resnet18":
        if pretrained_weights_path:
            model = models.resnet18(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif network_type == "resnet34":
        if pretrained_weights_path:
            model = models.resnet34(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif network_type == "resnet50":
        if pretrained_weights_path:
            model = models.resnet50(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif network_type == "vgg16":
        if pretrained_weights_path:
            model = models.vgg16(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif network_type == "vgg19":
        if pretrained_weights_path:
            model = models.vgg19(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    elif network_type == "vit_base":
        if pretrained_weights_path:
            model = models.vit_b_16(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif network_type == "vit_small":
        from torchvision.models.vision_transformer import VisionTransformer
        if pretrained_weights_path:
            model = VisionTransformer(
                image_size=224, patch_size=16,
                num_layers=12, num_heads=6,
                hidden_dim=384, mlp_dim=1536,
            )
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            # No pretrained ImageNet weights available for ViT-Small in torchvision
            model = VisionTransformer(
                image_size=224, patch_size=16,
                num_layers=12, num_heads=6,
                hidden_dim=384, mlp_dim=1536,
            )
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif network_type == "cct_7":
        try:
            from .cct import CCT
        except ImportError:
            from cct import CCT
        model = CCT(
            num_classes=num_classes,
            embed_dim=256,
            n_conv_layers=1,
            n_transformer_layers=7,
            n_heads=4,
            mlp_ratio=2,
            dropout=0.1,
        )
        if pretrained_weights_path:
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)

    elif network_type == "efficientnet_b0":
        if pretrained_weights_path:
            model = models.efficientnet_b0(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif network_type == "efficientnet_b1":
        if pretrained_weights_path:
            model = models.efficientnet_b1(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.efficientnet_b1(
                weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif network_type == "densenet121":
        if pretrained_weights_path:
            model = models.densenet121(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif network_type == "densenet169":
        if pretrained_weights_path:
            model = models.densenet169(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.densenet169(
                weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif network_type == "densenet201":
        if pretrained_weights_path:
            model = models.densenet201(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.densenet201(
                weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif network_type == "mobilenet_v2":
        if pretrained_weights_path:
            model = models.mobilenet_v2(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.mobilenet_v2(
                weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif network_type == "mobilenet_v3_small":
        if pretrained_weights_path:
            model = models.mobilenet_v3_small(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif network_type == "mobilenet_v3_large":
        if pretrained_weights_path:
            model = models.mobilenet_v3_large(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif network_type == "convnext_tiny":
        if pretrained_weights_path:
            model = models.convnext_tiny(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif network_type == "convnext_small":
        if pretrained_weights_path:
            model = models.convnext_small(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.convnext_small(
                weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    elif network_type == "inception_v3":
        if pretrained_weights_path:
            model = models.inception_v3(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif network_type == "swin_tiny":
        if pretrained_weights_path:
            model = models.swin_t(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.swin_t(
                weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif network_type == "swin_small":
        if pretrained_weights_path:
            model = models.swin_s(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.swin_s(
                weights=models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif network_type == "maxvit_t":
        if pretrained_weights_path:
            model = models.maxvit_t(weights=None)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
        else:
            model = models.maxvit_t(
                weights=models.MaxVit_T_Weights.IMAGENET1K_V1 if pretrained else None
            )
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)

    elif network_type == "unet":
        # U-Net implementation for classification tasks
        # Adapted from segmentation architecture with global pooling
        from .unet import UNet
        model = UNet(num_classes=num_classes, pretrained=pretrained)

    else:
        raise ValueError(f"Unknown network type: {network_type}")

    return model
