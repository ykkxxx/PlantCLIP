"""CNN 基线模型：ResNet50 与 ViT-B/16（均来自 torchvision，ImageNet 预训练）。"""

import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes, pretrained=True):
    """ResNet50 分类模型，替换最后的全连接层。

    Args:
        num_classes: 输出类别数
        pretrained: 是否加载 ImageNet 预训练权重
    """
    if pretrained:
        try:
            # torchvision >= 0.13 的新 API
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except (TypeError, AttributeError):
            # 旧版 torchvision 只支持 pretrained 参数
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit_b16(num_classes, pretrained=True):
    """ViT-B/16 分类模型，替换最后的分类头。

    Args:
        num_classes: 输出类别数
        pretrained: 是否加载 ImageNet 预训练权重
    """
    if pretrained:
        try:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        except (TypeError, AttributeError):
            model = models.vit_b_16(pretrained=True)
    else:
        model = models.vit_b_16(weights=None)

    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
