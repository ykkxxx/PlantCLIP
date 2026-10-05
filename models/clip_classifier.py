"""CLIP 微调基线（文档第 4.1 节 Baseline D）。

在 open_clip 预训练模型之上加一个线性分类头，支持两种模式：

- mode="linear"   ：冻结图像编码器，只训练线性头（linear probe）
- mode="finetune" ：图像编码器 + 线性头全量微调

对外接口与 torchvision 模型完全一致（forward(images) -> logits），
因此 train.py / evaluate.py 不需要任何改动即可复用。

说明：本基线**只使用图像编码器**，文本编码器留给后续的 Prompt Learning 改进。
"""

import torch
import torch.nn as nn

from models.clip_model import load_clip


class CLIPClassifier(nn.Module):
    """CLIP 图像编码器 + 线性分类头。

    Args:
        num_classes: 类别数
        backbone: open_clip 主干名（"ViT-B-16" / "ViT-B-32"，连字符写法）
        pretrained: 预训练权重来源（"openai"）
        mode: "linear"（冻结主干，只训头）或 "finetune"（全量微调）
        dropout: 分类头前的 dropout，默认 0（即不使用）
    """

    def __init__(self, num_classes, backbone="ViT-B-16", pretrained="openai",
                 mode="finetune", dropout=0.0):
        super().__init__()
        if mode not in ("linear", "finetune"):
            raise ValueError(f"不支持的 mode='{mode}'，可选：linear / finetune")

        self.mode = mode
        self.backbone_name = backbone

        # freeze=False：是否冻结交给下面的 mode 决定
        clip, _ = load_clip(backbone, pretrained, device="cpu", freeze=False)
        self.clip = clip

        # 图像特征维度：open_clip 的 ViT 在 visual 上暴露 output_dim
        # （投影后的维度，ViT-B/16 与 ViT-B/32 都是 512）
        feat_dim = getattr(clip.visual, "output_dim", None)
        if feat_dim is None:  # 兼容旧版 open_clip
            feat_dim = clip.text_projection.shape[1]

        # 参数名前缀为 head.*，与 train.py 的 HEAD_PREFIXES 配合，
        # 使分类头能用 head_lr_mult 倍的 lr 单独成组（头是随机初始化，需要更大 lr）。
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # 文本编码器本基线不使用：它不参与前向、不产生梯度，但会混进
        # "可训练参数量"里污染与 Adapter / Prompt Learning 的对比，所以始终冻结。
        if getattr(clip, "text", None) is not None:
            for param in clip.text.parameters():
                param.requires_grad_(False)

        # linear 模式：图像编码器也冻结，只训线性头
        if mode == "linear":
            for param in clip.visual.parameters():
                param.requires_grad_(False)

    def train(self, mode=True):
        """切换训练/评估模式。

        linear 模式下主干是冻结的，必须始终留在 eval 状态：否则 dropout 等
        层会在训练期引入随机性，与"冻结特征做线性探测"的前提不符。
        """
        super().train(mode)
        if self.mode == "linear":
            self.clip.eval()
        return self

    def forward(self, images):
        features = self.clip.encode_image(images)
        return self.head(features.float())

    def trainable_parameter_count(self):
        """可训练参数量（论文里对比 Adapter / Prompt Learning 的关键指标）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
