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

        # 冻结策略：先全部冻结，再按模式解冻。
        # 这里刻意不按属性名去找文本塔（open_clip 各版本对它的命名不一致），
        # 只主动解冻 visual 与 head，其余（文本编码器、logit_scale 等）一律冻结。
        # 文本塔在本基线不参与前向，若不解冻它，"可训练参数量"会凭空多出 63M，
        # 污染与 Adapter / Prompt Learning 的对比。
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(True)
        if mode == "finetune":
            for param in self.clip.visual.parameters():
                param.requires_grad_(True)

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
