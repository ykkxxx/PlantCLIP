"""基于 open_clip 的分类模型，承载 Baseline D 与后续改进方向。

三种模式（由配置里的 model.mode 选择）：

- mode="linear"   ：冻结图像编码器，只训练线性头（linear probe，可训练参数 0.01%）
- mode="finetune" ：图像编码器 + 线性头全量微调（Baseline D，可训练参数 57.7%）
- mode="adapter"  ：冻结图像编码器，在每层插入 Adapter，只训 Adapter + 线性头
                    （改进方向一，可训练参数约 0.8%）

对外接口与 torchvision 模型完全一致（forward(images) -> logits），
因此 train.py / evaluate.py 不需要任何改动即可复用。

说明：以上三种模式都**只使用图像编码器**，文本编码器始终冻结，
留给后续的 Prompt Learning 改进方向。
"""

import torch.nn as nn

from models.adapters import Adapter, AdapterBlock, get_visual_blocks
from models.clip_model import load_clip

MODES = ("linear", "finetune", "adapter")


class CLIPClassifier(nn.Module):
    """CLIP 图像编码器 + 线性分类头（可选 Adapter）。

    Args:
        num_classes: 类别数
        backbone: open_clip 主干名（"ViT-B-16" / "ViT-B-32"，连字符写法）
        pretrained: 预训练权重来源（"openai"）
        mode: "linear" / "finetune" / "adapter"
        dropout: 分类头前的 dropout，默认 0（即不使用）
        bottleneck: Adapter 瓶颈维度（仅 mode="adapter" 有效）
        alpha: Adapter 残差缩放系数（仅 mode="adapter" 有效）
    """

    def __init__(self, num_classes, backbone="ViT-B-16", pretrained="openai",
                 mode="finetune", dropout=0.0, bottleneck=64, alpha=0.2):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"不支持的 mode='{mode}'，可选：{' / '.join(MODES)}")

        self.mode = mode
        self.backbone_name = backbone

        # freeze=False：是否冻结交给下面的 mode 决定
        clip, _ = load_clip(backbone, pretrained, device="cpu", freeze=False)
        self.clip = clip

        # 图像特征维度：open_clip 的 ViT 在 visual 上暴露 output_dim
        # （投影后的维度，ViT-B/16 与 ViT-B/32 都是 512）。回退分支只为兼容老版本。
        feat_dim = getattr(clip.visual, "output_dim", None)
        if feat_dim is None:
            projection = getattr(clip, "text_projection", None)
            feat_dim = projection.shape[1] if projection is not None else clip.visual.width

        # 参数名前缀为 head.*，与 train.py 的 HEAD_PREFIXES 配合，
        # 使分类头能用 head_lr_mult 倍的 lr 单独成组（头是随机初始化，需要更大 lr）。
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # 冻结策略：先全部冻结，再按模式解冻。
        # 这里刻意不按属性名去找文本塔（open_clip 各版本对它的命名不一致），
        # 只主动解冻需要训练的部分，其余（文本编码器、logit_scale 等）一律冻结。
        # 文本塔不参与前向，若不解冻它，"可训练参数量"会凭空多出 63M，
        # 污染与 Adapter / Prompt Learning 的对比。
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.head.parameters():
            param.requires_grad_(True)

        if mode == "finetune":
            for param in self.clip.visual.parameters():
                param.requires_grad_(True)
        elif mode == "adapter":
            self._insert_adapters(bottleneck, alpha)

    def _insert_adapters(self, bottleneck, alpha):
        """把图像塔的每个 block 换成 AdapterBlock，并只解冻其中的 Adapter 参数。

        注意必须先加载预训练权重再调用（__init__ 里 load_clip 已经做完），
        否则包装后的参数名对不上。
        """
        visual = self.clip.visual
        # 兼容 visual.blocks 与 visual.transformer.resblocks 两种结构
        blocks = get_visual_blocks(visual)

        # token 维度：优先用 patch embedding 的输出通道（最稳定），退回 width
        if hasattr(visual, "conv1"):
            dim = int(visual.conv1.weight.shape[0])
        else:
            dim = int(visual.width)

        for i, block in enumerate(blocks):
            blocks[i] = AdapterBlock(block, dim, bottleneck, alpha)

        # 解冻 Adapter（head 在上面已经解冻，主干保持冻结）
        for module in visual.modules():
            if isinstance(module, Adapter):
                for param in module.parameters():
                    param.requires_grad_(True)

    def train(self, mode=True):
        """切换训练/评估模式。

        linear / adapter 模式下主干是冻结的，必须始终留在 eval 状态：否则
        dropout 等层会在训练期引入随机性，与"冻结特征"的前提不符。
        （Adapter 内部只有 Linear + GELU，没有 dropout，一并置 eval 无影响。）
        """
        super().train(mode)
        if self.mode in ("linear", "adapter"):
            self.clip.eval()
        return self

    def forward(self, images):
        features = self.clip.encode_image(images)
        return self.head(features.float())

    def trainable_parameter_count(self):
        """可训练参数量（论文里对比 Adapter / Prompt Learning 的关键指标）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
