"""CLIP 图像塔的结构定位工具 + Adapter 模块。

Adapter（文档第 4.2 节，改进方向一）是 CLIP-Adapter 式的残差瓶颈，插在每个
Transformer block 的输出上：

    x' = x + alpha * W_up(GELU(W_down(x)))

W_up 零初始化，因此刚插入时 Adapter 输出恒为 0、整体是恒等映射 —— 训练从
预训练模型原样开始，不会因为随机初始化的新模块破坏已有特征。

配合冻结的主干，可训练参数只有约 120 万（占 CLIP 总量 0.8%）。

文件开头的 get_visual_blocks / get_visual_transformer / get_num_tokens 是
**结构定位工具**：open_clip 各版本对图像塔的命名不一致，集中放在这里，
Adapter 与 Attention 两条改进方向共用同一套探测逻辑。
"""

import torch.nn as nn


def get_visual_blocks(visual):
    """取图像塔里 Transformer block 的序列（nn.Sequential，可按下标替换）。

    open_clip 各版本结构不同，两条路径都要覆盖：
    - visual.blocks                 （部分版本 / 变体）
    - visual.transformer.resblocks  （经典结构）

    用途：插入 Adapter（需要按下标替换每个 block）。
    抓取整段 token 序列请用 get_visual_transformer。
    """
    blocks = getattr(visual, "blocks", None)
    if blocks is not None:
        return blocks

    transformer = getattr(visual, "transformer", None)
    if transformer is not None:
        blocks = getattr(transformer, "resblocks", None)
    if blocks is None:
        available = [name for name, _ in visual.named_children()]
        raise AttributeError(
            f"无法定位图像塔的 Transformer blocks，可用的子模块：{available}"
        )
    return blocks


def get_visual_transformer(visual):
    """取图像塔里**包住所有 block 的那一层**（用于 hook 抓完整 token 序列）。

    经典结构是 visual.transformer（Transformer 模块，输出 (N, L, D) 的
    全部 token，含 CLS）；部分变体把 block 直接放在 visual.blocks 下，
    此时返回 blocks 本身。
    """
    transformer = getattr(visual, "transformer", None)
    if transformer is not None:
        return transformer

    blocks = getattr(visual, "blocks", None)
    if blocks is not None:
        return blocks

    available = [name for name, _ in visual.named_children()]
    raise AttributeError(
        f"无法定位图像塔的 Transformer 栈，可用的子模块：{available}"
    )


def get_num_tokens(visual):
    """图像塔的 token 总数（含 CLS），用于判断 hook 输出的维度顺序。

    标准 open_clip 的 VisionTransformer 带 positional_embedding 缓冲，形状
    (num_patches + 1, width)；退化版本退回 grid_size。ViT-B/16 @224 为 197
    （196 个 patch + 1 个 CLS）。
    """
    pos = getattr(visual, "positional_embedding", None)
    if pos is not None:
        return int(pos.shape[0])

    grid = getattr(visual, "grid_size", None)
    if grid is not None:
        return int(grid) ** 2 + 1

    available = [name for name, _ in visual.named_children()]
    raise AttributeError(
        f"无法确定图像塔的 token 数（既没有 positional_embedding 也没有 "
        f"grid_size），visual 的子模块：{available}"
    )


class Adapter(nn.Module):
    """瓶颈残差 Adapter。

    Args:
        dim: 输入/输出维度（ViT-B 的图像塔为 768）
        bottleneck: 瓶颈维度，越小参数越少
        alpha: 残差缩放系数
    """

    def __init__(self, dim, bottleneck=64, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)

        # 零初始化 up：初始时 Adapter 输出为 0，等价于恒等映射。
        # down 保持默认初始化即可（第 0 步它梯度为 0，up 更新后就开始学了）。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.alpha * self.up(self.act(self.down(x)))


class AdapterBlock(nn.Module):
    """把原 Transformer block 包一层，其输出再过一个 Adapter。

    用包装而不是改 open_clip 源码：原 block 的参数名只是多了一层 ``block.``
    前缀，预训练权重在包装之前就已加载完毕，不受影响。
    """

    def __init__(self, block, dim, bottleneck=64, alpha=0.2):
        super().__init__()
        self.block = block
        self.adapter = Adapter(dim, bottleneck, alpha)

    def forward(self, x, *args, **kwargs):
        # 透传 *args/**kwargs：经典结构的 Transformer.forward 会调
        # block(x, attn_mask=...)，签名必须能吃下这个参数
        out = self.block(x, *args, **kwargs)
        if isinstance(out, tuple):  # 兼容个别返回 tuple 的 open_clip 版本
            out = out[0]
        return self.adapter(out)
