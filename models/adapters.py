"""CLIP 图像塔里的 Adapter 模块（文档第 4.2 节，改进方向一）。

结构是 CLIP-Adapter 式的残差瓶颈，插在每个 Transformer block 的输出上：

    x' = x + alpha * W_up(GELU(W_down(x)))

W_up 零初始化，因此刚插入时 Adapter 输出恒为 0、整体是恒等映射 —— 训练从
预训练模型原样开始，不会因为随机初始化的新模块破坏已有特征。

配合冻结的主干，可训练参数只有约 120 万（占 CLIP 总量 0.8%）。
"""

import torch.nn as nn


def get_visual_blocks(visual):
    """取图像塔里 Transformer block 的序列（nn.Sequential，可按下标替换）。

    open_clip 各版本结构不同，两条路径都要覆盖：
    - visual.blocks                 （部分版本 / 变体）
    - visual.transformer.resblocks  （经典结构）
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


class Adapter(nn.Module):
    """瓶颈残差 Adapter。

    Args:
        dim: 输入/输出维度（ViT-B 为 768）
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
