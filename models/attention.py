"""Attention 改进方向：注意力池化分类头（文档第 4.4 节，改进方向三）。

动机
----
CLIP 的 ViT 用 **CLS token** 表征整张图，那是一个全局聚合向量。PlantVillage
是单叶图像，病斑只占叶片上一小块区域，CLS 会把病斑信号和健康叶肉摊平在一起。

本方向**不动主干**：图像塔照旧全部冻结，只把"取 CLS"换成"用一个可学习
的 query 对全部 196 个 patch token 做注意力池化"，让模型自己学该盯哪块区域：

    attn   = softmax( (W_q · q) (W_k · x)^T / sqrt(d) )
    pooled = attn · (W_v · x)

可训练参数只有 query + W_q/W_k/W_v（512 空间）与分类头，约 0.81M，占 CLIP
总量 0.54% —— 介于线性探测（0.013%）与 Adapter（0.81%）之间。

实现要点（open_clip 3.3.0 实测结构）
------------------------------------
`visual` 的子模块是 ['conv1', 'patch_dropout', 'ln_pre', 'transformer',
'ln_post']，注意**没有 proj 子模块** —— proj 是挂在 visual 上的 nn.Parameter
矩阵 (width, output_dim) = (768, 512)。encode_image 内部实际是：

    x = self.transformer(x)          # (N, 197, 768)，含 CLS
    x = self.ln_post(x[:, 0, :])     # 只把 CLS 过 LayerNorm
    x = x @ self.proj                # 投到 512 维共享空间

因此这里在 ``visual.transformer`` 上挂 forward hook 抓下整段 token，然后
**照抄 CLS 那条路径、但作用在所有 patch 上**（ln_post → @ proj），最后交给
注意力池化。conv1 / patchify / CLS / 位置编码全部复用 visual.forward，
不重写任何版本相关的细节 —— 与 CoOp 用 hook 绕开 `.text`/`.blocks` 是同一思路。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapters import get_num_tokens, get_visual_transformer
from models.clip_model import load_clip


class AttentionPool(nn.Module):
    """用单个可学习 query 对 patch token 做注意力池化。

    Args:
        dim: token 维度（投影到 CLIP 共享空间后为 512）
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.last_attn = None   # (N, P)，已 detach，供后续画热力图用

    def forward(self, patches):
        """patches: (N, P, D) -> (N, D)"""
        n = patches.shape[0]
        q = self.wq(self.query).expand(n, -1, -1)                        # (N, 1, D)
        k = self.wk(patches)                                             # (N, P, D)
        v = self.wv(patches)                                             # (N, P, D)

        attn = torch.matmul(q, k.transpose(1, 2)) / (self.dim ** 0.5)     # (N, 1, P)
        attn = F.softmax(attn, dim=-1)

        # detach 后再存：否则会把反向图一直挂住，训练时白占显存
        self.last_attn = attn.detach().squeeze(1)                        # (N, P)
        return torch.matmul(attn, v).squeeze(1)                          # (N, D)


class AttnClassifier(nn.Module):
    """冻结的 CLIP 图像塔 + 注意力池化 + 线性分类头。

    对外接口与 torchvision 模型一致（forward(images) -> logits），
    train.py / evaluate.py 不需要改动即可复用。

    Args:
        num_classes: 类别数
        backbone: open_clip 主干名（"ViT-B-16"，连字符写法）
        pretrained: 预训练权重来源（"openai"）
        dropout: 分类头前的 dropout，默认 0（即不使用）
    """

    def __init__(self, num_classes, backbone="ViT-B-16", pretrained="openai",
                 dropout=0.0):
        super().__init__()
        clip, _ = load_clip(backbone, pretrained, device="cpu", freeze=False)

        # 整个 CLIP（图像塔 + 文本塔）全部冻结，可训练的只有下面的 pool 与 head
        for param in clip.parameters():
            param.requires_grad_(False)

        self.mode = "attn"
        self.backbone_name = backbone
        self.clip = clip

        visual = clip.visual
        # token 维度：优先用 patch embedding 的输出通道（最稳定），退回 width
        if hasattr(visual, "conv1"):
            width = int(visual.conv1.weight.shape[0])
        else:
            width = int(visual.width)

        # 投影矩阵：open_clip 把它挂成 visual 上的 Parameter，而不是子模块
        proj = getattr(visual, "proj", None)
        if proj is not None:
            assert int(proj.shape[0]) == width, (
                f"proj 的输入维度 {int(proj.shape[0])} 与 patch embedding 的输出 "
                f"通道 {width} 不一致，图像塔结构与预期不符"
            )
            feat_dim = int(proj.shape[-1])
        else:
            # 没有投影矩阵时，token 直接就是 width 维
            feat_dim = width

        self.num_tokens = get_num_tokens(visual)
        self.token_dim = width          # 图像塔内部 token 维度（768），仅诊断用
        self.pool = AttentionPool(feat_dim)
        # 参数名前缀 head.*，与 train.py 的 HEAD_PREFIXES 配合
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        for param in self.pool.parameters():
            param.requires_grad_(True)
        for param in self.head.parameters():
            param.requires_grad_(True)

        # 抓取主干输出的容器：不是 buffer，不需要进 state_dict
        self._patch_tokens = None
        get_visual_transformer(visual).register_forward_hook(self._capture)

    # ---------------- hook ----------------
    def _capture(self, module, inputs, output):
        """forward hook：把 Transformer 栈的输出（含 CLS 的完整序列）存下来。"""
        self._patch_tokens = output

    def _batch_first(self, tokens):
        """统一成 (N, L, D)。

        open_clip 的 Transformer 返回 (L, N, D)（长度维在前），但个别版本返回
        (N, L, D)。这里用"哪一维等于 token 数"来判断，不依赖版本约定。
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[1] != self.num_tokens and tokens.shape[0] == self.num_tokens:
            tokens = tokens.permute(1, 0, 2)
        return tokens

    # ---------------- 前向 ----------------
    def forward(self, images):
        # 走标准的 encode_image：hook 顺手把整段 token 截下来，
        # 它自己返回的池化特征（CLS 那条路）这里用不到。
        self.clip.encode_image(images)

        if self._patch_tokens is None:
            raise RuntimeError(
                "没抓到图像塔的 token —— forward hook 未触发，主干结构与预期不符"
            )
        tokens = self._batch_first(self._patch_tokens)
        self._patch_tokens = None   # 用完即放，避免长期持有激活

        visual = self.clip.visual
        # 照抄 CLS 的路径（ln_post -> proj），但作用在所有 token 上
        tokens = visual.ln_post(tokens)
        patches = tokens[:, 1:, :]                       # 去掉 CLS，只留 patch

        proj = getattr(visual, "proj", None)
        if proj is not None:
            patches = patches @ proj                     # -> (N, P, feat_dim)

        return self.head(self.pool(patches).float())

    def train(self, mode=True):
        """主干全冻，必须始终留在 eval 状态。

        否则 patch_dropout 等层会在训练期引入随机性，与"冻结特征、只学池化"
        的前提不符。
        """
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameter_count(self):
        """可训练参数量（论文里对比三条改进方向的关键指标）。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
