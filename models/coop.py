"""Prompt Learning 改进方向：CoOp（Context Optimization，文档第 4.3 节）。

思路：不训练任何网络权重，只在文本侧学一组可学习的上下文向量（context），
拼在类别名前面，让冻结的文本编码器自己"学会"适合本任务的类别描述：

    [SOT] + [ctx_1 ... ctx_n_ctx] + [类别名 token] + [EOT]

可训练参数只有 n_ctx × 512 个（n_ctx=16 时 8192 个，占 CLIP 总量的 0.005%），
图像塔与文本塔全部冻结。与 Adapter 的区别在于：Adapter 改图像侧，CoOp 改文本侧。

实现说明（open_clip 3.3.0 实测结构）：
该版本里 ``CLIP`` 类**直接继承自 TextTransformer**，文本塔的组件就是 clip
自己的属性 —— ``clip.token_embedding`` / ``clip.transformer`` / ``clip.ln_final``
/ ``clip.text_projection``，并没有 ``clip.text`` 这一层。

因此这里在 ``clip.token_embedding`` 上挂一个 forward hook，把它输出的第
1..n_ctx 个位置替换成可学习的 ctx 向量，其余流程原样交给 ``clip.encode_text()``
—— EOT 定位、attn_mask、text_projection 等版本相关细节都不用自己写。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip_model import load_clip
from utils.prompts import is_healthy, parse_class_name


def default_class_texts(class_names):
    """PlantVillage 目录名 -> 自然语言类别描述（CoOp prompt 里 {class} 那一段）。

    Apple___Apple_scab            -> "apple leaf with apple scab"
    Potato___healthy              -> "healthy potato leaf"
    """
    texts = []
    for raw in class_names:
        plant, disease = parse_class_name(raw)
        if is_healthy(disease):
            texts.append(f"healthy {plant} leaf")
        else:
            texts.append(f"{plant} leaf with {disease}")
    return texts


class CoOpClassifier(nn.Module):
    """可学习上下文 + 冻结的 CLIP 双塔。

    对外接口与 torchvision 模型一致（forward(images) -> logits），
    train.py / evaluate.py 不需要改动即可复用。

    Args:
        class_names: 类别名列表，顺序与标签 0..C-1 一致
        backbone: open_clip 主干名
        pretrained: 预训练权重来源
        n_ctx: 上下文向量个数
        ctx_init: 用来初始化上下文的文本；None 表示随机初始化
    """

    def __init__(self, class_names, backbone="ViT-B-16", pretrained="openai",
                 n_ctx=16, ctx_init="a photo of a"):
        super().__init__()
        clip, tokenizer = load_clip(backbone, pretrained, device="cpu", freeze=False)

        # 整个 CLIP（图像塔 + 文本塔）全部冻结，可训练的只有 ctx 与 logit_scale
        for param in clip.parameters():
            param.requires_grad_(False)

        self.mode = "coop"
        self.backbone_name = backbone
        self.clip = clip
        self.n_ctx = int(n_ctx)

        width = int(clip.token_embedding.embedding_dim)
        context_length = int(clip.positional_embedding.shape[0])

        # ---------------- 可学习上下文 ----------------
        if ctx_init:
            init_tokens = tokenizer(ctx_init)                     # (1, L)
            end = int(init_tokens[0].argmax())                    # EOT 位置
            with torch.no_grad():
                init_emb = clip.token_embedding(init_tokens[0, 1:end])   # (L, D)
            ctx = init_emb[:self.n_ctx]
            if ctx.shape[0] < self.n_ctx:   # 初始化文本不够长，剩下的用正态分布补齐
                rest = torch.empty(self.n_ctx - ctx.shape[0], width).normal_(0.0, 0.02)
                ctx = torch.cat([ctx, rest], dim=0)
        else:
            ctx = torch.empty(self.n_ctx, width).normal_(0.0, 0.02)

        self.ctx = nn.Parameter(ctx.clone())                      # (n_ctx, D)

        # ---------------- 固定的类别名 token ----------------
        texts = [f"{text}." for text in default_class_texts(class_names)]
        tokens = tokenizer(texts)                # (C, L)，同批内右对齐补 0
        # SimpleTokenizer 用 0 做 padding，而真实 token 的 id 都 > 0
        # （<|startoftext|>=49406 起算），所以 ctx 占位也用 0 不会干扰 argmax。
        pad_id = 0
        eot_pos = tokens.argmax(dim=-1)          # EOT 是每行 id 最大的位置

        seqs, eot_indices = [], []
        for c in range(len(texts)):
            end = int(eot_pos[c])
            body = tokens[c, 1:end]              # 去掉 SOT 与 EOT，只留类别名
            seq = torch.cat([
                tokens[c, :1],                                          # SOT
                torch.full((self.n_ctx,), pad_id, dtype=torch.long),    # ctx 占位
                body,
                tokens[c, end:end + 1],                                 # EOT
            ])
            if seq.shape[0] > context_length:
                raise ValueError(
                    f"类别 '{texts[c]}' 的 prompt 长度 {seq.shape[0]} 超过 "
                    f"{context_length}，请调小 n_ctx"
                )
            eot_indices.append(self.n_ctx + body.shape[0] + 1)
            if seq.shape[0] < context_length:
                seq = torch.cat([seq, torch.full(
                    (context_length - seq.shape[0],), pad_id, dtype=torch.long)])
            seqs.append(seq)

        self.register_buffer("token_ids", torch.stack(seqs))
        self.register_buffer("eot_index", torch.tensor(eot_indices))

        # 温度系数参与训练（官方 CoOp 的做法）：它直接决定相似度 softmax 的
        # 尖锐程度，冻结在预训练值上时梯度太弱、上下文学不动。
        scale = getattr(clip, "logit_scale", None)
        if isinstance(scale, nn.Parameter):
            scale.requires_grad_(True)

        # 最后注册 hook：上面初始化 ctx 时也调用了 token_embedding，
        # 那时 ctx 还没准备好，不能被替换。
        clip.token_embedding.register_forward_hook(self._inject_ctx)

    # ---------------- 文本侧 ----------------
    def _inject_ctx(self, module, inputs, output):
        """forward hook：把 token_embedding 输出里第 1..n_ctx 位换成可学习的 ctx。

        output 形状 (C, L, D)：保留第 0 位（SOT）与 n_ctx+1 之后的部分，
        中间 n_ctx 个位置用当前上下文替换。长度不变，因此后面的
        positional_embedding 相加、EOT 定位都照常工作。
        """
        ctx = self.ctx.to(dtype=output.dtype, device=output.device)
        ctx = ctx.unsqueeze(0).expand(output.shape[0], -1, -1)
        return torch.cat([output[:, :1, :], ctx, output[:, self.n_ctx + 1:, :]], dim=1)

    def text_features(self):
        """(C, D) 的 L2 归一化文本特征。每次前向都重算，因为 ctx 在更新。"""
        return F.normalize(self.clip.encode_text(self.token_ids).float(), dim=-1)

    # ---------------- 图像侧 ----------------
    def forward(self, images):
        image_feats = F.normalize(self.clip.encode_image(images).float(), dim=-1)
        logits = image_feats @ self.text_features().t()

        scale = getattr(self.clip, "logit_scale", None)
        if scale is None:
            return logits * 100.0
        # clamp 防止训练中温度失控（官方实现同样会截断）
        return logits * scale.exp().clamp(max=100.0).to(logits.dtype)

    def train(self, mode=True):
        """主干（图像塔 + 文本塔）全部冻结，始终 eval。

        可训练的 ctx 挂在 self 上，不受影响；logit_scale 是 Parameter，
        eval() 也不会改变它。
        """
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
