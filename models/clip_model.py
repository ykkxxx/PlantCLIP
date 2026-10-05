"""CLIP 模型封装（基于 OpenCLIP）。

注意：open_clip 的模型名用**连字符** —— 'ViT-B-32'，而不是文档里写的 'ViT-B/32'。
这里做了容错转换，两种写法都能接受。

用法：
    model, tokenizer = load_clip("ViT-B-32", "openai", device)
    text_feats  = encode_texts(model, tokenizer, prompts_per_class, device)
    img_feats, labels = encode_images(model, loader, device)
"""

import torch
import torch.nn.functional as F


def normalize_backbone_name(name):
    """'ViT-B/32' -> 'ViT-B-32'（open_clip 用连字符命名）。"""
    return str(name).strip().replace("/", "-")


def load_clip(backbone="ViT-B-32", pretrained="openai", device="cuda"):
    """加载 CLIP 模型与 tokenizer，冻结全部参数并切到 eval 模式。

    Returns:
        (model, tokenizer)
    """
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError(
            "未安装 open_clip，请先执行：pip install open-clip-torch\n"
            "若下载预训练权重太慢，可先设置镜像：export HF_ENDPOINT=https://hf-mirror.com"
        ) from exc

    backbone = normalize_backbone_name(backbone)
    model, _, _ = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(backbone)

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, prompts_per_class, device):
    """把每个类别的 prompt 编码成文本特征。

    一个类若有多条模板，先各自 L2 归一化再取平均，最后再归一化一次
    （prompt ensembling，通常比单条模板更稳）。

    Args:
        prompts_per_class: prompts[i] = 第 i 类的 prompt 文本列表

    Returns:
        (C, D) 的 L2 归一化文本特征
    """
    class_feats = []
    for texts in prompts_per_class:
        tokens = tokenizer(texts).to(device)
        feats = F.normalize(model.encode_text(tokens).float(), dim=-1)
        class_feats.append(F.normalize(feats.mean(dim=0), dim=-1))
    return torch.stack(class_feats)


@torch.no_grad()
def encode_images(model, loader, device):
    """编码整个 DataLoader 的图像。

    Returns:
        (image_feats, labels)，特征已 L2 归一化
    """
    all_feats, all_labels = [], []
    for images, labels in loader:
        feats = model.encode_image(images.to(device, non_blocking=True))
        all_feats.append(F.normalize(feats.float(), dim=-1))
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def get_logit_scale(model, default=100.0):
    """CLIP 的温度系数（logit_scale 取 exp）。取不到时回退到默认值 100。"""
    scale = getattr(model, "logit_scale", None)
    if scale is None:
        return default
    return float(scale.exp().item())
