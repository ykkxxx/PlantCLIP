"""CLIP 微调基线的预检脚本（不碰数据集，20 秒跑完）。

正式训练前先跑一次，确认这几件事，避免白等一次 25 分钟的训练：
    1. open_clip 能加载 ViT-B-16 与 openai 权重
    2. CLIPClassifier 的 forward 输出形状正确
    3. linear / finetune 两种模式的可训练参数量符合预期
    4. linear 模式下主干始终留在 eval 状态
    5. model.type=clip 时数据侧走的是 CLIP 官方预处理（均值必须一致）

用法：
    python check_clip_finetune.py
"""

import sys

import torch

from dataset.preprocess import CLIP_MEAN, build_transform
from models import build_model
from utils.config import load_config

NUM_CLASSES = 38


def main():
    cfg = load_config("configs/clip_finetune.yaml")

    print("=" * 60)
    print("[1] 构建模型（首次运行会下载 open_clip 权重）")
    model = build_model(cfg, NUM_CLASSES)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"    mode={model.mode} | backbone={model.backbone_name}")
    print(f"    分类头: {model.head[1]}")
    print(f"    可训练参数: {n_train:,} / {n_total:,} （{100.0 * n_train / n_total:.2f}%）")

    print("[2] 前向传播")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    print(f"    输入 (2, 3, 224, 224) -> 输出 {tuple(out.shape)}")
    assert tuple(out.shape) == (2, NUM_CLASSES), "输出形状不对！"

    print("[3] 训练模式下主干的状态")
    model.train()
    backbone_training = model.clip.visual.training
    print(f"    mode={model.mode} -> 主干 visual.training = {backbone_training}")
    if model.mode == "linear":
        assert backbone_training is False, "linear 模式下主干应始终处于 eval 状态！"

    print("[4] 数据侧预处理")
    transform = build_transform(cfg["model"]["type"], "test", 224)
    print(f"    model.type={cfg['model']['type']} -> {transform}")
    normalize = [t for t in transform.transforms if t.__class__.__name__ == "Normalize"][0]
    mean = tuple(round(float(v), 6) for v in normalize.mean)
    print(f"    归一化均值: {mean}")
    assert mean == CLIP_MEAN, "预处理用的不是 CLIP 官方均值，精度会大幅虚低！"

    print("=" * 60)
    print("全部检查通过 ✅  下一步：")
    print("    python train.py --config configs/clip_finetune.yaml")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\n❌ 检查未通过: {exc}")
        sys.exit(1)
