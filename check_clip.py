"""CLIP 分类模型的预检脚本（不碰图像数据，20 秒跑完）。

正式训练前先跑一次，确认这几件事，避免白等一次几十分钟的训练：
    1. open_clip 能加载主干与 openai 权重
    2. 各模式的冻结策略正确（该冻结的必须全冻、该解冻的必须解开）
    3. forward 输出形状正确
    4. linear / adapter / coop 模式下主干始终留在 eval 状态
    5. model.type=clip 时数据侧走的是 CLIP 官方预处理（均值必须一致）

用法：
    python check_clip.py                                     # 默认查微调配置
    python check_clip.py --config configs/clip_adapter.yaml  # 查 Adapter 配置
    python check_clip.py --config configs/clip_coop.yaml     # 查 CoOp 配置
"""

import argparse
import os
import sys

import torch

from dataset.preprocess import CLIP_MEAN, build_transform
from models import build_model
from models.adapters import Adapter, get_visual_blocks
from utils.config import load_class_names, load_config

# class_names.json 缺失时的占位类别数（正常流程下由 create_datasets 落盘）
DEFAULT_NUM_CLASSES = 38


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP 分类模型预检")
    parser.add_argument("--config", type=str, default="configs/clip_finetune.yaml",
                        help="要检查的实验配置（相对项目根目录）")
    return parser.parse_args()


def check_freeze_policy(model):
    """按模式检查可训练参数分布，返回 (可训练总数, 总参数, 顶层分布)。

    不按属性名去引用文本塔 —— open_clip 各版本对它的命名不一致（3.3.0 里
    CLIP 直接继承 TextTransformer，根本没有 clip.text），只做方向性判断：
    "视觉塔之外不该有可训练参数" / "除 ctx 外不该有可训练参数"。
    """
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    n_train = sum(trainable.values())
    n_total = sum(p.numel() for p in model.parameters())

    buckets = {}
    for name, numel in trainable.items():
        top = name.split(".")[0]
        buckets[top] = buckets.get(top, 0) + numel

    clip_trainable = sorted(n for n in trainable if n.startswith("clip."))
    vision_only = all(n.startswith("clip.visual.") for n in clip_trainable)

    if model.mode == "linear":
        assert set(buckets) == {"head"}, f"linear 只应训练 head，实际 {sorted(buckets)}"

    elif model.mode == "adapter":
        assert set(buckets) == {"clip", "head"}, \
            f"adapter 应只训练 clip + head，实际 {sorted(buckets)}"
        assert vision_only, \
            f"adapter 下视觉塔之外不应有可训练参数：{clip_trainable[:3]}"
        # 每个 Transformer block 应恰好插一个 Adapter
        n_adapters = len([m for m in model.modules() if isinstance(m, Adapter)])
        depth = len(get_visual_blocks(model.clip.visual))
        print(f"    Adapter: 插入 {n_adapters} 个（主干深度 {depth}）")
        assert n_adapters == depth, f"Adapter 数量应为 {depth}，实际 {n_adapters}！"

    elif model.mode == "finetune":
        assert set(buckets) <= {"clip", "head"}, \
            f"finetune 只应有 clip + head，实际 {sorted(buckets)}"
        assert vision_only, \
            f"finetune 下文本塔必须冻结，但 {clip_trainable[:3]} 可训练"

    elif model.mode == "coop":
        assert "ctx" in buckets, "CoOp 的 ctx 必须可训练！"
        assert not clip_trainable or (
            len(clip_trainable) == 1 and clip_trainable[0].endswith("logit_scale")
        ), f"CoOp 的图像/文本塔必须全冻，只有 logit_scale 例外：{clip_trainable[:3]}"
        print(f"    CoOp: n_ctx={model.n_ctx} | ctx 形状 {tuple(model.ctx.shape)} "
              f"| 图像塔深度 {len(get_visual_blocks(model.clip.visual))}（全冻）")

    return n_train, n_total, buckets


def main():
    args = parse_args()
    cfg = load_config(args.config)

    print("=" * 60)
    print(f"配置文件: {args.config}")

    # CoOp 的文本 prompt 直接依赖类别名，顺手用它确定类别数（不再写死 38）
    try:
        class_names = load_class_names(cfg)
    except (FileNotFoundError, OSError):
        split_dir = cfg["data"].get("split_dir", "split")
        print(f"    ⚠ 未找到 {os.path.join(split_dir, 'class_names.json')}，"
              f"暂用 {DEFAULT_NUM_CLASSES} 个占位类别名")
        class_names = [f"class_{i}" for i in range(DEFAULT_NUM_CLASSES)]
    num_classes = len(class_names)

    print("[1] 构建模型（首次运行会下载 open_clip 权重）")
    model = build_model(cfg, num_classes, class_names)
    n_train, n_total, buckets = check_freeze_policy(model)

    print(f"    mode={model.mode} | backbone={model.backbone_name} | 类别数: {num_classes}")
    if hasattr(model, "head"):
        print(f"    分类头: {model.head[1]}")
    print(f"    可训练参数: {n_train:,} / {n_total:,} （{100.0 * n_train / n_total:.4f}%）")
    print("    可训练分布: " + (", ".join(f"{k}={v:,}" for k, v in buckets.items()) or "无"))

    print("[2] 前向传播")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    print(f"    输入 (2, 3, 224, 224) -> 输出 {tuple(out.shape)}")
    assert tuple(out.shape) == (2, num_classes), "输出形状不对！"

    if model.mode == "coop":
        # CoOp 的类别 token 必须按类别数逐条构造，顺序与标签一一对应
        assert model.token_ids.shape[0] == num_classes, \
            f"token_ids 应有 {num_classes} 行（每类一行），实际 {model.token_ids.shape[0]}"
        length = int(model.token_ids.shape[1])
        assert int(model.eot_index.max()) < length, "EOT 位置超出序列长度，prompt 构造有误！"
        print(f"    token_ids {tuple(model.token_ids.shape)} | 序列长度 {length} | "
              f"前 3 类的 EOT 位置 {model.eot_index[:3].tolist()}"
              f"（= n_ctx({model.n_ctx}) + 类别名长度 + 1）")

    print("[3] 训练模式下主干的状态")
    model.train()
    backbone_training = model.clip.visual.training
    print(f"    mode={model.mode} -> 主干 visual.training = {backbone_training}")
    if model.mode in ("linear", "adapter", "coop"):
        assert backbone_training is False, f"{model.mode} 模式下主干应始终处于 eval 状态！"

    print("[4] 数据侧预处理")
    transform = build_transform(cfg["model"]["type"], "test", 224)
    print(f"    model.type={cfg['model']['type']} -> {transform}")
    normalize = [t for t in transform.transforms if t.__class__.__name__ == "Normalize"][0]
    mean = [float(v) for v in normalize.mean]
    print(f"    归一化均值: {mean}")
    # 逐项按容差比较（不要取整后再判等，0.481455 != 0.48145466）
    assert len(mean) == len(CLIP_MEAN) and all(
        abs(a - b) < 1e-5 for a, b in zip(mean, CLIP_MEAN)
    ), "预处理用的不是 CLIP 官方均值，精度会大幅虚低！"

    print("=" * 60)
    print("全部检查通过 ✅  下一步：")
    print(f"    python train.py --config {args.config}")
    print(f"    python evaluate.py --config {args.config}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\n❌ 检查未通过: {exc}")
        sys.exit(1)
