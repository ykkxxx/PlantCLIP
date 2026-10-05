"""训练主脚本（CNN 基线：ResNet50 / ViT-B/16）。

满足文档 §11 的要求：GPU 训练、50 轮、自动保存 best 模型、
loss 与 accuracy 记录、early stopping；训练结束后输出论文用曲线图。

用法：
    python train.py --config configs/resnet50.yaml
    python train.py --config configs/vit_b16.yaml

    # 调试时可临时覆盖轮数/批大小（不修改配置文件）
    python train.py --config configs/resnet50.yaml --epochs 1 --batch-size 8
"""

import argparse
import csv
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset.dataset import create_datasets
from models import build_model
from utils.config import load_config, resolve_output_dirs
from utils.logger import setup_logger
from utils.seed import set_seed

# 分类头的参数名前缀（resnet50 为 fc.*，vit_b_16 为 heads.*，CLIP 为 head.*）
HEAD_PREFIXES = ("fc.", "heads.", "head.")


def parse_args():
    parser = argparse.ArgumentParser(description="PlantVillage CNN 基线训练")
    parser.add_argument("--config", type=str, default="configs/resnet50.yaml",
                        help="实验配置文件（相对项目根目录）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖配置中的训练轮数（调试用）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="覆盖配置中的 batch size（调试用）")
    parser.add_argument("--shots", type=int, default=None,
                        help="少样本设定：训练集每类只用 N 张（默认用全部）。"
                             "实验名会自动加 _Nshot 后缀，结果存到独立目录")
    return parser.parse_args()


def take_few_shot(dataset, shots, seed=42):
    """从训练集里每类随机抽 shots 张，构造少样本训练子集。

    分层抽样 + 固定 seed，保证可复现。返回的是原 dataset 的 Subset，
    因此 transform（含训练增强）保持不变；验证/测试集不受影响，仍用全部数据。
    """
    rng = random.Random(seed)
    by_class = {}
    for i, label in enumerate(dataset.labels):
        by_class.setdefault(label, []).append(i)

    keep = []
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        keep += indices[:shots]
    return Subset(dataset, sorted(keep))


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回 (平均loss, 准确率)。"""
    model.train()
    total_loss, total_correct, total_num = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_num += batch_size
    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate_split(model, loader, criterion, device):
    """在验证集上评估，返回 (平均loss, 准确率)。"""
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_num += batch_size
    return total_loss / total_num, total_correct / total_num


def plot_curve(x, y, title, ylabel, save_path):
    """画单条曲线并保存（论文用图）。"""
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg["experiment"]["name"]
    tcfg = cfg["training"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.batch_size is not None:
        tcfg["batch_size"] = args.batch_size

    # 少样本实验用独立的实验名，避免覆盖全量实验的 checkpoint / 结果 / 日志
    if args.shots:
        exp_name = f"{exp_name}_{args.shots}shot"
        cfg["experiment"]["name"] = exp_name

    results_dir, ckpt_dir, log_dir = resolve_output_dirs(cfg)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    set_seed(tcfg.get("seed", 42))
    logger = setup_logger(exp_name, str(log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info(f"实验名称 : {exp_name}")
    logger.info(f"配置文件 : {args.config}")
    logger.info(f"运行设备 : {device}")
    if device.type == "cuda":
        logger.info(f"GPU      : {torch.cuda.get_device_name(0)}")
    logger.info("=" * 60)

    # ---------------- 数据 ----------------
    datasets = create_datasets(cfg)
    # 类别数由数据本身推断，不信任配置里的硬编码 38
    num_classes = max(datasets["train"].labels) + 1
    logger.info(f"类别数（由数据推断）: {num_classes}")

    if args.shots:
        datasets["train"] = take_few_shot(datasets["train"], args.shots, tcfg.get("seed", 42))
        logger.info(f"少样本设定：训练集每类 {args.shots} 张，共 {len(datasets['train'])} 张"
                    f"（验证/测试集仍为全量）")

    batch_size = tcfg.get("batch_size", 64)
    num_workers = cfg["data"].get("num_workers", 8)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # ---------------- 模型 ----------------
    model = build_model(cfg, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # 可训练参数量：CLIP 微调（全量）与 Adapter / Prompt Learning（少量）对比的核心指标
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {n_trainable:,} / {n_total:,} "
                f"（{100.0 * n_trainable / max(n_total, 1):.2f}%）")

    optimizer_name = str(tcfg.get("optimizer", "adam")).lower()
    lr = tcfg.get("lr", 1e-3)
    weight_decay = tcfg.get("weight_decay", 1e-4)
    head_lr_mult = float(tcfg.get("head_lr_mult", 1.0))

    # 分类头是随机初始化的，而主干是预训练权重：两者用同一个学习率时，
    # 头学不动、主干先被破坏。head_lr_mult > 1 时把头单独分一组用更大的 lr。
    # 默认 1.0，退化为单参数组（ResNet50 基线的行为完全不变）。
    if head_lr_mult > 1.0:
        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            (head_params if name.startswith(HEAD_PREFIXES) else backbone_params).append(param)
        params = [
            {"params": backbone_params, "lr": lr},
            {"params": head_params, "lr": lr * head_lr_mult},
        ]
        logger.info(f"分类头单独分组: 主干 lr={lr:.2e} / 头 lr={lr * head_lr_mult:.2e} "
                    f"（主干 {len(backbone_params)} 个张量，头 {len(head_params)} 个张量）")
    else:
        params = model.parameters()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    epochs = tcfg.get("epochs", 50)
    warmup_epochs = int(tcfg.get("warmup_epochs", 0))
    scheduler_name = str(tcfg.get("lr_scheduler", "cosine")).lower()

    # warmup：前 warmup_epochs 轮 lr 从 1/warmup 线性爬到满值，之后按 scheduler 衰减。
    # LambdaLR 对每个参数组按各自基准 lr 等比缩放，因此与上面的分组 lr 兼容。
    if warmup_epochs > 0:
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            if scheduler_name == "cosine":
                progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    else:
        scheduler = None

    patience = tcfg.get("early_stopping_patience", 10)
    logger.info(f"模型: {cfg['model'].get('name')} | 优化器: {optimizer_name} | lr: {lr} "
                f"| epochs: {epochs} | batch: {batch_size} | warmup: {warmup_epochs} "
                f"| early_stop: {patience}")

    # ---------------- 可选 TensorBoard ----------------
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_dir / exp_name))
        logger.info(f"TensorBoard 已启用: {log_dir / exp_name}")
    except Exception:
        logger.info("未启用 TensorBoard（tensorboard 未安装或不可用，不影响训练）")

    # ---------------- 训练循环 ----------------
    best_val_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_split(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": current_lr,
        })
        logger.info(f"Epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} "
                    f"train_acc {train_acc:.4f} | val_loss {val_loss:.4f} "
                    f"val_acc {val_acc:.4f} | lr {current_lr:.2e}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("acc/train", train_acc, epoch)
            writer.add_scalar("acc/val", val_acc, epoch)

        # 按验证集准确率保存 best 模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": cfg["model"].get("name"),
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "config": cfg,
            }, os.path.join(ckpt_dir, "best_model.pth"))
            logger.info(f"  ↳ 保存 best 模型（val_acc={val_acc:.4f}）")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping：验证集连续 {patience} 轮未提升，"
                            f"在 epoch {epoch} 提前停止")
                break

    if writer is not None:
        writer.close()

    elapsed = time.time() - start_time
    logger.info(f"训练结束：用时 {elapsed:.1f}s，best epoch={best_epoch}，"
                f"best val_acc={best_val_acc:.4f}")

    # ---------------- 保存历史与曲线 ----------------
    history_path = os.path.join(results_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        csv_writer.writeheader()
        csv_writer.writerows(history)
    logger.info(f"训练历史已保存: {history_path}")

    with open(os.path.join(results_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "experiment": exp_name,
            "model": cfg["model"].get("name"),
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "elapsed_seconds": round(elapsed, 1),
        }, f, ensure_ascii=False, indent=2)

    epochs_axis = [h["epoch"] for h in history]
    plot_curve(epochs_axis, [h["train_loss"] for h in history],
               "Train Loss", "Loss", os.path.join(results_dir, "train_loss.png"))
    plot_curve(epochs_axis, [h["val_loss"] for h in history],
               "Val Loss", "Loss", os.path.join(results_dir, "val_loss.png"))
    plot_curve(epochs_axis, [h["train_acc"] for h in history],
               "Train Accuracy", "Accuracy", os.path.join(results_dir, "train_accuracy.png"))
    plot_curve(epochs_axis, [h["val_acc"] for h in history],
               "Val Accuracy", "Accuracy", os.path.join(results_dir, "val_accuracy.png"))
    logger.info(f"训练曲线已保存至: {results_dir}")
    logger.info(f"best 模型: {os.path.join(ckpt_dir, 'best_model.pth')}")
    logger.info(f"下一步评估: python evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
