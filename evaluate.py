"""测试集评估脚本：指标、混淆矩阵、分类报告，并汇总到 experiment_results.csv。

满足文档 §6 / §12 的输出要求。评估复用与训练完全相同的 split/ 划分。

用法：
    python evaluate.py --config configs/resnet50.yaml
    python evaluate.py --config configs/vit_b16.yaml
    # 指定其他 checkpoint 或评估验证集
    python evaluate.py --config configs/resnet50.yaml --checkpoint checkpoints/resnet50/best_model.pth
    python evaluate.py --config configs/resnet50.yaml --split val
"""

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import create_datasets
from models import build_model
from utils.config import (PROJECT_ROOT, load_class_names, load_config,
                          resolve_output_dirs)
from utils.logger import setup_logger
from utils.metrics import build_report
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="PlantVillage 测试集评估")
    parser.add_argument("--config", type=str, default="configs/resnet50.yaml",
                        help="实验配置文件（相对项目根目录）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 路径，默认 checkpoints/<exp>/best_model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="评估哪个划分，默认 test")
    parser.add_argument("--shots", type=int, default=None,
                        help="少样本实验：与 train.py 的 --shots 保持一致，"
                             "用于定位 checkpoints/<实验名>_Nshot/best_model.pth")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, loader, device):
    """在数据集上推理，返回 (preds, labels) 两个 numpy 数组。"""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(cm, class_names, save_path, normalize=True):
    """绘制混淆矩阵（默认按行归一化，便于观察每类的误判分布）。"""
    cm_plot = cm.astype(np.float64)
    if normalize:
        cm_plot = cm_plot / np.maximum(cm_plot.sum(axis=1, keepdims=True), 1)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.5)))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (row-normalized)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def append_summary(path, row):
    """把一行结果写入汇总 CSV（列名取并集，并按 experiment+split 去重覆盖）。"""
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    fieldnames = list(row.keys())
    rows = []

    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]
            for key in (reader.fieldnames or []):
                if key not in fieldnames:
                    fieldnames.append(key)
        # 同一实验同一划分的旧记录先移除，避免重复行
        rows = [r for r in rows
                if not (r.get("experiment") == row["experiment"] and r.get("split") == row["split"])]

    rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg["experiment"]["name"]
    # 与 train.py 保持一致：少样本实验读同名后缀的 checkpoint 与结果目录
    if args.shots:
        exp_name = f"{exp_name}_{args.shots}shot"
        cfg["experiment"]["name"] = exp_name

    results_dir, ckpt_dir, log_dir = resolve_output_dirs(cfg)
    os.makedirs(results_dir, exist_ok=True)

    set_seed(cfg["training"].get("seed", 42))
    logger = setup_logger(exp_name, str(log_dir))

    ckpt_path = args.checkpoint or os.path.join(str(ckpt_dir), "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}\n请先运行 train.py 训练模型。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info(f"评估实验 : {exp_name} | 划分: {args.split} | 设备: {device}")
    logger.info(f"checkpoint: {ckpt_path}")
    logger.info("=" * 60)

    # ---------------- 数据（复用训练时的同一划分）----------------
    datasets = create_datasets(cfg)
    class_names = load_class_names(cfg)
    num_classes = len(class_names)

    batch_size = cfg["training"].get("batch_size", 64)
    loader = DataLoader(datasets[args.split], batch_size=batch_size, shuffle=False,
                        num_workers=cfg["data"].get("num_workers", 8), pin_memory=True)

    # ---------------- 模型与权重 ----------------
    model = build_model(cfg, num_classes, class_names).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"已加载权重：epoch={ckpt.get('epoch')} val_acc={ckpt.get('val_acc')}")

    # ---------------- 推理与指标 ----------------
    preds, labels = collect_predictions(model, loader, device)
    report_text, metrics, cm = build_report(class_names, preds, labels)

    report_path = os.path.join(results_dir, f"classification_report_{args.split}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    logger.info(f"分类报告已保存: {report_path}")
    print("\n" + report_text + "\n")

    cm_path = os.path.join(results_dir, f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(cm, class_names, cm_path)
    logger.info(f"混淆矩阵已保存: {cm_path}")

    # ---------------- 汇总到跨实验对比表 ----------------
    summary_path = PROJECT_ROOT / cfg["output"].get("results_dir", "results") / "experiment_results.csv"
    row = {
        "experiment": exp_name,
        "split": args.split,
        "model": cfg["model"].get("name"),
        "epoch": ckpt.get("epoch"),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    append_summary(summary_path, row)
    logger.info(f"实验汇总已更新: {summary_path}")
    logger.info(f"Accuracy={metrics['accuracy']:.4f} | macro-F1={metrics['macro_f1']:.4f} "
                f"| weighted-F1={metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
