"""CLIP zero-shot 评估（Baseline C，文档第 4.1 节）。

不做任何训练：直接用 CLIP 的图像-文本相似度做 38 类分类。

流程：
    1. 类别名 -> 自然语言 prompt（utils/prompts.py）
    2. 文本编码器 -> 每个类别的文本特征
    3. 图像编码器 -> 测试集图像特征
    4. 相似度最高者即预测类别

用法：
    python evaluate_zeroshot.py --config configs/clip_zeroshot.yaml --split test
    # 只用前 256 张图快速验证流程（调试用）
    python evaluate_zeroshot.py --config configs/clip_zeroshot.yaml --limit 256
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片

import torch
from torch.utils.data import DataLoader, Subset

from dataset.dataset import create_datasets
from evaluate import append_summary, load_class_names, plot_confusion_matrix
from models.clip_model import (encode_images, encode_texts, get_logit_scale,
                               load_clip, text_feature_similarity)
from utils.config import PROJECT_ROOT, load_config, resolve_output_dirs
from utils.logger import setup_logger
from utils.metrics import build_report
from utils.prompts import build_class_prompts
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP zero-shot 评估")
    parser.add_argument("--config", type=str, default="configs/clip_zeroshot.yaml",
                        help="实验配置文件（相对项目根目录）")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="评估哪个划分，默认 test")
    parser.add_argument("--limit", type=int, default=None,
                        help="只用前 N 张图快速验证（调试用）")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg["experiment"]["name"]
    results_dir, _, log_dir = resolve_output_dirs(cfg)
    os.makedirs(results_dir, exist_ok=True)

    set_seed(cfg["training"].get("seed", 42))
    logger = setup_logger(exp_name, str(log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    logger.info("=" * 60)
    logger.info(f"CLIP zero-shot | 实验: {exp_name} | 划分: {args.split} | 设备: {device}")
    logger.info("=" * 60)

    # ---------------- 数据（复用训练时的同一划分；transform 由 model.type=clip 决定）----
    datasets = create_datasets(cfg)
    class_names = load_class_names(cfg)
    logger.info(f"类别数: {len(class_names)}")

    split_ds = datasets[args.split]
    if args.limit:
        split_ds = Subset(split_ds, range(min(args.limit, len(split_ds))))
        logger.info(f"[调试] 只使用前 {len(split_ds)} 张图")

    loader = DataLoader(split_ds, batch_size=cfg["training"].get("batch_size", 64),
                        shuffle=False, num_workers=cfg["data"].get("num_workers", 8),
                        pin_memory=True)

    # ---------------- 文本侧：类别 prompt -> 文本特征 ----------------
    prompt_cfg = cfg.get("prompt", {})
    prompts_per_class = build_class_prompts(
        class_names,
        disease_templates=(prompt_cfg.get("disease_templates")
                           or prompt_cfg.get("templates")
                           or prompt_cfg.get("template")),
        healthy_templates=prompt_cfg.get("healthy_templates"),
    )
    logger.info(f"每类模板数: {len(prompts_per_class[0])}")
    for name, texts in list(zip(class_names, prompts_per_class))[:2]:
        logger.info(f"  {name}  ->  {texts[0]}")

    model, tokenizer = load_clip(model_cfg.get("clip_backbone", "ViT-B-32"),
                                 model_cfg.get("pretrained", "openai"), device)
    logger.info(f"CLIP 主干: {model_cfg.get('clip_backbone')} | "
                f"预训练: {model_cfg.get('pretrained')}")

    text_feats = encode_texts(model, tokenizer, prompts_per_class, device)

    # 自检：正常应明显小于 1；接近 1 说明文本特征塌缩，zero-shot 必然失效
    text_sim = text_feature_similarity(text_feats)
    logger.info(f"类别文本特征平均两两相似度: {text_sim:.4f}（正常应远小于 1）")
    if text_sim > 0.95:
        logger.warning("⚠ 文本特征高度相似，疑似塌缩 —— 检查上方是否出现 "
                       "'QuickGELU mismatch' 警告")

    # ---------------- 图像侧 + 相似度分类 ----------------
    image_feats, labels = encode_images(model, loader, device)
    logit_scale = get_logit_scale(model)
    logits = logit_scale * (image_feats @ text_feats.t())

    preds = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    n_pred_classes = len(set(preds.tolist()))
    logger.info(f"预测覆盖类别数: {n_pred_classes}/{len(class_names)}"
                f"（远小于类别总数说明模型退化地只预测少数几类）")

    # ---------------- 指标与输出 ----------------
    report_text, metrics, cm = build_report(class_names, preds, labels)

    report_path = os.path.join(results_dir, f"classification_report_{args.split}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print("\n" + report_text + "\n")

    cm_path = os.path.join(results_dir, f"confusion_matrix_{args.split}.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    summary_path = PROJECT_ROOT / cfg["output"].get("results_dir", "results") / "experiment_results.csv"
    append_summary(summary_path, {
        "experiment": exp_name,
        "split": args.split,
        "model": f"clip-{model_cfg.get('clip_backbone')}-zeroshot",
        "epoch": 0,
        **{k: round(v, 4) for k, v in metrics.items()},
    })

    logger.info(f"分类报告: {report_path}")
    logger.info(f"混淆矩阵: {cm_path}")
    logger.info(f"汇总表  : {summary_path}")
    logger.info(f"Accuracy={metrics['accuracy']:.4f} | macro-F1={metrics['macro_f1']:.4f} "
                f"| weighted-F1={metrics['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
