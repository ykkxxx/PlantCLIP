"""注意力池化的诊断脚本：注意力到底是"盯住了病斑"，还是塌成了均匀分布。

判别问题
--------
Attention 方向 5-shot 跑出 0.7621，与线性探测（0.7653）基本持平 —— 但两者
参数量差了 41 倍（0.81M vs 19K）。这个结果有两种完全不同的解释，对应论文里
两种不同的写法：

  (a) 注意力确实集中到了病斑上，只是"盯对地方"这件事对分类没有额外帮助
      （CLIP 的 CLS 池化其实已经够用）→ 结论是"全局池化已足够，注意力
      池化未带来增益"，是方法层面的结论；
  (b) 注意力压根没学出结构，塌成了近似均匀分布 → 结论是"池化层未学出有效
      的空间先验"，是训练/容量层面的问题，不是方法问题。

本脚本用四个数字把两者分开，顺手出论文可解释性图（三条改进方向里唯一
能出的图）：

  1. 平均最大注意力权重    —— 均匀分布下恒为 1/196 ≈ 0.0051
  2. 注意力熵 / log(196)   —— 均匀分布为 1.0，越小越集中
  3. 覆盖 50% 注意力质量所需 patch 数 —— 均匀分布为 98
  4. 中心 4x4 与四角的平均权重之比    —— 明显 >1 说明学的是"看中间"的
     位置先验，而不是"看病斑"的内容先验

不需要重训：AttentionPool 前向时已把 last_attn（detach 过的 (N, 196)）存好，
所以几秒钟跑完。

用法：
    python check_attn.py --config configs/clip_attn.yaml --shots 5
    python check_attn.py --config configs/clip_attn.yaml --shots 5 --split val
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset.dataset import create_datasets
from evaluate import restore_model_config
from models import build_model
from utils.config import load_class_names, load_config, resolve_output_dirs

UNIFORM_ENTROPY_TOL = 0.98   # 熵比高于此值即认为注意力已塌成均匀分布
WEAK_ENTROPY_TOL = 0.90      # 低于此值认为注意力明显集中
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(description="注意力池化诊断")
    parser.add_argument("--config", type=str, default="configs/clip_attn.yaml",
                        help="实验配置文件（相对项目根目录）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint 路径，默认 checkpoints/<exp>/best_model.pth")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="在哪个划分上统计，默认 test")
    parser.add_argument("--shots", type=int, default=None,
                        help="少样本实验：与 train.py 的 --shots 保持一致")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="实验名后缀：与 train.py 的 --exp-suffix 保持一致")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="热力图里各展示几张正确/错误样本（默认 3+3）")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


@torch.no_grad()
def collect_attention(model, loader, device):
    """跑完整个划分，收集注意力权重 / 预测 / 标签 / 置信度。

    Returns:
        (attn (N, P), preds (N,), labels (N,), conf (N,))，全部在 CPU 上
    """
    model.eval()
    attn_all, preds_all, labels_all, conf_all = [], [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        if model.pool.last_attn is None:
            raise RuntimeError("pool.last_attn 为空 —— 注意力池化层没有被执行？")
        probs = logits.softmax(dim=1)
        attn_all.append(model.pool.last_attn.float().cpu())
        conf_all.append(probs.max(dim=1).values.cpu())
        preds_all.append(probs.argmax(dim=1).cpu())
        labels_all.append(labels)
    return (torch.cat(attn_all), torch.cat(preds_all),
            torch.cat(labels_all), torch.cat(conf_all))


def attention_stats(attn, grid):
    """把 (N, P) 的注意力权重压成几个可判读的标量。

    attn 的每一行都是 softmax 的输出，即一个归一化的概率分布。
    """
    n, p = attn.shape
    max_weight = attn.max(dim=1).values
    entropy = -(attn * attn.clamp_min(EPS).log()).sum(dim=1)

    # 从大到小累加，看要多少个 patch 才能覆盖一半的注意力质量
    sorted_w, _ = attn.sort(dim=1, descending=True)
    n50 = (sorted_w.cumsum(dim=1) < 0.5).sum(dim=1) + 1

    # 中心 vs 四角：区分"看中间"的位置先验和"看病斑"的内容先验
    mean_map = attn.mean(dim=0).reshape(grid, grid)
    side = min(4, grid)
    half = max(0, (grid - side) // 2)
    center = float(mean_map[half:half + side, half:half + side].mean())
    corners = float(torch.stack([
        mean_map[0, 0], mean_map[0, -1], mean_map[-1, 0], mean_map[-1, -1]
    ]).mean())

    uniform_entropy = math.log(p)
    return {
        "num_samples": n,
        "num_patches": p,
        "max_weight": float(max_weight.mean()),
        "max_weight_uniform": 1.0 / p,
        "entropy": float(entropy.mean()),
        "entropy_uniform": uniform_entropy,
        "entropy_ratio": float(entropy.mean()) / uniform_entropy,
        "patches_for_50pct": float(n50.float().mean()),
        "patches_for_50pct_uniform": p / 2.0,
        "center_over_corners": center / max(corners, EPS),
        # 逐样本最大权重的中位数：均值会被少数极尖的样本拉高，中位数更稳
        "max_weight_median": float(max_weight.median()),
    }


def build_report(stats, split, exp_name, ckpt_epoch, ckpt_val_acc):
    """把统计量整理成可保存的文本报告。"""
    ratio = stats["entropy_ratio"]
    if ratio > UNIFORM_ENTROPY_TOL:
        gain = "几乎完全均匀 —— 池化层没有学出空间结构，这是训练/容量问题，不是方法问题"
    elif ratio > WEAK_ENTROPY_TOL:
        gain = "只有很弱的集中 —— 有结构但很淡，仍贴着均匀分布"
    else:
        gain = "明显集中 —— 池化层确实学出了空间偏好"

    coc = stats["center_over_corners"]
    if coc > 1.5:
        bias = f"中心权重是四角的 {coc:.2f} 倍 —— 更像'看中间'的位置先验"
    elif coc < 0.67:
        bias = f"中心权重只有四角的 {coc:.2f} 倍 —— 偏好边缘，反常，值得查"
    else:
        bias = f"中心/四角 = {coc:.2f} —— 没有明显的中心偏置"

    lines = [
        f"注意力池化诊断 | 实验: {exp_name} | 划分: {split}",
        f"checkpoint: epoch={ckpt_epoch} val_acc={ckpt_val_acc}",
        f"样本数: {stats['num_samples']} | patch 数: {stats['num_patches']}",
        "",
        "指标                             实测        均匀分布基线",
        f"  平均最大注意力权重            {stats['max_weight']:.5f}     "
        f"{stats['max_weight_uniform']:.5f}",
        f"  最大注意力权重中位数          {stats['max_weight_median']:.5f}     "
        f"{stats['max_weight_uniform']:.5f}",
        f"  注意力熵                      {stats['entropy']:.4f}     "
        f"{stats['entropy_uniform']:.4f}",
        f"  熵 / log(P)                   {stats['entropy_ratio']:.4f}     "
        f"1.0000",
        f"  覆盖 50% 质量所需 patch 数    {stats['patches_for_50pct']:.1f}        "
        f"{stats['patches_for_50pct_uniform']:.1f}",
        "",
        f"结论: {gain}",
        f"分布: {bias}",
        "",
    ]
    return "\n".join(lines)


def _to_display(img_tensor, mean, std):
    """反归一化回 [0,1] 的 HWC numpy，用于叠加显示。"""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = (img_tensor.cpu() * s + m).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _upsample(heat, grid, size):
    """(grid, grid) -> (size, size)，最近邻放大（224/14 = 16，整除）。"""
    factor = max(1, size // grid)
    return np.kron(heat, np.ones((factor, factor)))


def _short(name):
    """类别名太长，图里放不下：Tomato___Late_blight -> Tomato / Late blight。"""
    return name.replace("___", " / ").replace("_", " ")


def plot_samples(dataset, indices, attn, preds, labels, conf, class_names, path,
                 grid, mean, std):
    """正确 / 错误各展示若干张：上排原图，下排注意力叠加。"""
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 6.4), squeeze=False)

    for col, idx in enumerate(indices):
        idx = int(idx)
        image, _ = dataset[idx]        # 走数据集自己的 transform，与模型看到的一致
        size = int(image.shape[-1])
        img = _to_display(image, mean, std)

        heat = attn[idx].reshape(grid, grid).numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        heat = _upsample(heat, grid, size)

        correct = preds[idx] == labels[idx]
        color = "green" if correct else "crimson"
        axes[0][col].set_title(
            f"{'OK' if correct else 'WRONG'}  {conf[idx]:.2f}\n"
            f"true: {_short(class_names[labels[idx]])}\n"
            f"pred: {_short(class_names[preds[idx]])}",
            fontsize=7, color=color,
        )
        for row in range(2):
            axes[row][col].imshow(img)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
        axes[1][col].imshow(heat, cmap="inferno", alpha=0.5)
        axes[0][col].set_ylabel("image" if col == 0 else "")
        axes[1][col].set_ylabel("attention" if col == 0 else "")

    fig.suptitle("Attention pooling, most confident samples "
                 "(left: correct, right: wrong)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_mean_map(attn, path, grid):
    """全样本的平均注意力图，以及逐样本最大权重的分布。"""
    p = attn.shape[1]
    mean_map = attn.mean(dim=0).reshape(grid, grid).numpy()
    rel = mean_map / (1.0 / p)             # 1.0 = 与均匀分布持平
    dev = max(float(np.abs(rel - 1.0).max()), 1e-6)

    max_w = attn.max(dim=1).values.numpy()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    im0 = axes[0].imshow(mean_map, cmap="inferno")
    axes[0].set_title(f"Mean attention map ({grid}x{grid})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rel, cmap="RdBu_r", vmin=1 - dev, vmax=1 + dev)
    axes[1].set_title("Same, relative to uniform (red = above average)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].hist(max_w, bins=40, color="steelblue")
    axes[2].axvline(1.0 / p, color="crimson", linestyle="--",
                    label=f"uniform = 1/{p} = {1.0 / p:.5f}")
    axes[2].axvline(float(np.median(max_w)), color="black", linestyle="-",
                    label=f"median = {np.median(max_w):.5f}")
    axes[2].set_xlabel("max attention weight per sample")
    axes[2].set_ylabel("count")
    axes[2].set_title("Per-sample peak attention")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg["experiment"]["name"]
    # 与 train.py / evaluate.py 保持一致的后缀规则
    if args.shots:
        exp_name = f"{exp_name}_{args.shots}shot"
        cfg["experiment"]["name"] = exp_name
    if args.exp_suffix:
        exp_name = f"{exp_name}_{args.exp_suffix}"
        cfg["experiment"]["name"] = exp_name

    results_dir, ckpt_dir, _ = resolve_output_dirs(cfg)
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = args.checkpoint or os.path.join(str(ckpt_dir), "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {ckpt_path}\n请先运行 train.py 训练模型。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = create_datasets(cfg)
    class_names = load_class_names(cfg)
    num_classes = len(class_names)

    ckpt = torch.load(ckpt_path, map_location=device)
    restore_model_config(cfg, ckpt)
    model = build_model(cfg, num_classes, class_names).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if getattr(model, "mode", None) != "attn":
        raise SystemExit(
            f"该 checkpoint 的 model.mode = {getattr(model, 'mode', None)!r}，"
            f"不是 attn —— 本脚本只诊断注意力池化。"
            f"请确认 --config 指向 configs/clip_attn.yaml。"
        )

    patches = model.num_tokens - 1
    grid = math.isqrt(patches)
    assert grid * grid == patches, (
        f"patch 数 {patches} 不是完全平方数，无法还原成方阵热力图"
    )

    dataset = datasets[args.split]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg["data"].get("num_workers", 8), pin_memory=True)

    print("=" * 64)
    print(f"注意力池化诊断 | 实验: {exp_name} | 划分: {args.split} | 设备: {device}")
    print(f"checkpoint: {ckpt_path}")

    attn, preds, labels, conf = collect_attention(model, loader, device)
    stats = attention_stats(attn, grid)

    report = build_report(stats, args.split, exp_name,
                          ckpt.get("epoch"), round(float(ckpt.get("val_acc", 0.0)), 4))
    print(report)

    report_path = os.path.join(results_dir, f"attn_analysis_{args.split}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"诊断报告已保存: {report_path}")

    # ---------------- 可解释性图 ----------------
    normalize = [t for t in dataset.transform.transforms if isinstance(t, T.Normalize)]
    if not normalize:
        raise SystemExit("数据集的 transform 里没有 Normalize，无法还原图像显示")
    mean = [float(v) for v in normalize[0].mean]
    std = [float(v) for v in normalize[0].std]

    k = max(1, args.num_samples)
    correct_idx = (preds == labels).nonzero(as_tuple=True)[0]
    wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
    # 取"错得最自信"的：那是最能说明模型到底在看哪儿的样本；
    # 正确样本同理取最自信的。都是确定性的，不引入随机数。
    correct_pick = correct_idx[conf[correct_idx].argsort(descending=True)][:k]
    wrong_pick = wrong_idx[conf[wrong_idx].argsort(descending=True)][:k]
    indices = torch.cat([correct_pick, wrong_pick])

    if len(indices) == 0:
        print("没有可展示的样本，跳过热力图。")
        return
    if len(wrong_pick) == 0:
        print(f"⚠ 该划分上没有错分样本（正确率 100%？），只展示正确样本。")

    heat_path = os.path.join(results_dir, "attention_heatmap.png")
    plot_samples(dataset, indices, attn, preds, labels, conf, class_names,
                 heat_path, grid, mean, std)
    print(f"注意力热力图已保存: {heat_path}")

    map_path = os.path.join(results_dir, "attention_mean_map.png")
    plot_mean_map(attn, map_path, grid)
    print(f"平均注意力图已保存: {map_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
