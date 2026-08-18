"""PlantVillage 数据集加载、统计与划分。

功能：
- PlantVillageDataset : 基于 (image_path, label) 列表的轻量 Dataset
- build_index         : 用 torchvision ImageFolder 建立索引与类别名
- stratified_split    : 按类别分层划分 Train/Val/Test (70/15/15)，固定随机种子
- save/load split     : 划分索引落盘，保证多实验共享同一份划分
- create_datasets     : 按配置一键构建 train/val/test Dataset（CNN 或 CLIP 预处理）
- run_statistics      : 数据统计（类别数、每类样本数）并保存类别分布图

注意：
- 图像统一为 224x224 RGB；CNN 与 CLIP 使用不同预处理，见 preprocess.py
- 独立运行验证：python -m dataset.dataset --root <data_root>
"""

import json
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from dataset.preprocess import build_transform


class PlantVillageDataset(Dataset):
    """基于 (image_path, label) 列表的轻量 Dataset。

    Args:
        paths: 图像绝对路径列表
        labels: 与 paths 对应的标签列表
        indices: 可选，仅取其中部分索引（用于按划分取子集）
        transform: torchvision transform，None 时返回 PIL 图像
    """

    def __init__(self, paths, labels, indices=None, transform=None):
        if indices is None:
            indices = list(range(len(paths)))
        self.paths = [paths[i] for i in indices]
        self.labels = [labels[i] for i in indices]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # PlantVillage 图像均为 JPG，统一转 RGB
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def build_index(data_root):
    """读取 PlantVillage 数据集，返回 (paths, labels, class_names)。

    class_names 按目录名排序（与 torchvision ImageFolder.classes 一致），
    标签 0..C-1 与 class_names 一一对应，顺序稳定，便于索引复用。
    """
    dataset = datasets.ImageFolder(root=data_root)
    paths = [p for p, _ in dataset.samples]
    labels = [l for _, l in dataset.samples]
    return paths, labels, dataset.classes


def stratified_split(labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """按类别分层划分索引，保证每类样本在三个子集中均有代表。

    对样本极少的类别做兜底：
    - 该类样本 <= 1 张：全部进训练集
    - 该类样本 >= 2 张：保证测试集至少 1 张
    - 该类样本 >= 3 张：训练/验证/测试均至少 1 张
    同一 seed 下结果完全可复现（文档第 7 节要求）。

    Returns:
        (train_idx, val_idx, test_idx)，均为升序排列的索引列表
    """
    rng = random.Random(seed)
    num_classes = max(labels) + 1
    train_idx, val_idx, test_idx = [], [], []

    for c in range(num_classes):
        cls_idx = [i for i, l in enumerate(labels) if l == c]
        rng.shuffle(cls_idx)
        n = len(cls_idx)

        if n <= 1:  # 样本过少，全部进训练集
            train_idx += cls_idx
            continue

        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        else:  # n == 2：训练 1 / 测试 1
            n_train, n_val = 1, 0

        n_test = n - n_train - n_val
        train_idx += cls_idx[:n_train]
        val_idx += cls_idx[n_train:n_train + n_val]
        test_idx += cls_idx[n_train + n_val:]

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def save_split_indices(indices_dict, class_names, save_dir):
    """将划分索引与类别名保存为 json，便于复用与复现。"""
    os.makedirs(save_dir, exist_ok=True)
    for name, idx in indices_dict.items():
        with open(os.path.join(save_dir, f"{name}_indices.json"), "w") as f:
            json.dump(idx, f)
    with open(os.path.join(save_dir, "class_names.json"), "w") as f:
        json.dump(list(class_names), f, ensure_ascii=False, indent=2)
    print(f"[dataset] 划分索引已保存至 {save_dir}/")


def load_split_indices(save_dir):
    """读取已保存的划分索引；文件缺失时返回 None。"""
    names = ("train", "val", "test")
    if not all(os.path.exists(os.path.join(save_dir, f"{n}_indices.json")) for n in names):
        return None
    indices = {}
    for n in names:
        with open(os.path.join(save_dir, f"{n}_indices.json"), "r") as f:
            indices[n] = json.load(f)
    return indices


def create_datasets(cfg, force_resplit=False):
    """按配置构建 train/val/test Dataset。

    优先复用已保存的划分索引（保证所有实验使用同一份数据划分）；
    首次运行或 force_resplit=True 时重新划分并落盘到 split/。

    Args:
        cfg: 配置字典，需含 data.root；可选 data.image_size / data.split /
             data.split_dir / model.type（"cnn" 或 "clip"）
        force_resplit: 强制重新划分

    Returns:
        {"train": Dataset, "val": Dataset, "test": Dataset}
    """
    data_root = os.path.expanduser(cfg["data"]["root"])
    image_size = cfg["data"].get("image_size", 224)
    model_type = cfg["model"].get("type", "cnn")
    split_cfg = cfg["data"].get("split", {})
    split_dir = cfg["data"].get("split_dir", "split")

    train_ratio = split_cfg.get("train_ratio", 0.7)
    val_ratio = split_cfg.get("val_ratio", 0.15)
    test_ratio = split_cfg.get("test_ratio", 0.15)
    seed = split_cfg.get("seed", 42)

    paths, labels, class_names = build_index(data_root)
    print(f"[dataset] 数据根目录: {data_root}")
    print(f"[dataset] 总样本数: {len(paths)}，类别数: {len(class_names)}")

    indices = None if force_resplit else load_split_indices(split_dir)

    if indices is None:
        train_idx, val_idx, test_idx = stratified_split(
            labels, train_ratio, val_ratio, test_ratio, seed
        )
        indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        save_split_indices(indices, class_names, split_dir)
    else:
        print(f"[dataset] 复用已有划分索引: {split_dir}/")
        train_idx, val_idx, test_idx = indices["train"], indices["val"], indices["test"]

    datasets_dict = {}
    for split_name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        transform = build_transform(model_type, split_name, image_size)
        datasets_dict[split_name] = PlantVillageDataset(
            paths, labels, indices=idx, transform=transform
        )
        print(f"[dataset] {split_name:<5s}: {len(idx):>6d} 样本")

    return datasets_dict


def run_statistics(data_root, save_path="results/data_distribution.png"):
    """统计类别数、每类样本数，并保存类别分布条形图。

    Args:
        data_root: 数据集根目录
        save_path: 分布图保存路径

    Returns:
        (total, num_classes, per_class_counts)
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    _, labels, class_names = build_index(data_root)
    counts = Counter(labels)

    print(f"\n[统计] 类别数: {len(class_names)}，总样本数: {len(labels)}\n")
    for i, name in enumerate(class_names):
        print(f"  {i:>3d}  {name:<45s}  {counts[i]:>6d}")
    print()

    # 类别分布条形图
    ordered_counts = [counts[i] for i in range(len(class_names))]
    plt.figure(figsize=(max(10, len(class_names) * 0.6), 5))
    plt.bar(range(len(class_names)), ordered_counts)
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.ylabel("样本数量")
    plt.title(f"PlantVillage 类别分布 (共 {len(class_names)} 类，{len(labels)} 张)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[统计] 类别分布图已保存: {save_path}")

    return len(labels), len(class_names), dict(counts)


if __name__ == "__main__":
    # 独立验证：python -m dataset.dataset --root <data_root>
    import argparse

    parser = argparse.ArgumentParser(description="PlantVillage 数据统计与划分验证")
    parser.add_argument(
        "--root", type=str,
        default="~/autodl-tmp/PlantVillage-Dataset-master/raw/color",
    )
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    run_statistics(root)

    # 划分演示（固定 seed，可重复运行验证结果一致）
    _, labels, _ = build_index(root)
    tr, va, te = stratified_split(labels)
    total = len(labels)
    print(f"[划分演示] train={len(tr)}  val={len(va)}  test={len(te)}")
    print(f"[划分演示] 比例: {len(tr)/total:.1%} / {len(va)/total:.1%} / {len(te)/total:.1%}")
