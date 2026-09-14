"""分类指标计算（纯 numpy 实现，不依赖 scikit-learn）。

包含：Accuracy、混淆矩阵、Per-class Accuracy、Precision/Recall/F1
（逐类 + macro + weighted），以及生成论文用分类报告文本。
"""

import numpy as np


def accuracy(preds, labels):
    """整体准确率。"""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    return float((preds == labels).mean())


def confusion_matrix(preds, labels, num_classes):
    """构建混淆矩阵，cm[i, j] = 真实类别 i 被预测为类别 j 的样本数。

    Args:
        preds: 预测标签数组
        labels: 真实标签数组
        num_classes: 类别总数

    Returns:
        (num_classes, num_classes) 的 int64 矩阵
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(labels), np.asarray(preds)):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm):
    """每类准确率（即该类召回率）：对角线 / 该类真实样本总数。"""
    total = cm.sum(axis=1)
    correct = np.diag(cm)
    return np.where(total > 0, correct / np.maximum(total, 1), 0.0)


def precision_recall_f1(cm):
    """由混淆矩阵计算逐类 P/R/F1，以及 macro / weighted 平均。

    macro 平均只统计「真实样本数 > 0」的类别，避免数据集中不存在的类别
    拉低平均值（与 sklearn 的默认行为一致）。

    Returns:
        dict: {
            "per_class": (precision, recall, f1) 三个数组,
            "macro": (P, R, F1),
            "weighted": (P, R, F1),
        }
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    pred_sum = cm.sum(axis=0)   # 每列和：被预测为该类的样本总数
    true_sum = cm.sum(axis=1)   # 每行和：该类的真实样本总数

    precision = np.where(pred_sum > 0, tp / np.maximum(pred_sum, 1), 0.0)
    recall = np.where(true_sum > 0, tp / np.maximum(true_sum, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / np.maximum(denom, 1e-12), 0.0)

    present = true_sum > 0
    if present.any():
        macro = (float(precision[present].mean()),
                 float(recall[present].mean()),
                 float(f1[present].mean()))
    else:
        macro = (0.0, 0.0, 0.0)

    total = true_sum.sum()
    weights = true_sum / total if total > 0 else np.zeros_like(true_sum)
    weighted = (float((precision * weights).sum()),
                float((recall * weights).sum()),
                float((f1 * weights).sum()))

    return {"per_class": (precision, recall, f1), "macro": macro, "weighted": weighted}


def build_report(class_names, preds, labels):
    """生成分类报告文本与指标字典（论文 §6 要求的完整输出）。

    Args:
        class_names: 类别名列表（顺序与标签 0..C-1 对应）
        preds: 预测标签数组
        labels: 真实标签数组

    Returns:
        (report_text, metrics_dict, cm)
    """
    num_classes = len(class_names)
    cm = confusion_matrix(preds, labels, num_classes)
    acc = accuracy(preds, labels)
    prf = precision_recall_f1(cm)
    pca = per_class_accuracy(cm)

    precision, recall, f1 = prf["per_class"]
    macro = prf["macro"]
    weighted = prf["weighted"]

    lines = [
        "=" * 82,
        "Classification Report",
        "=" * 82,
        f"Overall Accuracy  : {acc:.4f}",
        f"Macro    P/R/F1   : {macro[0]:.4f} / {macro[1]:.4f} / {macro[2]:.4f}",
        f"Weighted P/R/F1   : {weighted[0]:.4f} / {weighted[1]:.4f} / {weighted[2]:.4f}",
        "-" * 82,
        f"{'class':<42s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'acc':>10s}",
        "-" * 82,
    ]
    for i, name in enumerate(class_names):
        lines.append(f"{name:<42s}{precision[i]:>10.4f}{recall[i]:>10.4f}"
                     f"{f1[i]:>10.4f}{pca[i]:>10.4f}")
    lines.append("=" * 82)

    metrics = {
        "accuracy": acc,
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_f1": macro[2],
        "weighted_precision": weighted[0],
        "weighted_recall": weighted[1],
        "weighted_f1": weighted[2],
    }
    return "\n".join(lines), metrics, cm
