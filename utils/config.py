"""配置加载：以 base.yaml 为默认值，用实验配置覆盖。

这样每个实验配置文件只需写与默认不同的字段，同时保证合并后的 cfg
一定含有 data / model / training / output / experiment 各节，
避免下游 create_datasets(cfg) 因缺少 model 节而 KeyError。
"""

import os
from pathlib import Path

import yaml

# 项目根目录（utils/ 的上一级），无论从哪个 cwd 运行都能正确定位
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base, override):
    """递归合并 dict，override 中的值优先。"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path, base_name="base.yaml"):
    """加载实验配置并与 base.yaml 合并。

    Args:
        config_path: 实验配置路径，可写相对项目根的路径（如 "configs/resnet50.yaml"）
        base_name: 基础配置文件名（位于 configs/ 下）

    Returns:
        合并后的配置 dict
    """
    with open(CONFIG_DIR / base_name, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    exp_path = Path(config_path)
    if not exp_path.is_absolute():
        candidate = PROJECT_ROOT / exp_path
        exp_path = candidate if candidate.exists() else exp_path
    with open(exp_path, "r", encoding="utf-8") as f:
        exp_cfg = yaml.safe_load(f) or {}

    cfg = deep_merge(cfg, exp_cfg)

    # ---- 补齐必需字段，保证下游不 KeyError ----
    cfg.setdefault("model", {})
    cfg["model"].setdefault("type", "cnn")          # cnn / clip
    cfg.setdefault("training", {})
    cfg.setdefault("output", {})
    cfg.setdefault("data", {})
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("name", "exp")

    # ---- 把划分目录固定到项目根下 ----
    # create_datasets 里 split_dir 是相对路径、按 cwd 解析；这里转成绝对路径，
    # 保证在任何目录下运行 train/evaluate 都读写同一份划分，实验之间才可比。
    split_dir = cfg["data"].get("split_dir", "split")
    if not os.path.isabs(split_dir):
        cfg["data"]["split_dir"] = str(PROJECT_ROOT / split_dir)

    return cfg


def resolve_output_dirs(cfg):
    """返回 (results_dir, checkpoint_dir, log_dir) 三个绝对路径。

    结果与权重按 experiment.name 分子目录，避免多个基线互相覆盖；
    日志统一放在 logs/ 下。
    """
    output = cfg.get("output", {})
    exp_name = cfg["experiment"]["name"]
    results_dir = PROJECT_ROOT / output.get("results_dir", "results") / exp_name
    checkpoint_dir = PROJECT_ROOT / output.get("checkpoint_dir", "checkpoints") / exp_name
    log_dir = PROJECT_ROOT / output.get("log_dir", "logs")
    return results_dir, checkpoint_dir, log_dir
