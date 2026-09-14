"""模型构建入口。

当前仅支持第一阶段的两个 CNN 基线；CLIP 相关封装（clip_model.py /
adapter.py）将在第二阶段加入。
"""

from models.cnn import build_resnet50, build_vit_b16

# model.name -> 构建函数
_BUILDERS = {
    "resnet50": build_resnet50,
    "vit_b_16": build_vit_b16,
    "vit_b16": build_vit_b16,   # 兼容配置文件里的简写
}


def build_model(cfg, num_classes):
    """按配置构建模型。

    Args:
        cfg: 配置 dict，需含 model.name / model.type / model.pretrained
        num_classes: 类别数（由数据集推断后传入，而非信任配置里的硬编码值）

    Returns:
        nn.Module
    """
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "cnn")
    name = model_cfg.get("name", "resnet50")

    if model_type == "clip":
        raise NotImplementedError(
            "CLIP 模型将在第二阶段实现（models/clip_model.py）。"
            "当前 train.py 只支持 CNN 基线：resnet50 / vit_b_16。"
        )

    if name not in _BUILDERS:
        raise ValueError(f"未知模型 '{name}'，可选：{sorted(_BUILDERS)}")

    pretrained = bool(model_cfg.get("pretrained", True))
    return _BUILDERS[name](num_classes=num_classes, pretrained=pretrained)


__all__ = ["build_model", "build_resnet50", "build_vit_b16"]
