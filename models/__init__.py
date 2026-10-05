"""模型构建入口。

- CNN 基线：ResNet50 / ViT-B/16（torchvision + ImageNet 预训练）
- CLIP 微调 / 线性探测 / Adapter：CLIPClassifier（open_clip + 线性分类头）
- Prompt Learning：CoOpClassifier（冻住双塔，只学文本侧上下文向量）

三种产出的都是普通 nn.Module（forward(images) -> logits），
train.py / evaluate.py 共用同一套训练与评估流程。
"""

from models.clip_classifier import CLIPClassifier
from models.cnn import build_resnet50, build_vit_b16
from models.coop import CoOpClassifier

# model.name -> 构建函数（仅 CNN 基线；CLIP 走 build_model 里的独立分支）
_BUILDERS = {
    "resnet50": build_resnet50,
    "vit_b_16": build_vit_b16,
    "vit_b16": build_vit_b16,   # 兼容配置文件里的简写
}


def build_model(cfg, num_classes, class_names=None):
    """按配置构建模型。

    Args:
        cfg: 配置 dict，需含 model.type / model.name
        num_classes: 类别数（由数据集推断后传入，而非信任配置里的硬编码值）
        class_names: 类别名列表（顺序与标签一致）。只有 CoOp 需要它来构造
            文本 prompt，其余模型传 None 即可。

    Returns:
        nn.Module
    """
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "cnn")
    name = model_cfg.get("name", "resnet50")

    # CLIP 分支：model.type=clip 同时决定数据侧改用 CLIP 官方预处理
    # （见 dataset/preprocess.py::build_transform），两者必须一致。
    if model_type == "clip":
        mode = model_cfg.get("mode", "finetune")

        if mode == "coop":
            if not class_names:
                raise ValueError(
                    "mode='coop' 需要 class_names（文本侧要按类别名构造 prompt），"
                    "请确认 split/class_names.json 存在且已传入 build_model"
                )
            return CoOpClassifier(
                class_names=class_names,
                backbone=model_cfg.get("clip_backbone", "ViT-B-16"),
                pretrained=model_cfg.get("pretrained", "openai"),
                n_ctx=int(model_cfg.get("n_ctx", 16)),
                ctx_init=model_cfg.get("ctx_init", "a photo of a"),
            )

        return CLIPClassifier(
            num_classes=num_classes,
            backbone=model_cfg.get("clip_backbone", "ViT-B-16"),
            pretrained=model_cfg.get("pretrained", "openai"),
            mode=mode,
            dropout=float(model_cfg.get("dropout", 0.0)),
            bottleneck=int(model_cfg.get("adapter_bottleneck", 64)),
            alpha=float(model_cfg.get("adapter_alpha", 0.2)),
        )

    if name not in _BUILDERS:
        raise ValueError(f"未知模型 '{name}'，可选：{sorted(_BUILDERS)}")

    pretrained = bool(model_cfg.get("pretrained", True))
    return _BUILDERS[name](num_classes=num_classes, pretrained=pretrained)


__all__ = ["build_model", "build_resnet50", "build_vit_b16",
           "CLIPClassifier", "CoOpClassifier"]
