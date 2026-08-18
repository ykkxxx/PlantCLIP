"""图像预处理（transform）定义。

区分两条 pipeline：
- cnn  : ResNet / ViT（torchvision）使用的 ImageNet 归一化与增强
- clip : CLIP（OpenCLIP ViT-B/32）官方预处理，含专用 mean/std 与 BICUBIC 插值

输入统一为 224x224 RGB（文档第 8 节要求）。
"""

import torchvision.transforms as T

# -------------------- 归一化参数 --------------------
# ImageNet 统计量（CNN 模型用）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP 官方预处理统计量（OpenAI CLIP / OpenCLIP 通用）
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_cnn_transform(split="train", image_size=224):
    """CNN 模型变换（ImageNet 归一化）。

    训练: 随机裁剪 + 水平翻转 + 旋转 + 颜色抖动（缓解病害数据光照/角度差异）
    验证/测试: Resize 256 -> CenterCrop 224（标准评估流程）
    """
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_clip_transform(split="train", image_size=224):
    """CLIP 模型变换（对齐 OpenAI CLIP 官方预处理，BICUBIC 插值）。

    训练: RandomResizedCrop(scale=(0.08, 1.0)) + 水平翻转
    验证/测试: Resize(image_size) -> CenterCrop(image_size)
    """
    bicubic = T.InterpolationMode.BICUBIC
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=bicubic),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return T.Compose([
        T.Resize(image_size, interpolation=bicubic),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_transform(model_type="cnn", split="train", image_size=224):
    """统一入口：按模型类型返回对应 transform。

    Args:
        model_type: "cnn"（ResNet/ViT）或 "clip"（CLIP）
        split: "train" / "val" / "test"
        image_size: 输入分辨率，默认 224

    Returns:
        torchvision.transforms.Compose
    """
    if model_type == "clip":
        return build_clip_transform(split, image_size)
    return build_cnn_transform(split, image_size)
