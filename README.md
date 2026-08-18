# PlantDisease-CLIP

**基于 CLIP 的植物病害识别研究**（硕士论文实验代码）

使用视觉-语言模型（Vision-Language Model）在公开农产品病害数据集 **PlantVillage** 上进行病害分类，并设计改进方法提升分类性能。本项目代码按科研论文实验规范组织，强调可复现、参数明确、输出完整。

> 说明：论文中数据集统一视为"桥梁病害识别"对象，本仓库代码以 PlantVillage 作为公开替代数据集。

---

## 1. 环境要求

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX 4090D ×1（24GB） |
| CPU / 内存 | 16 vCPU / 62GB |
| 操作系统 | Ubuntu 20.04 |
| Python | 3.8 |
| PyTorch | 1.10.0 |
| CUDA | 11.3 |

代码运行于 AutoDL 云端（本地 Windows 只写代码不上传运行）。

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 目录结构

```
PlantDisease-CLIP
├── configs/              # 实验配置文件（yaml）
│   ├── base.yaml            # 公共基础配置
│   ├── resnet50.yaml        # Baseline A：ResNet50
│   ├── vit_b16.yaml         # Baseline B：ViT-B/16
│   ├── clip_zeroshot.yaml   # Baseline C：CLIP Zero-shot
│   └── clip_finetune.yaml   # Baseline D：CLIP Fine-tuning
├── dataset/              # 数据模块（本阶段已完成）
│   ├── dataset.py           # 数据集加载 / 统计 / 划分
│   └── preprocess.py        # CNN / CLIP 两种预处理
├── models/               # 待实现：clip_model.py, adapter.py
├── train.py              # 待实现：通用训练脚本
├── test.py               # 待实现
├── evaluate.py           # 待实现：指标与图表
├── utils/                # 待实现：种子 / 指标 / 可视化
├── results/              # 实验输出（论文用）
├── split/                # 数据划分索引（自动生成）
├── requirements.txt
└── README.md
```

## 4. 数据集

- **名称**：PlantVillage Dataset
- **路径**：`~/autodl-tmp/PlantVillage-Dataset-master/raw/color`
- **格式**：标准 ImageFolder（`root/类别名/图像.JPG`）
- **类别数**：约 38 类（Apple、Tomato、Potato、Corn、Grape、Peach、Pepper、Strawberry、Soybean、Orange 等）
- **输入**：224 × 224 RGB 图像 → 输出对应病害类别

```bash
color/
├── Apple___Apple_scab/
│   └── xxx.JPG
├── Tomato___Late_blight/
└── ...
```

## 5. 数据划分

- 比例：**Train 70% / Val 15% / Test 15%**
- 方式：**按类别分层（stratified）**划分，保证每类样本在三个子集中均有代表；对样本极少的类别有兜底逻辑（见 `dataset.py::stratified_split`）
- 随机种子：**固定 seed=42**，划分结果可复现
- 索引落盘：首次运行时将划分索引保存到 `split/`，后续所有实验（CNN / CLIP）**复用同一份划分**，保证对比实验公平性

| 文件 | 说明 |
|---|---|
| `split/train_indices.json` | 训练集索引 |
| `split/val_indices.json` | 验证集索引 |
| `split/test_indices.json` | 测试集索引 |
| `split/class_names.json` | 类别名（与标签一一对应） |

如需重新划分：删除 `split/` 目录，或调用 `create_datasets(cfg, force_resplit=True)`。

## 6. 配置说明

采用 **yaml 配置 + 实验覆盖** 模式：`train.py`（第二阶段实现）先加载 `configs/base.yaml` 作为默认值，再用各实验配置文件覆盖差异字段。因此实验配置只需写与默认不同的内容。

| 配置文件 | 模型 | 用途 |
|---|---|---|
| `base.yaml` | — | 数据路径、划分、输出目录、通用训练参数 |
| `resnet50.yaml` | ResNet50（ImageNet 预训练） | 传统 CNN 基线 |
| `vit_b16.yaml` | ViT-B/16（ImageNet 预训练） | Transformer 视觉基线 |
| `clip_zeroshot.yaml` | OpenCLIP ViT-B/32（冻结） | 原始 CLIP 迁移能力验证 |
| `clip_finetune.yaml` | OpenCLIP ViT-B/32（微调） | 领域微调效果验证 |

关键配置项：

- `model.type`: `cnn`（ResNet/ViT）或 `clip`（CLIP），决定使用哪种预处理
- `data.root`: 数据集根目录
- `data.split.seed`: 划分随机种子（默认 42）
- `prompt.template`: CLIP 文本模板，`{}` 替换为具体病害类别，如 `a photo of a diseased {} leaf`
- `training.*`: 训练超参（epochs=50、batch_size、lr 等）

## 7. Dataset 模块使用

### 快速验证（统计 + 划分演示）

```bash
python -m dataset.dataset --root ~/autodl-tmp/PlantVillage-Dataset-master/raw/color
```

会输出：类别数、每类样本数、类别分布图（`results/data_distribution.png`）、划分后三个子集大小及比例。

### 在代码中使用

```python
import yaml
from dataset.dataset import create_datasets

with open("configs/resnet50.yaml", "r") as f:
    cfg = yaml.safe_load(f)

datasets = create_datasets(cfg)      # {"train": ..., "val": ..., "test": ...}

from torch.utils.data import DataLoader
train_loader = DataLoader(datasets["train"], batch_size=64, shuffle=True, num_workers=8)
```

> CLIP 实验时将配置中 `model.type` 设为 `clip`，自动使用 CLIP 预处理（含专用归一化参数与 BICUBIC 插值）。

### 预处理策略

| 模型 | 训练 | 验证/测试 |
|---|---|---|
| CNN | RandomResizedCrop(224) + 翻转 + 旋转 + 颜色抖动 | Resize(256) + CenterCrop(224) |
| CLIP | RandomResizedCrop(224, bicubic) + 翻转 | Resize(224) + CenterCrop(224) |

两者归一化参数不同（ImageNet 统计量 vs CLIP 统计量），已在 `dataset/preprocess.py` 中区分。

## 8. 当前阶段状态（Phase 1）

**已实现：**

- ✅ `configs/`：base + 4 个实验配置
- ✅ `dataset/`：数据读取、数据统计、分层划分（70/15/15、固定种子）、CNN/CLIP 两种 Dataset
- ✅ `requirements.txt`、`.gitignore`、`README.md`

**待实现（Phase 1 后半段）：**

- ⏳ `train.py`：通用训练框架（GPU / 50 epochs / 保存 best 模型 / loss 与 accuracy 记录 / early stopping）
- ⏳ `test.py`、`evaluate.py`：指标计算与论文图表输出
- ⏳ `models/`：ResNet50 / ViT / CLIP 模型封装
- ⏳ Baseline A–D 训练与结果整理

## 9. 实验输出约定（论文用）

训练完成后统一生成到 `results/`：

```
results/
├── train_loss.png          # 训练 loss 曲线
├── val_accuracy.png        # 验证 accuracy 曲线
├── confusion_matrix.png    # 混淆矩阵
├── classification_report.txt  # Accuracy/Precision/Recall/F1 + Per-class
├── data_distribution.png   # 数据分布图
└── experiment_results.csv  # 各实验指标汇总（论文对比表用）
```

模型权重保存到 `checkpoints/best_model.pth`。

## 10. 复现性

- 数据划分：固定 seed=42 + 分层划分 + 索引落盘复用
- 训练：`training.seed` 固定（后续 train.py 中统一设置 torch/numpy/random 种子）
- 每个实验一个独立 yaml 配置，记录全部超参，可直接复现

## 11. 后续阶段

| 阶段 | 内容 |
|---|---|
| Phase 1 | 数据读取 / 统计 / 划分 / CLIP Dataset / **Baseline 训练** |
| Phase 2 | CLIP Fine-tuning / Adapter / Prompt 实验 |
| Phase 3 | 改进模型 / 消融实验 / 可视化分析 |
