"""把 PlantVillage 的类别名转成 CLIP 可用的自然语言 prompt。

PlantVillage 的目录名形如：
    Apple___Apple_scab
    Potato___healthy
    Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot

不能直接塞进模板，有两个原因：
1. 下划线和三重下划线不是自然语言，CLIP 的文本编码器读不出语义；
2. **healthy 类会被 "a photo of a diseased {} leaf" 描述成"患病"，语义完全反了**
   —— 这会让健康类几乎全部误判。

所以先把类别名解析成 (植物, 病害)，健康类走单独的模板。
"""

import re

# 植物名规范化：把括号里的补充说明去掉，统一成 CLIP 语料里常见的说法
_PLANT_ALIASES = {
    "corn (maize)": "corn",
    "pepper, bell": "bell pepper",
    "cherry (including sour)": "cherry",
}

# 病害名规范化：PlantVillage 的目录名里有几个直译成英文不通顺，
# 会显著拉低 zero-shot 效果，这里手工改成自然的说法。
_DISEASE_ALIASES = {
    "cercospora leaf spot gray leaf spot": "cercospora leaf spot and gray leaf spot",
    "spider mites two-spotted spider mite": "two-spotted spider mite damage",
    "haunglongbing (citrus greening)": "citrus greening",
    "esca (black measles)": "esca black measles",
    "tomato yellow leaf curl virus": "yellow leaf curl virus",
    "tomato mosaic virus": "mosaic virus",
}

# 患病类模板（多条用于 prompt ensembling）
DEFAULT_DISEASE_TEMPLATES = [
    "a photo of a {plant} leaf with {disease}",
    "a close-up photo of {disease} on a {plant} leaf",
    "a {plant} leaf infected with {disease}",
]

# 健康类模板：必须单独一套，否则会把健康叶片描述成病害
DEFAULT_HEALTHY_TEMPLATES = [
    "a photo of a healthy {plant} leaf",
    "a close-up photo of a healthy {plant} leaf",
]


def parse_class_name(raw):
    """把类别目录名解析成 (植物名, 病害名)，都已转小写。

    >>> parse_class_name("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
    ('corn', 'cercospora leaf spot gray leaf spot')
    >>> parse_class_name("Potato___healthy")
    ('potato', 'healthy')
    """
    if "___" in raw:
        plant_part, disease_part = raw.split("___", 1)
    else:
        plant_part, disease_part = raw, ""

    plant = re.sub(r"\s+", " ", plant_part.replace("_", " ")).strip().lower()
    plant = _PLANT_ALIASES.get(plant, plant)

    disease = re.sub(r"\s+", " ", disease_part.replace("_", " ")).strip().lower()
    disease = _DISEASE_ALIASES.get(disease, disease)
    return plant, disease


def is_healthy(disease):
    """健康类（含没有病害名的情况）走单独的模板。"""
    return disease in ("", "healthy")


def _fix_articles(text):
    """把 "a apple leaf" 修正成 "an apple leaf"。

    模板里写死了不定冠词 "a"，但 apple / orange / esca 等词以元音开头。
    只匹配独立的单词 "a"，因此不会误伤 "leaf"、"esca" 这类词内部的字母。
    """
    return re.sub(r"\ba (?=[aeiou])", "an ", text)


def format_template(template, plant, disease):
    """同时兼容两种写法：

    - 命名占位符（推荐）："a photo of a {plant} leaf with {disease}"
    - 文档里的单占位符：  "a photo of a diseased {} leaf"
    """
    if "{plant}" in template or "{disease}" in template:
        text = template.format(plant=plant, disease=disease)
    else:
        text = template.format(disease)
    return _fix_articles(re.sub(r"\s+", " ", text)).strip()


def build_class_prompts(class_names, disease_templates=None, healthy_templates=None):
    """为每个类别生成 prompt 文本。

    Args:
        class_names: 类别名列表，顺序与标签 0..C-1 对应
        disease_templates: 患病类模板（str 或 list），默认用 DEFAULT_DISEASE_TEMPLATES
        healthy_templates: 健康类模板（str 或 list），默认用 DEFAULT_HEALTHY_TEMPLATES

    Returns:
        prompts[i] = 第 i 类的 prompt 文本列表（长度 = 该类用到的模板数）
    """
    disease_templates = disease_templates or DEFAULT_DISEASE_TEMPLATES
    healthy_templates = healthy_templates or DEFAULT_HEALTHY_TEMPLATES
    if isinstance(disease_templates, str):
        disease_templates = [disease_templates]
    if isinstance(healthy_templates, str):
        healthy_templates = [healthy_templates]

    prompts = []
    for raw in class_names:
        plant, disease = parse_class_name(raw)
        templates = healthy_templates if is_healthy(disease) else disease_templates
        prompts.append([format_template(t, plant, disease) for t in templates])
    return prompts
