# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件实现当前项目使用的双模态成对检测数据集。
功能：从 `images/` 中读取 UV 图像、从 `images_white/` 中读取 White 图像，并结合
      `labels/` 下的 YOLO 标注文件，组织成 RF-DETR 训练与评估可直接使用的数据结构。

结构概览：
  第一部分：导入依赖
  第二部分：YOLO 标签解析与类别名读取工具
  第三部分：双模态数据集类 `DualModalYoloDetection`
  第四部分：供 CocoEvaluator 使用的 COCO 兼容接口 `_DualCocoLikeAPI`
  第五部分：数据集构建函数 `build_dual_dataset`

设计原则：
  - UV 是主模态，因此标注、框坐标、`orig_size`、`size` 都以 UV 图像为准。
  - White 仅作为与 UV 一一配对的辅助输入图像，不独立持有检测标签。
  - 数据增强入口接受 `(img_uv, img_white, target)`，保证主模态语义始终清晰。
"""

# ========== 第一部分：导入依赖 ==========
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from torchvision.datasets import VisionDataset

from custom.dataset_auto_coco import resolve_roboflow_coco_dataset_dir
from custom.dataset_layout import (
    is_probable_uv_image,
    list_image_files,
    resolve_split_layout,
    resolve_white_path_for_uv,
)


# ========== 第二部分：YOLO 标签解析与类别名读取工具 ==========
def parse_yolo_label(label_path: str, img_w: int, img_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    读取单张图像对应的 YOLO 标签文件，并转换为 RF-DETR 更容易消费的 `xyxy` 框格式。

    YOLO 每一行的格式是：
        class_id center_x center_y width height
    其中坐标和宽高都已经归一化到 `[0, 1]`。

    Args:
        label_path:
            标签文件路径。
        img_w:
            UV 图像宽度，单位为像素。
        img_h:
            UV 图像高度，单位为像素。

    Returns:
        boxes:
            形状为 `(N, 4)` 的浮点张量，格式为 `xyxy`。
        classes:
            形状为 `(N,)` 的整型张量，对应每个框的类别 id。
    """
    boxes: List[List[float]] = []
    classes: List[int] = []

    # 如果标签文件不存在，直接返回空张量。
    # 这样可以兼容“存在图片但没有目标”的样本。
    if not os.path.exists(label_path):
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    with open(label_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            cls_id = int(parts[0])

            # YOLO 坐标是归一化的，因此需要恢复到像素坐标。
            center_x = float(parts[1]) * img_w
            center_y = float(parts[2]) * img_h
            box_w = float(parts[3]) * img_w
            box_h = float(parts[4]) * img_h

            # 将中心点表示转换为左上角 / 右下角表示。
            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0
            x2 = center_x + box_w / 2.0
            y2 = center_y + box_h / 2.0

            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)

    # 如果文件存在但没有有效标注，同样返回空张量。
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.int64)


def load_class_names(data_yaml_path: str) -> List[str]:
    """
    从 `dataset_dual.yaml` 中读取类别名称列表。

    支持两种常见写法：
      1. `names` 是字典：`{0: "NPML", 1: "PML", 2: "PM"}`
      2. `names` 是列表：`["NPML", "PML", "PM"]`
    """
    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[key] for key in sorted(names.keys())]
    return list(names)


# ========== 第三部分：双模态数据集类 ==========
class DualModalYoloDetection(VisionDataset):
    """
    双模态成对检测数据集。

    目录约定如下：
        dataset_dir/
            images/{train,val}/pair_xxxx_uv.bmp
            images_white/{train,val}/pair_xxxx_white.bmp
            labels/{train,val}/pair_xxxx_uv.txt
            dataset_dual.yaml

    每个样本返回：
        `(img_uv, img_white, target)`

    其中：
      - `img_uv` 是主模态图像
      - `img_white` 是辅助模态图像
      - `target` 中的框与尺寸信息全部以 UV 图像为准
    """

    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",
        transforms=None,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__(dataset_dir)

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self._transforms = transforms

        # ---------- 第一步：确定类别名 ----------
        yaml_path = self.dataset_dir / "dataset_dual.yaml"
        if class_names is not None:
            self.classes = class_names
        else:
            self.classes = load_class_names(str(yaml_path))

        self.coco_root = resolve_roboflow_coco_dataset_dir(
            dataset_dir=self.dataset_dir,
            class_names=self.classes,
            log_prefix="[Dataset-Dual]",
        )
        self._coco_image_id_by_path: Dict[str, int] = {}

        # ---------- 第二步：扫描成对样本 ----------
        layout = resolve_split_layout(
            dataset_dir=self.dataset_dir,
            split=split,
            require_white=True,
            require_labels=True,
        )
        uv_dir = layout.uv_dir
        white_dir = layout.white_dir
        label_dir = layout.label_dir

        self.pairs: List[Dict[str, str]] = []

        # 遍历 UV 图像目录，以 UV 文件名为主索引 White 图像与标签。
        for image_path in list_image_files(uv_dir):
            if not is_probable_uv_image(image_path):
                continue

            white_path = resolve_white_path_for_uv(image_path, white_dir)
            label_path = label_dir / f"{image_path.stem}.txt"

            # 只有同时存在配对 White 图像时，才认为这是一个有效样本。
            if white_path is not None and white_path.exists():
                self.pairs.append(
                    {
                        "uv": str(image_path),
                        "white": str(white_path),
                        "label": str(label_path),
                    }
                )

        # 如果一个有效 pair 都没有，直接报错，避免后续静默训练空数据。
        assert len(self.pairs) > 0, (
            f"在 {uv_dir} 中没有找到任何有效的 UV/White 成对样本。"
            f"请检查 images/、images_white/ 和 labels/ 的目录结构。"
        )

        # `ids` 保持和样本索引一致，供评估逻辑使用。
        self.ids = list(range(len(self.pairs)))

        # 提前构建 COCO 兼容 API，方便直接接入 RF-DETR 现有评估流程。
        self.coco = self._build_coco_api()
        if self._coco_image_id_by_path:
            self.ids = [
                self._coco_image_id_by_path.get(str(Path(pair["uv"]).resolve()), idx)
                for idx, pair in enumerate(self.pairs)
            ]

    def __len__(self) -> int:
        """返回数据集样本数。"""
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        读取一个双模态样本。

        返回结构：
            `(img_uv, img_white, target)`

        其中 `target` 的字段与 RF-DETR 原始数据流保持兼容。
        """
        pair = self.pairs[idx]

        # ---------- 第一步：读取配对图像 ----------
        # 统一转为 RGB，保证后续 transform 和 backbone 输入格式稳定。
        img_uv = Image.open(pair["uv"]).convert("RGB")
        img_white = Image.open(pair["white"]).convert("RGB")

        # ---------- 第二步：读取 UV 标注 ----------
        # 因为标注定义在 UV 图像上，所以宽高也使用 UV 图像尺寸。
        width, height = img_uv.size
        image_id_value = self._coco_image_id_by_path.get(str(Path(pair["uv"]).resolve()), idx)
        boxes, classes = self._load_boxes_and_classes(
            label_path=pair["label"],
            image_id=image_id_value,
            width=width,
            height=height,
        )

        # ---------- 第三步：对框做基本清洗 ----------
        # 先 clamp 到图像边界内，再过滤无效框。
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # `area` 与 `iscrowd` 是 COCO 兼容评估接口常见字段。
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(classes),), dtype=torch.int64)

        # `orig_size` 和 `size` 在增强前相同；
        # 后续若 transform 改变尺寸，会重新写入 `size`。
        target = {
            "boxes": boxes,
            "labels": classes,
            "image_id": torch.tensor([image_id_value]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
        }

        # ---------- 第四步：执行双模态同步增强 ----------
        if self._transforms is not None:
            img_uv, img_white, target = self._transforms(img_uv, img_white, target)

        return img_uv, img_white, target

    def _load_boxes_and_classes(
        self,
        label_path: str,
        image_id: int,
        width: int,
        height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._coco_image_id_by_path:
            ann_ids = self.coco.getAnnIds(imgIds=[image_id])
            annotations = self.coco.loadAnns(ann_ids)
            if len(annotations) == 0:
                return (
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.int64),
                )

            boxes: List[List[float]] = []
            classes: List[int] = []
            for annotation in annotations:
                x1, y1, box_w, box_h = annotation["bbox"]
                boxes.append([x1, y1, x1 + box_w, y1 + box_h])
                classes.append(int(annotation["category_id"]))

            return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
                classes, dtype=torch.int64
            )

        return parse_yolo_label(label_path, width, height)

    def _build_coco_api(self):
        """
        优先复用真实 COCO 标注；若不存在，再回退到本地构造的兼容接口。

        RF-DETR 的评估流程依赖 COCO API 风格的数据访问接口，
        这里优先使用自动检测/生成出的 COCO 标注，以保证单模态与双模态评估口径一致。
        """
        annotation_path = self._resolve_coco_annotation_file()
        if annotation_path is not None and annotation_path.exists():
            coco = COCO(str(annotation_path))
            self._coco_image_id_by_path = {
                self._normalize_coco_image_path(image["file_name"]): int(image["id"])
                for image in coco.dataset.get("images", [])
            }
            return coco

        self._coco_image_id_by_path = {}
        return _DualCocoLikeAPI(self.classes, self.pairs)

    def _resolve_coco_annotation_file(self) -> Optional[Path]:
        split_to_dir = {"train": "train", "val": "valid", "test": "test"}
        coco_split = split_to_dir.get(self.split, self.split)
        annotation_path = self.coco_root / coco_split / "_annotations.coco.json"
        if annotation_path.exists():
            return annotation_path
        return None

    def _normalize_coco_image_path(self, file_name: str) -> str:
        image_path = Path(file_name)
        if image_path.is_absolute():
            return str(image_path.resolve())

        split_to_dir = {"train": "train", "val": "valid", "test": "test"}
        coco_split = split_to_dir.get(self.split, self.split)
        return str((self.coco_root / coco_split / image_path).resolve())


# ========== 第四部分：供 CocoEvaluator 使用的 COCO 兼容接口 ==========
class _DualCocoLikeAPI:
    """
    一个最小版本的 COCO 风格接口。

    目标不是完整复刻 pycocotools，而是提供 `CocoEvaluator` 在当前项目中实际会访问到的字段和方法。
    """

    def __init__(self, classes: List[str], pairs: List[Dict[str, str]]):
        self.classes = classes
        self.pairs = pairs

        # 先构建标准 COCO 风格字典。
        self.dataset = self._build_coco_dataset()

        # 再建立若干常用索引，方便后续按 id 快速查找。
        self.imgs = {img["id"]: img for img in self.dataset["images"]}
        self.anns = {ann["id"]: ann for ann in self.dataset["annotations"]}
        self.cats = {cat["id"]: cat for cat in self.dataset["categories"]}

        # ---------- 构建 imgToAnns ----------
        self.imgToAnns: Dict[int, List[dict]] = {}
        for ann in self.dataset["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []
            self.imgToAnns[img_id].append(ann)

        # 确保没有标注的图像也有空列表，避免评估阶段 KeyError。
        for img_id in self.imgs:
            if img_id not in self.imgToAnns:
                self.imgToAnns[img_id] = []

        # ---------- 构建 catToImgs ----------
        self.catToImgs: Dict[int, List[int]] = {cat_id: [] for cat_id in self.cats}
        for ann in self.dataset["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if img_id not in self.catToImgs.get(cat_id, []):
                self.catToImgs.setdefault(cat_id, []).append(img_id)

    def _build_coco_dataset(self) -> dict:
        """
        将当前双模态数据集组织为 COCO 风格的 `dataset` 字典。

        这里图像信息仍然以 UV 图像为主，因为标注和评估都在 UV 坐标系下进行。
        """
        images = []
        annotations = []
        categories = []

        # 先写类别表。
        for idx, name in enumerate(self.classes):
            categories.append({"id": idx, "name": name, "supercategory": "none"})

        ann_id = 0
        for img_id, pair in enumerate(self.pairs):
            # 读取 UV 图像尺寸，作为该样本的唯一尺寸基准。
            img = Image.open(pair["uv"])
            width, height = img.size
            img.close()

            images.append(
                {
                    "id": img_id,
                    "file_name": os.path.basename(pair["uv"]),
                    "height": height,
                    "width": width,
                }
            )

            # 将 YOLO 标签逐条转成 COCO annotation。
            boxes, classes = parse_yolo_label(pair["label"], width, height)
            for index in range(len(classes)):
                x1, y1, x2, y2 = boxes[index].tolist()
                box_w = x2 - x1
                box_h = y2 - y1

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(classes[index].item()),
                        "bbox": [x1, y1, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        return {"images": images, "annotations": annotations, "categories": categories}

    # ---------- 以下方法提供 COCO API 兼容访问 ----------
    def getImgIds(self, imgIds=None, catIds=None):
        """
        返回满足筛选条件的图像 id 列表。

        该方法名称沿用 COCO API 原始命名，以减少评估适配代码。
        """
        imgIds = imgIds or []
        catIds = catIds or []

        result = set(imgIds) if imgIds else set(self.imgs.keys())
        if catIds:
            matching = set()
            for cat_id in catIds:
                matching.update(self.catToImgs.get(cat_id, []))
            result &= matching
        return list(result)

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        """返回满足筛选条件的类别 id 列表。"""
        catNms = catNms or []
        catIds = catIds or []
        cats = self.dataset["categories"]

        if catNms:
            cats = [cat for cat in cats if cat["name"] in catNms]
        if catIds:
            cats = [cat for cat in cats if cat["id"] in catIds]
        return [cat["id"] for cat in cats]

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        """返回满足筛选条件的 annotation id 列表。"""
        imgIds = imgIds or []
        catIds = catIds or []
        areaRng = areaRng or []

        if len(imgIds) == 0:
            anns = self.dataset["annotations"]
        else:
            anns = []
            for img_id in imgIds:
                anns.extend(self.imgToAnns.get(img_id, []))

        if catIds:
            anns = [ann for ann in anns if ann["category_id"] in catIds]
        if len(areaRng) == 2:
            anns = [ann for ann in anns if areaRng[0] <= ann["area"] <= areaRng[1]]
        if iscrowd is not None:
            anns = [ann for ann in anns if ann["iscrowd"] == iscrowd]
        return [ann["id"] for ann in anns]

    def loadAnns(self, ids=None):
        """按 id 返回 annotation 列表。"""
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.anns[idx] for idx in ids if idx in self.anns]

    def loadCats(self, ids=None):
        """按 id 返回 category 列表。"""
        if ids is None:
            return list(self.cats.values())
        ids = ids if isinstance(ids, list) else [ids]
        return [self.cats[idx] for idx in ids if idx in self.cats]

    def loadImgs(self, ids=None):
        """按 id 返回 image 元信息列表。"""
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.imgs[idx] for idx in ids if idx in self.imgs]


# ========== 第五部分：数据集构建函数 ==========
def build_dual_dataset(
    image_set: str,
    dataset_dir: str,
    resolution: int,
    class_names: Optional[List[str]] = None,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> DualModalYoloDetection:
    """
    根据 RF-DETR 训练/验证阶段约定构建双模态数据集实例。

    Args:
        image_set:
            数据集划分名称，通常是 `train` / `val` / `test`。
        dataset_dir:
            数据集根目录。
        resolution:
            目标输入分辨率。
        class_names:
            可选的类别名覆盖；若不提供，则从 yaml 自动读取。
        multi_scale:
            是否启用多尺度训练。
        expanded_scales:
            是否启用扩展尺度集合。
        patch_size:
            backbone patch 大小。
        num_windows:
            windowed attention 使用的窗口数。

    Returns:
        `DualModalYoloDetection` 实例。
    """
    # 延迟导入增强模块，避免在只做静态分析或导入本文件时引入不必要依赖。
    from custom.dual_transforms import make_dual_transforms

    transforms = make_dual_transforms(
        image_set=image_set,
        resolution=resolution,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        patch_size=patch_size,
        num_windows=num_windows,
    )

    # 为了兼容 RF-DETR 现有调用习惯，这里把 `test` 映射到 `val`。
    split_map = {"train": "train", "val": "val", "test": "val"}
    split = split_map.get(image_set, image_set)

    dataset = DualModalYoloDetection(
        dataset_dir=dataset_dir,
        split=split,
        transforms=transforms,
        class_names=class_names,
    )

    return dataset
