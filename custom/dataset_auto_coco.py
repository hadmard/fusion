"""
文件说明：本文件负责为当前 UV/White 成对数据集自动探测或生成 COCO 缓存。
功能说明：当训练/评估流程要求 Roboflow COCO 目录结构时，本文件会优先复用已有 COCO
缓存；若原始数据仍是 `images + images_white + labels` 的成对布局，则自动转换出
 `_auto_coco/` 缓存，避免手工维护第二份标注。

结构概览：
  第一部分：常量与基础读取工具
  第二部分：COCO 根目录解析入口
  第三部分：成对数据集转 COCO 缓存
  第四部分：缓存补全与元数据写入
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import yaml
from PIL import Image

from custom.dataset_layout import is_probable_uv_image, list_image_files, resolve_split_layout

AUTO_COCO_DIRNAME = "_auto_coco"
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg"}


# ========== 第一部分：常量与基础读取工具 ==========
def load_class_names_from_yaml(yaml_path: Path) -> list[str]:
    """从 `dataset_dual.yaml` 中读取类别名称，兼容 dict/list 两种写法。"""
    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    names = data.get("names", {})
    if isinstance(names, dict):
        return [names[key] for key in sorted(names.keys())]
    return list(names)


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> tuple[list[list[float]], list[int]]:
    """把 YOLO 归一化标注恢复为像素级 `xyxy` 框，供 COCO 缓存生成使用。"""
    boxes: list[list[float]] = []
    classes: list[int] = []

    if not label_path.exists():
        return boxes, classes

    with open(label_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            cls_id = int(parts[0])
            center_x = float(parts[1]) * img_w
            center_y = float(parts[2]) * img_h
            box_w = float(parts[3]) * img_w
            box_h = float(parts[4]) * img_h

            x1 = center_x - box_w / 2.0
            y1 = center_y - box_h / 2.0
            x2 = center_x + box_w / 2.0
            y2 = center_y + box_h / 2.0

            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)

    return boxes, classes


def is_standard_roboflow_coco_root(dataset_dir: str | Path) -> bool:
    """判断目录是否已经是 RF-DETR 可直接消费的 Roboflow COCO 结构。"""
    root = Path(dataset_dir)
    return (root / "train" / "_annotations.coco.json").exists() and (
        root / "valid" / "_annotations.coco.json"
    ).exists()


# ========== 第二部分：COCO 根目录解析入口 ==========
def resolve_roboflow_coco_dataset_dir(
    dataset_dir: str | Path,
    class_names: Optional[list[str]] = None,
    log_prefix: str = "[Dataset]",
) -> Path:
    """
    返回训练流程应当使用的 COCO 根目录。

    优先级：
      1. 原始目录本身已经是 COCO 布局，直接复用。
      2. `_auto_coco/` 缓存已经存在，直接复用。
      3. 否则从成对数据集布局生成一份缓存。
    """
    source_root = Path(dataset_dir).resolve()
    cache_root = source_root / AUTO_COCO_DIRNAME

    if is_standard_roboflow_coco_root(source_root):
        _ensure_test_annotation(source_root, source_root / "valid" / "_annotations.coco.json")
        print(f"{log_prefix} Detected COCO dataset at: {source_root}")
        return source_root

    if _is_generated_coco_cache(cache_root):
        _ensure_test_annotation(cache_root, cache_root / "valid" / "_annotations.coco.json")
        print(f"{log_prefix} Reusing cached COCO dataset at: {cache_root}")
        return cache_root

    print(f"{log_prefix} No COCO annotations found. Converting paired dataset to COCO cache...")
    return convert_paired_dataset_to_coco_cache(
        dataset_dir=source_root,
        output_dir=cache_root,
        class_names=class_names,
        log_prefix=log_prefix,
    )


# ========== 第三部分：成对数据集转 COCO 缓存 ==========
def convert_paired_dataset_to_coco_cache(
    dataset_dir: str | Path,
    output_dir: str | Path,
    class_names: Optional[list[str]] = None,
    log_prefix: str = "[Dataset]",
) -> Path:
    """把 `images/images_white/labels` 布局转换成 Roboflow 风格的 COCO 缓存。"""
    source_root = Path(dataset_dir).resolve()
    output_root = Path(output_dir).resolve()

    _validate_paired_layout(source_root)
    classes = class_names or load_class_names_from_yaml(source_root / "dataset_dual.yaml")

    output_root.mkdir(parents=True, exist_ok=True)

    split_specs = {
        "train": "train",
        "valid": "val",
    }
    for coco_split, source_split in split_specs.items():
        split_dir = output_root / coco_split
        split_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = split_dir / "_annotations.coco.json"
        dataset = _build_coco_dataset_for_split(source_root, source_split, classes)
        with open(annotation_path, "w", encoding="utf-8") as handle:
            json.dump(dataset, handle, ensure_ascii=False, indent=2)

    _ensure_test_annotation(output_root, output_root / "valid" / "_annotations.coco.json")
    _write_cache_metadata(output_root, source_root, classes)

    print(f"{log_prefix} COCO cache generated at: {output_root}")
    return output_root


def _validate_paired_layout(source_root: Path) -> None:
    """在真正生成缓存前先做目录完整性检查，兼容标准布局与服务器图片布局。"""
    yaml_path = source_root / "dataset_dual.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {yaml_path}")

    for split in ["train", "val"]:
        resolve_split_layout(
            dataset_dir=source_root,
            split=split,
            require_white=True,
            require_labels=True,
        )


def _is_generated_coco_cache(cache_root: Path) -> bool:
    """判断 `_auto_coco/` 是否已经包含可复用的最小 COCO 标注集。"""
    required = [
        cache_root / "train" / "_annotations.coco.json",
        cache_root / "valid" / "_annotations.coco.json",
    ]
    return all(path.exists() for path in required)


def _build_coco_dataset_for_split(
    source_root: Path,
    source_split: str,
    classes: Iterable[str],
) -> dict:
    """为单个 split 生成 COCO `images / annotations / categories` 结构。"""
    layout = resolve_split_layout(
        dataset_dir=source_root,
        split=source_split,
        require_white=True,
        require_labels=True,
    )
    uv_dir = layout.uv_dir
    label_dir = layout.label_dir

    images = []
    annotations = []
    categories = [
        {"id": class_id, "name": class_name, "supercategory": "none"}
        for class_id, class_name in enumerate(classes)
    ]

    ann_id = 0
    image_id = 0
    for image_path in list_image_files(uv_dir):
        if not is_probable_uv_image(image_path):
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        images.append(
            {
                "id": image_id,
                # 使用绝对路径，避免训练入口切换工作目录后找不到缓存中的图片。
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{image_path.stem}.txt"
        boxes, label_ids = parse_yolo_label(label_path, width, height)
        for box, label_id in zip(boxes, label_ids):
            x1, y1, x2, y2 = box
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            if box_w <= 0 or box_h <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label_id),
                    "bbox": [float(x1), float(y1), float(box_w), float(box_h)],
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    if not images:
        raise FileNotFoundError(f"No UV images found under {uv_dir}")

    return {
        "info": {
            "description": "Auto-generated COCO cache for paired UV/White dataset",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_split": source_split,
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ========== 第四部分：缓存补全与元数据写入 ==========
def _ensure_test_annotation(root: Path, fallback_annotation: Path) -> None:
    """
    为缺失 `test/` 标注的场景补一份兜底文件。

    RF-DETR 的部分 Roboflow 数据流默认会探测 `test/_annotations.coco.json`。
    这里复用 valid 标注是为了兼容训练/评估入口，而不是为了引入新的测试集语义。
    """
    test_dir = root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_annotation = test_dir / "_annotations.coco.json"
    if test_annotation.exists():
        return

    with open(fallback_annotation, "r", encoding="utf-8") as handle:
        dataset = json.load(handle)
    dataset.setdefault("info", {})
    dataset["info"]["source_split"] = dataset["info"].get("source_split", "val")
    dataset["info"]["test_fallback"] = "valid"

    with open(test_annotation, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)


def _write_cache_metadata(output_root: Path, source_root: Path, classes: list[str]) -> None:
    """记录缓存来源与类别信息，方便后续判断缓存是否可信、是否需要重建。"""
    metadata = {
        "source_root": str(source_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "class_names": classes,
    }
    with open(output_root / "_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
