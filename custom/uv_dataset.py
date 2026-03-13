# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
uv单模态用的
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

from custom.dual_dataset import load_class_names, parse_yolo_label
from rfdetr.datasets.coco import make_coco_transforms_square_div_64


class UVYoloDetection(VisionDataset):
    """Single-modal UV dataset backed by the paired UV/White directory layout."""

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

        yaml_path = self.dataset_dir / "dataset_dual.yaml"
        if class_names is not None:
            self.classes = class_names
        else:
            self.classes = load_class_names(str(yaml_path))

        uv_dir = self.dataset_dir / "images" / split
        label_dir = self.dataset_dir / "labels" / split

        self.samples: List[Dict[str, str]] = []
        for filename in sorted(os.listdir(uv_dir)):
            if not filename.endswith((".bmp", ".png", ".jpg", ".jpeg")):
                continue

            stem = Path(filename).stem
            if not stem.endswith("_uv"):
                continue

            self.samples.append(
                {
                    "uv": str(uv_dir / filename),
                    "label": str(label_dir / f"{stem}.txt"),
                }
            )

        assert len(self.samples) > 0, (
            f"No valid UV samples found in {uv_dir}. "
            f"Expected files like pair_0001_uv.bmp."
        )

        self.ids = list(range(len(self.samples)))
        self.coco = _UVCocoLikeAPI(self.classes, self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        img_uv = Image.open(sample["uv"]).convert("RGB")
        width, height = img_uv.size
        boxes, classes = parse_yolo_label(sample["label"], width, height)

        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(classes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": classes,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(height), int(width)]),
            "size": torch.as_tensor([int(height), int(width)]),
        }

        if self._transforms is not None:
            img_uv, target = self._transforms(img_uv, target)

        return img_uv, target


class _UVCocoLikeAPI:
    """Minimal COCO-like wrapper used by RF-DETR evaluation."""

    def __init__(self, classes: List[str], samples: List[Dict[str, str]]):
        self.classes = classes
        self.samples = samples
        self.dataset = self._build_coco_dataset()
        self.imgs = {img["id"]: img for img in self.dataset["images"]}
        self.anns = {ann["id"]: ann for ann in self.dataset["annotations"]}
        self.cats = {cat["id"]: cat for cat in self.dataset["categories"]}

        self.imgToAnns: Dict[int, List[dict]] = {}
        for ann in self.dataset["annotations"]:
            self.imgToAnns.setdefault(ann["image_id"], []).append(ann)
        for img_id in self.imgs:
            self.imgToAnns.setdefault(img_id, [])

        self.catToImgs: Dict[int, List[int]] = {cat_id: [] for cat_id in self.cats}
        for ann in self.dataset["annotations"]:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if img_id not in self.catToImgs.get(cat_id, []):
                self.catToImgs.setdefault(cat_id, []).append(img_id)

    def _build_coco_dataset(self) -> dict:
        images = []
        annotations = []
        categories = []

        for idx, name in enumerate(self.classes):
            categories.append({"id": idx, "name": name, "supercategory": "none"})

        ann_id = 0
        for img_id, sample in enumerate(self.samples):
            img = Image.open(sample["uv"])
            width, height = img.size
            img.close()

            images.append(
                {
                    "id": img_id,
                    "file_name": os.path.basename(sample["uv"]),
                    "height": height,
                    "width": width,
                }
            )

            boxes, classes = parse_yolo_label(sample["label"], width, height)
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

    def getImgIds(self, imgIds=None, catIds=None):
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
        catNms = catNms or []
        catIds = catIds or []
        cats = self.dataset["categories"]
        if catNms:
            cats = [cat for cat in cats if cat["name"] in catNms]
        if catIds:
            cats = [cat for cat in cats if cat["id"] in catIds]
        return [cat["id"] for cat in cats]

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
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
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.anns[idx] for idx in ids if idx in self.anns]

    def loadCats(self, ids=None):
        if ids is None:
            return list(self.cats.values())
        ids = ids if isinstance(ids, list) else [ids]
        return [self.cats[idx] for idx in ids if idx in self.cats]

    def loadImgs(self, ids=None):
        if ids is None:
            return []
        ids = ids if isinstance(ids, list) else [ids]
        return [self.imgs[idx] for idx in ids if idx in self.imgs]


def build_uv_dataset(
    image_set: str,
    dataset_dir: str,
    resolution: int,
    class_names: Optional[List[str]] = None,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> UVYoloDetection:
    """Build the UV-only dataset using the same paired dataset root."""

    transforms = make_coco_transforms_square_div_64(
        image_set=image_set,
        resolution=resolution,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        patch_size=patch_size,
        num_windows=num_windows,
    )

    split_map = {"train": "train", "val": "val", "test": "val"}
    split = split_map.get(image_set, image_set)

    return UVYoloDetection(
        dataset_dir=dataset_dir,
        split=split,
        transforms=transforms,
        class_names=class_names,
    )
