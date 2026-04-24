"""
文件说明：本文件定义 UV/White 双模态训练与验证阶段的数据增强。
功能：在保证 UV 主模态语义始终清晰的前提下，对 UV 与 White 图像执行同步几何增强、
      固定的 dataset 侧尺寸对齐、张量化与归一化，并同步维护检测框坐标。

----------------------------------------------------------------------
当前训练阶段增强配置（依执行顺序）：

  几何增强：
    1. DualRandomHorizontalFlip         p=0.5
       UV 与 White 同步水平翻转，框坐标同步修正。

    2. DualRandomSelect                 p=0.5 走 PML 裁剪分支，0.5 走普通随机裁剪分支
       分支A：DualPMLGuidedCrop(label=1) — 围绕 PML 框外扩裁剪
       分支B：DualRandomCrop（384~600）— 普通随机裁剪

    3. DualResizePad
       保持裁剪区域纵横比，仅在裁剪区域大于基准画布时等比缩小，
       再同步 padding 到 batch resize 的基准画布。

  batch 级 resize：
    4. 多尺度随机 resize 不在 dataset 单图阶段执行，而是在训练循环中
       对整个 NestedTensor batch 使用同一个 scale 执行。

  最终后处理：
    5. DualToTensor + DualNormalize
       转张量并以 ImageNet 均值/方差标准化；框转为归一化 cxcywh。


结构概览：
  第一部分：导入依赖
  第二部分：增强实现（可多个）
  第三部分：同步几何增强
  第四部分：张量化与归一化
  第五部分：实现函数 `make_dual_transforms`

核心约束：
  - 所有 transform 的输入输出签名统一为：
        `(img_uv, img_white, target) -> (img_uv, img_white, target)`
  - 所有边界框变换都以 UV 图像为基准，因为标注来源于 UV。
  - 当前训练链路不启用模态特异性外观扰动，只保留同步 flip/crop。
"""

# ========== 第一部分：导入依赖 ==========
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision.transforms as TT
import torchvision.transforms.functional as F

from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.box_ops import box_xyxy_to_cxcywh
from rfdetr.util.misc import interpolate


def hflip(image, target):
    """对 UV 图像做水平翻转，并同步更新 target 中的几何字段。"""
    flipped_image = F.hflip(image)

    if target is None:
        return flipped_image, None

    target = target.copy()
    width, _ = image.size

    if "boxes" in target:
        boxes = target["boxes"].clone()
        x_min = boxes[:, 0].clone()
        x_max = boxes[:, 2].clone()
        boxes[:, 0] = width - x_max
        boxes[:, 2] = width - x_min
        target["boxes"] = boxes

    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return flipped_image, target


def crop(image, target, region):
    """对 UV 图像做裁剪，并把 boxes/area/masks 同步到裁剪后的坐标系。"""
    top, left, height, width = region
    cropped_image = F.crop(image, top, left, height, width)

    if target is None:
        return cropped_image, None

    target = target.copy()
    target["size"] = torch.as_tensor([int(height), int(width)])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        offset = torch.as_tensor(
            [left, top, left, top],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        max_xy = torch.as_tensor(
            [width, height],
            dtype=boxes.dtype,
            device=boxes.device,
        )

        cropped_boxes = boxes - offset
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_xy)
        cropped_boxes = cropped_boxes.clamp(min=0)

        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        fields.append("boxes")

    if "masks" in target:
        target["masks"] = target["masks"][:, top : top + height, left : left + width]
        fields.append("masks")

    if "boxes" in target or "masks" in target:
        if "boxes" in target:
            reshaped_boxes = target["boxes"].reshape(-1, 2, 2)
            keep = torch.all(reshaped_boxes[:, 1, :] > reshaped_boxes[:, 0, :], dim=1)
        else:
            keep = target["masks"].flatten(1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def resize_pad(image, target, size: int, fill: int = 0):
    """
    等比缩小过大的图像，再居中 padding 到正方形尺寸。

    小于目标尺寸的 crop 不做放大，只 padding。这样 PML/PM 局部裁剪不会被
    额外插值放大，但相对整图 resize 仍保留更多原始细节。
    """
    original_width, original_height = image.size
    scale = min(
        1.0,
        float(size) / max(original_width, 1),
        float(size) / max(original_height, 1),
    )
    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    if new_width != original_width or new_height != original_height:
        image = F.resize(image, (new_height, new_width))

    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    pad_right = size - new_width - pad_left
    pad_bottom = size - new_height - pad_top
    image = F.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)

    if target is None:
        return image, None

    target = target.copy()

    if "boxes" in target:
        boxes = target["boxes"] * torch.as_tensor(
            [scale, scale, scale, scale],
            dtype=target["boxes"].dtype,
            device=target["boxes"].device,
        )
        boxes = boxes + torch.as_tensor(
            [pad_left, pad_top, pad_left, pad_top],
            dtype=boxes.dtype,
            device=boxes.device,
        )
        target["boxes"] = boxes

    if "area" in target:
        target["area"] = target["area"] * (scale * scale)

    if "masks" in target:
        masks = target["masks"]
        if new_width != original_width or new_height != original_height:
            masks = interpolate(
                masks[:, None].float(),
                (new_height, new_width),
                mode="nearest",
            )[:, 0] > 0.5
        target["masks"] = torch_F.pad(
            masks,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=False,
        )

    target["size"] = torch.as_tensor([int(size), int(size)])
    return image, target


# ========== 第二部分：增强实现（可多个） ==========
class DualCompose:
    """
    双模态版本的 Compose。

    作用与 `torchvision.transforms.Compose` 类似，
    但这里每个变换都要同时接收：
        `img_uv, img_white, target`
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img_uv, img_white, target):
        # 依次执行每个变换，并把上一个变换的输出交给下一个变换。
        for transform in self.transforms:
            img_uv, img_white, target = transform(img_uv, img_white, target)
        return img_uv, img_white, target

    def __repr__(self) -> str:
        lines = [self.__class__.__name__ + "("]
        for transform in self.transforms:
            lines.append(f"    {transform}")
        lines.append(")")
        return "\n".join(lines)


# ========== 第三部分：同步几何增强 ==========
class DualRandomHorizontalFlip:
    """对 UV 与 White 同步执行水平翻转，并同步修正 UV 框坐标。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img_uv, img_white, target):
        if random.random() < self.p:
            # UV 侧使用项目已有的 `hflip`，这样可以自动同步 target 中的 boxes。
            img_uv, target = hflip(img_uv, target)

            # White 侧只需要做同样的几何变换，不需要单独改框。
            img_white = F.hflip(img_white)
        return img_uv, img_white, target


class DualSquareResize:
    """
    将 UV 与 White 同步缩放到正方形尺寸。

    这是当前训练与验证流程中最稳定的尺寸对齐方式。
    """

    def __init__(self, sizes: List[int]):
        self.sizes = sizes

    def __call__(self, img_uv, img_white, target):
        size = random.choice(self.sizes)

        # 先记录 UV 原始尺寸，用于同步缩放框坐标和 area。
        orig_w_uv, orig_h_uv = img_uv.size
        img_uv_resized = F.resize(img_uv, (size, size))
        ratio_width = size / max(orig_w_uv, 1)
        ratio_height = size / max(orig_h_uv, 1)

        target = target.copy()

        # 按缩放比例更新框坐标。
        if "boxes" in target:
            target["boxes"] = target["boxes"] * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height],
                dtype=torch.float32,
            )

        # `area` 也需要同步缩放。
        if "area" in target:
            target["area"] = target["area"] * (ratio_width * ratio_height)

        # `size` 表示变换后的尺寸。
        target["size"] = torch.tensor([size, size])

        # 如果做实例分割，则掩码也要同步缩放。
        if "masks" in target:
            target["masks"] = interpolate(
                target["masks"][:, None].float(), (size, size), mode="nearest"
            )[:, 0] > 0.5

        # White 图像跟随 UV 一起调整到完全相同的输入尺寸。
        img_white_resized = F.resize(img_white, (size, size))
        return img_uv_resized, img_white_resized, target


class DualResizePad:
    """
    将 UV 与 White 同步 padding 到正方形尺寸。

    若输入图像任意边长超过目标尺寸，则先等比缩小到可放入画布；
    若输入已经小于目标尺寸，则不放大，只居中 padding。
    """

    def __init__(self, sizes: List[int], fill: int = 0):
        self.sizes = sizes
        self.fill = fill

    def __call__(self, img_uv, img_white, target):
        size = random.choice(self.sizes)
        img_uv, target = resize_pad(img_uv, target, size=size, fill=self.fill)
        img_white, _ = resize_pad(img_white, None, size=size, fill=self.fill)
        return img_uv, img_white, target


class DualRandomSelect:
    """在两条增强分支之间随机选择一条执行。"""

    def __init__(self, transforms1, transforms2, p: float = 0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img_uv, img_white, target):
        if random.random() < self.p:
            return self.transforms1(img_uv, img_white, target)
        return self.transforms2(img_uv, img_white, target)


class DualRandomCrop:
    """对两路图像执行同步随机裁剪，并以 UV 框为准修正 target。"""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_uv, img_white, target):
        max_crop_w = min(img_uv.width, self.max_size)
        max_crop_h = min(img_uv.height, self.max_size)
        min_crop_w = min(self.min_size, max_crop_w)
        min_crop_h = min(self.min_size, max_crop_h)

        crop_w = random.randint(min_crop_w, max_crop_w)
        crop_h = random.randint(min_crop_h, max_crop_h)

        # 使用 UV 图像来采样裁剪区域，保证框变换与标签基准一致。
        region = TT.RandomCrop.get_params(img_uv, [crop_h, crop_w])
        img_uv, target = crop(img_uv, target, region)
        img_white = F.crop(img_white, *region)
        return img_uv, img_white, target


class DualPMLGuidedCrop:
    """
    围绕 PML 框做同步裁剪。

    当前类别约定是 `["NPML", "PML", "PM"]`，因此默认 `label=1`。
    裁剪区域以随机选中的一个 PML 框为中心，并按框宽高做少量外扩。
    """

    def __init__(
        self,
        label: int = 1,
        margin_ratio: Tuple[float, float] = (0.15, 0.35),
        fallback_to_random_crop: bool = True,
        random_crop_min_size: int = 384,
        random_crop_max_size: int = 600,
    ):
        self.label = label
        self.margin_ratio = margin_ratio
        self.fallback_to_random_crop = fallback_to_random_crop
        self.fallback_crop = DualRandomCrop(
            min_size=random_crop_min_size,
            max_size=random_crop_max_size,
        )

    def __call__(self, img_uv, img_white, target):
        if target is None or "boxes" not in target or target["boxes"].numel() == 0:
            return img_uv, img_white, target

        labels = target.get("labels")
        if labels is None:
            return img_uv, img_white, target

        pml_mask = labels == self.label
        if not bool(pml_mask.any().item()):
            if self.fallback_to_random_crop:
                return self.fallback_crop(img_uv, img_white, target)
            return img_uv, img_white, target

        pml_boxes = target["boxes"][pml_mask]
        pml_box = pml_boxes[random.randrange(len(pml_boxes))]
        x1, y1, x2, y2 = [float(value) for value in pml_box.tolist()]
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        margin = random.uniform(*self.margin_ratio)
        margin_x = box_w * margin
        margin_y = box_h * margin

        img_w, img_h = img_uv.size
        left = max(0, int(np.floor(x1 - margin_x)))
        top = max(0, int(np.floor(y1 - margin_y)))
        right = min(img_w, int(np.ceil(x2 + margin_x)))
        bottom = min(img_h, int(np.ceil(y2 + margin_y)))

        crop_w = max(1, right - left)
        crop_h = max(1, bottom - top)
        region = (top, left, crop_h, crop_w)

        img_uv, target = crop(img_uv, target, region)
        img_white = F.crop(img_white, *region)
        return img_uv, img_white, target


# ========== 第四部分：张量化与归一化 ==========
class DualToTensor:
    """将两路 PIL 图像同时转为张量。"""

    def __call__(self, img_uv, img_white, target):
        return F.to_tensor(img_uv), F.to_tensor(img_white), target


class DualNormalize:
    """
    对两路张量图像做标准化，并把框从 `xyxy` 转为归一化后的 `cxcywh`。

    这样可以直接对接 RF-DETR 训练目标格式。
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, img_uv, img_white, target):
        img_uv = F.normalize(img_uv, mean=self.mean, std=self.std)
        img_white = F.normalize(img_white, mean=self.mean, std=self.std)

        # 推理时若 target 为空，直接返回图像即可。
        if target is None:
            return img_uv, img_white, None

        target = target.copy()
        height, width = img_uv.shape[-2:]

        # RF-DETR 训练期框格式使用归一化后的 `cxcywh`。
        if "boxes" in target:
            boxes = box_xyxy_to_cxcywh(target["boxes"])
            boxes = boxes / torch.tensor([width, height, width, height], dtype=torch.float32)
            target["boxes"] = boxes

        return img_uv, img_white, target


# ========== 第五部分：实现函数 ==========
def make_dual_transforms(
    image_set: str,
    resolution: int,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> DualCompose:
    """
    构建与当前阶段匹配的双模态增强。

    训练阶段使用更强的数据增强；
    验证/测试阶段只保留确定性的 resize + normalize。
    """
    # 最终的公共后处理：张量化 + 标准化。
    normalize = DualCompose(
        [
            DualToTensor(),
            DualNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Dataset-side resize is fixed. Random scale jitter is handled later in
    # train_one_epoch by resizing the whole NestedTensor batch with one scale.
    batch_resize_base_size = resolution

    # 默认只使用单一输入分辨率。
    scales = [resolution]

    # 如果启用多尺度训练，则根据 RF-DETR 现有工具函数计算可用尺度集合。
    if multi_scale:
        scales = compute_multi_scale_scales(
            resolution, expanded_scales, patch_size, num_windows
        )
        if scales:
            batch_resize_base_size = scales[-1]
        print(
            "[DualTransforms] batch-level multi-scale sizes: "
            f"{scales}; dataset base size: {batch_resize_base_size}"
        )

    if image_set == "train":
        crop_max_size = min(600, batch_resize_base_size)
        crop_min_size = min(384, crop_max_size)
        return DualCompose(
            [
                DualRandomHorizontalFlip(p=0.5),
                DualRandomSelect(
                    DualPMLGuidedCrop(
                        label=1,
                        margin_ratio=(0.15, 0.35),
                        random_crop_min_size=crop_min_size,
                        random_crop_max_size=crop_max_size,
                    ),
                    DualCompose(
                        [
                            DualRandomCrop(
                                min_size=crop_min_size,
                                max_size=crop_max_size,
                            ),
                        ]
                    ),
                    p=0.5,
                ),
                DualResizePad([batch_resize_base_size]),
                normalize,
            ]
        )

    # 验证与测试阶段保持尽量确定、可复现的图像预处理。
    return DualCompose(
        [
            DualSquareResize([resolution]),
            normalize,
        ]
    )
