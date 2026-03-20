# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件定义 UV/White 双模态训练与验证阶段的数据增强。
功能：在保证 UV 主模态语义始终清晰的前提下，对 UV 与 White 图像执行同步几何增强、
      模态特异性光学扰动、张量化与归一化，并同步维护检测框坐标。

----------------------------------------------------------------------
当前训练阶段增强配置（依执行顺序）：

  几何增强：
    1. DualRandomHorizontalFlip         p=0.5
       UV 与 White 同步水平翻转，框坐标同步修正。

    2. DualRandomSelect                 p=0.7 走分支A，0.3 走分支B
       分支A：DualSquareResize — 直接缩放到目标分辨率
       分支B：DualPMFocusCrop（p=1.0，围绕 PM 标签裁剪，
               scale 0.65~0.95，最少保留 4 框含 1 个 PM）
              → DualSquareResize — 裁剪后再缩放到目标分辨率

  外观增强（模态特异）：
    3. DualWhiteLightJitter             p=0.35
       仅作用于 White，亮度/对比度各 ±12%，饱和度 ±6%。
    4. DualUVFluorescenceJitter         p=0.4
       仅作用于 UV，模拟荧光强度波动（增益 0.92~1.12）、
       gamma 校正（0.9~1.12）、蓝通道增益（0.95~1.18）。
    5. DualGaussianBlur                 p=0.08，kernel=3
       UV 与 White 以独立概率施加高斯模糊。
    6. DualGaussianNoise                p=0.2，std 0.003~0.012
       UV 与 White 分别独立加噪（UV 噪声上限额外 ×1.5）。

  最终后处理：
    7. DualToTensor + DualNormalize
       转张量并以 ImageNet 均值/方差标准化；框转为归一化 cxcywh。


结构概览：
  第一部分：导入依赖
  第二部分：增强实现（可多个）
  第三部分：同步几何增强
  第四部分：模态相关光学扰动
  第五部分：张量化与归一化
  第六部分：实现函数 `make_dual_transforms`

核心约束：
  - 所有 transform 的输入输出签名统一为：
        `(img_uv, img_white, target) -> (img_uv, img_white, target)`
  - 所有边界框变换都以 UV 图像为基准，因为标注来源于 UV。
  - White 是辅助模态，因此允许做分支 dropout 等更贴近采集误差的扰动。
"""

# ========== 第一部分：导入依赖 ==========
import random
from typing import List, Optional, Tuple

import numpy as np
import PIL
import PIL.ImageFilter
import torch
import torchvision.transforms as TT
import torchvision.transforms.functional as F

from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.box_ops import box_xyxy_to_cxcywh
from rfdetr.util.misc import interpolate


def _normalize_resize_size(
    image_size: Tuple[int, int],
    size,
    max_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    兼容旧版 DETR 风格的 resize 入口。

    为什么把兼容逻辑留在 custom：
    - 训练链路当前仍依赖旧版 dual transforms 的函数签名；
    - `src/rfdetr` 已切换到新的 transforms 体系，不再暴露旧 helper；
    - 把这层适配局部化，能避免把兼容代码反向塞回主库。
    """
    if isinstance(size, (list, tuple)):
        if len(size) != 2:
            raise ValueError(f"resize size 必须是长度为 2 的序列，当前得到: {size}")
        return int(size[0]), int(size[1])

    width, height = image_size
    short_side = int(size)

    if max_size is not None:
        min_original = float(min(width, height))
        max_original = float(max(width, height))
        if max_original / max(min_original, 1.0) * short_side > max_size:
            short_side = int(round(max_size * min_original / max(max_original, 1.0)))

    if width <= height:
        new_width = short_side
        new_height = int(round(short_side * height / max(width, 1)))
    else:
        new_height = short_side
        new_width = int(round(short_side * width / max(height, 1)))

    return new_height, new_width


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


def resize(image, target, size, max_size: Optional[int] = None):
    """对 UV 图像做按比例 resize，并同步更新几何字段。"""
    new_height, new_width = _normalize_resize_size(image.size, size, max_size)
    resized_image = F.resize(image, (new_height, new_width))

    if target is None:
        return resized_image, None

    target = target.copy()
    original_width, original_height = image.size
    ratio_width = new_width / max(original_width, 1)
    ratio_height = new_height / max(original_height, 1)

    if "boxes" in target:
        scale = torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height],
            dtype=target["boxes"].dtype,
            device=target["boxes"].device,
        )
        target["boxes"] = target["boxes"] * scale

    if "area" in target:
        target["area"] = target["area"] * (ratio_width * ratio_height)

    target["size"] = torch.as_tensor([int(new_height), int(new_width)])

    if "masks" in target:
        target["masks"] = interpolate(
            target["masks"][:, None].float(),
            (new_height, new_width),
            mode="nearest",
        )[:, 0] > 0.5

    return resized_image, target


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


class DualRandomResize:
    """
    随机缩放到一个候选尺寸。

    UV 侧使用项目已有的 `resize` 工具函数，这样 target 中的框与尺寸信息会自动同步更新；
    White 侧则跟随 UV 的结果尺寸直接 resize。
    """

    def __init__(self, sizes: List[int], max_size: Optional[int] = None):
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img_uv, img_white, target):
        size = random.choice(self.sizes)

        # UV resize 后，target["size"] 会被更新为新的高宽。
        img_uv, target = resize(img_uv, target, size, self.max_size)
        height, width = target["size"].tolist()

        # White 使用完全相同的尺寸，保证两路特征能够配对。
        img_white = F.resize(img_white, (int(height), int(width)))
        return img_uv, img_white, target


class DualRandomSizeCrop:
    """对两路图像执行同步随机裁剪，并以 UV 框为准修正 target。"""

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_uv, img_white, target):
        crop_w = random.randint(self.min_size, min(img_uv.width, self.max_size))
        crop_h = random.randint(self.min_size, min(img_uv.height, self.max_size))

        # 使用 UV 图像来采样裁剪区域，保证框变换与标签基准一致。
        region = TT.RandomCrop.get_params(img_uv, [crop_h, crop_w])
        img_uv, target = crop(img_uv, target, region)
        img_white = F.crop(img_white, *region)
        return img_uv, img_white, target


class DualPMFocusCrop:
    """
    面向 PM 小目标的安全裁剪。

    该任务里的 PM 框很小，如果直接沿用 COCO 风格的大幅随机裁剪，
    很容易把病斑裁没，或者把叶片上下文裁得过碎。
    这里优先做“大范围但有目标约束”的裁剪，并尽量围绕 PM 取样。
    """

    def __init__(
        self,
        p: float = 0.3,
        min_crop_scale: float = 0.65,
        max_crop_scale: float = 0.95,
        min_aspect: float = 0.85,
        max_aspect: float = 1.2,
        attempts: int = 12,
        min_kept_boxes: int = 4,
        focus_label: Optional[int] = 2,
        min_focus_boxes: int = 1,
        focus_probability: float = 0.8,
    ):
        self.p = p
        self.min_crop_scale = min_crop_scale
        self.max_crop_scale = max_crop_scale
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.attempts = attempts
        self.min_kept_boxes = min_kept_boxes
        self.focus_label = focus_label
        self.min_focus_boxes = min_focus_boxes
        self.focus_probability = focus_probability

    def __call__(self, img_uv, img_white, target):
        if random.random() >= self.p:
            return img_uv, img_white, target

        if target is None or "boxes" not in target or target["boxes"].numel() == 0:
            return img_uv, img_white, target

        img_w, img_h = img_uv.size
        labels = target.get("labels")
        focus_boxes = None
        if labels is not None and self.focus_label is not None:
            focus_mask = labels == self.focus_label
            if bool(focus_mask.any().item()):
                focus_boxes = target["boxes"][focus_mask]

        min_crop_w = max(1, int(img_w * self.min_crop_scale))
        min_crop_h = max(1, int(img_h * self.min_crop_scale))
        max_crop_w = max(min_crop_w, int(img_w * self.max_crop_scale))
        max_crop_h = max(min_crop_h, int(img_h * self.max_crop_scale))

        for _ in range(self.attempts):
            crop_w = random.randint(min_crop_w, min(max_crop_w, img_w))
            crop_h = random.randint(min_crop_h, min(max_crop_h, img_h))

            aspect = crop_w / max(crop_h, 1)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            if focus_boxes is not None and random.random() < self.focus_probability:
                focus_box = focus_boxes[random.randrange(len(focus_boxes))]
                center_x = int(round(float((focus_box[0] + focus_box[2]) * 0.5)))
                center_y = int(round(float((focus_box[1] + focus_box[3]) * 0.5)))

                left_min = max(0, center_x - crop_w + 1)
                left_max = min(center_x, img_w - crop_w)
                top_min = max(0, center_y - crop_h + 1)
                top_max = min(center_y, img_h - crop_h)

                left = (
                    random.randint(left_min, left_max)
                    if left_min <= left_max
                    else random.randint(0, img_w - crop_w)
                )
                top = (
                    random.randint(top_min, top_max)
                    if top_min <= top_max
                    else random.randint(0, img_h - crop_h)
                )
            else:
                left = random.randint(0, img_w - crop_w)
                top = random.randint(0, img_h - crop_h)

            region = (top, left, crop_h, crop_w)
            img_uv_cropped, target_cropped = crop(img_uv, target, region)

            if int(target_cropped["boxes"].shape[0]) < self.min_kept_boxes:
                continue

            if focus_boxes is not None:
                kept_focus = int((target_cropped["labels"] == self.focus_label).sum().item())
                if kept_focus < self.min_focus_boxes:
                    continue

            img_white_cropped = F.crop(img_white, *region)
            return img_uv_cropped, img_white_cropped, target_cropped

        return img_uv, img_white, target


# ========== 第四部分：模态相关光学扰动 ==========
class DualWhiteLightJitter:
    """
    仅对白光分支做温和照明扰动。

    白光图承担叶片轮廓、叶脉和整体组织状态信息，
    所以只模拟亮度/对比度/轻微饱和度变化，避免大幅颜色偏移。
    """

    def __init__(
        self,
        brightness: float = 0.12,
        contrast: float = 0.12,
        saturation: float = 0.06,
        hue: float = 0.0,
        p: float = 0.35,
    ):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img_uv, img_white, target):
        if random.random() < self.p:
            img_white = TT.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )(img_white)
        return img_uv, img_white, target


class DualUVFluorescenceJitter:
    """
    仅作用在 UV 图像上的荧光强度扰动。

    目的是模拟不同拍摄条件下的荧光亮度、响应强度和蓝通道变化。
    """

    def __init__(
        self,
        p: float = 0.4,
        intensity_gain: Tuple[float, float] = (0.92, 1.12),
        gamma_range: Tuple[float, float] = (0.9, 1.12),
        blue_gain: Tuple[float, float] = (0.95, 1.18),
    ):
        self.p = p
        self.intensity_gain = intensity_gain
        self.gamma_range = gamma_range
        self.blue_gain = blue_gain

    def __call__(self, img_uv, img_white, target):
        if random.random() >= self.p:
            return img_uv, img_white, target

        # 转为 [0, 1] 浮点数组，便于做连续强度扰动。
        uv_arr = np.asarray(img_uv).astype(np.float32) / 255.0

        gain = random.uniform(*self.intensity_gain)
        gamma = random.uniform(*self.gamma_range)
        blue_gain = random.uniform(*self.blue_gain)

        # 先做整体增益，再做 gamma，再额外拉伸蓝通道。
        uv_arr = np.clip(uv_arr * gain, 0.0, 1.0)
        uv_arr = np.clip(np.power(uv_arr, gamma), 0.0, 1.0)
        uv_arr[..., 2] = np.clip(uv_arr[..., 2] * blue_gain, 0.0, 1.0)

        img_uv = PIL.Image.fromarray((uv_arr * 255.0).astype(np.uint8))
        return img_uv, img_white, target


class DualGaussianBlur:
    """分别对两路图像以独立概率施加高斯模糊。"""

    def __init__(self, kernel_sizes: List[int] = [3], p: float = 0.08):
        self.kernel_sizes = kernel_sizes
        self.p = p

    def __call__(self, img_uv, img_white, target):
        if random.random() < self.p:
            img_white = img_white.filter(
                PIL.ImageFilter.GaussianBlur(radius=random.choice(self.kernel_sizes) // 2)
            )

        if random.random() < self.p:
            img_uv = img_uv.filter(
                PIL.ImageFilter.GaussianBlur(radius=random.choice(self.kernel_sizes) // 2)
            )

        return img_uv, img_white, target


class DualGaussianNoise:
    """分别为 UV 与 White 添加高斯噪声，以模拟成像噪声。"""

    def __init__(self, std_range: Tuple[float, float] = (0.003, 0.012), p: float = 0.2):
        self.std_range = std_range
        self.p = p

    def __call__(self, img_uv, img_white, target):
        # White 与 UV 分支分别独立决定是否加噪。
        if random.random() < self.p:
            img_white = self._add_noise(img_white, random.uniform(*self.std_range))

        # UV 的噪声范围允许稍微更强一些，用于模拟 UV 成像波动。
        if random.random() < min(self.p * 1.2, 1.0):
            img_uv = self._add_noise(
                img_uv,
                random.uniform(self.std_range[0], self.std_range[1] * 1.5),
            )

        return img_uv, img_white, target

    @staticmethod
    def _add_noise(img: PIL.Image.Image, std: float) -> PIL.Image.Image:
        """给单张 PIL 图像叠加高斯噪声。"""
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.randn(*arr.shape).astype(np.float32) * std
        arr = np.clip(arr + noise, 0.0, 1.0)
        return PIL.Image.fromarray((arr * 255.0).astype(np.uint8))


# ========== 第五部分：张量化与归一化 ==========
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


# ========== 第六部分：实现函数 ==========
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

    # 默认只使用单一输入分辨率。
    scales = [resolution]

    # 如果启用多尺度训练，则根据 RF-DETR 现有工具函数计算可用尺度集合。
    if multi_scale:
        scales = compute_multi_scale_scales(
            resolution, expanded_scales, patch_size, num_windows
        )
        scales = [scale for scale in scales if scale >= resolution]
        if not scales:
            scales = [resolution]
        print(f"[DualTransforms] multi-scale sizes: {scales}")

    if image_set == "train":
        return DualCompose(
            [
                # ---------- 几何增强 ----------
                DualRandomHorizontalFlip(p=0.5),
                DualRandomSelect(
                    DualSquareResize(scales),
                    DualCompose(
                        [
                            DualPMFocusCrop(
                                p=1.0,
                                min_crop_scale=0.65,
                                max_crop_scale=0.95,
                                min_aspect=0.85,
                                max_aspect=1.2,
                                attempts=12,
                                min_kept_boxes=4,
                                focus_label=2,
                                min_focus_boxes=1,
                                focus_probability=0.8,
                            ),
                            DualSquareResize(scales),
                        ]
                    ),
                    p=0.7,
                ),
                # ---------- 外观增强 ----------
                DualWhiteLightJitter(
                    brightness=0.12,
                    contrast=0.12,
                    saturation=0.06,
                    hue=0.0,
                    p=0.35,
                ),
                DualUVFluorescenceJitter(
                    p=0.4,
                    intensity_gain=(0.92, 1.12),
                    gamma_range=(0.9, 1.12),
                    blue_gain=(0.95, 1.18),
                ),
                DualGaussianBlur(kernel_sizes=[3], p=0.08),
                DualGaussianNoise(std_range=(0.003, 0.012), p=0.2),
                # ---------- 最终张量化 ----------
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
