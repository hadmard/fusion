# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件是 `rfdetr` 顶层包入口。
功能说明：负责设置少量导入期环境兼容项、导出公开模型类，并延迟暴露训练相关可选符号。

结构概览：
  第一部分：导入期环境兼容
  第二部分：公开模型导出
  第三部分：延迟属性解析
"""

import os

import torch

# ========== 第一部分：导入期环境兼容 ==========
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def _ensure_torchvision_nms_schema() -> None:
    """
    在缺少 torchvision 原生算子注册的环境里预先声明 `torchvision::nms`。

    某些 Windows / CPU 环境中，torchvision 在导入 `_meta_registrations`
    时会先为 `torchvision::nms` 注册 fake/meta 实现；如果底层 schema 尚未定义，
    导入会直接失败，进而让整个 `import rfdetr` 中断。这里仅补 schema，不提供
    真正实现，这样至少能让依赖 transforms/datasets 的测试与源码导入继续进行。
    """
    try:
        torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::nms", "Meta")
        return
    except RuntimeError:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")


_ensure_torchvision_nms_schema()

# ========== 第二部分：公开模型导出 ==========
from rfdetr.detr import (
    RFDETRBase,  # DEPRECATED # noqa: F401
    RFDETRLarge,
    RFDETRLargeDeprecated,  # DEPRECATED # noqa: F401
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegPreview,  # DEPRECATED # noqa: F401
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)

__all__ = [
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]

# Lazily resolved names: avoids eager pytorch_lightning import at `import rfdetr` time.
_LAZY_TRAINING = frozenset({"RFDETRModule", "RFDETRDataModule", "build_trainer"})
_PLUS_EXPORTS = frozenset({"RFDETR2XLarge", "RFDETRXLarge"})


# ========== 第三部分：延迟属性解析 ==========
def __getattr__(name: str):
    """Resolve PTL and plus-only exports lazily, raising only on explicit access."""
    if name in _LAZY_TRAINING:
        from rfdetr import training as _training

        value = getattr(_training, name)
        globals()[name] = value
        return value

    if name in _PLUS_EXPORTS:
        from rfdetr.platform import _INSTALL_MSG
        from rfdetr.platform import models as _platform_models

        if hasattr(_platform_models, name):
            value = getattr(_platform_models, name)
            globals()[name] = value
            return value

        raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
