# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：
    本包初始化文件，负责向外暴露 RF-DETR 的高层模型类接口，并做少量环境变量兼容设置。

结构概览：
    第一部分：版权与许可声明
    第二部分：环境兼容性设置（如 `PYTORCH_ENABLE_MPS_FALLBACK`）
    第三部分：从 `rfdetr.detr` 与平台模型中导出常用模型类（便于 `from rfdetr import RFDETRBase`）

代码段说明：
    - 本文件不包含模型实现，仅做符号重导出（re-export），以提供简洁的公共 API。
    - 注释变更仅为中文说明，未更改导入或运行逻辑。
"""

# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetr.detr import (
        RFDETRBase,
        RFDETRLarge,
        RFDETRLargeDeprecated,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSeg2XLarge,
        RFDETRSegLarge,
        RFDETRSegMedium,
        RFDETRSegNano,
        RFDETRSegPreview,
        RFDETRSegSmall,
        RFDETRSegXLarge,
        RFDETRSmall,
)
from rfdetr.platform.models import (
        RFDETR2XLarge,
        RFDETRXLarge,
)
