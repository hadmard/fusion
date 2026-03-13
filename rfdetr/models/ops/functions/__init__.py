# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
文件说明：
	`rfdetr.models.ops.functions` 包的初始化，用于导出纯 Python 实现的 ms-deform-attn 回退函数（`ms_deform_attn_core_pytorch`）。

结构概览：
	- 从 `ms_deform_attn_func` 导出 `ms_deform_attn_core_pytorch`，作为 CUDA 扩展不可用时的回退实现

注意：仅为包级别添加中文说明，不修改导出行为。
"""
from rfdetr.models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
