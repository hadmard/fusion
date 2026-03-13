# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：
    定义 backbone 抽象基类 `BackboneBase`，供具体骨干实现继承并实现 `get_named_param_lr_pairs` 等接口。

结构概览：
    - `BackboneBase`：抽象基类，声明了获取命名参数与学习率配对的方法签名（用于参数分组与学习率调度）

注意：仅增加说明性注释，未更改类接口或实现。
"""

from torch import nn


class BackboneBase(nn.Module):
        def __init__(self):
                super().__init__()

        def get_named_param_lr_pairs(self, args, prefix:str):
                raise NotImplementedError
