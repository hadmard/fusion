# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：
	`rfdetr.models` 包的初始化文件，向外暴露模型构建与后处理接口，便于上层 `rfdetr.main` 或 `rfdetr.detr` 直接导入。

结构概览：
	- 从 `lwdetr` 导出 `PostProcess`、`build_criterion_and_postprocessors`、`build_model` 等函数/类

注意：仅添加中文说明，不更改导出行为。
"""

from rfdetr.models.lwdetr import PostProcess, build_criterion_and_postprocessors, build_model
