# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：定义模块，初始化init

子模块职责：
  - custom.core：模型核心结构，包括跨模态融合模块与双模态检测器。
  - custom.data：数据管线，包括双模态数据集、同步增强、collate 与 COCO 自动适配。
  - custom.runtime：运行兼容层，负责对接 src/rfdetr 的旧式 Model / args 接口。
  - custom.train：提供独立训练脚本入口。

"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ========== 第一部分：源码路径兼容 ==========
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"


def _prepend_sys_path(path: Path) -> None:
    """
    文件说明：把给定路径稳定地放到 `sys.path` 前部。
    功能说明：将重复出现在训练入口脚本里的路径注入逻辑收敛到包级 helper，
    这样后续若路径规则有变，只需要在一个地方维护。
    """
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def prepare_project_environment(*, change_cwd: bool = False) -> Path:
    """
    文件说明：为项目中的自定义训练脚本统一准备运行环境。
    功能说明：补齐项目根目录和 `src/` 路径，并在需要时切回项目根目录作为工作目录。

    这样做的原因：
    1. 当前主线训练与兼容入口此前各自复制了一份环境准备逻辑。
    2. 这些重复代码很容易在未来只改一处，造成入口脚本之间静默漂移。
    3. 收敛到包级 helper 后，入口脚本可以只保留自己的主流程。
    """
    _prepend_sys_path(_PROJECT_ROOT)
    _prepend_sys_path(_SRC_ROOT)

    if change_cwd:
        os.chdir(_PROJECT_ROOT)

    return _PROJECT_ROOT


# 保持现有行为：只要以 `custom.*` 形式导入，就自动补齐 `src/` 路径。
_prepend_sys_path(_SRC_ROOT)
