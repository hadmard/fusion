# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件是 UV-only 训练模式的兼容启动入口。
功能说明：它本身不维护训练参数，而是复用统一训练启动器 `run_train.py`，只额外指定
 `modality_mode="uv_only"` 与 UV-only 的输出目录、日志前缀。

结构概览：
  第一部分：路径初始化
  第二部分：UV-only 启动入口
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ========== 第一部分：路径初始化 ==========
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)


# ========== 第二部分：UV-only 启动入口 ==========
if __name__ == "__main__":
    from custom.train.run_train import run_training

    # 保留单独脚本而不是只靠命令行参数，是为了让历史使用方式与现有 README/命令保持兼容。
    run_training(
        modality_mode="uv_only",
        output_base_dir="output/train_uv",
        log_prefix="[Train-UV]",
    )
