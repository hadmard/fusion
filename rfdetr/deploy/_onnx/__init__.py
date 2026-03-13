# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：
	ONNX 导出优化器与自定义符号（symbolic）注册相关的包装模块，提供对 ONNX 模型的常见简化与自定义操作符到 ONNX 的映射。

结构概览：
	- `optimizer`：ONNX 优化器与模型变换工具
	- `symbolic`：自定义运算符的 ONNX 符号注册表
	- 导出 `OnnxOptimizer` 与 `CustomOpSymbolicRegistry` 以便上层调用

注意：仅添加说明性注释，不修改导出与注册逻辑。
"""
from rfdetr.deploy._onnx import optimizer, symbolic
from rfdetr.deploy._onnx.optimizer import OnnxOptimizer
from rfdetr.deploy._onnx.symbolic import CustomOpSymbolicRegistry
