# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""
文件说明：
    本模块实现了一个简单的自定义 ONNX 符号（symbolic）与优化器注册表 `CustomOpSymbolicRegistry`，供 ONNX 导出时注册自定义转换函数使用。

结构概览：
    - `CustomOpSymbolicRegistry`：类级别容器，用于收集注册的优化器函数
    - `register_optimizer`：装饰器工厂，用于将函数注册到该表

注意：仅为模块添加中文说明，不修改注册器行为。
"""



class CustomOpSymbolicRegistry:
    # _SYMBOLICS = {}
    _OPTIMIZER = []

    @classmethod
    def optimizer(cls, fn):
        cls._OPTIMIZER.append(fn)


def register_optimizer():
    def optimizer_wrapper(fn):
        CustomOpSymbolicRegistry.optimizer(fn)
        return fn
    return optimizer_wrapper
