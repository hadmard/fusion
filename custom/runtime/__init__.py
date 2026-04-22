"""
文件说明：custom 运行兼容子包。
功能说明：集中放置当前 custom 训练链路与 src/rfdetr 之间的适配层。
"""

from custom.runtime.rfdetr_compat import Model, build_dataset, populate_args

__all__ = [
    "Model",
    "build_dataset",
    "populate_args",
]
