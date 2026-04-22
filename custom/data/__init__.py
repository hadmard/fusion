"""
文件说明：custom 数据管线子包。
功能说明：集中放置双模态数据集、同步增强、collate 与 COCO 自动适配逻辑。
"""

from custom.data.dual_collate import dual_collate_fn
from custom.data.dual_dataset import DualModalYoloDetection, build_dual_dataset

__all__ = [
    "DualModalYoloDetection",
    "build_dual_dataset",
    "dual_collate_fn",
]
