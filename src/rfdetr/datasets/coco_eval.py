"""
文件说明：本文件为旧 `rfdetr.datasets.coco_eval` 导入路径提供兼容层。
功能说明：把新版评估实现 `rfdetr.evaluation.coco_eval` 中的 `CocoEvaluator`
重新导出给仍依赖旧路径的训练/评估代码。

结构概览：
  第一部分：弃用提示
  第二部分：符号转发
"""

from rfdetr.utilities.decorators import _warn_deprecated_module

# ========== 第一部分：弃用提示 ==========
_warn_deprecated_module("rfdetr.datasets.coco_eval", "rfdetr.evaluation.coco_eval")

# ========== 第二部分：符号转发 ==========
from rfdetr.evaluation.coco_eval import CocoEvaluator  # noqa: F401, E402
