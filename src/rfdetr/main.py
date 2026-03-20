"""
文件说明：本文件为旧 `rfdetr.main` 导入路径提供兼容层。
功能说明：把当前仓库仍在使用的旧 `populate_args / Model` 接口转发到
`custom.rfdetr_compat`，避免 custom 训练/推理入口在上游目录迁移后全部失效。

结构概览：
  第一部分：兼容符号转发
"""

# ========== 第一部分：兼容符号转发 ==========
from custom.rfdetr_compat import Model, build_dataset, populate_args  # noqa: F401
