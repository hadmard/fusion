"""
文件说明：本文件为旧 `rfdetr.cli.main` 路径提供兼容入口。
功能说明：旧版本通过 `rfdetr.cli.main:trainer` 暴露控制台脚本；新版已经改为
`rfdetr.cli:main`。这里保留一个极薄 shim，避免旧脚本或外部调用直接断裂。

结构概览：
  第一部分：新版入口转发
  第二部分：旧 `trainer` 名称兼容
"""

from rfdetr.cli import main


def trainer() -> None:
    """兼容旧 console entry point 名称。"""
    main()
