"""
文件说明：该文件是 `regular_defor.pth` 的专用评估入口。
功能说明：默认读取根目录 `eval/model/regular_defor.pth`，并调用统一评估核心在成对测试集上输出完整评估结果。

结构概览：
  第一部分：导入公共评估入口
  第二部分：默认模型配置
  第三部分：脚本入口
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom.eval_entry_common import run_eval_entry


# ========== 第一部分：导入公共评估入口 ==========
DEFAULT_MODEL_NAME = "regular_defor.pth"
PERSONALIZED_DEFAULT_ARGS: list[str] = []


# ========== 第二部分：默认模型配置 ==========
def main() -> None:
    run_eval_entry(
        default_model_name=DEFAULT_MODEL_NAME,
        script_description="评估 `regular_defor.pth`，默认从 `eval/model/` 读取对应权重。",
        personalized_args=PERSONALIZED_DEFAULT_ARGS,
    )


# ========== 第三部分：脚本入口 ==========
if __name__ == "__main__":
    main()
