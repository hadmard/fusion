"""
文件说明：该文件是 `checkpoint_best_ema.pth` 的专用评估入口。
功能说明：默认读取根目录 `eval/model/checkpoint_best_ema.pth`，并调用内部评估核心在成对测试集上输出完整评估结果。

使用方式：
  1. 把 `checkpoint_best_ema.pth` 放进根目录 `eval/model/` 文件夹。
  2. 直接运行：
     - `conda run -n rfdetr python eval/run_eval_bestema.py`
  3. 如需临时改成别的权重文件名，也可以追加：
     - `conda run -n rfdetr python eval/run_eval_bestema.py --model-name bestema_v2.pth`
  4. 如需显式指定测试集，也可以继续追加参数：
     - `conda run -n rfdetr python eval/run_eval_bestema.py --uv-dir D:\desktop\fusion--\test_uv --white-dir D:\desktop\fusion--\test_white --label-dir D:\desktop\fusion--\test_label`
  5. 输出会写到：
     - `output/eval/<时间戳>/`

结构概览：
  第一部分：导入公共评估入口
  第二部分：脚本入口
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom.eval_entry_common import run_eval_entry


# ========== 第一部分：导入公共评估入口 ==========
DEFAULT_MODEL_NAME = "checkpoint_best_ema.pth"
PERSONALIZED_DEFAULT_ARGS: list[str] = []


# ========== 第二部分：脚本入口 ==========
def main() -> None:
    run_eval_entry(
        default_model_name=DEFAULT_MODEL_NAME,
        script_description="评估 `checkpoint_best_ema.pth`，默认从 `eval/model/` 读取对应权重。",
        personalized_args=PERSONALIZED_DEFAULT_ARGS,
    )


if __name__ == "__main__":
    main()
