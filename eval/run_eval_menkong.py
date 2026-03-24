"""
文件说明：该文件是 `menkong.pth` 的专用评估入口。
功能说明：固定读取根目录 `eval/menkong.pth`，并调用内部评估核心在成对测试集上输出完整评估结果。
说明：该脚本会自动触发 `legacy_gate_model.py` 的旧门控兼容结构，不需要手工干预。

使用方式：
  1. 把 `menkong.pth` 放进根目录 `eval/` 文件夹。
  2. 直接运行：
     - `conda run -n rfdetr python eval/run_eval_menkong.py`
  3. 如需显式指定测试集，也可以追加参数：
     - `conda run -n rfdetr python eval/run_eval_menkong.py --uv-dir D:\desktop\fusion--\test_uv --white-dir D:\desktop\fusion--\test_white --label-dir D:\desktop\fusion--\test_label`
  4. 输出会写到：
     - `output/eval/<时间戳>/`

结构概览：
  第一部分：导入依赖与路径常量
  第二部分：脚本入口
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# ========== 第一部分：导入依赖与路径常量 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = EVAL_DIR / "eval_core.py"
CHECKPOINT_PATH = EVAL_DIR / "menkong.pth"


# ========== 第二部分：脚本入口 ==========
def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"未找到权重：{CHECKPOINT_PATH}。请先把 `menkong.pth` 放进根目录 `eval/`。")

    command = [
        sys.executable,
        str(CORE_SCRIPT),
        "--checkpoint",
        str(CHECKPOINT_PATH),
        *sys.argv[1:],
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
