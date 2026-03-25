"""
文件说明：该文件用于封装根目录 `eval/` 下评估入口脚本共享的公共逻辑。
功能说明：统一解析 `--model-name` 参数、解析 `eval/model/` 中的权重路径，并调用 `custom/eval_runtime.py`。

结构概览：
  第一部分：导入依赖与路径常量
  第二部分：参数解析
  第三部分：对外评估入口
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from custom.model_registry import MODEL_DIR, resolve_model_checkpoint_path


# ========== 第一部分：导入依赖与路径常量 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_DIR = Path(__file__).resolve().parent
CORE_SCRIPT = CUSTOM_DIR / "eval_runtime.py"


# ========== 第二部分：参数解析 ==========
def _parse_args(default_model_name: str, script_description: str) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=script_description)
    parser.add_argument(
        "--model-name",
        type=str,
        default=default_model_name,
        help=(
            "要评估的模型权重文件名，默认从 `eval/model/` 中读取。"
            f"默认值：{default_model_name}"
        ),
    )
    parser.add_argument(
        "--list-model-dir",
        action="store_true",
        help="仅打印当前模型目录路径，便于确认权重应该放在哪里。",
    )
    return parser.parse_known_args()


# ========== 第三部分：对外评估入口 ==========
def run_eval_entry(
    *,
    default_model_name: str,
    script_description: str,
    personalized_args: Sequence[str] | None = None,
) -> None:
    args, passthrough_args = _parse_args(
        default_model_name=default_model_name,
        script_description=script_description,
    )

    if args.list_model_dir:
        print(MODEL_DIR.resolve())
        return

    checkpoint_path = resolve_model_checkpoint_path(model_name=args.model_name)
    command = [
        sys.executable,
        str(CORE_SCRIPT),
        "--checkpoint",
        str(checkpoint_path),
        *(list(personalized_args) if personalized_args is not None else []),
        *passthrough_args,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
