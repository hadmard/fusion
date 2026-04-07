"""
文件说明：该文件用于批量评估 `eval/model/` 目录中的当前主模型集合。
功能说明：统一使用同一套测试集路径和评估核心，按固定顺序逐个评估并汇总输出目录，避免手动逐条命令导致口径漂移。

结构概览：
  第一部分：导入依赖与常量
  第二部分：参数解析
  第三部分：批量评估执行
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ========== 第一部分：导入依赖与常量 ==========
DEFAULT_MODEL_NAMES = [
    "uv_single.pth",
    "menkong.pth",
    "kimi.pth",
    "multi_feature.pth",
    "high_resolution.pth",
]


# ========== 第二部分：参数解析 ==========
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评估 `eval/model/` 目录中的主模型集合，并固定使用同一测试集口径。"
    )
    parser.add_argument(
        "--uv-dir",
        type=str,
        default="test/uvtest",
        help="UV 测试图片目录，默认使用根目录 `test/uvtest`。",
    )
    parser.add_argument(
        "--white-dir",
        type=str,
        default="test/whitetest",
        help="White 测试图片目录，默认使用根目录 `test/whitetest`。",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="test/labeltest",
        help="测试标签目录，默认使用根目录 `test/labeltest`。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行设备，默认使用 `cuda`。",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="统一的置信度阈值，默认 0.5。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="仅评估前 N 对图片；0 表示全量。",
    )
    return parser.parse_args()


# ========== 第三部分：批量评估执行 ==========
def main() -> None:
    args = _parse_args()
    runtime_script = PROJECT_ROOT / "custom" / "eval_runtime.py"
    summary_lines: list[str] = []

    for model_name in DEFAULT_MODEL_NAMES:
        command = [
            sys.executable,
            str(runtime_script),
            "--checkpoint",
            model_name,
            "--uv-dir",
            args.uv_dir,
            "--white-dir",
            args.white_dir,
            "--label-dir",
            args.label_dir,
            "--device",
            args.device,
            "--confidence-threshold",
            str(args.confidence_threshold),
        ]
        if args.max_images > 0:
            command.extend(["--max-images", str(args.max_images)])

        print(f"[Batch] Start evaluating: {model_name}")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        summary_path = ""
        for line in completed.stdout.splitlines():
            if line.startswith("[Report] Summary saved to:"):
                summary_path = line.split(":", 1)[1].strip()
        summary_lines.append(f"{model_name} -> {summary_path}")

        # 统一把子进程输出回显出来，方便直接查错和回看表格指标。
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n")

    print("[Batch] Finished.")
    for line in summary_lines:
        print(f"  {line}")


if __name__ == "__main__":
    main()
