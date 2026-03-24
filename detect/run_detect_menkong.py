"""
文件说明：该文件是 `menkong.pth` 的专用双模态检测入口。
功能说明：固定读取根目录 `eval/menkong.pth`，并对 `detect/image_uv/` 与 `detect/image_white/` 中的成对图片做检测。
说明：当脚本检测到 `menkong.pth` 时，会自动走 `eval/legacy_gate_model.py` 的旧门控兼容结构。

使用方式：
  1. 把 `menkong.pth` 放到根目录 `eval/` 文件夹。
  2. 把 UV 图片放到 `detect/image_uv/`，文件名示例：`pair_0001_uv.bmp`
  3. 把 White 图片放到 `detect/image_white/`，文件名示例：`pair_0001_white.bmp`
  4. 直接运行：
     - `conda run -n rfdetr python detect/run_detect_menkong.py`
  5. 输出会写到：
     - `output/detect/<时间戳>/menkong/`

结构概览：
  第一部分：导入公共检测逻辑
  第二部分：脚本入口
"""

from __future__ import annotations

from detect_common import run_dual_modal_detection


# ========== 第一部分：导入公共检测逻辑 ==========
CHECKPOINT_NAME = "menkong.pth"


# ========== 第二部分：脚本入口 ==========
def main() -> None:
    run_dual_modal_detection(
        checkpoint_name=CHECKPOINT_NAME,
        script_name="run_detect_menkong.py",
    )


if __name__ == "__main__":
    main()
