"""
文件说明：该文件用于封装根目录 `detect/` 下双模态检测入口共享的公共逻辑。
功能说明：统一读取 `detect/image_uv` 与 `detect/image_white` 的配对图片，加载 `eval/` 中指定的双模态权重，
并把检测结果输出到 `output/detect/<时间戳>/`。

结构概览：
  第一部分：导入依赖与路径常量
  第二部分：输出目录与结果保存
  第三部分：对外检测入口
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


# ========== 第一部分：导入依赖与路径常量 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DETECT_DIR = Path(__file__).resolve().parent
IMAGE_UV_DIR = DETECT_DIR / "image_uv"
IMAGE_WHITE_DIR = DETECT_DIR / "image_white"
OUTPUT_BASE_DIR = PROJECT_ROOT / "output" / "detect"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from custom.eval_runtime import _build_image_pairs, _load_checkpoint_runtime, _run_single_pair
from custom.model_registry import resolve_model_checkpoint_path


# ========== 第二部分：输出目录与结果保存 ==========
def _build_output_dir(checkpoint_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = OUTPUT_BASE_DIR / timestamp / Path(checkpoint_name).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _draw_detection_preview(
    uv_path: Path,
    detections: list[dict[str, Any]],
    class_names: list[str],
    save_path: Path,
) -> None:
    image = Image.open(uv_path).convert("RGB")
    drawer = ImageDraw.Draw(image)

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        class_id = int(detection["class_id"])
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
        score = float(detection["confidence"])
        drawer.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        drawer.text((x1 + 2, max(y1 - 14, 0)), f"{class_name} {score:.3f}", fill="red")

    image.save(save_path)


def _save_detection_report(
    output_dir: Path,
    checkpoint_path: Path,
    runtime_meta: dict[str, Any],
    per_image_results: list[dict[str, Any]],
) -> None:
    summary_report = {
        "model_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "runtime": {
            "device": runtime_meta["device"],
            "resolution": runtime_meta["resolution"],
            "dual_modal": runtime_meta["dual_modal"],
            "use_white": runtime_meta["use_white"],
            "fusion_type": runtime_meta["fusion_type"],
            "fusion_num_layers": runtime_meta["fusion_num_layers"],
            "architecture_variant": runtime_meta["architecture_variant"],
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        },
        "detect_source_dirs": {
            "uv_dir": str(IMAGE_UV_DIR.resolve()),
            "white_dir": str(IMAGE_WHITE_DIR.resolve()),
        },
        "num_pairs": len(per_image_results),
        "total_detections": sum(item["num_detections"] for item in per_image_results),
    }

    summary_path = output_dir / "summary_report.json"
    details_path = output_dir / "per_image_detections.json"
    summary_path.write_text(json.dumps(summary_report, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path.write_text(json.dumps(per_image_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Detect] Summary saved to: {summary_path}")
    print(f"[Detect] Per-image detections saved to: {details_path}")


# ========== 第三部分：对外检测入口 ==========
def run_dual_modal_detection(*, checkpoint_name: str, script_name: str) -> Path:
    checkpoint_path = resolve_model_checkpoint_path(model_name=checkpoint_name)

    if not IMAGE_UV_DIR.exists() or not IMAGE_WHITE_DIR.exists():
        raise FileNotFoundError(
            "未找到检测图片目录。请确认根目录下存在 `detect/image_uv/` 和 `detect/image_white/`。"
        )

    image_pairs = _build_image_pairs(uv_dir=IMAGE_UV_DIR, white_dir=IMAGE_WHITE_DIR, label_dir=None)
    if not image_pairs:
        raise RuntimeError(
            "未找到可配对的检测图片。请把 `*_uv.*` 放进 `detect/image_uv/`，"
            "把同名前缀的 `*_white.*` 放进 `detect/image_white/`。"
        )

    output_dir = _build_output_dir(checkpoint_name)
    model, postprocess, runtime_meta = _load_checkpoint_runtime(
        checkpoint_path=checkpoint_path,
        device=DEFAULT_DEVICE,
    )
    if not runtime_meta["dual_modal"] or not runtime_meta["use_white"]:
        raise RuntimeError(
            f"`{script_name}` 只支持双模态权重，但 `{checkpoint_name}` 当前不是双模态结构。"
        )

    per_image_results: list[dict[str, Any]] = []
    for image_pair in image_pairs:
        pair_result, _, _ = _run_single_pair(
            model=model,
            postprocess=postprocess,
            image_pair=image_pair,
            resolution=runtime_meta["resolution"],
            device=DEFAULT_DEVICE,
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            dual_modal=True,
        )
        per_image_results.append(pair_result)

        preview_path = output_dir / "visualizations" / f"{image_pair.pair_id}_uv.jpg"
        _draw_detection_preview(
            uv_path=image_pair.uv_path,
            detections=pair_result["detections"],
            class_names=runtime_meta["class_names"],
            save_path=preview_path,
        )

    _save_detection_report(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        runtime_meta=runtime_meta,
        per_image_results=per_image_results,
    )
    print(f"[Detect] Visualizations saved to: {output_dir / 'visualizations'}")
    return output_dir
