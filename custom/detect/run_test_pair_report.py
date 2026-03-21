"""
文件说明：本文件用于测试仓库根目录的 best 权重在成对测试图片上的推理表现，并输出一份参数报告。
功能说明：默认读取根目录 `checkpoint_best_ema.pth`，使用 `datasets/test_uv` 与 `datasets/test_white`
中的同名配对图片做双模态推理，统计模型参数量、模型文件大小、GFLOPs、FPS，
并保存每张图片的检测结果到 JSON。

结构概览：
  第一部分：导入依赖与环境准备
  第二部分：参数解析与路径工具
  第三部分：模型加载与前处理
  第四部分：单对图片推理与统计
  第五部分：结果汇总与落盘
  第六部分：脚本入口
"""

from __future__ import annotations

import argparse
import json
import tempfile
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

_FILE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_FILE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILE_PROJECT_ROOT))

from custom import prepare_project_environment

# 统一把项目根目录与 `src/` 加入运行环境，避免脚本单独执行时导入失败。
PROJECT_ROOT = prepare_project_environment(change_cwd=True)

from custom.dual_dataset import load_class_names, parse_yolo_label
from custom.dual_model import build_dual_model
from custom.rfdetr_compat import populate_args
from rfdetr.evaluation.coco_eval import patched_pycocotools_summarize
from rfdetr.models.lwdetr import PostProcess


# ========== 第二部分：参数解析与路径工具 ==========
@dataclass
class ImagePair:
    """表示一组按文件名前缀配对成功的 UV / White 图片。"""

    pair_id: str
    uv_path: Path
    white_path: Path
    label_path: Path | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试根目录 best 权重在 test_uv / test_white 成对图片上的推理表现。"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="权重路径；默认自动优先寻找仓库根目录的 checkpoint_best_ema.pth。",
    )
    parser.add_argument(
        "--uv-dir",
        type=str,
        default="datasets/test_uv",
        help="UV 测试图片目录。",
    )
    parser.add_argument(
        "--white-dir",
        type=str,
        default="datasets/test_white",
        help="White 测试图片目录。",
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        default="datasets/test_label",
        help="测试标签目录；若存在同名标签，则自动计算 AP/Precision/Recall/F1。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="结果输出目录；默认写到 output/test_pair_report/<时间戳>。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备，默认优先 cuda。",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="全局置信度阈值。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="仅测试前 N 对图片；0 表示全部。",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="正式计时前的 warmup 次数，用于减少首轮波动。",
    )
    return parser.parse_args()


def _resolve_checkpoint_path(explicit_path: str | None) -> Path:
    """
    解析实际要使用的 checkpoint。

    这里优先根目录的 `checkpoint_best_ema.pth`，因为用户明确说的是“根目录那个 best 权重文件”。
    """
    if explicit_path is not None:
        checkpoint_path = Path(explicit_path)
        if checkpoint_path.exists():
            return checkpoint_path.resolve()
        raise FileNotFoundError(f"指定的 checkpoint 不存在：{checkpoint_path}")

    candidates = [
        PROJECT_ROOT / "checkpoint_best_ema.pth",
        PROJECT_ROOT / "checkpoint_best_regular.pth",
        PROJECT_ROOT / "output" / "checkpoint_best_ema.pth",
        PROJECT_ROOT / "output" / "checkpoint_best_regular.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "未找到可用的 best checkpoint。已检查："
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _build_image_pairs(
    uv_dir: Path,
    white_dir: Path,
    label_dir: Path | None,
) -> list[ImagePair]:
    """
    从 `test_uv` 与 `test_white` 中构建同名前缀配对。

    这里不是直接比完整文件名，因为一个是 `_uv.bmp`，一个是 `_white.bmp`；
    真正稳定的配对键是 `pair_xxxx` 这段前缀。
    """
    if not uv_dir.exists():
        raise FileNotFoundError(f"UV 图片目录不存在：{uv_dir}")
    if not white_dir.exists():
        raise FileNotFoundError(f"White 图片目录不存在：{white_dir}")

    uv_map = {
        image_path.stem.removesuffix("_uv"): image_path
        for image_path in sorted(uv_dir.glob("*_uv.*"))
        if image_path.is_file()
    }
    white_map = {
        image_path.stem.removesuffix("_white"): image_path
        for image_path in sorted(white_dir.glob("*_white.*"))
        if image_path.is_file()
    }

    common_ids = sorted(set(uv_map) & set(white_map))
    if not common_ids:
        raise RuntimeError(
            f"在 {uv_dir} 和 {white_dir} 中没有找到任何同名前缀的配对图片。"
        )

    return [
        ImagePair(
            pair_id=pair_id,
            uv_path=uv_map[pair_id],
            white_path=white_map[pair_id],
            label_path=(label_dir / f"{uv_map[pair_id].stem}.txt") if label_dir is not None else None,
        )
        for pair_id in common_ids
    ]


def _resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        output_dir = Path(output_dir_arg)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = PROJECT_ROOT / "output" / "test_pair_report" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _safe_getattr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _build_runtime_class_names(checkpoint_args: Any, runtime_num_classes: int) -> list[str]:
    """
    尽量继承 checkpoint 自带类别名；若数量不匹配，再退化为通用占位名。

    这样做是为了兼容历史上出现过的“args.num_classes 与检测头维度不一致”情况。
    """
    class_names = list(_safe_getattr(checkpoint_args, "class_names", []) or [])
    if len(class_names) == runtime_num_classes:
        return class_names
    return [f"class_{index}" for index in range(runtime_num_classes)]


def _load_dataset_class_names(dataset_yaml_path: Path, fallback_num_classes: int) -> list[str]:
    if dataset_yaml_path.exists():
        try:
            class_names = load_class_names(str(dataset_yaml_path))
            if class_names:
                return class_names
        except Exception:  # noqa: BLE001
            pass
    return [f"class_{index}" for index in range(fallback_num_classes)]


# ========== 第三部分：模型加载与前处理 ==========
def _load_checkpoint_runtime(
    checkpoint_path: Path,
    device: str,
) -> tuple[torch.nn.Module, PostProcess, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    checkpoint_args = checkpoint.get("args")

    checkpoint_num_classes = int(state_dict["class_embed.bias"].shape[0]) - 1
    class_names = _build_runtime_class_names(checkpoint_args, checkpoint_num_classes)

    runtime_args = populate_args(
        num_classes=checkpoint_num_classes,
        class_names=class_names,
        pretrain_weights=None,
        resolution=int(_safe_getattr(checkpoint_args, "resolution", 560)),
        use_white=bool(_safe_getattr(checkpoint_args, "use_white", True)),
        fusion_type=str(_safe_getattr(checkpoint_args, "fusion_type", "uv_queries_white")),
        fusion_num_layers=int(_safe_getattr(checkpoint_args, "fusion_num_layers", 1)),
        device=device,
        dual_modal=True,
    )

    model = build_dual_model(runtime_args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    postprocess = PostProcess(num_select=runtime_args.num_select)

    runtime_meta = {
        "checkpoint_path": str(checkpoint_path),
        "resolution": int(runtime_args.resolution),
        "class_names": class_names,
        "num_classes": int(runtime_args.num_classes),
        "use_white": bool(runtime_args.use_white),
        "fusion_type": str(runtime_args.fusion_type),
        "fusion_num_layers": int(runtime_args.fusion_num_layers),
        "device": device,
    }
    return model, postprocess, runtime_meta


def _load_and_preprocess_pair(
    image_pair: ImagePair,
    resolution: int,
    device: str,
) -> tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor, torch.Tensor]:
    img_uv = Image.open(image_pair.uv_path).convert("RGB")
    img_white = Image.open(image_pair.white_path).convert("RGB")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    uv_tensor = TF.resize(TF.normalize(TF.to_tensor(img_uv), mean, std), [resolution, resolution])
    white_tensor = TF.resize(
        TF.normalize(TF.to_tensor(img_white), mean, std),
        [resolution, resolution],
    )

    target_sizes = torch.tensor([[img_uv.size[1], img_uv.size[0]]], device=device)
    return (
        img_uv,
        img_white,
        uv_tensor.unsqueeze(0).to(device),
        white_tensor.unsqueeze(0).to(device),
        target_sizes,
    )


def _filter_predictions(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    confidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = torch.where(scores >= confidence_threshold)[0]
    return boxes[keep], scores[keep], labels[keep]


def _maybe_compute_gflops(
    model: torch.nn.Module,
    resolution: int,
    device: str,
) -> tuple[float | None, str | None]:
    """
    尝试计算 GFLOPs。

    这里使用可选依赖 `thop`；如果环境没有安装或当前模型不支持 profile，
    就返回空值而不是让整份报告失败。
    """
    try:
        from thop import profile
    except ImportError:
        return None, "环境中未安装 thop，GFLOPs 未计算。"

    sample_uv = torch.randn(1, 3, resolution, resolution, device=device)
    sample_white = torch.randn(1, 3, resolution, resolution, device=device)

    try:
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(sample_uv, sample_white), verbose=False)
        # NOTE: approximate FLOPs, may be inaccurate for custom ops.
        # 常见论文表格更倾向写 FLOPs，因此这里按 2 * MACs 换算。
        gflops = float(macs * 2.0 / 1e9)
        return gflops, None
    except Exception as exc:  # noqa: BLE001
        return None, f"GFLOPs 计算失败：{exc}"


# ========== 第四部分：单对图片推理与统计 ==========
def _run_single_pair(
    model: torch.nn.Module,
    postprocess: PostProcess,
    image_pair: ImagePair,
    resolution: int,
    device: str,
    confidence_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    img_uv, img_white, batch_uv, batch_white, target_sizes = _load_and_preprocess_pair(
        image_pair=image_pair,
        resolution=resolution,
        device=device,
    )

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
        outputs = model(batch_uv, batch_white)
        result = postprocess(outputs, target_sizes)[0]
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    raw_boxes = result["boxes"].detach().cpu().float()
    raw_scores = result["scores"].detach().cpu().float()
    raw_labels = result["labels"].detach().cpu().long()

    boxes, scores, labels = _filter_predictions(
        boxes=raw_boxes,
        scores=raw_scores,
        labels=raw_labels,
        confidence_threshold=confidence_threshold,
    )

    detections = []
    for index in range(len(scores)):
        detections.append(
            {
                "class_id": int(labels[index].item()),
                "confidence": float(scores[index].item()),
                "bbox_xyxy": [float(value) for value in boxes[index].tolist()],
            }
        )

    pair_result = {
        "pair_id": image_pair.pair_id,
        "uv_path": str(image_pair.uv_path.resolve()),
        "white_path": str(image_pair.white_path.resolve()),
        "uv_image_size": {"width": img_uv.size[0], "height": img_uv.size[1]},
        "white_image_size": {"width": img_white.size[0], "height": img_white.size[1]},
        "num_detections": len(detections),
        "detections": detections,
        "inference_seconds": elapsed_seconds,
    }
    metric_prediction = {
        "pair_id": image_pair.pair_id,
        "boxes": raw_boxes,
        "scores": raw_scores,
        "labels": raw_labels,
    }
    return pair_result, metric_prediction, elapsed_seconds


def _run_warmup(
    model: torch.nn.Module,
    postprocess: PostProcess,
    image_pair: ImagePair,
    resolution: int,
    device: str,
    warmup_steps: int,
) -> None:
    for _ in range(max(warmup_steps, 0)):
        _run_single_pair(
            model=model,
            postprocess=postprocess,
            image_pair=image_pair,
            resolution=resolution,
            device=device,
            confidence_threshold=0.0,
        )


def _count_detections_by_class(
    per_image_results: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in per_image_results:
        for detection in item["detections"]:
            class_id = detection["class_id"]
            class_name = (
                class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
            )
            counter[class_name] += 1
    return dict(counter)


def _has_complete_labels(image_pairs: list[ImagePair]) -> bool:
    if not image_pairs:
        return False
    return all(image_pair.label_path is not None and image_pair.label_path.exists() for image_pair in image_pairs)


def _build_test_coco_dataset(
    image_pairs: list[ImagePair],
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    images = []
    annotations = []
    categories = [
        {"id": class_id, "name": class_name, "supercategory": "none"}
        for class_id, class_name in enumerate(class_names)
    ]
    image_id_by_pair: dict[str, int] = {}
    annotation_id = 1

    for image_id, image_pair in enumerate(image_pairs):
        with Image.open(image_pair.uv_path) as image:
            width, height = image.size

        image_id_by_pair[image_pair.pair_id] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": str(image_pair.uv_path.resolve()),
                "width": width,
                "height": height,
            }
        )

        boxes, classes = parse_yolo_label(str(image_pair.label_path), width, height)
        for index in range(len(classes)):
            x1, y1, x2, y2 = boxes[index].tolist()
            box_w = x2 - x1
            box_h = y2 - y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(classes[index].item()),
                    "bbox": [float(x1), float(y1), float(box_w), float(box_h)],
                    "area": float(box_w * box_h),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }, image_id_by_pair


def _build_coco_detection_results(
    metric_predictions: list[dict[str, Any]],
    image_id_by_pair: dict[str, int],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for image_prediction in metric_predictions:
        image_id = image_id_by_pair[image_prediction["pair_id"]]
        boxes = image_prediction["boxes"]
        scores = image_prediction["scores"]
        labels = image_prediction["labels"]
        for index in range(len(scores)):
            x1, y1, x2, y2 = [float(value) for value in boxes[index].tolist()]
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(labels[index].item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[index].item()),
                }
            )
    return detections


def _compute_detection_metrics(
    image_pairs: list[ImagePair],
    metric_predictions: list[dict[str, Any]],
    class_names: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    if not _has_complete_labels(image_pairs):
        return None, "测试标签不完整，跳过 AP/Precision/Recall/F1 计算。"

    coco_gt_dataset, image_id_by_pair = _build_test_coco_dataset(
        image_pairs=image_pairs,
        class_names=class_names,
    )
    detections = _build_coco_detection_results(
        metric_predictions=metric_predictions,
        image_id_by_pair=image_id_by_pair,
    )

    coco_gt = COCO()
    coco_gt.dataset = coco_gt_dataset
    coco_gt.createIndex()

    if detections:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            json.dump(detections, temp_file, ensure_ascii=False)
            temp_path = temp_file.name
        coco_dt = coco_gt.loadRes(temp_path)
        Path(temp_path).unlink(missing_ok=True)
    else:
        return None, "模型在当前阈值下没有预测框，无法计算 COCO 指标。"

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = sorted(image_id_by_pair.values())
    coco_eval.params.catIds = [category["id"] for category in coco_gt_dataset["categories"]]
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    patched_pycocotools_summarize(coco_eval)
    point_metrics = _compute_point_metrics_at_iou_and_score(
        coco_eval=coco_eval,
        iou_threshold=0.5,
        score_threshold=0.5,
        max_dets=100,
    )
    return {
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "mAP": float(coco_eval.stats[0]),
        "Precision@0.5": float(point_metrics["precision"]),
        "Recall@0.5": float(point_metrics["recall"]),
        "F1@0.5": float(point_metrics["f1_score"]),
        "details": {
            "definition": {
                "precision_recall_f1": "Precision@IoU=0.5, score=0.5, maxDets=100",
                "ap": "COCO AP with maxDets=100",
            },
            "counts": point_metrics["counts"],
            "class_map": point_metrics["class_map"],
        },
    }, None


def _compute_point_metrics_at_iou_and_score(
    coco_eval: COCOeval,
    iou_threshold: float,
    score_threshold: float,
    max_dets: int,
) -> dict[str, Any]:
    iou_index = int(np.argwhere(np.isclose(coco_eval.params.iouThrs, iou_threshold)).item())
    max_det_index = int(np.argwhere(np.array(coco_eval.params.maxDets) == max_dets).item())
    area_all = tuple(coco_eval.params.areaRng[0])
    iou75_index = int(np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.75)).item())
    area_index = 0

    eval_by_cat_area_img: dict[int, dict[tuple[float, float], dict[int, Any]]] = {}
    for entry in coco_eval.evalImgs:
        if entry is None:
            continue
        cat_id = int(entry["category_id"])
        image_id = int(entry["image_id"])
        area_rng = tuple(entry["aRng"])
        eval_by_cat_area_img.setdefault(cat_id, {}).setdefault(area_rng, {})[image_id] = entry

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    class_map: list[dict[str, Any]] = []
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_eval.cocoGt.loadCats(coco_eval.params.catIds)}

    for cat_index, cat_id in enumerate(coco_eval.params.catIds):
        total_gt = 0
        tp = 0
        fp = 0

        for image_id in coco_eval.params.imgIds:
            entry = eval_by_cat_area_img.get(int(cat_id), {}).get(area_all, {}).get(int(image_id))
            if entry is None:
                continue

            total_gt += int(np.sum(~np.array(entry["gtIgnore"], dtype=bool)))
            dt_scores = np.array(entry["dtScores"])
            dt_matches = np.array(entry["dtMatches"])[iou_index]
            dt_ignore = np.array(entry["dtIgnore"])[iou_index].astype(bool)

            keep = (dt_scores >= score_threshold) & (~dt_ignore)
            tp += int(np.sum(dt_matches[keep] != 0))
            fp += int(np.sum(dt_matches[keep] == 0))

        fn = max(total_gt - tp, 0)
        true_positives += tp
        false_positives += fp
        false_negatives += fn

        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if (class_precision + class_recall) > 0
            else 0.0
        )

        precision_slice = coco_eval.eval["precision"][:, :, cat_index, area_index, max_det_index]
        precision_masked = np.where(precision_slice > -1, precision_slice, np.nan)
        ap_per_iou = np.nanmean(precision_masked, axis=1)
        class_map.append(
            {
                "class": cat_id_to_name.get(int(cat_id), f"class_{cat_id}"),
                "AP50": float(np.nanmean(precision_masked[iou_index])),
                "AP75": float(np.nanmean(precision_masked[iou75_index])),
                "mAP": float(np.nanmean(ap_per_iou)),
                "Precision@0.5": float(class_precision),
                "Recall@0.5": float(class_recall),
                "F1@0.5": float(class_f1),
                "counts": {
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                },
            }
        )

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "class_map": class_map,
        "counts": {
            "tp": int(true_positives),
            "fp": int(false_positives),
            "fn": int(false_negatives),
        },
    }


# ========== 第五部分：结果汇总与落盘 ==========
def _build_summary_report(
    checkpoint_path: Path,
    runtime_meta: dict[str, Any],
    per_image_results: list[dict[str, Any]],
    total_elapsed_seconds: float,
    gflops: float | None,
    gflops_note: str | None,
    metric_values: dict[str, Any] | None,
    metrics_note: str | None,
) -> dict[str, Any]:
    total_images = len(per_image_results)
    total_detections = sum(item["num_detections"] for item in per_image_results)
    parameter_count = int(runtime_meta["parameter_count"])

    fps = float(total_images / total_elapsed_seconds) if total_elapsed_seconds > 0 else None

    # 这里直接用 checkpoint 文件大小，更符合“最终权重文件体积”这个表格字段语义。
    model_size_mb = float(checkpoint_path.stat().st_size / (1024 * 1024))

    return {
        "model_name": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path),
        "test_dataset": {
            "uv_dir": str((PROJECT_ROOT / runtime_meta["uv_dir"]).resolve()),
            "white_dir": str((PROJECT_ROOT / runtime_meta["white_dir"]).resolve()),
            "num_pairs": total_images,
        },
        "runtime": {
            "device": runtime_meta["device"],
            "resolution": runtime_meta["resolution"],
            "use_white": runtime_meta["use_white"],
            "fusion_type": runtime_meta["fusion_type"],
            "fusion_num_layers": runtime_meta["fusion_num_layers"],
            "confidence_threshold": runtime_meta["confidence_threshold"],
        },
        "table_like_metrics": {
            "AP50": None if metric_values is None else metric_values["AP50"],
            "AP75": None if metric_values is None else metric_values["AP75"],
            "mAP": None if metric_values is None else metric_values["mAP"],
            "Precision@0.5": None if metric_values is None else metric_values["Precision@0.5"],
            "Recall@0.5": None if metric_values is None else metric_values["Recall@0.5"],
            "F1@0.5": None if metric_values is None else metric_values["F1@0.5"],
            "FPS": fps,
            "GFLOPs": gflops,
            "Parameters(total)": int(parameter_count),
            "Model Size": model_size_mb,
        },
        "metrics_note": metrics_note,
        "gflops_note": gflops_note,
        "results_json": None if metric_values is None else metric_values["details"],
        "detection_summary": {
            "total_detections": total_detections,
            "detections_by_class": _count_detections_by_class(
                per_image_results=per_image_results,
                class_names=runtime_meta["class_names"],
            ),
        },
    }


def _save_report(
    output_dir: Path,
    summary_report: dict[str, Any],
    per_image_results: list[dict[str, Any]],
) -> None:
    summary_path = output_dir / "summary_report.json"
    detections_path = output_dir / "per_image_detections.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary_report, file, ensure_ascii=False, indent=2)

    with detections_path.open("w", encoding="utf-8") as file:
        json.dump(per_image_results, file, ensure_ascii=False, indent=2)

    print(f"[Report] Summary saved to: {summary_path}")
    print(f"[Report] Per-image detections saved to: {detections_path}")


# ========== 第六部分：脚本入口 ==========
def main() -> Path:
    args = _parse_args()

    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    uv_dir = Path(args.uv_dir)
    white_dir = Path(args.white_dir)
    label_dir = Path(args.label_dir) if args.label_dir else None
    output_dir = _resolve_output_dir(args.output_dir)
    image_pairs = _build_image_pairs(uv_dir=uv_dir, white_dir=white_dir, label_dir=label_dir)

    if args.max_images > 0:
        image_pairs = image_pairs[: args.max_images]

    model, postprocess, runtime_meta = _load_checkpoint_runtime(
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    runtime_meta["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    runtime_meta["uv_dir"] = str(uv_dir)
    runtime_meta["white_dir"] = str(white_dir)
    runtime_meta["label_dir"] = None if label_dir is None else str(label_dir)
    runtime_meta["confidence_threshold"] = float(args.confidence_threshold)
    runtime_meta["class_names"] = _load_dataset_class_names(
        PROJECT_ROOT / "datasets" / "dataset_dual.yaml",
        runtime_meta["num_classes"],
    )

    if image_pairs:
        _run_warmup(
            model=model,
            postprocess=postprocess,
            image_pair=image_pairs[0],
            resolution=runtime_meta["resolution"],
            device=args.device,
            warmup_steps=args.warmup,
        )

    gflops, gflops_note = _maybe_compute_gflops(
        model=model,
        resolution=runtime_meta["resolution"],
        device=args.device,
    )

    per_image_results: list[dict[str, Any]] = []
    metric_predictions: list[dict[str, Any]] = []
    total_elapsed_seconds = 0.0

    for index, image_pair in enumerate(image_pairs, start=1):
        pair_result, metric_prediction, elapsed_seconds = _run_single_pair(
            model=model,
            postprocess=postprocess,
            image_pair=image_pair,
            resolution=runtime_meta["resolution"],
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )
        per_image_results.append(pair_result)
        metric_predictions.append(metric_prediction)
        total_elapsed_seconds += elapsed_seconds

        if index % 10 == 0 or index == len(image_pairs):
            print(f"[Progress] {index}/{len(image_pairs)} pairs done.")

    metric_values, metrics_note = _compute_detection_metrics(
        image_pairs=image_pairs,
        metric_predictions=metric_predictions,
        class_names=runtime_meta["class_names"],
    )
    if metrics_note is None:
        metrics_note = "已基于 datasets/test_label 真实计算 AP、Precision、Recall 和 F1。"

    summary_report = _build_summary_report(
        checkpoint_path=checkpoint_path,
        runtime_meta=runtime_meta,
        per_image_results=per_image_results,
        total_elapsed_seconds=total_elapsed_seconds,
        gflops=gflops,
        gflops_note=gflops_note,
        metric_values=metric_values,
        metrics_note=metrics_note,
    )

    _save_report(
        output_dir=output_dir,
        summary_report=summary_report,
        per_image_results=per_image_results,
    )

    print("[Done] Table-like metrics:")
    for key, value in summary_report["table_like_metrics"].items():
        print(f"  {key}: {value}")
    return output_dir


if __name__ == "__main__":
    main()
