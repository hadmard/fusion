# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件是双模态 RF-DETR 的推理脚本。
功能：加载训练得到的双模态 checkpoint，对 UV/White 成对图像执行推理，做阈值过滤和 NMS，
      并保存可视化结果和 `detections.json`。

结构概览：
  第一部分：导入依赖与路径设置
  第二部分：推理配置区
  第三部分：后处理工具函数
  第四部分：模型加载与单样本推理
  第五部分：结果可视化
  第六部分：主执行流程

设计约束：
  - UV 是主模态，White 是辅助模态。
  - 当 `FUSION_TYPE == "none"` 或 `USE_WHITE == False` 时，脚本支持退化为 UV-only baseline 推理。
  - 结果框、类别和坐标系始终以 UV 图像为准。
"""

# ========== 第一部分：导入依赖与路径设置 ==========
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from custom.dataset_layout import (
    is_probable_uv_image,
    list_image_files,
    resolve_split_layout,
    resolve_white_path_for_uv,
)

# 将项目根目录加入路径，保证脚本运行时能正确导入 `custom/` 和 `rfdetr/`。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 切换工作目录，确保相对路径全部以项目根目录为基准。
os.chdir(_PROJECT_ROOT)


# ========== 第二部分：推理配置区 ==========
# ---------- 模型权重 ----------
CHECKPOINT_PATH = "output/train/2026-03-08_100100/checkpoint_best_total.pth"

# ---------- 模态与融合配置 ----------
USE_WHITE = True
FUSION_TYPE = "uv_queries_white"
FUSION_NUM_LAYERS = 1

# ---------- 类别与输入路径 ----------
NUM_CLASSES = 3
CLASS_NAMES = ["NPML", "PML", "PM"]
DETECT_DATASET_DIR = "detect_datasets"
DETECT_SPLIT = "test"

# ---------- 后处理阈值 ----------
CONFIDENCE_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.50
CLASS_CONF_THRESHOLDS: Dict[str, float] = {
    "NPML": 0.50,
    "PML": 0.50,
    "PM": 0.40,  # PM 允许相对更低阈值，以提升召回
}

# ---------- 输出配置 ----------
DEVICE = "cuda"
DRAW_BOXES = True
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
SAVE_JSON = True
OUTPUT_BASE_DIR = "output/detect"


# ========== 第三部分：后处理工具函数 ==========
def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    返回可用的 checkpoint 路径；若显式路径失效，则自动回退到最新的 best checkpoint。
    """
    explicit_path = Path(checkpoint_path)
    if explicit_path.exists():
        return str(explicit_path)

    candidates = sorted(
        Path("output/train").glob("*/checkpoint_best_regular.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        fallback = candidates[0]
        print(
            f"[Detect] Configured checkpoint not found: {checkpoint_path}\n"
            f"[Detect] Fallback to latest checkpoint: {fallback}"
        )
        return str(fallback)

    raise FileNotFoundError(
        "No usable checkpoint found. "
        f"Configured path: {checkpoint_path}. "
        "Also failed to find output/train/*/checkpoint_best_regular.pth."
    )


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    计算两组 `xyxy` 框之间的 IoU 矩阵。

    这里不依赖 torchvision 的 NMS/IoU 运算符，
    是为了让脚本在环境不完整时仍尽量可运行。
    """
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    用纯 PyTorch 实现一个简单的 NMS。

    返回保留下来的索引。
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    # 先按分数从高到低排序。
    order = torch.argsort(scores, descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        # 当前最高分框直接保留。
        current_idx = int(order[0].item())
        keep.append(current_idx)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = _box_iou(boxes[current_idx].unsqueeze(0), boxes[rest]).squeeze(0)

        # 只保留与当前框 IoU 不超过阈值的剩余框。
        order = rest[ious <= iou_threshold]

    return torch.as_tensor(keep, dtype=torch.long)


def _filter_with_threshold_and_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    class_thresholds: Dict[str, float],
    global_threshold: float,
    nms_iou_threshold: float,
) -> torch.Tensor:
    """
    先按类别阈值过滤，再按类内执行 NMS。

    返回最终保留的原始索引。
    """
    if scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    # 先排除无效类别，避免背景类或越界类别进入后处理。
    valid_class = (labels >= 0) & (labels < len(class_names))
    if not bool(valid_class.any()):
        return torch.empty((0,), dtype=torch.long)

    valid_idx = torch.where(valid_class)[0]
    boxes_v = boxes[valid_idx]
    scores_v = scores[valid_idx]
    labels_v = labels[valid_idx]

    # 每个类别允许使用不同阈值。
    per_cls_threshold = torch.as_tensor(
        [
            class_thresholds.get(class_names[int(label.item())], global_threshold)
            for label in labels_v
        ],
        dtype=scores_v.dtype,
    )

    conf_keep = torch.where(scores_v >= per_cls_threshold)[0]
    if conf_keep.numel() == 0:
        return torch.empty((0,), dtype=torch.long)

    boxes_v = boxes_v[conf_keep]
    scores_v = scores_v[conf_keep]
    labels_v = labels_v[conf_keep]
    idx_v = valid_idx[conf_keep]

    # 对每个类别独立做 NMS。
    classwise_keep: List[torch.Tensor] = []
    for cls_id in torch.unique(labels_v):
        cls_indices = torch.where(labels_v == cls_id)[0]
        keep_local = _nms(boxes_v[cls_indices], scores_v[cls_indices], nms_iou_threshold)
        classwise_keep.append(idx_v[cls_indices[keep_local]])

    if not classwise_keep:
        return torch.empty((0,), dtype=torch.long)

    final_idx = torch.cat(classwise_keep, dim=0)
    final_scores = scores[final_idx]
    return final_idx[torch.argsort(final_scores, descending=True)]


# ========== 第四部分：模型加载与单样本推理 ==========
def _resolve_white_path(uv_path: str) -> str | None:
    """兼容新旧目录结构，根据 UV 文件路径找到对应 white 图。"""
    uv_path_obj = Path(uv_path)

    sibling_m_dir = uv_path_obj.parent.with_name(f"{uv_path_obj.parent.name}_m")
    for white_dir in [uv_path_obj.parent, sibling_m_dir]:
        white_path = resolve_white_path_for_uv(uv_path_obj, white_dir if white_dir.exists() else None)
        if white_path is not None:
            return str(white_path)
    return None


def _load_model_and_postprocess(device: str):
    """
    加载双模态模型与 RF-DETR 的后处理器。

    这里显式把 `use_white / fusion_type / fusion_num_layers` 注入模型配置，
    以保证推理脚本与训练脚本的实验设置保持一致。
    """
    from custom.dual_model import build_dual_model
    from rfdetr.config import RFDETRBaseConfig
    from rfdetr.main import populate_args
    from rfdetr.models.lwdetr import PostProcess

    model_cfg = RFDETRBaseConfig(num_classes=NUM_CLASSES, pretrain_weights=None)
    model_kwargs = model_cfg.model_dump()
    model_kwargs["use_white"] = USE_WHITE
    model_kwargs["fusion_type"] = FUSION_TYPE
    model_kwargs["fusion_num_layers"] = FUSION_NUM_LAYERS

    args = populate_args(**model_kwargs)
    args.dual_modal = True
    args.device = device

    checkpoint_path = _resolve_checkpoint_path(CHECKPOINT_PATH)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    checkpoint_num_classes = state_dict["class_embed.bias"].shape[0]

    # RF-DETR 内部构建模型时会自动对 `num_classes` 做 +1（包含 no-object）。
    args.num_classes = checkpoint_num_classes - 1

    model = build_dual_model(args).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    postprocess = PostProcess(num_select=model_cfg.num_select)
    return model, postprocess, model_cfg.resolution


def _run_single_inference(
    model: torch.nn.Module,
    postprocess,
    img_uv: Image.Image,
    img_white: Image.Image | None,
    resolution: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对单对图像执行一次推理，并返回过滤后的框、分数和类别。
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 结果需要映射回 UV 原始尺寸，因此记录 UV 图像大小。
    orig_h, orig_w = img_uv.size[1], img_uv.size[0]

    # UV 始终作为主输入。
    tensor_uv = TF.resize(TF.normalize(TF.to_tensor(img_uv), mean, std), [resolution, resolution])
    batch_uv = tensor_uv.unsqueeze(0).to(device)
    target_sizes = torch.tensor([[orig_h, orig_w]], device=device)

    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=use_amp):
        if USE_WHITE and FUSION_TYPE != "none":
            if img_white is None:
                raise ValueError("White image is required when fusion is enabled.")

            tensor_white = TF.resize(
                TF.normalize(TF.to_tensor(img_white), mean, std),
                [resolution, resolution],
            )
            batch_white = tensor_white.unsqueeze(0).to(device)
            outputs = model(batch_uv, batch_white)
        else:
            outputs = model(batch_uv)

    # 将模型输出解码回图像坐标系。
    result = postprocess(outputs, target_sizes)[0]
    boxes = result["boxes"].detach().cpu().float()
    scores = result["scores"].detach().cpu().float()
    labels = result["labels"].detach().cpu().long()

    # 执行按类阈值过滤 + 类内 NMS。
    keep = _filter_with_threshold_and_nms(
        boxes=boxes,
        scores=scores,
        labels=labels,
        class_names=CLASS_NAMES,
        class_thresholds=CLASS_CONF_THRESHOLDS,
        global_threshold=CONFIDENCE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
    )

    return boxes[keep], scores[keep], labels[keep]


# ========== 第五部分：结果可视化 ==========
def _visualize_pair(
    img_white: Image.Image,
    img_uv: Image.Image,
    boxes_np: np.ndarray,
    scores_np: np.ndarray,
    labels_np: np.ndarray,
    out_path: str,
) -> None:
    """
    将 White/UV 图像并排可视化，并叠加检测框。

    优先使用 `supervision` 做可视化；若环境中没有该库，则退回到纯 PIL 实现。
    """
    panel_title_height = 28
    separator_width = 4

    detections = [
        f"{CLASS_NAMES[int(labels_np[i])]} {scores_np[i]:.2f}" for i in range(len(scores_np))
    ]

    def _add_panel_title(img: Image.Image, title: str) -> Image.Image:
        titled = Image.new("RGB", (img.width, img.height + panel_title_height), (255, 255, 255))
        titled.paste(img, (0, panel_title_height))
        draw = ImageDraw.Draw(titled)
        draw.text((8, 6), title, fill="black")
        return titled

    try:
        import supervision as sv

        if len(scores_np) > 0:
            sv_det = sv.Detections(
                xyxy=boxes_np,
                confidence=scores_np,
                class_id=labels_np.astype(int),
            )
        else:
            sv_det = sv.Detections.empty()

        box_annotator = sv.BoxAnnotator(thickness=BOX_THICKNESS)
        label_annotator = sv.LabelAnnotator(text_scale=TEXT_SCALE)

        white_arr = np.array(img_white)
        uv_arr = np.array(img_uv)

        if len(scores_np) > 0:
            white_arr = box_annotator.annotate(white_arr.copy(), sv_det)
            white_arr = label_annotator.annotate(white_arr, sv_det, labels=detections)
            uv_arr = box_annotator.annotate(uv_arr.copy(), sv_det)
            uv_arr = label_annotator.annotate(uv_arr, sv_det, labels=detections)

        white_arr = np.array(_add_panel_title(Image.fromarray(white_arr), "White"))
        uv_arr = np.array(_add_panel_title(Image.fromarray(uv_arr), "UV"))

        # 为了左右拼接，需要把两张图 padding 到相同高度。
        h_w, w_w = white_arr.shape[:2]
        h_u, w_u = uv_arr.shape[:2]
        h_max = max(h_w, h_u)

        if h_w < h_max:
            white_arr = np.vstack([white_arr, np.zeros((h_max - h_w, w_w, 3), dtype=np.uint8)])
        if h_u < h_max:
            uv_arr = np.vstack([uv_arr, np.zeros((h_max - h_u, w_u, 3), dtype=np.uint8)])

        separator = np.ones((h_max, separator_width, 3), dtype=np.uint8) * 255
        combined = np.hstack([uv_arr, separator, white_arr])
        Image.fromarray(combined).save(out_path, quality=95)
        return
    except ImportError:
        pass

    # ---------- 回退到 PIL 版本 ----------
    img_w = img_white.copy()
    img_u = img_uv.copy()
    draw_w = ImageDraw.Draw(img_w)
    draw_u = ImageDraw.Draw(img_u)

    for i in range(len(scores_np)):
        x1, y1, x2, y2 = boxes_np[i].tolist()
        text = f"{CLASS_NAMES[int(labels_np[i])]} {scores_np[i]:.2f}"

        # White / UV 两张图都画同一套框，方便人工对照。
        for draw in [draw_w, draw_u]:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=BOX_THICKNESS)
            draw.text((x1, max(y1 - 15, 0)), text, fill="red")

    img_w = _add_panel_title(img_w, "White")
    img_u = _add_panel_title(img_u, "UV")

    merged = Image.new(
        "RGB",
        (img_u.width + separator_width + img_w.width, max(img_u.height, img_w.height)),
        (255, 255, 255),
    )
    merged.paste(img_u, (0, 0))
    merged.paste(img_w, (img_u.width + separator_width, 0))
    merged.save(out_path, quality=95)


# ========== 第六部分：主执行流程 ==========
if __name__ == "__main__":
    # ---------- 第一步：创建输出目录 ----------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[Detect] Output dir: {output_dir}")

    # ---------- 第二步：加载模型 ----------
    model, postprocess, resolution = _load_model_and_postprocess(DEVICE)
    print("[Detect] Model loaded.")

    # ---------- 第三步：收集 UV 图像列表 ----------
    detect_root = Path(DETECT_DATASET_DIR)
    if list_image_files(detect_root):
        uv_files = [
            str(image_path)
            for image_path in list_image_files(detect_root)
            if is_probable_uv_image(image_path)
        ]
    else:
        layout = resolve_split_layout(
            dataset_dir=detect_root,
            split=DETECT_SPLIT,
            require_white=USE_WHITE and FUSION_TYPE != "none",
            require_labels=False,
        )
        uv_files = [
            str(image_path)
            for image_path in list_image_files(layout.uv_dir)
            if is_probable_uv_image(image_path)
        ]

    if not uv_files:
        print(f"[Detect] No UV images found in: {DETECT_DATASET_DIR}")
        print("[Detect] Expected either a direct UV image folder, or a dataset root with test/test_m style subdirs.")
        sys.exit(1)

    all_results = []
    processed = 0
    skipped = 0

    # ---------- 第四步：逐对执行推理 ----------
    for uv_path in uv_files:
        white_path = None
        if USE_WHITE and FUSION_TYPE != "none":
            white_path = _resolve_white_path(uv_path)

        if USE_WHITE and FUSION_TYPE != "none" and white_path is None:
            print(f"[Skip] Paired white image not found for: {os.path.basename(uv_path)}")
            skipped += 1
            continue

        img_uv = Image.open(uv_path).convert("RGB")
        img_white = Image.open(white_path).convert("RGB") if white_path is not None else None

        boxes_t, scores_t, labels_t = _run_single_inference(
            model=model,
            postprocess=postprocess,
            img_uv=img_uv,
            img_white=img_white,
            resolution=resolution,
            device=DEVICE,
        )

        boxes_np = boxes_t.numpy()
        scores_np = scores_t.numpy()
        labels_np = labels_t.numpy()

        # 将每张图的结果整理成可序列化字典，方便保存为 JSON。
        detections = []
        for i in range(len(scores_np)):
            cls_id = int(labels_np[i])
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id],
                    "confidence": float(scores_np[i]),
                    "bbox_xyxy": boxes_np[i].tolist(),
                }
            )

        filename = os.path.basename(uv_path)
        all_results.append(
            {
                "filename": filename,
                "uv_path": uv_path,
                "white_path": white_path,
                "num_detections": len(detections),
                "detections": detections,
            }
        )

        # 如启用绘图，则输出并排可视化图像。
        if DRAW_BOXES:
            out_img_path = os.path.join(output_dir, f"det_{Path(filename).stem}.jpg")
            _visualize_pair(
                img_white if img_white is not None else img_uv,
                img_uv,
                boxes_np,
                scores_np,
                labels_np,
                out_img_path,
            )

        processed += 1
        if processed % 20 == 0 or processed == len(uv_files):
            print(f"[Detect] Progress: {processed}/{len(uv_files)}")

    # ---------- 第五步：保存 JSON ----------
    if SAVE_JSON:
        json_path = os.path.join(output_dir, "detections.json")
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(all_results, file, ensure_ascii=False, indent=2)
        print(f"[Detect] JSON saved: {json_path}")

    # ---------- 第六步：打印汇总 ----------
    total_dets = sum(result["num_detections"] for result in all_results)
    print("[Detect] Done.")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Total detections: {total_dets}")
    print(f"  Output dir: {output_dir}")
