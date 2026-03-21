"""
文件说明：本文件为当前仓库保留旧式训练循环与评估循环的兼容实现。
功能说明：提供 `train_one_epoch` 与 `evaluate`，让仍依赖旧 `main.py` 风格
入口的 custom 双模态代码在新版 `src/rfdetr` 结构下继续工作。

结构概览：
  第一部分：导入依赖与 AMP 兼容
  第二部分：训练统计辅助函数
  第三部分：单轮训练循环
  第四部分：评估循环与扩展指标
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, DefaultDict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.util.misc import NestedTensor

try:
    from torch.amp import GradScaler, autocast

    _DEPRECATED_AMP = False
except ImportError:  # pragma: no cover - old torch fallback
    from torch.cuda.amp import GradScaler, autocast

    _DEPRECATED_AMP = True


# ========== 第二部分：训练统计辅助函数 ==========
def get_autocast_args(args):
    """统一不同 torch 版本下 autocast 的调用参数。"""
    if _DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": torch.bfloat16}
    return {"device_type": "cuda", "enabled": args.amp, "dtype": torch.bfloat16}


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
    """按置信度阈值扫描 macro-F1，用于生成更可读的 `results.json`。"""
    num_classes = len(per_class_data)
    results = []
    for conf_thresh in conf_thresholds:
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []
        for index in range(num_classes):
            data = per_class_data[index]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore
            valid_matches = matches[valid]
            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if classes_with_gt:
            macro_precision = np.mean([per_class_precisions[index] for index in classes_with_gt])
            macro_recall = np.mean([per_class_recalls[index] for index in classes_with_gt])
            macro_f1 = np.mean([per_class_f1s[index] for index in classes_with_gt])
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": float(conf_thresh),
                "macro_f1": float(macro_f1),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )
    return results


def coco_extended_metrics(coco_eval):
    """从 COCO evaluator 原始匹配结果中额外整理每类 precision/recall/F1。"""
    iou50_idx = np.argwhere(np.isclose(coco_eval.params.iouThrs, 0.50)).item()
    cat_ids = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_idx = 0
    maxdet_idx = 2

    eval_imgs_unflat = {}
    for entry in coco_eval.evalImgs:
        if entry is None:
            continue
        cat_id = entry["category_id"]
        area_rng = tuple(entry["aRng"])
        img_id = entry["image_id"]
        eval_imgs_unflat.setdefault(cat_id, {}).setdefault(area_rng, {})[img_id] = entry

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])
    per_class_data = []
    for cat_id in cat_ids:
        dt_scores = []
        dt_matches = []
        dt_ignore = []
        total_gt = 0
        for img_id in coco_eval.params.imgIds:
            entry = eval_imgs_unflat.get(cat_id, {}).get(area_rng_all, {}).get(img_id)
            if entry is None:
                continue
            total_gt += sum(1 for ignored in entry["gtIgnore"] if not ignored)
            for index in range(len(entry["dtIds"])):
                dt_scores.append(entry["dtScores"][index])
                dt_matches.append(entry["dtMatches"][iou50_idx, index])
                dt_ignore.append(entry["dtIgnore"][iou50_idx, index])
        per_class_data.append(
            {
                "scores": np.array(dt_scores),
                "matches": np.array(dt_matches),
                "ignore": np.array(dt_ignore, dtype=bool),
                "total_gt": total_gt,
            }
        )

    classes_with_gt = [index for index in range(num_classes) if per_class_data[index]["total_gt"] > 0]
    best = max(
        sweep_confidence_thresholds(per_class_data, np.linspace(0.0, 1.0, 101), classes_with_gt),
        key=lambda item: item["macro_f1"],
    )

    per_class = []
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_eval.cocoGt.loadCats(cat_ids)}
    for index, cat_id in enumerate(cat_ids):
        precision_slice = coco_eval.eval["precision"][:, :, index, area_idx, maxdet_idx]
        precision_masked = np.where(precision_slice > -1, precision_slice, np.nan)
        ap_per_iou = np.nanmean(precision_masked, axis=1)
        ap_50_95 = float(np.nanmean(ap_per_iou))
        ap_50 = float(np.nanmean(precision_masked[iou50_idx]))
        if any(np.isnan(value) for value in (ap_50_95, ap_50, best["per_class_prec"][index], best["per_class_rec"][index])):
            continue
        per_class.append(
            {
                "class": cat_id_to_name[int(cat_id)],
                "map@50:95": ap_50_95,
                "map@50": ap_50,
                "precision": float(best["per_class_prec"][index]),
                "recall": float(best["per_class_rec"][index]),
                "f1_score": float(best["per_class_f1"][index]),
            }
        )

    return {
        "summary": {
            "confidence_threshold": float(best["confidence_threshold"]),
            "map@50:95": float(coco_eval.stats[0]),
            "map@50": float(coco_eval.stats[1]),
            "precision": float(best["macro_precision"]),
            "recall": float(best["macro_recall"]),
            "f1_score": float(best["macro_f1"]),
        },
        "class_map": per_class,
    }


# ========== 第三部分：单轮训练循环 ==========
def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = None,
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] | None = None,
):
    """执行一轮旧式训练循环，兼容单模态与双模态 batch 结构。"""
    schedules = schedules or {}
    callbacks = callbacks or defaultdict(list)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = f"Epoch: [{epoch}]"
    print_freq = args.print_freq if args is not None else 10
    start_steps = epoch * num_training_steps_per_epoch

    scaler = GradScaler("cuda", enabled=args.amp) if not _DEPRECATED_AMP else GradScaler(enabled=args.amp)
    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 2:
            samples, targets = batch
            samples_uv = None
        else:
            samples, samples_uv, targets = batch

        global_step = start_steps + data_iter_step
        for callback in callbacks["on_train_batch_start"]:
            callback({"step": global_step, "model": model, "epoch": epoch})

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(global_step)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                if samples_uv is not None:
                    samples_uv.tensors = F.interpolate(samples_uv.tensors, size=scale, mode="bilinear", align_corners=False)
                    samples_uv.mask = F.interpolate(samples_uv.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()

        for index in range(args.grad_accum_steps):
            start_idx = index * sub_batch_size
            end_idx = start_idx + sub_batch_size
            current_samples = NestedTensor(samples.tensors[start_idx:end_idx], samples.mask[start_idx:end_idx]).to(device)
            if samples_uv is not None:
                current_samples_uv = NestedTensor(samples_uv.tensors[start_idx:end_idx], samples_uv.mask[start_idx:end_idx]).to(device)
            else:
                current_samples_uv = None
            current_targets = [{key: value.to(device) for key, value in target.items()} for target in targets[start_idx:end_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(current_samples, current_targets) if current_samples_uv is None else model(current_samples, current_samples_uv, current_targets)
                loss_dict = criterion(outputs, current_targets)
                weight_dict = criterion.weight_dict
                losses = sum((1 / args.grad_accum_steps) * loss_dict[key] * weight_dict[key] for key in loss_dict if key in weight_dict)
            scaler.scale(losses).backward()

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{key}_unscaled": value for key, value in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {key: value * weight_dict[key] for key, value in loss_dict_reduced.items() if key in weight_dict}
        loss_value = sum(loss_dict_reduced_scaled.values()).item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {key: meter.global_avg for key, meter in metric_logger.meters.items()}


# ========== 第四部分：评估循环与扩展指标 ==========
def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    """执行旧式评估循环，并返回与历史 `main.py` 一致的结果结构。"""
    model.eval()
    criterion.eval()
    if args.fp16_eval:
        model.half()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    coco_evaluator = CocoEvaluator(base_ds, ("bbox",) if not args.segmentation_head else ("bbox", "segm"), args.eval_max_dets)
    print_freq = args.print_freq if args is not None else 10

    for batch in metric_logger.log_every(data_loader, print_freq, "Test:"):
        if len(batch) == 2:
            samples, targets = batch
            samples_uv = None
        else:
            samples, samples_uv, targets = batch

        samples = samples.to(device)
        if samples_uv is not None:
            samples_uv = samples_uv.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()
            if samples_uv is not None:
                samples_uv.tensors = samples_uv.tensors.half()

        with autocast(**get_autocast_args(args)):
            outputs = model(samples) if samples_uv is None else model(samples, samples_uv)

        if args.fp16_eval:
            for key, value in list(outputs.items()):
                if key == "enc_outputs":
                    for sub_key in value:
                        value[sub_key] = value[sub_key].float()
                elif key == "aux_outputs":
                    for aux_index in range(len(value)):
                        for sub_key in value[aux_index]:
                            value[aux_index][sub_key] = value[aux_index][sub_key].float()
                else:
                    outputs[key] = value.float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {key: value * weight_dict[key] for key, value in loss_dict_reduced.items() if key in weight_dict}
        loss_dict_reduced_unscaled = {f"{key}_unscaled": value for key, value in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()), **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        results = {target["image_id"].item(): output for target, output in zip(targets, results_all)}
        coco_evaluator.update(results)

    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    bbox_eval = coco_evaluator.coco_eval["bbox"]
    stats["results_json"] = coco_extended_metrics(bbox_eval)
    stats["coco_eval_bbox"] = bbox_eval.stats.tolist()
    if args.segmentation_head and "segm" in coco_evaluator.coco_eval:
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
