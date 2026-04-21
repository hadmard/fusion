"""
文件说明：本文件为当前 custom 训练链路提供 PM / 小目标损失加权补丁。
功能说明：在不修改 `src/rfdetr` 主库 criterion 的前提下，运行时替换 criterion 中
分类与框回归损失的计算，让 PM 类和归一化面积较小的目标拥有更高监督权重。

结构概览：
  第一部分：导入依赖
  第二部分：实例权重计算
  第三部分：加权 loss 实现
  第四部分：对外补丁入口
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

from rfdetr.models.math import accuracy
from rfdetr.utilities import box_ops


# ========== 第二部分：实例权重计算 ==========
def _matched_instance_weights(
    criterion: Any,
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """
    根据匹配到的 GT 类别和尺寸生成实例权重。

    为什么只给匹配正样本加权：
    - 当前 PM 的主要矛盾是漏检和小框定位不足；
    - 直接提高整条 PM 通道的负样本权重，可能反而压低 PM 召回；
    - 因此这里只加权 Hungarian 匹配后的正样本监督。
    """
    pieces: list[torch.Tensor] = []
    pm_class_id = int(getattr(criterion, "pm_loss_class_id", 2))
    pm_weight = float(getattr(criterion, "pm_class_loss_weight", 1.0))
    small_weight = float(getattr(criterion, "small_object_loss_weight", 1.0))
    small_area_threshold = float(getattr(criterion, "small_object_area_threshold", 0.0))

    for target, (_, matched_target_indices) in zip(targets, indices):
        labels = target["labels"][matched_target_indices]
        boxes = target["boxes"][matched_target_indices]
        weights = torch.ones_like(labels, dtype=torch.float32, device=labels.device)

        if pm_weight > 1.0:
            pm_weights = torch.full_like(weights, pm_weight)
            weights = torch.where(labels == pm_class_id, pm_weights, weights)

        if small_weight > 1.0 and small_area_threshold > 0.0 and boxes.numel() > 0:
            # 训练 target 已在 Normalize 中转成归一化 cxcywh，w*h 可直接表示相对面积。
            areas = boxes[:, 2] * boxes[:, 3]
            small_weights = torch.full_like(weights, small_weight)
            weights = torch.where(areas <= small_area_threshold, torch.maximum(weights, small_weights), weights)

        pieces.append(weights)

    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


# ========== 第三部分：加权 loss 实现 ==========
def _weighted_loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    """
    PM / 小目标正样本加权版分类损失。

    当前训练默认使用 `ia_bce_loss=True`，这里完整保留原 IA-BCE 逻辑，只在匹配正样本
    对应的类别位置乘权重。若后续切到 varifocal / position-supervised，则回退到原始实现。
    """
    if self.use_varifocal_loss or self.use_position_supervised_loss:
        return self._pm_original_loss_labels(outputs, targets, indices, num_boxes, log=log)

    assert "pred_logits" in outputs
    src_logits = outputs["pred_logits"]
    idx = self._get_src_permutation_idx(indices)
    target_classes_o = torch.cat([target["labels"][matched] for target, (_, matched) in zip(targets, indices)])
    instance_weights = _matched_instance_weights(self, targets, indices).to(
        device=src_logits.device,
        dtype=src_logits.dtype,
    )

    if self.ia_bce_loss:
        alpha = self.focal_alpha
        gamma = 2
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([target["boxes"][matched] for target, (_, matched) in zip(targets, indices)], dim=0)

        if src_boxes.numel() == 0:
            loss_ce = src_logits.sum() * 0.0
        else:
            iou_targets = torch.diag(
                box_ops.box_iou(
                    box_ops.box_cxcywh_to_xyxy(src_boxes.detach()),
                    box_ops.box_cxcywh_to_xyxy(target_boxes),
                )[0]
            )
            pos_ious = iou_targets.clone().detach()
            prob = src_logits.sigmoid()
            pos_weights = torch.zeros_like(src_logits)
            neg_weights = prob**gamma

            pos_ind = [id_tensor for id_tensor in idx]
            pos_ind.append(target_classes_o)

            target = prob[tuple(pos_ind)].pow(alpha) * pos_ious.pow(1 - alpha)
            target = torch.clamp(target, 0.01).detach()

            pos_weights[tuple(pos_ind)] = target.to(pos_weights.dtype)
            neg_weights[tuple(pos_ind)] = 1 - target.to(neg_weights.dtype)
            loss_matrix = neg_weights * src_logits - F.logsigmoid(src_logits) * (pos_weights + neg_weights)

            if instance_weights.numel() > 0:
                element_weights = torch.ones_like(loss_matrix)
                element_weights[tuple(pos_ind)] = instance_weights.to(element_weights.dtype)
                loss_matrix = loss_matrix * element_weights
            loss_ce = loss_matrix.sum() / num_boxes
    else:
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        prob = src_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
        p_t = prob * target_classes_onehot + (1 - prob) * (1 - target_classes_onehot)
        loss_matrix = ce_loss * ((1 - p_t) ** 2)
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target_classes_onehot + (1 - self.focal_alpha) * (
                1 - target_classes_onehot
            )
            loss_matrix = alpha_t * loss_matrix

        if instance_weights.numel() > 0:
            pos_ind = [id_tensor for id_tensor in idx]
            pos_ind.append(target_classes_o)
            element_weights = torch.ones_like(loss_matrix)
            element_weights[tuple(pos_ind)] = instance_weights.to(element_weights.dtype)
            loss_matrix = loss_matrix * element_weights
        loss_ce = loss_matrix.mean(1).sum() / num_boxes

    losses = {"loss_ce": loss_ce}
    if log:
        losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    return losses


def _weighted_loss_boxes(self, outputs, targets, indices, num_boxes):
    """PM / 小目标正样本加权版 L1 与 GIoU 框回归损失。"""
    assert "pred_boxes" in outputs
    idx = self._get_src_permutation_idx(indices)
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat([target["boxes"][matched] for target, (_, matched) in zip(targets, indices)], dim=0)
    instance_weights = _matched_instance_weights(self, targets, indices).to(
        device=src_boxes.device,
        dtype=src_boxes.dtype,
    )

    if src_boxes.numel() == 0:
        zero = outputs["pred_boxes"].sum() * 0.0
        return {"loss_bbox": zero, "loss_giou": zero}

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum(dim=1)
    losses = {"loss_bbox": (loss_bbox * instance_weights).sum() / num_boxes}

    loss_giou = 1 - torch.diag(
        box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        )
    )
    losses["loss_giou"] = (loss_giou * instance_weights).sum() / num_boxes
    return losses


# ========== 第四部分：对外补丁入口 ==========
def apply_pm_loss_weighting(criterion: Any, args: Any) -> None:
    """根据训练参数决定是否给 criterion 打 PM / 小目标损失加权补丁。"""
    enabled = bool(getattr(args, "enable_pm_loss_weighting", False))
    if not enabled:
        return

    pm_weight = float(getattr(args, "pm_class_loss_weight", 1.0))
    small_weight = float(getattr(args, "small_object_loss_weight", 1.0))
    if pm_weight <= 1.0 and small_weight <= 1.0:
        return

    criterion.pm_loss_class_id = int(getattr(args, "pm_loss_class_id", 2))
    criterion.pm_class_loss_weight = pm_weight
    criterion.small_object_loss_weight = small_weight
    criterion.small_object_area_threshold = float(getattr(args, "small_object_area_threshold", 0.0))
    # 防止同一个 criterion 被重复打补丁时，把“原始实现”覆盖成已经加权后的方法。
    if not hasattr(criterion, "_pm_original_loss_labels"):
        criterion._pm_original_loss_labels = criterion.loss_labels
    if not hasattr(criterion, "_pm_original_loss_boxes"):
        criterion._pm_original_loss_boxes = criterion.loss_boxes
    criterion.loss_labels = MethodType(_weighted_loss_labels, criterion)
    criterion.loss_boxes = MethodType(_weighted_loss_boxes, criterion)

    print(
        "[PMLoss] enabled "
        f"pm_class_id={criterion.pm_loss_class_id}, "
        f"pm_weight={criterion.pm_class_loss_weight}, "
        f"small_weight={criterion.small_object_loss_weight}, "
        f"small_area_threshold={criterion.small_object_area_threshold}"
    )
