"""
文件说明：本文件实现 LaSt-ViT / LazyStrike 的核心聚合与训练辅助损失。
功能说明：按照论文思路对 ViT patch token 做频域低通稳定性评分、channel-wise Top-K
聚合 LazyStrike CLS，并提供一个可挂到 DETR criterion 上的 image-level 辅助监督。

结构概览：
  第一部分：导入依赖
  第二部分：LazyStrike 聚合工具
  第三部分：LazyStrike 聚合模块
  第四部分：Patch Score / Point-in-Box 诊断工具
  第五部分：辅助分类损失补丁
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Literal

import torch
from torch import nn
import torch.nn.functional as F


# ========== 第二部分：LazyStrike 聚合工具 ==========
def _feature_to_tokens(feature: torch.Tensor) -> tuple[torch.Tensor, dict[str, int | str]]:
    """
    将 dense feature 统一整理成 [B, N, C] patch token。

    为什么保留 grid 元信息：
    - 论文的 vote count 本质是 patch 级可视化；
    - 当前检测特征来自 [B, C, H, W]，后续需要恢复成 [B, H, W] heatmap。
    """
    if feature.dim() == 3:
        return feature, {"layout": "tokens", "num_tokens": feature.shape[1]}

    if feature.dim() == 4:
        _, _, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2).contiguous()
        return tokens, {"layout": "grid", "height": height, "width": width}

    raise ValueError(
        f"LazyStrike expects [B, N, C] or [B, C, H, W], got {tuple(feature.shape)}."
    )


def _tokens_to_map(values: torch.Tensor, layout: dict[str, int | str]) -> torch.Tensor:
    """将 [B, N] patch 值恢复为 heatmap；token 输入则原样返回。"""
    if layout["layout"] == "tokens":
        return values

    height = int(layout["height"])
    width = int(layout["width"])
    expected_tokens = height * width
    if values.shape[1] != expected_tokens:
        raise ValueError(
            f"Token count {values.shape[1]} does not match grid {height}x{width}."
        )
    return values.reshape(values.shape[0], height, width).contiguous()


def _build_gaussian_lowpass(
    channels: int,
    *,
    sigma_scale: float,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    """
    构造 channel 维度 FFT 使用的 Gaussian low-pass filter。

    论文和官方实现都在 channel 频域做低通；这里使用 `sqrt(C) * sigma_scale`
    作为默认带宽，保持和官方示例同量级，同时允许后续实验调节。
    """
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}.")

    sigma = max(float(channels) ** 0.5 * float(sigma_scale), eps)
    positions = torch.arange(channels, device=device, dtype=dtype)
    positions = positions - (float(channels) - 1.0) / 2.0
    kernel = torch.exp(-0.5 * (positions / sigma) ** 2)
    return kernel / kernel.max().clamp_min(eps)


def _resolve_topk(num_tokens: int, topk: int, topk_ratio: float) -> int:
    """根据绝对 K 或比例 K 得到最终 Top-K token 数。"""
    if topk > 0:
        resolved = topk
    else:
        resolved = int(round(float(num_tokens) * float(topk_ratio)))
    return max(1, min(int(resolved), int(num_tokens)))


# ========== 第三部分：LazyStrike 聚合模块 ==========
class LazyStrikeAggregator(nn.Module):
    """
    论文式 LazyStrike 聚合模块。

    输入：
      - [B, C, H, W] dense feature，或 [B, N, C] patch token

    输出：
      - `cls`：channel-wise Top-K 聚合得到的 LazyStrike CLS
      - `vote_count`：每个 patch 被多少个 channel 选中
      - `patch_score`：LazyStrike CLS 与每个 patch 的 cosine similarity
    """

    def __init__(
        self,
        *,
        topk: int = 0,
        topk_ratio: float = 0.25,
        sigma_scale: float = 1.0,
        score_numerator: Literal["original", "filtered"] = "original",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if topk < 0:
            raise ValueError(f"topk must be >= 0, got {topk}.")
        if topk == 0 and not (0.0 < topk_ratio <= 1.0):
            raise ValueError(f"topk_ratio must be in (0, 1], got {topk_ratio}.")
        if sigma_scale <= 0.0:
            raise ValueError(f"sigma_scale must be positive, got {sigma_scale}.")
        if score_numerator not in {"original", "filtered"}:
            raise ValueError(f"Unsupported score_numerator: {score_numerator}.")

        self.topk = int(topk)
        self.topk_ratio = float(topk_ratio)
        self.sigma_scale = float(sigma_scale)
        self.score_numerator = score_numerator
        self.eps = float(eps)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, layout = _feature_to_tokens(feature)
        original_dtype = tokens.dtype

        # torch.fft 对 bfloat16/float16 支持不稳定，显式转 fp32 保证 AMP 下可运行。
        tokens_fp32 = tokens.float()
        batch_size, num_tokens, channels = tokens_fp32.shape
        topk = _resolve_topk(num_tokens, self.topk, self.topk_ratio)

        lowpass = _build_gaussian_lowpass(
            channels,
            sigma_scale=self.sigma_scale,
            device=tokens_fp32.device,
            dtype=tokens_fp32.dtype,
            eps=self.eps,
        )

        spectrum = torch.fft.fft(tokens_fp32, dim=-1)
        spectrum = torch.fft.fftshift(spectrum, dim=-1)
        filtered = spectrum * lowpass.view(1, 1, channels)
        filtered = torch.fft.ifftshift(filtered, dim=-1)
        stable_tokens = torch.fft.ifft(filtered, dim=-1).real

        numerator = tokens_fp32 if self.score_numerator == "original" else stable_tokens
        stability = numerator / (stable_tokens.sub(tokens_fp32).abs().add(self.eps))

        # 对每个 channel 独立在 token 维度选 Top-K，这是论文聚合区别于普通池化的关键。
        _, indices = torch.topk(stability, k=topk, dim=1, largest=True, sorted=False)
        selected_tokens = torch.gather(tokens_fp32, dim=1, index=indices)
        lazy_cls = selected_tokens.mean(dim=1)

        flat_indices = indices.reshape(batch_size, -1)
        vote_count = torch.zeros(
            (batch_size, num_tokens),
            device=tokens_fp32.device,
            dtype=tokens_fp32.dtype,
        )
        vote_count.scatter_add_(
            dim=1,
            index=flat_indices,
            src=torch.ones_like(flat_indices, dtype=tokens_fp32.dtype),
        )
        patch_score = F.cosine_similarity(tokens_fp32, lazy_cls.unsqueeze(1), dim=-1)

        return {
            "cls": lazy_cls.to(original_dtype),
            "vote_count": _tokens_to_map(vote_count, layout),
            "patch_score": _tokens_to_map(patch_score, layout),
        }


# ========== 第四部分：Patch Score / Point-in-Box 诊断工具 ==========
@torch.no_grad()
def point_in_box_from_patch_score(
    patch_score: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    计算论文 Point-in-Box 诊断指标。

    输入约定：
      - `patch_score` 为 [B, H, W]，来自 LazyStrike CLS 与 patch token 的相似度；
      - `targets[*]["boxes"]` 为 RF-DETR 训练链路使用的归一化 cxcywh。

    为什么放在这里：
    - 论文用 PiB 判断最高 Patch Score 是否落在前景框内；
    - 当前项目已有检测框标签，后续可以直接用它判断 LazyStrike 是否真把语义拉回缺陷区域。
    """
    if patch_score.dim() != 3:
        raise ValueError(f"patch_score must be [B, H, W], got {tuple(patch_score.shape)}.")

    batch_size, height, width = patch_score.shape
    if batch_size != len(targets):
        raise ValueError(
            f"patch_score batch size {batch_size} does not match targets length {len(targets)}."
        )

    flat_indices = patch_score.flatten(1).argmax(dim=1)
    y_index = torch.div(flat_indices, width, rounding_mode="floor")
    x_index = flat_indices.remainder(width)
    x_center = (x_index.to(patch_score.dtype) + 0.5) / float(width)
    y_center = (y_index.to(patch_score.dtype) + 0.5) / float(height)

    in_box = torch.zeros(batch_size, device=patch_score.device, dtype=torch.bool)
    for batch_index, target in enumerate(targets):
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            continue
        boxes = boxes.to(device=patch_score.device, dtype=patch_score.dtype)
        x0 = boxes[:, 0] - boxes[:, 2] * 0.5
        y0 = boxes[:, 1] - boxes[:, 3] * 0.5
        x1 = boxes[:, 0] + boxes[:, 2] * 0.5
        y1 = boxes[:, 1] + boxes[:, 3] * 0.5
        inside = (
            (x_center[batch_index] >= x0)
            & (x_center[batch_index] <= x1)
            & (y_center[batch_index] >= y0)
            & (y_center[batch_index] <= y1)
        )
        in_box[batch_index] = inside.any()

    return {
        "in_box": in_box,
        "top_patch_xy": torch.stack((x_center, y_center), dim=-1),
        "pib": in_box.float().mean(),
    }


# ========== 第五部分：辅助分类损失补丁 ==========
def _targets_to_multihot(
    targets: list[dict[str, torch.Tensor]],
    *,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    将检测框类别折成 image-level multi-hot 标签。

    为什么使用 multi-hot：
    - 论文用图像级语义监督训练 CLS；
    - 当前任务一张图可能同时出现 NPML/PML/PM，多标签比单标签更符合检测数据。
    """
    labels = torch.zeros((len(targets), num_classes), device=device, dtype=dtype)
    for batch_index, target in enumerate(targets):
        target_labels = target.get("labels")
        if target_labels is None or target_labels.numel() == 0:
            continue
        valid_labels = target_labels[(target_labels >= 0) & (target_labels < num_classes)].long()
        if valid_labels.numel() == 0:
            continue
        labels[batch_index, valid_labels.unique()] = 1.0
    return labels


def _loss_lazystrike_cls(self, outputs, targets, indices=None, num_boxes=None):
    """LazyStrike CLS 的 image-level 多标签辅助分类损失。"""
    logits = outputs.get("lazystrike_logits")
    if logits is None:
        return {}

    num_classes = logits.shape[-1]
    target_labels = _targets_to_multihot(
        targets,
        num_classes=num_classes,
        device=logits.device,
        dtype=logits.dtype,
    )
    loss = F.binary_cross_entropy_with_logits(logits, target_labels, reduction="mean")
    return {"loss_lazystrike_cls": loss}


def _get_loss_with_lazystrike(self, loss, outputs, targets, indices, num_boxes, **kwargs):
    """在原 criterion loss 分发表中补充 LazyStrike 辅助损失。"""
    if loss == "lazystrike":
        return self.loss_lazystrike_cls(outputs, targets, indices, num_boxes)
    return self._lazystrike_original_get_loss(loss, outputs, targets, indices, num_boxes, **kwargs)


def apply_lazystrike_auxiliary_loss(criterion: Any, args: Any) -> None:
    """
    根据训练参数为 criterion 挂载 LazyStrike 辅助 loss。

    这里采用运行时补丁，是为了不直接改 `src/rfdetr` 主库 criterion 的核心逻辑，
    与当前 PM loss weighting 的接入风格保持一致。
    """
    enabled = bool(getattr(args, "lazystrike_enabled", False))
    loss_coef = float(getattr(args, "lazystrike_loss_coef", 0.0))
    if not enabled or loss_coef <= 0.0:
        return

    if "lazystrike" not in criterion.losses:
        criterion.losses.append("lazystrike")
    criterion.weight_dict["loss_lazystrike_cls"] = loss_coef

    if not hasattr(criterion, "_lazystrike_original_get_loss"):
        criterion._lazystrike_original_get_loss = criterion.get_loss
    criterion.loss_lazystrike_cls = MethodType(_loss_lazystrike_cls, criterion)
    criterion.get_loss = MethodType(_get_loss_with_lazystrike, criterion)

    print(
        "[LazyStrike] auxiliary loss enabled "
        f"loss_coef={loss_coef}"
    )
