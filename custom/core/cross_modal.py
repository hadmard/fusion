"""
文件说明：本文件实现当前仓库双模态主线使用的跨模态融合模块。
功能说明：在“UV 为主模态、White 为辅助模态”的前提下，提供当前主线路径所需的
same-depth one-read deformable UV<-White 跨模态融合实现。

结构概览：
  第一部分：导入依赖与常量
  第二部分：通用张量工具
  第三部分：通道投影
  第四部分：deformable 跨模态读取块
  第五部分：单层级跨模态融合
"""

# ========== 第一部分：导入依赖与常量 ==========
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from rfdetr.models.ops.modules import MSDeformAttn

FUSION_DIM = 256
EXPECTED_FUSION_LEVELS = 4
MemoryInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]


# ========== 第二部分：通用张量工具 ==========
def _to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    将输入统一转成 token 序列表示。

    支持两种输入：
      - [B, N, C]：已经是 token 序列，直接返回
      - [B, C, H, W]：flatten 成 [B, HW, C]
    """
    if x.dim() == 3:
        return x, {"layout": "tokens", "num_tokens": x.shape[1]}

    if x.dim() == 4:
        _, _, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, {"layout": "grid", "height": h, "width": w}

    raise ValueError(
        f"Expected a token sequence [B, N, C] or feature grid [B, C, H, W], got {tuple(x.shape)}."
    )


def _restore_layout(x: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    """
    将 token 序列恢复回原始布局。
    """
    if metadata["layout"] == "tokens":
        return x

    _, num_tokens, channels = x.shape
    height = metadata["height"]
    width = metadata["width"]
    expected_tokens = height * width

    if num_tokens != expected_tokens:
        raise ValueError(
            f"Token count {num_tokens} does not match patch grid {height}x{width} ({expected_tokens})."
        )

    return x.transpose(1, 2).reshape(x.shape[0], channels, height, width).contiguous()


def _flatten_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    将 padding mask 统一整理成 [B, N] 形式。
    """
    if mask is None:
        return None

    if mask.dim() == 2:
        return mask

    if mask.dim() == 3:
        return mask.flatten(1)

    raise ValueError(
        f"Expected padding mask [B, N] or [B, H, W], got {tuple(mask.shape)}."
    )


def _apply_padding_mask(tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """
    显式清零 padding 位置，避免这些位置参与后续 token 计算。
    """
    if padding_mask is None:
        return tokens
    return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def _validate_modal_shapes(query_tokens: torch.Tensor, memory_tokens: torch.Tensor) -> None:
    """
    检查跨模态读取两端在 batch 与 channel 维度上是否兼容。
    """
    if query_tokens.shape[0] != memory_tokens.shape[0]:
        raise ValueError(
            "UV and White batch sizes must match before fusion. "
            f"Got {query_tokens.shape[0]} and {memory_tokens.shape[0]}."
        )

    if query_tokens.shape[-1] != memory_tokens.shape[-1]:
        raise ValueError(
            "UV and White channel dimensions must match before fusion. "
            f"Got {query_tokens.shape[-1]} and {memory_tokens.shape[-1]}."
        )


def _build_reference_points_from_query_layout(
    *,
    layout_metadata: dict[str, Any],
    batch_size: int,
    num_memory_levels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    为 UV 查询网格生成归一化 reference points。

    当前主线路径只支持规则网格语义：
      - [B, C, H, W] 直接使用 H/W
      - [B, N, C] 要求 N 能还原为正方形 patch grid
    """
    layout = layout_metadata["layout"]
    if layout == "grid":
        height = int(layout_metadata["height"])
        width = int(layout_metadata["width"])
    elif layout == "tokens":
        num_tokens = int(layout_metadata["num_tokens"])
        side = int(round(float(num_tokens) ** 0.5))
        if side * side != num_tokens:
            raise ValueError(
                "Deformable fusion expects token inputs to form a square grid. "
                f"Got {num_tokens} tokens."
            )
        height = side
        width = side
    else:
        raise ValueError(f"Unsupported layout metadata: {layout_metadata}.")

    y_coords = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    x_coords = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    query_points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    return query_points.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, num_memory_levels, 1)


def _flatten_multi_level_memory_for_ms_deform_attn(
    memory_features: Sequence[torch.Tensor],
    memory_masks: Sequence[torch.Tensor | None] | None,
) -> MemoryInputs:
    """
    将 White feature 组装成 MSDeformAttn 所需的扁平输入格式。
    """
    feature_list = list(memory_features)
    if not feature_list:
        raise ValueError("memory_features must contain at least one tensor.")

    mask_list = _normalize_mask_group(
        masks=memory_masks,
        expected_length=len(feature_list),
    )

    flattened_features: list[torch.Tensor] = []
    flattened_masks: list[torch.Tensor | None] = []
    spatial_shapes: list[tuple[int, int]] = []

    reference_batch_size = feature_list[0].shape[0]
    reference_channels = feature_list[0].shape[1]

    for index, (feature, mask) in enumerate(zip(feature_list, mask_list)):
        if feature.dim() != 4:
            raise ValueError(
                f"memory_features[{index}] must be [B, C, H, W], got {tuple(feature.shape)}."
            )

        batch_size, channels, height, width = feature.shape
        if batch_size != reference_batch_size:
            raise ValueError(
                "All memory feature levels must share the same batch size. "
                f"Got {batch_size} at index {index}, expected {reference_batch_size}."
            )
        if channels != reference_channels:
            raise ValueError(
                "All memory feature levels must share the same channel dimension. "
                f"Got {channels} at index {index}, expected {reference_channels}."
            )

        spatial_shapes.append((height, width))
        flattened_features.append(feature.flatten(2).transpose(1, 2).contiguous())
        flattened_masks.append(_flatten_padding_mask(mask))

    input_flatten = torch.cat(flattened_features, dim=1)
    input_spatial_shapes = torch.as_tensor(
        spatial_shapes,
        dtype=torch.long,
        device=input_flatten.device,
    )
    level_sizes = input_spatial_shapes.prod(1)
    input_level_start_index = torch.cat(
        (
            level_sizes.new_zeros((1,)),
            level_sizes.cumsum(0)[:-1],
        )
    )

    if all(mask is None for mask in flattened_masks):
        input_padding_mask = None
    else:
        input_padding_mask = torch.cat(
            [
                mask
                if mask is not None
                else torch.zeros(
                    (reference_batch_size, height * width),
                    dtype=torch.bool,
                    device=input_flatten.device,
                )
                for mask, (height, width) in zip(flattened_masks, spatial_shapes)
            ],
            dim=1,
        )

    return input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask


def _normalize_feature_group(
    *,
    name: str,
    features: Sequence[torch.Tensor],
    expected_length: int,
) -> list[torch.Tensor]:
    """
    将输入特征列表转成 list，并明确检查长度。
    """
    feature_list = list(features)
    if len(feature_list) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} feature levels, got {len(feature_list)}."
        )
    return feature_list


def _normalize_mask_group(
    *,
    masks: Sequence[torch.Tensor | None] | None,
    expected_length: int,
) -> list[torch.Tensor | None]:
    """
    将可选 mask 组标准化成固定长度的 list。
    """
    if masks is None:
        return [None] * expected_length

    mask_list = list(masks)
    if len(mask_list) != expected_length:
        raise ValueError(
            f"Mask group must contain {expected_length} items, got {len(mask_list)}."
        )
    return mask_list


# ========== 第三部分：通道投影 ==========
class ChannelProjector(nn.Module):
    """
    将不同深度特征投到统一融合维度。
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim)

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(
                f"ChannelProjector expected last dim {self.in_dim}, got {tokens.shape[-1]}."
            )
        return self.proj(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, layout = _to_tokens(x)
        projected_tokens = self.forward_tokens(tokens)
        return _restore_layout(projected_tokens, layout)


# ========== 第四部分：deformable 跨模态读取块 ==========
class DeformableCrossModalReadBlock(nn.Module):
    """
    单次 UV-conditioned White deformable 读取块。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_levels: int = EXPECTED_FUSION_LEVELS,
        num_points: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"Feature dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attn = MSDeformAttn(
            d_model=dim,
            n_levels=num_levels,
            n_heads=num_heads,
            n_points=num_points,
        )
        self.dropout = nn.Dropout(dropout)

    def normalize_memory_tokens(self, memory_flatten: torch.Tensor) -> torch.Tensor:
        """
        对 White memory 做一次归一化，避免同一轮 read 内重复做 LayerNorm。
        """
        return self.memory_norm(memory_flatten)

    def forward_tokens(
        self,
        query_tokens: torch.Tensor,
        reference_points: torch.Tensor,
        memory_flatten: torch.Tensor,
        memory_spatial_shapes: torch.Tensor,
        memory_level_start_index: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
        memory_is_normalized: bool = False,
    ) -> torch.Tensor:
        _validate_modal_shapes(query_tokens, memory_flatten)

        query_tokens = self.query_norm(query_tokens)
        if not memory_is_normalized:
            memory_flatten = self.memory_norm(memory_flatten)
        attn_out = self.cross_attn(
            query=query_tokens,
            reference_points=reference_points,
            input_flatten=memory_flatten,
            input_spatial_shapes=memory_spatial_shapes,
            input_level_start_index=memory_level_start_index,
            input_padding_mask=memory_padding_mask,
        )
        return self.dropout(attn_out)


# ========== 第五部分：单层级跨模态融合 ==========
class SingleLevelCrossModalFusion(nn.Module):
    """
    Fuse one UV encoder feature with its matching White encoder feature.

    This is intentionally one read from one matching White memory level.
    """

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int = FUSION_DIM,
        num_heads: int = 8,
        num_points: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        residual_gate_init: float = 1e-3,
    ):
        super().__init__()

        self.input_projector = ChannelProjector(input_dim, fusion_dim)
        self.white_projector = ChannelProjector(input_dim, fusion_dim)
        self.read_block = DeformableCrossModalReadBlock(
            dim=fusion_dim,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=dropout,
        )

        mlp_hidden_dim = int(fusion_dim * mlp_ratio)
        self.final_ffn_norm = nn.LayerNorm(fusion_dim)
        self.final_ffn = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, fusion_dim),
            nn.Dropout(dropout),
        )
        self.output_projector = ChannelProjector(fusion_dim, input_dim)
        self.residual_gate = nn.Parameter(
            torch.tensor(float(residual_gate_init), dtype=torch.float32)
        )

    def forward(
        self,
        uv: torch.Tensor,
        white: torch.Tensor,
        *,
        uv_padding_mask: torch.Tensor | None = None,
        white_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uv_tokens, uv_layout = _to_tokens(uv)
        uv_mask = _flatten_padding_mask(uv_padding_mask)
        projected_uv = self.input_projector.forward_tokens(uv_tokens)
        projected_uv = _apply_padding_mask(projected_uv, uv_mask)

        projected_white = self.white_projector(white)

        reference_points = _build_reference_points_from_query_layout(
            layout_metadata=uv_layout,
            batch_size=uv_tokens.shape[0],
            num_memory_levels=1,
            device=projected_uv.device,
            dtype=projected_uv.dtype,
        )
        (
            memory_flatten,
            memory_spatial_shapes,
            memory_level_start_index,
            memory_padding_mask,
        ) = _flatten_multi_level_memory_for_ms_deform_attn(
            memory_features=[projected_white],
            memory_masks=[white_padding_mask],
        )
        normalized_memory_flatten = self.read_block.normalize_memory_tokens(memory_flatten)
        read_tokens = self.read_block.forward_tokens(
            query_tokens=projected_uv,
            reference_points=reference_points,
            memory_flatten=normalized_memory_flatten,
            memory_spatial_shapes=memory_spatial_shapes,
            memory_level_start_index=memory_level_start_index,
            memory_padding_mask=memory_padding_mask,
            memory_is_normalized=True,
        )

        fused_tokens = projected_uv + read_tokens
        fused_tokens = fused_tokens + self.final_ffn(self.final_ffn_norm(fused_tokens))
        fusion_delta = self.residual_gate.to(
            fused_tokens.dtype
        ) * self.output_projector.forward_tokens(fused_tokens)
        fused_tokens = uv_tokens + fusion_delta
        fused_tokens = _apply_padding_mask(fused_tokens, uv_mask)
        return _restore_layout(fused_tokens, uv_layout)


class MultiLevelCrossModalFusion(nn.Module):
    """
    Fuse each DINOv2 encoder depth with its matching White encoder depth.

    These 4 features are same-grid semantic depths, not a CNN/FPN pyramid. The
    module therefore avoids cross-depth mixing and applies one UV-conditioned
    White deformable read independently at each matching depth before projector.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        num_heads: int = 8,
        fusion_dim: int = FUSION_DIM,
        dropout: float = 0.0,
    ):
        super().__init__()

        input_dim_list = list(input_dims)
        if len(input_dim_list) != EXPECTED_FUSION_LEVELS:
            raise ValueError(
                "Deformable same-grid fusion expects exactly 4 encoder feature levels. "
                f"Got {len(input_dim_list)} levels."
            )

        self.level_fusions = nn.ModuleList(
            [
                SingleLevelCrossModalFusion(
                    input_dim=input_dim,
                    fusion_dim=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for input_dim in input_dim_list
            ]
        )

    def forward(
        self,
        uv_features: Sequence[torch.Tensor],
        white_features: Sequence[torch.Tensor],
        uv_padding_masks: Sequence[torch.Tensor | None] | None = None,
        white_padding_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> list[torch.Tensor]:
        uv_feature_list = _normalize_feature_group(
            name="uv_features",
            features=uv_features,
            expected_length=EXPECTED_FUSION_LEVELS,
        )
        white_feature_list = _normalize_feature_group(
            name="white_features",
            features=white_features,
            expected_length=EXPECTED_FUSION_LEVELS,
        )
        uv_mask_list = _normalize_mask_group(
            masks=uv_padding_masks,
            expected_length=EXPECTED_FUSION_LEVELS,
        )
        white_mask_list = _normalize_mask_group(
            masks=white_padding_masks,
            expected_length=EXPECTED_FUSION_LEVELS,
        )

        return [
            fusion_level(
                uv=uv_feature,
                white=white_feature,
                uv_padding_mask=uv_mask,
                white_padding_mask=white_mask,
            )
            for fusion_level, uv_feature, white_feature, uv_mask, white_mask in zip(
                self.level_fusions,
                uv_feature_list,
                white_feature_list,
                uv_mask_list,
                white_mask_list,
            )
        ]
