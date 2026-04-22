"""
文件说明：本文件实现当前仓库双模态主线使用的跨模态融合模块。
功能说明：在“UV 为主模态、White 为辅助模态”的前提下，提供当前主线路径所需的
deformable same-grid 多层级跨模态融合实现。

结构概览：
  第一部分：导入依赖与常量
  第二部分：通用张量工具
  第三部分：通道投影
  第四部分：deformable 跨模态读取块
  第五部分：深度 residual 聚合
  第六部分：当前主线路径的多层级跨模态融合
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
    显式清零 padding 位置，避免这些位置参与历史状态聚合。
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


def _rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    在 depth residual 聚合中使用无参数 RMSNorm。
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms


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


def _validate_same_grid_feature_group(
    *,
    name: str,
    features: Sequence[torch.Tensor],
) -> None:
    """
    当前主线只支持“同 patch 网格、不同 encoder 深度”的特征组。
    """
    signatures = []
    for index, feature in enumerate(features):
        if feature.dim() == 4:
            signatures.append(("grid", feature.shape[-2], feature.shape[-1]))
        elif feature.dim() == 3:
            signatures.append(("tokens", feature.shape[1]))
        else:
            raise ValueError(
                f"{name}[{index}] must be [B, N, C] or [B, C, H, W], got {tuple(feature.shape)}."
            )

    reference_signature = signatures[0]
    for index, signature in enumerate(signatures[1:], start=1):
        if signature != reference_signature:
            raise ValueError(
                f"{name} must share the same patch grid. "
                f"Expected {reference_signature}, got {signature} at index {index}."
            )


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


# ========== 第五部分：深度 residual 聚合 ==========
class DepthAttentionResidual(nn.Module):
    """
    使用可学习 pseudo-query 在 depth 方向聚合历史状态。
    """

    def __init__(self, dim: int, num_queries: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        history_states: list[torch.Tensor],
        query_index: int,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not history_states:
            raise ValueError("history_states must contain at least one tensor.")

        if query_index < 0 or query_index >= self.queries.shape[0]:
            raise ValueError(
                f"query_index {query_index} is out of range for {self.queries.shape[0]} depth queries."
            )

        if len(history_states) == 1:
            return _apply_padding_mask(history_states[0], padding_mask)

        values = torch.stack(history_states, dim=0)
        keys = _rms_norm_last_dim(values, eps=self.eps)
        query = self.queries[query_index]
        logits = torch.einsum("c,sbnc->sbn", query, keys)
        weights = torch.softmax(logits, dim=0)
        aggregated = torch.einsum("sbn,sbnc->bnc", weights, values)
        return _apply_padding_mask(aggregated, padding_mask)


# ========== 第六部分：当前主线路径的多层级跨模态融合 ==========
class DeformableSequentialCrossModalFusionLevel(nn.Module):
    """
    当前主线使用的单个 UV 分支融合层。

    设计约束：
      1. 第一轮 read 严格只读对应 White level
      2. 后续 read 才读全部 4 路 White memory
      3. 不再保留旧 checkpoint 兼容写法，结构以当前主线清晰为先
    """

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int = FUSION_DIM,
        num_heads: int = 8,
        num_reads: int = 4,
        num_points: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {num_reads}.")

        self.num_reads = num_reads
        self.num_full_memory_levels = EXPECTED_FUSION_LEVELS
        self.input_projector = ChannelProjector(input_dim, fusion_dim)
        self.output_projector = ChannelProjector(fusion_dim, input_dim)

        # 第一轮 read 只看对应 White level，因此 n_levels 明确设为 1。
        self.same_level_read_block = DeformableCrossModalReadBlock(
            dim=fusion_dim,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=dropout,
        )

        # 后续 read 再切换到完整的 4-level White memory。
        self.cross_level_read_blocks = nn.ModuleList(
            [
                DeformableCrossModalReadBlock(
                    dim=fusion_dim,
                    num_heads=num_heads,
                    num_levels=self.num_full_memory_levels,
                    num_points=num_points,
                    dropout=dropout,
                )
                for _ in range(max(num_reads - 1, 0))
            ]
        )
        self.depth_residual = DepthAttentionResidual(dim=fusion_dim, num_queries=num_reads)

        mlp_hidden_dim = int(fusion_dim * mlp_ratio)
        self.final_ffn_norm = nn.LayerNorm(fusion_dim)
        self.final_ffn = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, fusion_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        uv: torch.Tensor,
        *,
        uv_padding_mask: torch.Tensor | None = None,
        same_level_reference_points: torch.Tensor,
        same_level_memory_inputs: MemoryInputs,
        full_reference_points: torch.Tensor,
        full_memory_inputs: MemoryInputs,
    ) -> torch.Tensor:
        uv_tokens, uv_layout = _to_tokens(uv)
        uv_mask = _flatten_padding_mask(uv_padding_mask)
        h0 = self.input_projector.forward_tokens(uv_tokens)
        h0 = _apply_padding_mask(h0, uv_mask)

        history_states = [h0]
        for read_index in range(self.num_reads):
            if read_index == 0:
                current_state = h0
                current_block = self.same_level_read_block
                current_reference_points = same_level_reference_points
                current_memory_inputs = same_level_memory_inputs
            else:
                current_state = self.depth_residual(
                    history_states=history_states,
                    query_index=read_index - 1,
                    padding_mask=uv_mask,
                )
                current_block = self.cross_level_read_blocks[read_index - 1]
                current_reference_points = full_reference_points
                current_memory_inputs = full_memory_inputs

            (
                current_memory_flatten,
                current_memory_spatial_shapes,
                current_memory_level_start_index,
                current_memory_padding_mask,
            ) = current_memory_inputs
            normalized_memory_flatten = current_block.normalize_memory_tokens(current_memory_flatten)

            zi = current_block.forward_tokens(
                query_tokens=current_state,
                reference_points=current_reference_points,
                memory_flatten=normalized_memory_flatten,
                memory_spatial_shapes=current_memory_spatial_shapes,
                memory_level_start_index=current_memory_level_start_index,
                memory_padding_mask=current_memory_padding_mask,
                memory_is_normalized=True,
            )
            history_states.append(_apply_padding_mask(zi, uv_mask))

        h_out = self.depth_residual(
            history_states=history_states,
            query_index=self.num_reads - 1,
            padding_mask=uv_mask,
        )
        fused_tokens = h_out + self.final_ffn(self.final_ffn_norm(h_out))
        fused_tokens = self.output_projector.forward_tokens(fused_tokens)
        fused_tokens = _apply_padding_mask(fused_tokens, uv_mask)
        return _restore_layout(fused_tokens, uv_layout)


class MultiLevelCrossModalFusion(nn.Module):
    """
    管理 4 路 UV 与 4 路 White 的 same-grid deformable 融合。
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        num_heads: int = 8,
        num_reads: int = EXPECTED_FUSION_LEVELS,
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

        if num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {num_reads}.")

        self.white_projectors = nn.ModuleList(
            [ChannelProjector(in_dim=dim, out_dim=fusion_dim) for dim in input_dim_list]
        )
        self.level_fusions = nn.ModuleList(
            [
                DeformableSequentialCrossModalFusionLevel(
                    input_dim=input_dim_list[level_index],
                    fusion_dim=fusion_dim,
                    num_heads=num_heads,
                    num_reads=num_reads,
                    dropout=dropout,
                )
                for level_index in range(EXPECTED_FUSION_LEVELS)
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

        _validate_same_grid_feature_group(name="uv_features", features=uv_feature_list)
        _validate_same_grid_feature_group(name="white_features", features=white_feature_list)

        projected_white_features = [
            projector(feature)
            for projector, feature in zip(self.white_projectors, white_feature_list)
        ]
        _, reference_layout = _to_tokens(uv_feature_list[0])

        shared_same_level_reference_points = _build_reference_points_from_query_layout(
            layout_metadata=reference_layout,
            batch_size=uv_feature_list[0].shape[0],
            num_memory_levels=1,
            device=projected_white_features[0].device,
            dtype=projected_white_features[0].dtype,
        )
        shared_full_reference_points = _build_reference_points_from_query_layout(
            layout_metadata=reference_layout,
            batch_size=uv_feature_list[0].shape[0],
            num_memory_levels=EXPECTED_FUSION_LEVELS,
            device=projected_white_features[0].device,
            dtype=projected_white_features[0].dtype,
        )

        shared_full_memory_inputs = _flatten_multi_level_memory_for_ms_deform_attn(
            memory_features=projected_white_features,
            memory_masks=white_mask_list,
        )
        shared_same_level_memory_inputs = [
            _flatten_multi_level_memory_for_ms_deform_attn(
                memory_features=[projected_white_features[level_index]],
                memory_masks=[white_mask_list[level_index]],
            )
            for level_index in range(EXPECTED_FUSION_LEVELS)
        ]

        fused_features: list[torch.Tensor] = []
        for level_index, level_fusion in enumerate(self.level_fusions):
            fused_feature = level_fusion(
                uv=uv_feature_list[level_index],
                uv_padding_mask=uv_mask_list[level_index],
                same_level_reference_points=shared_same_level_reference_points,
                same_level_memory_inputs=shared_same_level_memory_inputs[level_index],
                full_reference_points=shared_full_reference_points,
                full_memory_inputs=shared_full_memory_inputs,
            )
            fused_features.append(fused_feature)

        return fused_features
