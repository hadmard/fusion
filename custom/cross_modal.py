"""
文件说明：本文件实现当前仓库的跨模态融合模块。
功能说明：在“UV 为主模态、White 为辅助模态”的前提下，保留一对一兼容融合接口，
并新增“同网格、多深度、顺序跨模态融合”主路径。

结构概览：
  第一部分：导入依赖与常量
  第二部分：通用张量工具
  第三部分：单次跨模态读取块与通道投影
  第四部分：深度 attention residual 聚合
  第五部分：旧版一对一融合兼容层
  第六部分：顺序多层级跨模态融合
  第七部分：兼容别名
"""

# ========== 第一部分：导入依赖与常量 ==========
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

FUSION_DIM = 256
EXPECTED_FUSION_LEVELS = 4
SAME_LEVEL_FIRST_READ_ORDERS = (
    (0, 1, 2, 3),
    (1, 0, 2, 3),
    (2, 1, 3, 0),
    (3, 2, 1, 0),
)


# ========== 第二部分：通用张量工具 ==========
def _to_tokens(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    将输入统一转成 token 序列表示。

    支持两种输入：
      - [B, N, C]：已经是 token 序列，直接返回
      - [B, C, H, W]：flatten 成 [B, HW, C]
    """
    if x.dim() == 3:
        return x, {"layout": "tokens"}

    if x.dim() == 4:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return tokens, {"layout": "grid", "channels": c, "height": h, "width": w}

    raise ValueError(
        f"Expected a token sequence [B, N, C] or feature grid [B, C, H, W], got {tuple(x.shape)}."
    )


def _restore_layout(x: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    """
    将 token 序列还原回原始布局。
    """
    if metadata["layout"] == "tokens":
        return x

    _, n, c = x.shape
    h = metadata["height"]
    w = metadata["width"]
    expected_tokens = h * w

    if n != expected_tokens:
        raise ValueError(
            f"UV token count {n} does not match UV patch grid {h}x{w} ({expected_tokens})."
        )

    return x.transpose(1, 2).reshape(x.shape[0], c, h, w).contiguous()


def _flatten_padding_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """
    将 padding mask 统一整理成 attention 可接受的 [B, N] 形式。
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
    将 padding 位置显式清零，避免这些位置在深度聚合时变成脏值来源。
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
    论文在深度 attention 中使用 RMSNorm(key)。
    这里实现一个无参数版本，避免引入额外模块状态，并保持和论文意图一致。
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms


def _normalize_feature_group(
    *,
    name: str,
    features: Sequence[torch.Tensor],
    expected_length: int,
) -> list[torch.Tensor]:
    """
    将输入的特征列表转成普通 list，并明确检查长度。
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
    第一版明确只支持“同网格、多深度”特征。

    如果输入不是同一个 patch 网格，当前 same-level-first 顺序融合就不再是
    “同网格跨深度”语义，因此这里直接报错，不静默兼容成别的结构。
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
                f"{name} must share the same patch grid in v1. "
                f"Expected all signatures to match {reference_signature}, got {signature} at index {index}."
            )


# ========== 第三部分：单次跨模态读取块与通道投影 ==========
class ChannelProjector(nn.Module):
    """
    负责把不同深度特征投到统一融合维度。

    这里使用逐 token 的 Linear。对 [B, C, H, W] 网格输入来说，它等价于
    “对每个空间位置做一次 1x1 conv 风格的通道线性变换”，只是实现上复用了
    token 视角，方便和后续 attention 逻辑直接拼接。
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


class CrossModalFusionBlock(nn.Module):
    """
    单次的 UV-conditioned White reading 模块。

    职责固定为：
        z_i = CrossAttn(q=current_uv_state, k=current_white_level, v=current_white_level)

    也就是说：
      - 它只负责“当前这一次从哪一路 White memory 读取信息”
      - 不在块内做固定 residual 相加
      - 不在块内做 FFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"Feature dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        self.uv_norm = nn.LayerNorm(dim)
        self.white_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward_tokens(
        self,
        uv_tokens: torch.Tensor,
        white_tokens: torch.Tensor,
        white_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_modal_shapes(uv_tokens, white_tokens)

        q = self.uv_norm(uv_tokens)
        k = self.white_norm(white_tokens)
        attn_out, _ = self.cross_attn(
            query=q,
            key=k,
            value=white_tokens,
            key_padding_mask=white_padding_mask,
            need_weights=False,
        )
        return attn_out

    def forward(
        self,
        uv: torch.Tensor,
        white: torch.Tensor,
        white_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uv_tokens, uv_layout = _to_tokens(uv)
        white_tokens, _ = _to_tokens(white)
        white_mask = _flatten_padding_mask(white_padding_mask)
        fused_uv = self.forward_tokens(
            uv_tokens=uv_tokens,
            white_tokens=white_tokens,
            white_padding_mask=white_mask,
        )
        return _restore_layout(fused_uv, uv_layout)


# ========== 第四部分：深度 attention residual 聚合 ==========
class DepthAttentionResidual(nn.Module):
    """
    使用 learnable pseudo-query 在深度方向聚合历史状态。

    本轮明确沿用这一设计，而不是从输入动态生成聚合权重，原因是：
      - 我们要把“去哪一路 White 读”与“读回来后信哪些历史状态”拆开
      - 当前阶段先验证固定结构本身，不额外引入输入相关的门控复杂度
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


# ========== 第五部分：旧版一对一融合兼容层 ==========
class CrossModalFusionStack(nn.Module):
    """
    兼容旧的一对一 UV<-White 多层融合接口。

    这个类保留旧名字，是为了避免仓库里可能残留的引用直接失效。
    当前主路径已经切换到 `MultiLevelCrossModalFusion`，这里只承担兼容职责。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.blocks = nn.ModuleList(
            [
                CrossModalFusionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.depth_residual = DepthAttentionResidual(dim=dim, num_queries=num_layers)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.final_ffn_norm = nn.LayerNorm(dim)
        self.final_ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        uv: torch.Tensor,
        white: torch.Tensor,
        uv_padding_mask: torch.Tensor | None = None,
        white_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uv_tokens, uv_layout = _to_tokens(uv)
        white_tokens, _ = _to_tokens(white)
        _validate_modal_shapes(uv_tokens, white_tokens)

        uv_mask = _flatten_padding_mask(uv_padding_mask)
        white_mask = _flatten_padding_mask(white_padding_mask)
        h0 = _apply_padding_mask(uv_tokens, uv_mask)
        history_states = [h0]

        for layer_index, block in enumerate(self.blocks):
            if layer_index == 0:
                hi = h0
            else:
                hi = self.depth_residual(
                    history_states=history_states,
                    query_index=layer_index - 1,
                    padding_mask=uv_mask,
                )

            zi = block.forward_tokens(
                uv_tokens=hi,
                white_tokens=white_tokens,
                white_padding_mask=white_mask,
            )
            history_states.append(_apply_padding_mask(zi, uv_mask))

        h_out = self.depth_residual(
            history_states=history_states,
            query_index=len(self.blocks) - 1,
            padding_mask=uv_mask,
        )
        fused_uv = h_out + self.final_ffn(self.final_ffn_norm(h_out))
        fused_uv = _apply_padding_mask(fused_uv, uv_mask)
        return _restore_layout(fused_uv, uv_layout)


# ========== 第六部分：顺序多层级跨模态融合 ==========
class SequentialCrossModalFusionLevel(nn.Module):
    """
    针对单个 UV 分支做“顺序读取 White 多层级 + AttnResidual 聚合”。

    核心流程固定为：
      1. 先把 U_l 投到统一融合维度
      2. 按 same-level-first 顺序依次读取 4 个 White 分支
      3. 第 2/3/4 次读取前，用 AttnResidual 聚合历史状态
      4. 最后再做一次 AttnResidual + 最终 FFN
      5. 投回原 UV 通道，保证 projector 接口不变
    """

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int = FUSION_DIM,
        num_heads: int = 8,
        num_reads: int = EXPECTED_FUSION_LEVELS,
        read_order: Sequence[int] | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {num_reads}.")

        self.num_reads = num_reads
        self.read_order = tuple(read_order) if read_order is not None else tuple(range(num_reads))

        if len(self.read_order) != self.num_reads:
            raise ValueError(
                f"read_order length ({len(self.read_order)}) must match num_reads ({self.num_reads})."
            )

        self.input_projector = ChannelProjector(input_dim, fusion_dim)
        self.output_projector = ChannelProjector(fusion_dim, input_dim)
        self.read_blocks = nn.ModuleList(
            [
                CrossModalFusionBlock(
                    dim=fusion_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_reads)
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
        projected_white_features: Sequence[torch.Tensor],
        uv_padding_mask: torch.Tensor | None = None,
        white_padding_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        white_feature_list = _normalize_feature_group(
            name="projected_white_features",
            features=projected_white_features,
            expected_length=self.num_reads,
        )
        white_mask_list = _normalize_mask_group(
            masks=white_padding_masks,
            expected_length=self.num_reads,
        )

        uv_tokens, uv_layout = _to_tokens(uv)
        uv_mask = _flatten_padding_mask(uv_padding_mask)
        h0 = self.input_projector.forward_tokens(uv_tokens)
        h0 = _apply_padding_mask(h0, uv_mask)

        white_tokens_list: list[torch.Tensor] = []
        flattened_white_masks: list[torch.Tensor | None] = []
        for white_feature, white_mask in zip(white_feature_list, white_mask_list):
            white_tokens, _ = _to_tokens(white_feature)
            _validate_modal_shapes(h0, white_tokens)
            white_tokens_list.append(white_tokens)
            flattened_white_masks.append(_flatten_padding_mask(white_mask))

        history_states = [h0]

        for read_index, white_level_index in enumerate(self.read_order):
            if read_index == 0:
                current_state = h0
            else:
                current_state = self.depth_residual(
                    history_states=history_states,
                    query_index=read_index - 1,
                    padding_mask=uv_mask,
                )

            zi = self.read_blocks[read_index].forward_tokens(
                uv_tokens=current_state,
                white_tokens=white_tokens_list[white_level_index],
                white_padding_mask=flattened_white_masks[white_level_index],
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
    管理 4 路 UV 与 4 路 White 的同网格顺序跨模态融合。

    第一版明确只支持：
      - 4 路 encoder depth features
      - same-level-first 固定读取顺序
      - 每个 UV level 独立去读 White 的 4 个 level
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
                "Sequential same-grid fusion v1 expects exactly 4 encoder feature levels. "
                f"Got {len(input_dim_list)} levels."
            )

        if num_reads != EXPECTED_FUSION_LEVELS:
            raise ValueError(
                "Sequential same-grid fusion v1 uses exactly 4 reads so it can query "
                "W2/W5/W8/W11 in a fixed order. "
                f"Got num_reads={num_reads}."
            )

        self.white_projectors = nn.ModuleList(
            [ChannelProjector(in_dim=dim, out_dim=fusion_dim) for dim in input_dim_list]
        )
        self.level_fusions = nn.ModuleList(
            [
                SequentialCrossModalFusionLevel(
                    input_dim=input_dim_list[level_index],
                    fusion_dim=fusion_dim,
                    num_heads=num_heads,
                    num_reads=num_reads,
                    read_order=SAME_LEVEL_FIRST_READ_ORDERS[level_index],
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

        fused_features: list[torch.Tensor] = []
        for level_index, level_fusion in enumerate(self.level_fusions):
            fused_feature = level_fusion(
                uv=uv_feature_list[level_index],
                projected_white_features=projected_white_features,
                uv_padding_mask=uv_mask_list[level_index],
                white_padding_masks=white_mask_list,
            )
            fused_features.append(fused_feature)

        return fused_features


# ========== 第七部分：兼容别名 ==========
CrossModalBlock = CrossModalFusionBlock
