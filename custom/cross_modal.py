# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件定义当前项目使用的最小可行跨模态融合模块。
功能：在“UV 为主模态、White 为辅助模态”的前提下，实现单向的 UV<-White cross-attention，
      并在多层融合时使用论文式的深度 attention residual 聚合。

结构概览：
  第一部分：导入依赖
  第二部分：通用张量工具
  第三部分：单层跨模态注意力块 CrossModalFusionBlock
  第四部分：深度 attention residual 聚合
  第五部分：多层堆叠封装 CrossModalFusionStack
  第六部分：兼容旧命名的别名

设计原则：
  - uv当作query，white当作key/value，保持单向读取关系，确保检测语义始终在UV分支上。
  - 单层块只负责 z_i = Attn(q=h_i, k=white, v=white)。
  - 多层时不再使用单层标量门控，而是让 stack 在深度维度上对前序层输出做 attention 聚合。
  - FFN 只在 stack 最后执行一次，用来对最终聚合结果做一次表征重组。
  - 当前 DualModalLWDETR 的实际接入点在 encoder 后、projector 前。
"""

# ========== 第一部分：导入依赖 ==========
from __future__ import annotations

from typing import Any

import torch
from torch import nn


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

    b, n, c = x.shape
    h = metadata["height"]
    w = metadata["width"]
    expected_tokens = h * w

    if n != expected_tokens:
        raise ValueError(
            f"UV token count {n} does not match UV patch grid {h}x{w} ({expected_tokens})."
        )

    return x.transpose(1, 2).reshape(b, c, h, w).contiguous()


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


def _validate_modal_shapes(uv_tokens: torch.Tensor, white_tokens: torch.Tensor) -> None:
    """
    检查双模态输入在 batch 和 channel 维度上是否一致。
    """
    if uv_tokens.shape[0] != white_tokens.shape[0]:
        raise ValueError(
            f"UV and White batch sizes must match, got {uv_tokens.shape[0]} and {white_tokens.shape[0]}."
        )

    if uv_tokens.shape[-1] != white_tokens.shape[-1]:
        raise ValueError(
            "UV and White channel dimensions must match before fusion. "
            f"Got {uv_tokens.shape[-1]} and {white_tokens.shape[-1]}."
        )


def _apply_padding_mask(tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """
    将 padding 位置显式清零，避免这些位置在深度聚合时变成脏值来源。
    """
    if padding_mask is None:
        return tokens
    return tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def _rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    论文在深度 attention 中使用 RMSNorm(key)。
    这里实现一个无参数版本，避免引入额外模块状态，并保持和论文意图一致。
    """
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms


# ========== 第三部分：单层跨模态融合块 ==========
class CrossModalFusionBlock(nn.Module):
    """
    单层的 UV-conditioned White reading 模块。

    这里的单层块只实现论文映射中的：
        z_i = Attn(q=h_i, k=white, v=white)

    也就是说：
      - 输入 h_i 由 stack 的深度 attention 聚合给出
      - 本模块只负责“当前层从 White 读取什么信息”
      - 不在块内做固定残差相加
      - 不在块内做 FFN

    参数说明：
        dim:
            特征维度，也就是 token/channel 的最后一维大小。
        num_heads:
            Multi-head attention 的头数。
        dropout:
            attention 内部的 dropout。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Multi-head attention 要求 dim 可以被 num_heads 整除。这里提前检查，避免模型跑到中途才因为维度不匹配而报错。
        if dim % num_heads != 0:
            raise ValueError(
                f"Feature dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        # 对跨模态 attention 保留 PreNorm。
        # 这里刻意让 q 和 k 先归一化，再做打分，避免 UV / White 两路特征因为数值尺度不同而让点积失真。
        self.uv_norm = nn.LayerNorm(dim)
        self.white_norm = nn.LayerNorm(dim)

        # 这里直接使用 PyTorch 自带的 MultiheadAttention。
        # batch_first=True 使输入输出形状统一为 [B, N, C]。
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
        """
        在 token 空间执行一次：
            z_i = Attn(q=LN(h_i), k=LN(white), v=white)

        这里 white 的 key 会做 LayerNorm，但 value 保持原值。
        原因是：
          - key 参与打分，需要抑制数值尺度差异
          - value 承担的是被取回的内容，保留原幅值更有利于表达“这一层更新量到底有多强”
        """
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
        """
        执行一次单层 UV<-White 跨模态 attention。

        Args:
            uv:
                当前层聚合后的 h_i
            white:
                White 辅助模态特征
            white_padding_mask:
                White 侧 padding mask

        Returns:
            与 UV 输入布局一致的 z_i。
        """
        uv_tokens, uv_layout = _to_tokens(uv)
        white_tokens, _ = _to_tokens(white)
        _validate_modal_shapes(uv_tokens, white_tokens)

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
    具体思路来源于kimi最新的那个论文，虽然那个是在llm里面的，但是未尝不可用于cv，其实很多cv的顶刊都是在模仿llm的新思路
    
    这里借的是 Attention Residuals 的核心思想：
      - 不再把前序层结果用固定权重硬加
      - 而是让当前层按 learned pseudo-query 从历史状态中选择性聚合

    由于当前仓库的 fusion stack 层数很小，这里直接使用 full AttnRes 风格，
    不额外做 block 压缩版本。
    """

    def __init__(self, dim: int, num_queries: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # 前 num_queries-1 个 query 给中间层深度聚合，最后 1 个给 stack 输出聚合。
        self.queries = nn.Parameter(torch.empty(num_queries, dim))
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(
        self,
        history_states: list[torch.Tensor],
        query_index: int,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        对 history_states 在深度维度上做 softmax attention。

        history_states 中每个元素的形状都是 [B, N, C]。
        attention 权重沿“深度”维度归一化，而不是沿 token 维度归一化。
        """
        if not history_states:
            raise ValueError("history_states must contain at least one tensor.")

        if query_index < 0 or query_index >= self.queries.shape[0]:
            raise ValueError(
                f"query_index {query_index} is out of range for {self.queries.shape[0]} depth queries."
            )

        if len(history_states) == 1:
            return _apply_padding_mask(history_states[0], padding_mask)

        values = torch.stack(history_states, dim=0)
        # 论文式写法是：
        #   - key 用 RMSNorm 后参与打分
        #   - value 保持原始历史表示本身
        keys = _rms_norm_last_dim(values, eps=self.eps)
        query = self.queries[query_index]

        # 对每个 token 位置分别在深度维度上做注意力。
        logits = torch.einsum("c,sbnc->sbn", query, keys)
        weights = torch.softmax(logits, dim=0)
        aggregated = torch.einsum("sbn,sbnc->bnc", weights, values)
        return _apply_padding_mask(aggregated, padding_mask)


# ========== 第五部分：多层堆叠封装 ==========
class CrossModalFusionStack(nn.Module):
    """
    将多个单层融合块顺序堆叠。

    与旧版不同的是：
      - 旧版：每一层只吃上一层输出，属于简单串行残差传播
      - 新版：每一层进入融合块前，先对所有前序状态做一次深度 attention 聚合

    也就是说，这里保留“多层融合”的外形，
    但层间信息传递方式从固定残差链改成了论文式的 attention residual。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 层数至少为 1；如果是 0，就不应该实例化这个模块。
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        # 这里直接复制多个同构 attention 块。
        # 每一层都保持相同的“UV 查询 White”语义，但是否读取哪一层历史状态，交给 depth attention 决定。
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
        # 对于 L 层 attention-only stack：
        #   - h_2 ... h_L 需要 L-1 个 query
        #   - 输出端 h_out 再需要 1 个 query
        # 因此总共需要 L 个 depth queries。
        self.depth_residual = DepthAttentionResidual(dim=dim, num_queries=num_layers)

        # FFN 只在 stack 最后出现一次。
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
        """
        顺序执行多层 UV<-White 融合。

        这里始终保持：
          - h_0 = UV 主分支初始表示
          - White 是固定的辅助 memory
          - 每层先由深度 attention 计算 h_i
          - 再由跨模态 attention 计算 z_i
          - 最后再对所有历史状态做一次深度聚合，并只在这里执行一次 FFN
        """
        uv_tokens, uv_layout = _to_tokens(uv)
        white_tokens, _ = _to_tokens(white)
        _validate_modal_shapes(uv_tokens, white_tokens)

        uv_mask = _flatten_padding_mask(uv_padding_mask)
        white_mask = _flatten_padding_mask(white_padding_mask)

        # h_0 直接来自 UV，本身就是深度聚合中的 identity source。
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

        # stack 输出端再做一次深度聚合，得到最终 h_out。
        h_out = self.depth_residual(
            history_states=history_states,
            query_index=len(self.blocks) - 1,
            padding_mask=uv_mask,
        )
        fused_uv = h_out + self.final_ffn(self.final_ffn_norm(h_out))
        fused_uv = _apply_padding_mask(fused_uv, uv_mask)
        return _restore_layout(fused_uv, uv_layout)


# ========== 第六部分：兼容旧命名的别名 ==========
# 为了避免旧测试或旧代码引用 CrossModalBlock 时直接失效，
# 这里保留一个兼容别名。
CrossModalBlock = CrossModalFusionBlock
