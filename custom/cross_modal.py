# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件定义当前项目使用的最小可行跨模态融合模块。
功能：在“UV 为主模态、White 为辅助模态”的前提下，实现单向的 UV<-White cross-attention.

结构概览：
  第一部分：导入依赖
  第二部分：单层跨模态融合块 CrossModalFusionBlock
  第三部分：多层堆叠封装 CrossModalFusionStack
  第四部分：兼容旧命名的别名

设计原则：
  - uv当作query，white当作key/value，保持单向读取关系，确保检测语义始终在UV分支上。
  - 先做一个LAYERNORM ,再做cross-attention,然后FFN.
  - 当前 DualModalLWDETR 的实际接入点在 encoder 后、projector 前。
"""

# ========== 第一部分：导入依赖 ==========
from __future__ import annotations

from typing import Any

import torch
from torch import nn


# ========== 第二部分：单层跨模态融合块 ==========
class CrossModalFusionBlock(nn.Module):
    """
    单层的 UV-conditioned White reading 模块。

    前向形式严格固定为：
        fused_uv = uv + CrossAttention(LN(uv), LN(white), LN(white))
        fused_uv = fused_uv + MLP(LN(fused_uv))

    也就是说：
      - UV 主动去读 White
      - White 只是辅助 memory
      - 检测语义始终留在 UV 分支上

    参数说明：
        dim:
            特征维度，也就是 token/channel 的最后一维大小。
        num_heads:
            Multi-head attention 的头数。
        mlp_ratio:
            FFN 隐藏层相对 dim 的扩张倍数。
        dropout:
            attention 和 FFN 内部的 dropout。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0, #惯例如此
        dropout: float = 0.0, #无门控时候用的，现在不使用了
    ):
        super().__init__()

        # Multi-head attention 要求 dim 可以被 num_heads 整除。这里提前检查，避免模型跑到中途才因为维度不匹配而报错。
        if dim % num_heads != 0:
            raise ValueError(
                f"Feature dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        # FFN的隐空间维度
        mlp_hidden_dim = int(dim * mlp_ratio)

        # layernorm
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

        # FFN 前再做一层 LayerNorm。
        self.ffn_norm = nn.LayerNorm(dim)

        # 定义FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # 门控系数，初始置零
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn  = nn.Parameter(torch.zeros(1))

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        将输入统一转成 token 序列表示。

        支持两种输入：
          - [B, N, C]：已经是 token 序列，直接返回
          - [B, C, H, W]：flatten 成 [B, HW, C]

        返回：
          - token 化后的张量
          - 还原布局所需的元信息
        """
        # 如果本来就是 token 序列，不做任何形状变化。
        if x.dim() == 3:
            return x, {"layout": "tokens"}

        # 如果是 patch grid / feature grid，则转到 token space。
        if x.dim() == 4:
            b, c, h, w = x.shape

            # [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
            # 这样就能直接送进标准的 attention。
            tokens = x.flatten(2).transpose(1, 2).contiguous()
            return tokens, {"layout": "grid", "channels": c, "height": h, "width": w}

        # 其它形状都不在当前最小实现的支持范围内。
        raise ValueError(
            f"Expected a token sequence [B, N, C] or feature grid [B, C, H, W], got {tuple(x.shape)}."
        )

    def _restore_layout(self, x: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        """
        虽然说，vit时候内部是bnc，但是原仓库给vit做了封装，输入和输出都变成了bhwc
        """
        # token 输入保持 token 输出，不做还原。
        if metadata["layout"] == "tokens":
            return x

        # 对于 grid 布局，需要把 token 数量还原回原来的 H*W。
        b, n, c = x.shape
        h = metadata["height"]
        w = metadata["width"]
        expected_tokens = h * w

        # 如果 token 数不对，说明前面某一步发生了不合法的 shape 改动。
        if n != expected_tokens:
            raise ValueError(
                f"UV token count {n} does not match UV patch grid {h}x{w} ({expected_tokens})."
            )

        # [B, HW, C] -> [B, C, HW] -> [B, C, H, W]
        return x.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def _flatten_padding_mask(self, mask: torch.Tensor | None) -> torch.Tensor | None:
        """
        将 padding mask 统一整理成 attention 可接受的 [B, N] 形式。

        支持：
          - None
          - [B, N]
          - [B, H, W]
        """
        if mask is None:
            return None

        # 已经是 token 级 mask，直接返回。
        if mask.dim() == 2:
            return mask

        # 如果是空间 mask，则展平成 token 级 mask。
        if mask.dim() == 3:
            return mask.flatten(1)

        raise ValueError(
            f"Expected padding mask [B, N] or [B, H, W], got {tuple(mask.shape)}."
        )

    def forward(
        self,
        uv: torch.Tensor,
        white: torch.Tensor,
        uv_padding_mask: torch.Tensor | None = None,
        white_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        执行一次单层 UV<-White 跨模态融合。

        Args:
            uv:
                UV 主模态特征
            white:
                White 辅助模态特征
            uv_padding_mask:
                UV 侧 padding mask
            white_padding_mask:
                White 侧 padding mask

        Returns:
            与 UV 输入布局一致的融合结果。
        """
        # 先把两路输入都规范到 token space，便于后续统一处理。
        uv_tokens, uv_layout = self._to_tokens(uv)
        white_tokens, _ = self._to_tokens(white)

        #检查👇
        # batch
        if uv_tokens.shape[0] != white_tokens.shape[0]:
            raise ValueError(
                f"UV and White batch sizes must match, got {uv_tokens.shape[0]} and {white_tokens.shape[0]}."
            )

        # channel
        if uv_tokens.shape[-1] != white_tokens.shape[-1]:
            raise ValueError(
                "UV and White channel dimensions must match before fusion. "
                f"Got {uv_tokens.shape[-1]} and {white_tokens.shape[-1]}."
            )

        # 将 mask 展平为 token 级形式。
        uv_mask = self._flatten_padding_mask(uv_padding_mask)
        white_mask = self._flatten_padding_mask(white_padding_mask)

        #   Query = UV
        #   Key   = White
        #   Value = White
        q = self.uv_norm(uv_tokens)
        kv = self.white_norm(white_tokens)
        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=white_mask,
            need_weights=False,
        )

        # 第一条残差：UV token 读取 White 后，经 alpha_attn 缩放后回写到 UV 自身。
        # alpha_attn 初始为 0，训练初期融合贡献为零，不冲击预训练权重；
        fused_uv = uv_tokens + self.alpha_attn * attn_out

        # 第二条残差：标准 Transformer FFN，同样经 alpha_ffn 缩放控制初期贡献量。
        fused_uv = fused_uv + self.alpha_ffn * self.ffn(self.ffn_norm(fused_uv))

        # 如果 UV 侧存在 padding 区域，则把这些位置清零，避免脏值继续流向后续模块。
        if uv_mask is not None:
            fused_uv = fused_uv.masked_fill(uv_mask.unsqueeze(-1), 0.0)

        # 输出布局与 UV 输入保持一致，方便无缝接回原有检测主干。
        return self._restore_layout(fused_uv, uv_layout)


# ========== 第三部分：多层堆叠封装 ==========
class CrossModalFusionStack(nn.Module):
    """
    将多个单层融合块顺序堆叠。
    默认是 1 层，但是类似decoder，多加几层或许会有更高的性能

    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 层数至少为 1；如果是 0，就不应该实例化这个模块。
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        # 这里直接复制多个同构 block。
        # 每一层都保持相同的“UV 查询 White”语义。
        self.blocks = nn.ModuleList(
            [
                CrossModalFusionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
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
          - UV 是不断被更新的主分支
          - White 是固定的辅助 memory
        """
        fused_uv = uv
        for block in self.blocks:
            fused_uv = block(
                fused_uv,
                white,
                uv_padding_mask=uv_padding_mask,
                white_padding_mask=white_padding_mask,
            )
        return fused_uv


# ========== 第四部分：兼容旧命名的别名 ==========
# 为了避免旧测试或旧代码引用 CrossModalBlock 时直接失效，
# 这里保留一个兼容别名。
CrossModalBlock = CrossModalFusionBlock
