"""
文件说明：该文件用于兼容历史门控融合权重的模型结构。
功能说明：提供旧版门控融合 block、stack 和双模态 RF-DETR 构建函数，
使历史 `menkong.pth` 这类权重能够在当前仓库里按原结构加载，再走当前统一评估脚本。
使用方式：本文件不是独立入口，不需要手工运行；根目录 `eval/run_eval_menkong.py` 在评估 `menkong.pth` 时会自动引用它。

结构概览：
  第一部分：导入依赖与常量
  第二部分：旧版门控融合模块
  第三部分：旧版双模态检测模型
  第四部分：模型构建入口
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.models.backbone import build_backbone
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.segmentation_head import SegmentationHead
from rfdetr.models.transformer import build_transformer
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list


# ========== 第一部分：导入依赖与常量 ==========
SUPPORTED_FUSION_TYPES = {"none", "uv_queries_white"}


# ========== 第二部分：旧版门控融合模块 ==========
class LegacyGateCrossModalFusionBlock(nn.Module):
    """
    历史门控版的单层 UV<-White 融合块。

    这套实现和当前主分支最大的区别在于：
    1. 先做一次标准 cross-attention 残差
    2. 再做一次 FFN 残差
    3. 两条残差都由可学习标量 `alpha_attn / alpha_ffn` 控制

    这两个门控参数正是历史 `menkong.pth` 中最关键的结构参数，
    如果当前模型结构里没有它们，权重即使能 `strict=False` 加载，
    也只是在“形状上兼容”，语义上已经不再是原模型。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"Feature dim ({dim}) must be divisible by num_heads ({num_heads})."
            )

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.uv_norm = nn.LayerNorm(dim)
        self.white_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # 历史权重中的关键门控参数。
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn = nn.Parameter(torch.zeros(1))

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, int | str]]:
        if x.dim() == 3:
            return x, {"layout": "tokens"}

        if x.dim() == 4:
            batch_size, channels, height, width = x.shape
            tokens = x.flatten(2).transpose(1, 2).contiguous()
            return tokens, {
                "layout": "grid",
                "channels": channels,
                "height": height,
                "width": width,
            }

        raise ValueError(
            f"Expected [B, N, C] or [B, C, H, W], got {tuple(x.shape)}."
        )

    def _restore_layout(self, x: torch.Tensor, metadata: dict[str, int | str]) -> torch.Tensor:
        if metadata["layout"] == "tokens":
            return x

        batch_size, token_count, channels = x.shape
        height = int(metadata["height"])
        width = int(metadata["width"])
        if token_count != height * width:
            raise ValueError(
                f"UV token count {token_count} does not match UV patch grid {height}x{width}."
            )
        return x.transpose(1, 2).reshape(batch_size, channels, height, width).contiguous()

    def _flatten_padding_mask(self, mask: torch.Tensor | None) -> torch.Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask
        if mask.dim() == 3:
            return mask.flatten(1)
        raise ValueError(f"Expected padding mask [B, N] or [B, H, W], got {tuple(mask.shape)}.")

    def forward(
        self,
        uv: torch.Tensor,
        white: torch.Tensor,
        uv_padding_mask: torch.Tensor | None = None,
        white_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uv_tokens, uv_layout = self._to_tokens(uv)
        white_tokens, _ = self._to_tokens(white)

        if uv_tokens.shape[0] != white_tokens.shape[0]:
            raise ValueError(
                f"UV and White batch sizes must match, got {uv_tokens.shape[0]} and {white_tokens.shape[0]}."
            )
        if uv_tokens.shape[-1] != white_tokens.shape[-1]:
            raise ValueError(
                "UV and White channel dimensions must match before fusion. "
                f"Got {uv_tokens.shape[-1]} and {white_tokens.shape[-1]}."
            )

        uv_mask = self._flatten_padding_mask(uv_padding_mask)
        white_mask = self._flatten_padding_mask(white_padding_mask)

        query = self.uv_norm(uv_tokens)
        key_value = self.white_norm(white_tokens)
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=white_mask,
            need_weights=False,
        )

        fused_uv = uv_tokens + self.alpha_attn * attn_out
        fused_uv = fused_uv + self.alpha_ffn * self.ffn(self.ffn_norm(fused_uv))

        if uv_mask is not None:
            fused_uv = fused_uv.masked_fill(uv_mask.unsqueeze(-1), 0.0)

        return self._restore_layout(fused_uv, uv_layout)


class LegacyGateCrossModalFusionStack(nn.Module):
    """历史门控版多层堆叠，层间采用简单串行传播。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        self.blocks = nn.ModuleList(
            [
                LegacyGateCrossModalFusionBlock(
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
        fused_uv = uv
        for block in self.blocks:
            fused_uv = block(
                fused_uv,
                white,
                uv_padding_mask=uv_padding_mask,
                white_padding_mask=white_padding_mask,
            )
        return fused_uv


# ========== 第三部分：旧版双模态检测模型 ==========
class LegacyGateDualModalLWDETR(LWDETR):
    """兼容历史门控融合权重的双模态 RF-DETR。"""

    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        segmentation_head: nn.Module | None,
        num_classes: int,
        num_queries: int,
        aux_loss: bool = False,
        group_detr: int = 1,
        two_stage: bool = False,
        lite_refpoint_refine: bool = False,
        bbox_reparam: bool = False,
        use_white: bool = True,
        fusion_type: str = "uv_queries_white",
        fusion_num_heads: int = 8,
        fusion_num_layers: int = 1,
    ) -> None:
        super().__init__(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=segmentation_head,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss,
            group_detr=group_detr,
            two_stage=two_stage,
            lite_refpoint_refine=lite_refpoint_refine,
            bbox_reparam=bbox_reparam,
        )

        if fusion_type not in SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"Unsupported fusion_type '{fusion_type}'. Expected one of {sorted(SUPPORTED_FUSION_TYPES)}."
            )
        if not use_white and fusion_type != "none":
            raise ValueError("use_white=False is only valid when fusion_type='none'.")

        self.use_white = bool(use_white)
        self.fusion_type = fusion_type
        self.fusion_enabled = self.use_white and self.fusion_type == "uv_queries_white"

        encoder_feature_dims = list(backbone[0].encoder._out_feature_channels)
        self.fusion_layers = (
            nn.ModuleList(
                [
                    LegacyGateCrossModalFusionStack(
                        dim=feature_dim,
                        num_heads=fusion_num_heads,
                        num_layers=fusion_num_layers,
                    )
                    for feature_dim in encoder_feature_dims
                ]
            )
            if self.fusion_enabled
            else None
        )

    def _prepare_inputs(
        self, samples: NestedTensor | List[torch.Tensor] | torch.Tensor
    ) -> NestedTensor:
        if isinstance(samples, (list, torch.Tensor)):
            return nested_tensor_from_tensor_list(samples)
        return samples

    def _extract_encoder_features(self, samples: NestedTensor) -> List[NestedTensor]:
        encoder_features = self.backbone[0].encoder(samples.tensors)
        encoded_nested: List[NestedTensor] = []
        for feat in encoder_features:
            mask = samples.mask
            assert mask is not None
            resized_mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            encoded_nested.append(NestedTensor(feat, resized_mask))
        return encoded_nested

    def _fuse_uv_with_white(
        self,
        uv_features: List[NestedTensor],
        white_features: List[NestedTensor],
    ) -> List[NestedTensor]:
        if self.fusion_layers is None:
            return uv_features

        if len(uv_features) != len(white_features):
            raise ValueError(
                f"UV/White feature levels mismatch: {len(uv_features)} vs {len(white_features)}."
            )
        if len(uv_features) != len(self.fusion_layers):
            raise ValueError(
                f"Fusion layers ({len(self.fusion_layers)}) do not match feature levels ({len(uv_features)})."
            )

        fused_features: List[NestedTensor] = []
        for uv_feat, white_feat, fusion_layer in zip(uv_features, white_features, self.fusion_layers):
            uv_src, uv_mask = uv_feat.decompose()
            white_src, white_mask = white_feat.decompose()
            fused_uv = fusion_layer(
                uv=uv_src,
                white=white_src,
                uv_padding_mask=uv_mask,
                white_padding_mask=white_mask,
            )
            fused_features.append(NestedTensor(fused_uv, uv_mask))
        return fused_features

    def _project_encoder_features(
        self,
        samples: NestedTensor,
        encoder_features: List[NestedTensor],
    ) -> tuple[List[NestedTensor], List[torch.Tensor]]:
        projected_tensors = self.backbone[0].projector([feature.tensors for feature in encoder_features])
        projected_features: List[NestedTensor] = []
        pos_embeddings: List[torch.Tensor] = []
        for feat in projected_tensors:
            mask = samples.mask
            assert mask is not None
            resized_mask = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            nested_feat = NestedTensor(feat, resized_mask)
            projected_features.append(nested_feat)
            pos_embeddings.append(
                self.backbone[1](nested_feat, align_dim_orders=False).to(feat.dtype)
            )
        return projected_features, pos_embeddings

    def forward(
        self,
        samples_uv: NestedTensor | List[torch.Tensor] | torch.Tensor,
        samples_white: NestedTensor | List[torch.Tensor] | torch.Tensor | None = None,
        targets=None,
    ):
        samples_uv = self._prepare_inputs(samples_uv)
        if samples_white is not None:
            samples_white = self._prepare_inputs(samples_white)

        if self.fusion_enabled:
            if samples_white is None:
                raise ValueError(
                    "fusion_type='uv_queries_white' requires samples_white to be provided."
                )
            uv_encoder_features = self._extract_encoder_features(samples_uv)
            white_encoder_features = self._extract_encoder_features(samples_white)
            fused_encoder_features = self._fuse_uv_with_white(uv_encoder_features, white_encoder_features)
            fused_features, pos_uv = self._project_encoder_features(samples_uv, fused_encoder_features)
        else:
            fused_features, pos_uv = self.backbone(samples_uv)

        srcs = []
        masks = []
        for feat in fused_features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        if self.segmentation_head is not None:
            seg_head_fwd = (
                self.segmentation_head.sparse_forward
                if self.training
                else self.segmentation_head.forward
            )

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, pos_uv, refpoint_embed_weight, query_feat_weight
        )

        outputs_masks = None
        if hs is not None:
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)
            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(
                    fused_features[0].tensors,
                    hs,
                    samples_uv.tensors.shape[-2:],
                )

            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class,
                    outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )
        else:
            out = {}

        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for group_index in range(group_detr):
                cls_enc_group = self.transformer.enc_out_class_embed[group_index](hs_enc_list[group_index])
                cls_enc.append(cls_enc_group)
            cls_enc = torch.cat(cls_enc, dim=1)

            if self.segmentation_head is not None:
                masks_enc = seg_head_fwd(
                    fused_features[0].tensors,
                    [hs_enc],
                    samples_uv.tensors.shape[-2:],
                    skip_blocks=True,
                )[0]

            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["enc_outputs"]["pred_masks"] = masks_enc
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                if self.segmentation_head is not None:
                    out["pred_masks"] = masks_enc

        return out


# ========== 第四部分：模型构建入口 ==========
def build_legacy_gate_dual_model(args) -> LegacyGateDualModalLWDETR:
    """
    按历史门控融合结构构建双模态模型。

    这里尽量复用当前仓库的 backbone / transformer / head 构建代码，
    只把跨模态融合部分替换回旧版门控实现，避免把整套历史仓库一起搬回来。
    """

    num_classes = args.num_classes + 1

    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.shape if hasattr(args, "shape") else (args.resolution, args.resolution),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )

    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)
    segmentation_head = (
        SegmentationHead(
            args.hidden_dim,
            args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )
        if args.segmentation_head
        else None
    )

    return LegacyGateDualModalLWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam,
        use_white=getattr(args, "use_white", True),
        fusion_type=getattr(args, "fusion_type", "uv_queries_white"),
        fusion_num_heads=getattr(args, "fusion_num_heads", getattr(args, "ca_nheads", 8)),
        fusion_num_layers=getattr(args, "fusion_num_layers", 1),
    )
