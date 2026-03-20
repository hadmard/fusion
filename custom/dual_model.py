
"""
文件说明：融合的执行模块，具体思路模块是cross_attn那个文件
功能：保持 RF-DETR 主体结构不大改的前提下，引入 UV 主模态、White 辅助模态的双输入前向，
      并在 encoder 后、projector 前 插入单向 UV <- White 跨模态融合模块。

结构概览：
  第一部分：导入依赖与常量
  第二部分：双模态模型类 DualModalLWDETR
  第三部分：模型构建函数 build_dual_model

"""

# ========== 第一部分：导入依赖与常量 ==========
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from custom.cross_modal import CrossModalFusionStack
from rfdetr.models.backbone import build_backbone
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.segmentation_head import SegmentationHead
from rfdetr.models.transformer import build_transformer
from rfdetr.util.misc import NestedTensor, nested_tensor_from_tensor_list

# 有点bug，不要使用单模态
SUPPORTED_FUSION_TYPES = {"none", "uv_queries_white"}


# ========== 第二部分：双模态模型类 ==========
class DualModalLWDETR(LWDETR):
    """
    在 LWDETR 基础上扩展出的双模态模型。

    模型对外保持的语义是：
      - samples_uv：UV 主模态输入
      - samples_white：White 辅助模态输入

    检测链路是：
        UV encoder -> (optional UV<-White fusion) -> projector -> transformer -> detection head
    """

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
        fusion_num_layers: int = 6,
    ):
        # 先初始化 RF-DETR 原始主干。
        # 这样可以最大限度复用已有检测头、transformer、two-stage 等逻辑。
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

        # ---------- 检查配置合法性 ----------
        if fusion_type not in SUPPORTED_FUSION_TYPES:
            raise ValueError(
                f"Unsupported fusion_type '{fusion_type}'. "
                f"Expected one of {sorted(SUPPORTED_FUSION_TYPES)}."
            )

        if not use_white and fusion_type != "none":
            raise ValueError(
                "use_white=False is only valid when fusion_type='none'."
            )

        self.use_white = bool(use_white)
        self.fusion_type = fusion_type

        self.fusion_enabled = self.use_white and self.fusion_type == "uv_queries_white"

        # 融合点前移到 projector 之前，因此这里按 encoder 输出层级创建融合层。
        # projector 会在融合后的 UV encoder features 之上继续构建多尺度检测特征。
        encoder_feature_dims = list(backbone[0].encoder._out_feature_channels)
        self.fusion_layers = (
            nn.ModuleList(
                [
                    CrossModalFusionStack(
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
        #将输入统一整理成 NestedTensor。
        if isinstance(samples, (list, torch.Tensor)):
            return nested_tensor_from_tensor_list(samples)
        return samples

    def _fuse_uv_with_white(
        self,
        uv_features: List[NestedTensor],
        white_features: List[NestedTensor],
    ) -> List[NestedTensor]:
        """
        对每个特征层执行一次 UV<-White 融合。

        输入输出都仍然保持 RF-DETR 原本使用的 List[NestedTensor] 结构，
        以便后续 transformer 逻辑无需任何改动。
        """
        if self.fusion_layers is None:
            return uv_features

        # 两个模态必须具有相同的特征层数。
        if len(uv_features) != len(white_features):
            raise ValueError(
                f"UV/White feature levels mismatch: {len(uv_features)} vs {len(white_features)}."
            )

        if len(uv_features) != len(self.fusion_layers):
            raise ValueError(
                f"Fusion layers ({len(self.fusion_layers)}) do not match feature levels ({len(uv_features)})."
            )

        fused_features: List[NestedTensor] = []

        # 对每个 level 独立融合，保持多尺度结构不被打乱。
        for uv_feat, white_feat, fusion_layer in zip(
            uv_features, white_features, self.fusion_layers
        ):
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

    def _extract_encoder_features(self, samples: NestedTensor) -> List[NestedTensor]:
        """
        只运行 backbone encoder，并为每个 encoder feature 补齐对应 mask。
        这样就可以在 projector 之前完成跨模态融合。
        """
        encoder_features = self.backbone[0].encoder(samples.tensors)
        encoded_nested: List[NestedTensor] = []

        for feat in encoder_features:
            mask = samples.mask
            assert mask is not None
            resized_mask = F.interpolate(
                mask[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
            encoded_nested.append(NestedTensor(feat, resized_mask))

        return encoded_nested

    def _project_encoder_features(
        self,
        samples: NestedTensor,
        encoder_features: List[NestedTensor],
    ) -> tuple[List[NestedTensor], List[torch.Tensor]]:
        """
        将融合后的 encoder features 送入 projector，并重新构建位置编码。
        """
        projected_tensors = self.backbone[0].projector(
            [feature.tensors for feature in encoder_features]
        )
        projected_features: List[NestedTensor] = []
        pos_embeddings: List[torch.Tensor] = []

        for feat in projected_tensors:
            mask = samples.mask
            assert mask is not None
            resized_mask = F.interpolate(
                mask[None].float(), size=feat.shape[-2:]
            ).to(torch.bool)[0]
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
        """
        双模态模型前向。

        流程分为四步：
          1. 只用 UV 取主干特征
          2. 如启用融合，再额外提取 White 特征
          3. 在 encoder 后、projector 前做 UV<-White 融合
          4. 后续 detection branch 仍然只走 fused UV
        """
        # ---------- 第一步：统一输入格式 ----------
        samples_uv = self._prepare_inputs(samples_uv)
        if samples_white is not None:
            samples_white = self._prepare_inputs(samples_white)

        # ---------- 第二步：提取 UV 主特征 ----------
        fused_features = None
        pos_uv = None

        # ---------- 第三步：可选地在 projector 前引入 White 辅助特征 ----------
        if self.fusion_enabled:
            if samples_white is None:
                raise ValueError(
                    "fusion_type='uv_queries_white' requires samples_white to be provided."
                )

            # 先取 encoder feature，再做 UV<-White 融合，最后才进入 projector。
            uv_encoder_features = self._extract_encoder_features(samples_uv)
            white_encoder_features = self._extract_encoder_features(samples_white)
            fused_encoder_features = self._fuse_uv_with_white(
                uv_encoder_features, white_encoder_features
            )
            fused_features, pos_uv = self._project_encoder_features(
                samples_uv, fused_encoder_features
            )
        else:
            # baseline 路径保持原始 RF-DETR 行为不变。
            fused_features, pos_uv = self.backbone(samples_uv)

        # ---------- 第四步：整理成 transformer 所需的输入 ----------
        srcs = []
        masks = []
        for feat in fused_features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
            assert mask is not None

        # 训练与推理阶段 query 数量略有不同：
        # 推理时只保留一个 group。
        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
            query_feat_weight = self.query_feat.weight[: self.num_queries]

        # 分割头在训练和推理阶段的前向接口略有区别。
        if self.segmentation_head is not None:
            seg_head_fwd = (
                self.segmentation_head.sparse_forward
                if self.training
                else self.segmentation_head.forward
            )

        # ---------- 第五步：进入 RF-DETR 原始 transformer ----------
        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            srcs, masks, pos_uv, refpoint_embed_weight, query_feat_weight
        )

        outputs_masks = None
        if hs is not None:
            # ---------- 第六步：预测框 ----------
            if self.bbox_reparam:
                outputs_coord_delta = self.bbox_embed(hs)
                outputs_coord_cxcy = (
                    outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:]
                    + ref_unsigmoid[..., :2]
                )
                outputs_coord_wh = (
                    outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                )
                outputs_coord = torch.concat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            # ---------- 第七步：预测类别 ----------
            outputs_class = self.class_embed(hs)

            # ---------- 第八步：可选分割头 ----------
            if self.segmentation_head is not None:
                outputs_masks = seg_head_fwd(
                    fused_features[0].tensors,
                    hs,
                    samples_uv.tensors.shape[-2:],
                )

            # 主输出使用最后一层 decoder 的结果。
            out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
            if self.segmentation_head is not None:
                out["pred_masks"] = outputs_masks[-1]

            # 如果启用 auxiliary loss，则把中间层输出也返回。
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_class,
                    outputs_coord,
                    outputs_masks if self.segmentation_head is not None else None,
                )
        else:
            out = {}

        # ---------- 第九步：two-stage 额外输出 ----------
        if self.two_stage:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []

            for g_idx in range(group_detr):
                cls_enc_gidx = self.transformer.enc_out_class_embed[g_idx](
                    hs_enc_list[g_idx]
                )
                cls_enc.append(cls_enc_gidx)
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

    def update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        """
        根据当前 RF-DETR / windowed DINOv2 的层级路径更新 drop-path。

        这里覆写父类实现，是因为当前 backbone 的层路径和原始实现略有不同。
        """
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]
        encoder = self.backbone[0].encoder
        layers = encoder.encoder.encoder.layer

        for i in range(min(vit_encoder_num_layers, len(layers))):
            if hasattr(layers[i], "drop_path") and hasattr(layers[i].drop_path, "drop_prob"):
                layers[i].drop_path.drop_prob = dp_rates[i]


# ========== 第三部分：模型构建函数 ==========
def build_dual_model(args):
    """
    根据 RF-DETR 参数对象构建双模态模型。

    这个函数尽量沿用 rfdetr.models.lwdetr.build_model 的构建方式，
    仅在最后一步替换为 DualModalLWDETR。
    """
    # RF-DETR 原本内部会自动给检测头类别数 +1（包含 no-object）。
    num_classes = args.num_classes + 1
    torch.device(args.device)

    # ---------- 第一步：构建 backbone ----------
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

    # ---------- 第二步：构建 transformer ----------
    args.num_feature_levels = len(args.projector_scale)
    transformer = build_transformer(args)

    # ---------- 第三步：可选分割头 ----------
    segmentation_head = (
        SegmentationHead(
            args.hidden_dim,
            args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
        )
        if args.segmentation_head
        else None
    )

    # ---------- 第四步：构建双模态检测器 ----------
    model = DualModalLWDETR(
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
    return model
