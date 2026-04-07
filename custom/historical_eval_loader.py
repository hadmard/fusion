"""
文件说明：该文件用于在评估阶段按历史 checkpoint 结构加载对应年代的双模态模型实现。
功能说明：负责识别 `fusion_layers.*`、`read_blocks.*`、旧门控等历史权重结构，并选择正确的建模路径，
避免当前主线模型用 `strict=False` 静默吞掉大量不匹配权重后仍继续评估。

结构概览：
  第一部分：导入依赖与结构常量
  第二部分：checkpoint 结构识别
  第三部分：历史模型构建与安全加载
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from custom.legacy_gate_model import build_legacy_gate_dual_model
from custom.rfdetr_compat import Model, populate_args
from rfdetr.models.lwdetr import PostProcess


# ========== 第一部分：导入依赖与结构常量 ==========
ARCH_VARIANT_CURRENT = "current"
ARCH_VARIANT_LEGACY_GATE = "legacy_gate"
ARCH_VARIANT_FUSION_LAYERS = "fusion_layers_attnres"
ARCH_VARIANT_SAME_GRID = "same_grid_readblocks"

_HEAD_PREFIXES = (
    "class_embed.",
    "transformer.enc_out_class_embed.",
)

_HISTORICAL_MODULES = {
    ARCH_VARIANT_FUSION_LAYERS: "githistory.fusion_layers_attnres.custom.dual_model",
    ARCH_VARIANT_SAME_GRID: "githistory.same_grid_readblocks.custom.dual_model",
}


# ========== 第二部分：checkpoint 结构识别 ==========
def detect_checkpoint_architecture_variant(checkpoint: dict[str, Any]) -> str:
    """
    根据 state_dict 键名识别 checkpoint 属于哪一代融合结构。

    为什么必须显式识别：
    - 这几代模型对融合层命名完全不同
    - 当前评估主线此前用 `strict=False` 会把大段不匹配权重静默吞掉
    - 一旦识别错误，结果表面上“能跑”，实质上是错模型在推理
    """
    model_state = checkpoint.get("model", {})
    keys = tuple(model_state.keys())

    if any(key.endswith("alpha_attn") or key.endswith("alpha_ffn") for key in keys):
        return ARCH_VARIANT_LEGACY_GATE

    if any(key.startswith("fusion_layers.") for key in keys):
        return ARCH_VARIANT_FUSION_LAYERS

    if any(".read_blocks." in key for key in keys):
        return ARCH_VARIANT_SAME_GRID

    return ARCH_VARIANT_CURRENT


# ========== 第三部分：历史模型构建与安全加载 ==========
def _build_runtime_args(
    *,
    checkpoint: dict[str, Any],
    class_names: list[str],
    resolution: int,
    device: str,
    use_white: bool,
    fusion_type: str,
    fusion_num_layers: int,
    dual_modal: bool,
) -> Any:
    checkpoint_args = checkpoint.get("args")
    args_dict = vars(checkpoint_args).copy() if checkpoint_args is not None else {}
    args_dict.update(
        {
            "num_classes": len(class_names),
            "class_names": class_names,
            "pretrain_weights": None,
            "resolution": resolution,
            "device": device,
            "dual_modal": dual_modal,
            "use_white": use_white,
            "fusion_type": fusion_type,
            "fusion_num_layers": fusion_num_layers,
        }
    )
    return populate_args(**args_dict)


def _build_historical_dual_model(architecture_variant: str, args: Any) -> torch.nn.Module:
    if architecture_variant == ARCH_VARIANT_LEGACY_GATE:
        return build_legacy_gate_dual_model(args)

    if architecture_variant in _HISTORICAL_MODULES:
        module = import_module(_HISTORICAL_MODULES[architecture_variant])
        return module.build_dual_model(args)

    raise ValueError(f"Unsupported historical architecture variant: {architecture_variant}")


def _is_allowed_head_key(key: str) -> bool:
    return key.startswith(_HEAD_PREFIXES)


def _load_runtime_state_dict(
    *,
    model: torch.nn.Module,
    model_state: dict[str, torch.Tensor],
    args: Any,
    architecture_variant: str,
) -> None:
    state_dict = model_state.copy()
    num_desired_queries = int(args.num_queries) * int(args.group_detr)

    for name in list(state_dict.keys()):
        if name.endswith("refpoint_embed.weight") or name.endswith("query_feat.weight"):
            state_dict[name] = state_dict[name][:num_desired_queries]

    checkpoint_num_classes = int(state_dict["class_embed.bias"].shape[0])
    runtime_num_classes = int(args.num_classes) + 1
    needs_head_resize = checkpoint_num_classes != runtime_num_classes

    if needs_head_resize:
        model.reinitialize_detection_head(checkpoint_num_classes)

    result = model.load_state_dict(state_dict, strict=False)

    if needs_head_resize:
        model.reinitialize_detection_head(runtime_num_classes)

    disallowed_missing = [key for key in result.missing_keys if not _is_allowed_head_key(key)]
    disallowed_unexpected = [key for key in result.unexpected_keys if not _is_allowed_head_key(key)]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            "历史评估模型与 checkpoint 仍未对齐，拒绝继续静默评估。"
            f"\narchitecture_variant={architecture_variant}"
            f"\nmissing(sample)={disallowed_missing[:20]}"
            f"\nunexpected(sample)={disallowed_unexpected[:20]}"
        )


def build_checkpoint_runtime(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    class_names: list[str],
    resolution: int,
    device: str,
    use_white: bool,
    fusion_type: str,
    fusion_num_layers: int,
    dual_modal: bool,
) -> tuple[torch.nn.Module, PostProcess, str]:
    architecture_variant = detect_checkpoint_architecture_variant(checkpoint)

    if architecture_variant == ARCH_VARIANT_CURRENT:
        model_wrapper = Model(
            num_classes=len(class_names),
            class_names=class_names,
            pretrain_weights=str(checkpoint_path),
            resolution=resolution,
            use_white=use_white,
            fusion_type=fusion_type,
            fusion_num_layers=fusion_num_layers,
            device=device,
            dual_modal=dual_modal,
        )
        model = model_wrapper.model.to(device)
        model.eval()
        return model, model_wrapper.postprocess, architecture_variant

    args = _build_runtime_args(
        checkpoint=checkpoint,
        class_names=class_names,
        resolution=resolution,
        device=device,
        use_white=use_white,
        fusion_type=fusion_type,
        fusion_num_layers=fusion_num_layers,
        dual_modal=dual_modal,
    )
    model = _build_historical_dual_model(architecture_variant=architecture_variant, args=args).to(device)
    _load_runtime_state_dict(
        model=model,
        model_state=checkpoint["model"],
        args=args,
        architecture_variant=architecture_variant,
    )
    model = model.to(device)
    model.eval()
    postprocess = PostProcess(num_select=args.num_select)
    return model, postprocess, architecture_variant
