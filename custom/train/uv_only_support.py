"""
文件说明：本文件提供 UV-only 训练模式所需的运行时补丁。
功能说明：当前仓库的主训练流更多是围绕双模态入口扩展的，而 UV-only 仍然要复用成对
数据目录与同一套训练启动器；因此这里用最小 monkey patch 把数据集构建和 drop-path
更新逻辑修补到 UV-only 也能稳定跑通。

结构概览：
  第一部分：对外补丁入口
  第二部分：UV-only 数据集构建补丁
  第三部分：drop-path 兼容补丁
"""

from __future__ import annotations


# ========== 第一部分：对外补丁入口 ==========
def patch_uv_only_training_support() -> None:
    """集中应用 UV-only 训练路径依赖的两个运行时补丁。"""
    _patch_uv_dataset_builder()
    _patch_drop_path_update()


# ========== 第二部分：UV-only 数据集构建补丁 ==========
def _patch_uv_dataset_builder() -> None:
    from copy import copy

    import rfdetr.main as rfdetr_main
    from custom.dataset_auto_coco import resolve_roboflow_coco_dataset_dir
    from rfdetr.datasets import build_roboflow

    def _build_dataset_for_uv_only(image_set, args, resolution):
        # UV-only 仍沿用 paired dataset 根目录；这里先解析成 COCO 根目录，
        # 再把 patched args 交回原本的 Roboflow 构建流程，避免复制上游实现。
        coco_root = resolve_roboflow_coco_dataset_dir(
            dataset_dir=args.dataset_dir,
            class_names=getattr(args, "class_names", None),
            log_prefix="[Dataset-UV]",
        )
        patched_args = copy(args)
        patched_args.dataset_dir = str(coco_root)
        patched_args.dataset_file = "roboflow"
        return build_roboflow(image_set=image_set, args=patched_args, resolution=resolution)

    rfdetr_main.build_dataset = _build_dataset_for_uv_only


# ========== 第三部分：drop-path 兼容补丁 ==========
def _patch_drop_path_update() -> None:
    import torch
    from rfdetr.models.lwdetr import LWDETR

    def _safe_update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        # 不同 backbone/封装层级里 blocks 的挂载路径不完全一致。
        # 这里按候选路径依次探测，是为了兼容当前仓库已有权重和模型封装，而不是假设单一路径。
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)
        ]

        encoder = self.backbone[0].encoder
        block_candidates = [
            getattr(encoder, "blocks", None),
            getattr(getattr(encoder, "trunk", None), "blocks", None),
            getattr(getattr(encoder, "encoder", None), "blocks", None),
            getattr(
                getattr(getattr(encoder, "encoder", None), "trunk", None),
                "blocks",
                None,
            ),
            getattr(
                getattr(getattr(encoder, "model", None), "encoder", None),
                "layer",
                None,
            ),
        ]
        blocks = next(
            (candidate for candidate in block_candidates if candidate is not None),
            None,
        )
        if blocks is None:
            return

        for i in range(min(vit_encoder_num_layers, len(blocks))):
            drop_path = getattr(blocks[i], "drop_path", None)
            if hasattr(drop_path, "drop_prob"):
                drop_path.drop_prob = dp_rates[i]

    LWDETR.update_drop_path = _safe_update_drop_path
