# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：
    本模块实现获取模型参数字典的工具函数，供优化器构造时使用（例如按 backbone / decoder / 其它参数分组并设置不同学习率）。

结构概览：
    第一部分：ViT 专用的学习率/权重衰减策略函数（`get_vit_lr_decay_rate`, `get_vit_weight_decay_rate`）
    第二部分：主函数 `get_param_dict`，将模型参数按模块拆分并返回适用于 `torch.optim` 的参数字典列表

原始英文说明（保留）：
Functions to get params dict
"""
from typing import Any, Dict, List

import torch.nn as nn

from rfdetr.models.backbone import Joiner


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name: parameter name.
        lr_decay_rate: base lr decay rate.
        num_layers: number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """
    Calculate weight decay rate for different ViT parameters.

    Args:
        name: parameter name.
        weight_decay_rate: base weight decay rate.
    Returns:
        weight decay rate for the given parameter.
    """
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name):
        weight_decay_rate = 0.
    print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate


def get_param_dict(args: Any, model_without_ddp: nn.Module) -> List[Dict[str, Any]]:
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = model_without_ddp.backbone[0]
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = 'transformer.decoder'
    decoder_params = [
        p
        for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad
    ]

    decoder_param_lr_pairs = [
        {"params": param, "lr": args.lr * args.lr_component_decay}
        for param in decoder_params
    ]

    # 为双模态 fusion_layers 单独分组，赋予更高的 base_lr。
    # fusion_layers 是 Zero-Init 的全新参数，需要比预训练的 backbone/decoder 更高的学习率
    # 才能在有限 epoch 内充分学习如何利用白光辅助信息。
    # fusion_lr_mult 默认为 10，即 fusion 层的 base_lr = lr * 10。
    fusion_param_names: set = set()
    fusion_param_lr_pairs: List[Dict[str, Any]] = []
    if hasattr(model_without_ddp, 'fusion_layers') and model_without_ddp.fusion_layers is not None:
        fusion_lr_mult = getattr(args, 'fusion_lr_mult', 10.0)
        fusion_lr = args.lr * fusion_lr_mult
        for n, p in model_without_ddp.named_parameters():
            if n.startswith('fusion_layers') and p.requires_grad:
                fusion_param_names.add(n)
                fusion_param_lr_pairs.append({"params": p, "lr": fusion_lr})

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters() if (
            n not in backbone_named_param_lr_pairs
            and decoder_key not in n
            and n not in fusion_param_names
            and p.requires_grad)
    ]
    other_param_dicts = [
        {"params": param, "lr": args.lr}
        for param in other_params
    ]

    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs + fusion_param_lr_pairs
    )

    return final_param_dicts
