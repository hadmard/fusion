"""
文件说明：本文件保留旧 `rfdetr.util.get_param_dicts` 的直接可用实现。
功能说明：为仍依赖旧训练入口的代码提供参数分组逻辑，同时避免导入
`rfdetr.training` 包时强制要求 `pytorch_lightning`。

结构概览：
  第一部分：导入依赖
  第二部分：学习率/权重衰减辅助函数
  第三部分：参数分组函数
"""

from typing import Any, Dict, List

import torch.nn as nn

from rfdetr.models.backbone import Joiner
from rfdetr.utilities.logger import get_logger

logger = get_logger()


# ========== 第二部分：学习率/权重衰减辅助函数 ==========
def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 1.0, num_layers: int = 12) -> float:
    """按 ViT 层深计算 layer-wise 学习率衰减。"""
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    logger.debug("name: %s, lr_decay: %s", name, lr_decay_rate ** (num_layers + 1 - layer_id))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name: str, weight_decay_rate: float = 1.0) -> float:
    """按参数类型决定是否关闭权重衰减。"""
    if ("gamma" in name) or ("pos_embed" in name) or ("rel_pos" in name) or ("bias" in name) or ("norm" in name):
        weight_decay_rate = 0.0
    logger.debug("name: %s, weight_decay rate: %s", name, weight_decay_rate)
    return weight_decay_rate


# ========== 第三部分：参数分组函数 ==========
def get_param_dict(args: Any, model_without_ddp: nn.Module) -> List[Dict[str, Any]]:
    """返回兼容旧训练入口的优化器参数分组。"""
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = model_without_ddp.backbone[0]
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = "transformer.decoder"
    decoder_params = [parameter for name, parameter in model_without_ddp.named_parameters() if decoder_key in name and parameter.requires_grad]
    decoder_param_lr_pairs = [{"params": parameter, "lr": args.lr * args.lr_component_decay} for parameter in decoder_params]

    other_params = [
        parameter
        for name, parameter in model_without_ddp.named_parameters()
        if name not in backbone_named_param_lr_pairs and decoder_key not in name and parameter.requires_grad
    ]
    other_param_dicts = [{"params": parameter, "lr": args.lr} for parameter in other_params]
    return other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs
