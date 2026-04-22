
"""
文件说明：本文件实现双模态数据在 DataLoader 中的 collate_fn。
功能：把数据集返回的 (img_uv, img_white, target) 三元组整理成训练循环可直接消费的
      (samples_uv, samples_white, targets) 结构，其中前两项会被打包成 NestedTensor。

结构概览：
  第一部分：导入依赖
  第二部分：双模态 collate 函数

设计说明：
  - UV 放在前面，是因为当前项目中 UV 是主模态，White 是辅助模态。
  - 本函数不做任何图像增强或数值变换，只负责 batch 组织。
  - 之所以在这里转成 NestedTensor，是因为 RF-DETR 主训练循环直接依赖这种输入格式。
"""

# ========== 第一部分：导入依赖 ==========
from typing import Any, List, Tuple

from rfdetr.util.misc import nested_tensor_from_tensor_list


# ========== 第二部分：双模态 collate 函数 ==========
def dual_collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    """
    将双模态样本列表拼接为一个批次。

    Args:
        batch:
            DataLoader 传入的原始批次列表。
            每个元素的结构应为：
                (img_uv, img_white, target)

    Returns:
        tuple:
            返回结构为：
                (samples_uv, samples_white, targets)
            其中：
              - samples_uv 是 UV 图像构成的 NestedTensor
              - samples_white 是 White 图像构成的 NestedTensor
              - targets 保持为字典列表，供检测损失直接使用
    """
    # 先按列重组 batch。
    # 例如原来是：
    #   [(uv1, white1, t1), (uv2, white2, t2)]
    # 重组后变成：
    #   [(uv1, uv2), (white1, white2), (t1, t2)]
    batch = list(zip(*batch))

    # 将 UV 模态的图像列表打包为 NestedTensor。
    # 这样后续 backbone / model 前向就能直接读取 .tensors 和 .mask。
    batch[0] = nested_tensor_from_tensor_list(batch[0])

    # 将 White 模态图像也按同样方式打包。
    # 两个模态分别构建 NestedTensor，避免在这里做早期融合。
    batch[1] = nested_tensor_from_tensor_list(batch[1])

    # targets 保持原始 tuple/list 结构即可，训练循环会逐条送入 device。
    return tuple(batch)
