# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：custom 是当前项目中专门承载双模态检测定制逻辑的扩展包。
功能：将与 UV/White 双输入、跨模态融合、定制数据集、训练脚本、推理脚本相关的代码集中放在
      custom/ 目录下，尽量不直接侵入 rfdetr/ 主干实现，方便后续做消融实验和持续迭代。

结构概览：
  第一部分：包级说明与设计目标
  第二部分：子模块职责划分说明

子模块职责：
  - custom.cross_modal：定义 UV 主模态读取 White 辅助模态信息的跨模态融合模块。
  - custom.dual_model：在 RF-DETR 主干上接入双模态前向流程与融合逻辑。
  - custom.dual_dataset：实现成对 UV/White 数据集读取与 COCO 兼容评估接口。
  - custom.dual_transforms：实现双模态同步增强，保证框始终以 UV 标注为准。
  - custom.dual_collate：将 batch 组织成适合 RF-DETR 训练循环的结构。
  - custom.train / custom.detect：提供独立训练与推理脚本入口。

"""