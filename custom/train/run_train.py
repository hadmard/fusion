# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件是当前双模态 RF-DETR 实验的训练启动脚本。
功能：集中维护训练所需的实验参数，创建输出目录，构建双模态模型，并调用 RF-DETR
      原有训练循环启动训练。

结构概览：
  第一部分：导入依赖与路径设置
  第二部分：实验参数区
  第三部分：训练启动主流程

使用方式：
  - 直接运行本脚本即可启动训练。
  - 如需做 ablation，uv单模态用另外一个py文件，这个双模态脚本目前有点乱

约束说明：
  - 本脚本只负责“实验级配置与启动”，不改动 RF-DETR 主训练框架。
  - UV 是主模态，White 是辅助模态；因此 baseline 与 fusion 都通过少量开关切换。
"""

# ========== 第一部分：导入依赖与路径设置 ==========
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 将项目根目录加入 sys.path，确保脚本可以直接导入 custom/ 与 rfdetr/。
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 切换工作目录，确保相对路径（例如 datasets、output/train）都以项目根目录为基准。
os.chdir(_PROJECT_ROOT)


# ========== 第二部分：实验参数区 ==========
# ---------- 数据集配置 ----------
DATASET_DIR = "datasets"
CLASS_NAMES = ["NPML", "PML", "PM"]
NUM_CLASSES = 3

# ---------- 模型与融合配置 ----------
PRETRAIN_WEIGHTS = "rf-detr-base.pth"
DUAL_MODAL = True

# 是否启用 White 辅助模态。
# 若设为 False，则可得到 UV-only baseline。
USE_WHITE = True

# 融合类型：
#   - "none"              -> 不做跨模态融合
#   - "uv_queries_white"  -> 启用 UV<-White 单向 cross-attention
FUSION_TYPE = "uv_queries_white"

# 融合层数。当前最小可行基线通常保持为 1。
FUSION_NUM_LAYERS = 1

# 融合层（fusion_layers）使用与 decoder 相同的基础学习率 LR（1e-4）。
# populate_args 将 fusion_layers 归入"other params"组，自动继承 LR，无需额外倍增。

# 如需从历史 checkpoint 续训，可在这里指定路径。
# 注意：cross_modal.py 加入 alpha_attn/alpha_ffn Zero-Init 参数后，
# 旧 checkpoint 缺少这两个键，需要从头训练（不可续旧 checkpoint）。
RESUME = ""

# ---------- 训练超参数 ----------
EPOCHS = 120
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LR = 1e-4
LR_ENCODER = 1.5e-4
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1

# ---------- 正则化相关 ----------
DROPOUT = 0.1
DROP_PATH = 0.1

# ---------- 训练策略 ----------
USE_EMA = True
MULTI_SCALE = False
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 1
LR_MIN_FACTOR = 0.0
RESUME_LOAD_LR_SCHEDULER = False

# ---------- 评估 / 系统配置 ----------
EVAL_MAX_DETS = 500
RUN_TEST = False
NUM_WORKERS = 0
DEVICE = "cuda"

# ---------- 输出目录 ----------
OUTPUT_BASE_DIR = "output/train"


# ========== 第三部分：辅助函数 ==========

def _find_latest_valid_resume_checkpoint():
    """自动扫描 OUTPUT_BASE_DIR，找到 epoch 最大的有效 checkpoint 文件路径。"""
    import torch

    base_dir = Path(OUTPUT_BASE_DIR)
    if not base_dir.exists():
        return ""

    # 先按目录的最后修改时间排序，找到最新的训练目录，
    # 再在该目录内选 epoch 最大的 checkpoint。
    # 这样可以避免跨目录比较 epoch 而误选旧训练目录的问题。
    train_dirs = sorted(
        [p for p in base_dir.iterdir() if p.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    for train_dir in train_dirs:
        dir_checkpoints = sorted(
            train_dir.glob("checkpoint*.pth"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        best_path = ""
        best_epoch = -1
        for checkpoint_path in dir_checkpoints:
            try:
                checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            except Exception:
                continue
            epoch = int(checkpoint.get("epoch", -1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = str(checkpoint_path)
        if best_path:
            return best_path

    return ""


# ========== 第四部分：训练启动主流程 ==========
if __name__ == "__main__":
    from rfdetr.config import RFDETRBaseConfig
    from rfdetr.main import Model

    # ---------- 第一步：确定续训路径与输出目录 ----------
    resume_path = RESUME

    if resume_path:
        output_dir = str(Path(resume_path).parent)
        print(f"[Train] Resume from: {resume_path}")
        print(f"[Train] Continue writing to: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Train] Output dir: {output_dir}")

    # ---------- 第二步：准备模型配置 ----------
    # 这里使用 RF-DETR 的配置类来构建一个基础模型配置，
    # 再额外注入当前双模态实验所需的 fusion 开关。
    model_cfg = RFDETRBaseConfig(
        num_classes=NUM_CLASSES,
        pretrain_weights=PRETRAIN_WEIGHTS,
        use_white=USE_WHITE,
        fusion_type=FUSION_TYPE,
        fusion_num_layers=FUSION_NUM_LAYERS,
    )
    model_kwargs = model_cfg.model_dump()
    model_kwargs["dual_modal"] = DUAL_MODAL

    # ---------- 第三步：实例化模型 ----------
    model = Model(**model_kwargs)

    # RF-DETR 的训练回调是一个按事件名组织的字典。
    callbacks = defaultdict(list)

    # ---------- 第四步：组织训练参数 ----------
    # 这里的思路是：
    #   - 手动写出当前实验最关心的参数
    #   - 其余结构性参数从 model_kwargs 自动透传
    train_kwargs = {
        "callbacks": callbacks,
        "dataset_dir": DATASET_DIR,
        "dataset_file": "roboflow",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "dual_modal": DUAL_MODAL,
        "use_white": USE_WHITE,
        "fusion_type": FUSION_TYPE,
        "fusion_num_layers": FUSION_NUM_LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "lr": LR,
        "lr_encoder": LR_ENCODER,
        "weight_decay": WEIGHT_DECAY,
        "clip_max_norm": CLIP_MAX_NORM,
        "dropout": DROPOUT,
        "drop_path": DROP_PATH,
        "use_ema": USE_EMA,
        "multi_scale": MULTI_SCALE,
        "lr_scheduler": LR_SCHEDULER,
        "warmup_epochs": WARMUP_EPOCHS,
        "lr_min_factor": LR_MIN_FACTOR,
        "resume_load_lr_scheduler": RESUME_LOAD_LR_SCHEDULER,
        "num_workers": NUM_WORKERS,
        "device": DEVICE,
        "eval_max_dets": EVAL_MAX_DETS,
        "run_test": RUN_TEST,
        "segmentation_head": False,
        "mask_downsample_ratio": 4,
        "output_dir": output_dir,
    }

    # ---------- 第五步：把模型结构参数透传给训练入口 ----------
    # 这样可以避免重复手写 hidden_dim / resolution / patch_size 等参数。
    exclude_keys = {
        "num_classes",
        "pretrain_weights",
        "device",
        "license",
        "dual_modal",
        "segmentation_head",
        "mask_downsample_ratio",
    }
    for key, value in model_kwargs.items():
        if key not in exclude_keys and key not in train_kwargs:
            train_kwargs[key] = value

    # ---------- 第六步：可选续训 ----------
    if resume_path:
        train_kwargs["resume"] = resume_path
    elif RESUME:
        train_kwargs["resume"] = RESUME

    # ---------- 第七步：打印实验摘要 ----------
    effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(
        f"[Train] epochs={EPOCHS}, batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}={effective_batch}, "
        f"lr={LR}, scheduler={LR_SCHEDULER}, resume={bool(RESUME)}, "
        f"dual_modal={DUAL_MODAL}, use_white={USE_WHITE}, fusion_type={FUSION_TYPE}"
    )

    # ---------- 第八步：正式启动训练 ----------
    model.train(**train_kwargs)
    print(f"[Train] Done. Outputs saved to: {output_dir}")
