# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件是当前双模态训练主线的统一启动入口。
功能说明：集中维护当前实验主线使用的参数，并显式定义当前结构与预训练策略的关系，
避免训练入口继续混入已经不再使用的旧结构假设。

结构概览：
  第一部分：导入依赖与路径初始化
  第二部分：实验参数区
  第三部分：训练主流程
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from custom import prepare_project_environment

# ========== 第一部分：导入依赖与路径初始化 ==========
prepare_project_environment(change_cwd=True)


# ========== 第二部分：实验参数区 ==========
# Dataset
DATASET_DIR = "datasets"
CLASS_NAMES = ["NPML", "PML", "PM"]
NUM_CLASSES = 3

# Model
PRETRAIN_WEIGHTS = "rf-detr-base.pth"
USE_WHITE = True
FUSION_TYPE = "uv_queries_white"
FUSION_NUM_LAYERS = 4
PROJECTOR_SCALE = ["P3", "P4"]
RESOLUTION = 672
POSITIONAL_ENCODING_SIZE = 37

# Resume
RESUME = ""

# Training
EPOCHS = 160
BATCH_SIZE = 6
GRAD_ACCUM_STEPS = 2
MAX_TRAIN_BATCHES = 0
MAX_VAL_BATCHES = 0
MAX_TEST_BATCHES = 0
LR = 1.2e-4
LR_ENCODER = 1.8e-4
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1

# Regularization
DROPOUT = 0.1
DROP_PATH = 0.1

# Strategy
USE_EMA = True
MULTI_SCALE = True
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 3
LR_MIN_FACTOR = 0.0
RESUME_LOAD_LR_SCHEDULER = False
SQUARE_RESIZE_DIV_64 = True

# Runtime
EVAL_MAX_DETS = 500
RUN_TEST = False
# Windows 下 dataloader 多进程更容易触发 spawn 问题，默认更保守。
NUM_WORKERS = 4 if os.name == "nt" else 8
DEVICE = "cuda"
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Output
OUTPUT_BASE_DIR = "output/train"


# ========== 第三部分：训练主流程 ==========
def _resolve_pretrain_weights(projector_scale: list[str]) -> str | None:
    """
    当前 RF-DETR 整模型预训练权重只匹配单 `P4` 检测头结构。

    当前实验主线改成 `P3 + P4` 后，继续加载整模型权重会在 projector 与 decoder
    deformable attention 上发生 shape mismatch。因此这里显式回退到
    `pretrain_weights=None`，只保留 DINOv2 backbone 预训练。
    """
    if list(projector_scale) != ["P4"]:
        return None
    return PRETRAIN_WEIGHTS


def run_training(
    output_base_dir: str | None = None,
    log_prefix: str | None = None,
):
    """启动双模态训练，并返回本次运行的输出目录。"""
    from rfdetr.config import RFDETRBaseConfig
    from rfdetr.main import Model

    dual_modal = True
    use_white = USE_WHITE
    fusion_type = FUSION_TYPE
    pretrain_weights = _resolve_pretrain_weights(PROJECTOR_SCALE)
    output_dir_base = output_base_dir or OUTPUT_BASE_DIR
    log_tag = log_prefix or "[Train]"

    resume_path = RESUME
    if resume_path:
        output_dir = str(Path(resume_path).parent)
        print(f"{log_tag} Resume from: {resume_path}")
        print(f"{log_tag} Continue writing to: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(output_dir_base, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"{log_tag} Output dir: {output_dir}")

    model_cfg = RFDETRBaseConfig(
        num_classes=NUM_CLASSES,
        pretrain_weights=pretrain_weights,
        use_white=use_white,
        fusion_type=fusion_type,
        fusion_num_layers=FUSION_NUM_LAYERS,
        projector_scale=PROJECTOR_SCALE,
        resolution=RESOLUTION,
        positional_encoding_size=POSITIONAL_ENCODING_SIZE,
    )
    model_kwargs = model_cfg.model_dump()
    model_kwargs["dual_modal"] = dual_modal

    model = Model(**model_kwargs)
    callbacks = defaultdict(list)

    train_kwargs = {
        "callbacks": callbacks,
        "dataset_dir": DATASET_DIR,
        "dataset_file": "roboflow",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "dual_modal": dual_modal,
        "use_white": use_white,
        "fusion_type": fusion_type,
        "fusion_num_layers": FUSION_NUM_LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "max_train_batches": MAX_TRAIN_BATCHES,
        "max_val_batches": MAX_VAL_BATCHES,
        "max_test_batches": MAX_TEST_BATCHES,
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
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "eval_max_dets": EVAL_MAX_DETS,
        "run_test": RUN_TEST,
        "segmentation_head": False,
        "mask_downsample_ratio": 4,
        "output_dir": output_dir,
        "square_resize_div_64": SQUARE_RESIZE_DIV_64,
    }

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

    if resume_path:
        train_kwargs["resume"] = resume_path

    effective_batch = BATCH_SIZE * GRAD_ACCUM_STEPS
    print(
        f"{log_tag} resolution={RESOLUTION}, epochs={EPOCHS}, "
        f"batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}={effective_batch}, "
        f"max_train_batches={MAX_TRAIN_BATCHES}, max_val_batches={MAX_VAL_BATCHES}, "
        f"lr={LR}, scheduler={LR_SCHEDULER}, workers={NUM_WORKERS}, "
        f"pin_memory={PIN_MEMORY}, persistent_workers={PERSISTENT_WORKERS}, "
        f"resume={bool(resume_path)}, dual_modal={dual_modal}, "
        f"use_white={use_white}, fusion_type={fusion_type}, "
        f"projector_scale={PROJECTOR_SCALE}, "
        f"pretrain_weights={pretrain_weights or 'dinov2-only'}"
    )

    model.train(**train_kwargs)
    print(f"{log_tag} Done. Outputs saved to: {output_dir}")
    return output_dir


def main() -> str:
    """
    与 `if __name__ == "__main__"` 解耦的训练脚本入口。
    """
    return run_training()


if __name__ == "__main__":
    main()
