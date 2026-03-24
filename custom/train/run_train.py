# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：本文件是当前 `custom/train` 下的统一训练启动器。
功能说明：集中维护双模态与 UV-only 两种实验模式共享的大部分训练参数，并在运行时按
模式切换数据集补丁、融合配置、输出目录与日志前缀，减少两套脚本长期漂移。

结构概览：
  第一部分：导入依赖与路径初始化
  第二部分：实验参数区
  第三部分：模式解析
  第四部分：训练主流程
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

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
MODALITY_MODE = "dual_uv_white"
SUPPORTED_MODALITY_MODES = {"dual_uv_white", "uv_only"}
USE_WHITE = True
FUSION_TYPE = "uv_queries_white"
FUSION_NUM_LAYERS = 4

# Resume
RESUME = ""

# Training
# 当前默认参数按这台 9800X3D + 96GB RAM + RTX 5090 机器的“稳健长跑”思路收紧：
# - 保持 batch=6 不动
# - 用 grad accum 把有效 batch 提到 12，而不是继续放大单卡 batch
# - warmup 拉长一点，给双模态和多尺度更稳的起步空间
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
# 恢复到正常训练阶段更常用的 worker 数；若本地 spawn 不稳可再手动降回 0。
NUM_WORKERS = 4
DEVICE = "cuda"
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

# Output
OUTPUT_BASE_DIRS = {
    "dual_uv_white": "output/train",
    "uv_only": "output/train_uv",
}


# ========== 第三部分：模式解析 ==========
def _resolve_mode_settings(modality_mode: str) -> dict[str, Any]:
    """
    把模式名解析成训练主流程真正需要的一组开关。

    这里不直接把 `DUAL_MODAL`、`USE_WHITE` 等常量散落在多个脚本里，
    是为了让 UV-only 与双模态只在入口层做一次分流，避免参数漂移。
    """
    if modality_mode not in SUPPORTED_MODALITY_MODES:
        raise ValueError(
            f"Unsupported modality mode '{modality_mode}'. "
            f"Expected one of {sorted(SUPPORTED_MODALITY_MODES)}."
        )

    if modality_mode == "uv_only":
        return {
            "dual_modal": False,
            "use_white": False,
            "fusion_type": "none",
            "output_base_dir": OUTPUT_BASE_DIRS["uv_only"],
            "log_prefix": "[Train-UV]",
        }

    return {
        "dual_modal": True,
        "use_white": USE_WHITE,
        "fusion_type": FUSION_TYPE,
        "output_base_dir": OUTPUT_BASE_DIRS["dual_uv_white"],
        "log_prefix": "[Train]",
    }


# ========== 第四部分：训练主流程 ==========
def run_training(
    modality_mode: str | None = None,
    output_base_dir: str | None = None,
    log_prefix: str | None = None,
):
    """按指定模式启动训练，并返回本次运行的输出目录。"""
    from rfdetr.config import RFDETRBaseConfig
    from rfdetr.main import Model

    resolved_mode = modality_mode or MODALITY_MODE
    mode_settings = _resolve_mode_settings(resolved_mode)
    dual_modal = bool(mode_settings["dual_modal"])
    use_white = bool(mode_settings["use_white"])
    fusion_type = str(mode_settings["fusion_type"])
    output_dir_base = output_base_dir or str(mode_settings["output_base_dir"])
    log_tag = log_prefix or str(mode_settings["log_prefix"])

    if not dual_modal:
        # UV-only 仍复用 paired dataset 根目录，因此需要在启动前补齐数据与模型兼容补丁。
        from custom.train.uv_only_support import patch_uv_only_training_support

        patch_uv_only_training_support()

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

    # 先用 RF-DETR 自带配置类产出基础模型参数，再补上 custom 模态开关，
    # 可以减少和上游 `RFDETRBaseConfig` 的重复维护。
    model_cfg = RFDETRBaseConfig(
        num_classes=NUM_CLASSES,
        pretrain_weights=PRETRAIN_WEIGHTS,
        use_white=use_white,
        fusion_type=fusion_type,
        fusion_num_layers=FUSION_NUM_LAYERS,
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

    # 训练入口需要的通用参数手工列出；其余模型结构参数从 config 透传，
    # 这样后续如果上游配置类新增字段，这里通常不需要同步手改第二遍。
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
        f"{log_tag} mode={resolved_mode}, epochs={EPOCHS}, "
        f"batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}={effective_batch}, "
        f"max_train_batches={MAX_TRAIN_BATCHES}, max_val_batches={MAX_VAL_BATCHES}, "
        f"lr={LR}, scheduler={LR_SCHEDULER}, workers={NUM_WORKERS}, "
        f"pin_memory={PIN_MEMORY}, persistent_workers={PERSISTENT_WORKERS}, "
        f"resume={bool(resume_path)}, "
        f"dual_modal={dual_modal}, use_white={use_white}, fusion_type={fusion_type}"
    )

    model.train(**train_kwargs)
    print(f"{log_tag} Done. Outputs saved to: {output_dir}")
    return output_dir


def main() -> str:
    """
    文件说明：提供与 `if __name__ == "__main__"` 解耦的训练脚本入口。
    功能说明：让模块导入、命令行执行和后续可能的脚本复用都走同一条启动路径，
    减少入口逻辑散落在文件尾部的情况。
    """
    return run_training()


if __name__ == "__main__":
    main()
