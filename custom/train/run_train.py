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
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Keep this before any torch import in this launcher or torchrun children.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
USE_DINOV2_PRETRAIN = True
USE_WHITE = True
FUSION_TYPE = "uv_queries_white"
PROJECTOR_SCALE = ["P3", "P4"]
RESOLUTION = 672
POSITIONAL_ENCODING_SIZE = 37
GRADIENT_CHECKPOINTING = True

# Resume
RESUME = ""

# Training
EPOCHS = 160
BATCH_SIZE = 6
GRAD_ACCUM_STEPS = 3
MAX_TRAIN_BATCHES = 0
MAX_VAL_BATCHES = 0
MAX_TEST_BATCHES = 0
LR = 1.4e-4
LR_ENCODER = 2.0e-4
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1

# Regularization
DROPOUT = 0.1
DROP_PATH = 0.1

# Strategy
USE_EMA = True
MULTI_SCALE = True
EXPANDED_SCALES = True
DO_RANDOM_RESIZE_VIA_PADDING = False
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 5
LR_MIN_FACTOR = 0.05
RESUME_LOAD_LR_SCHEDULER = False
SQUARE_RESIZE_DIV_64 = True

# Runtime
EVAL_MAX_DETS = 500
RUN_TEST = True
NUM_GPUS = 2
# Windows 下 dataloader 多进程更容易触发 spawn 问题，默认更保守。
NUM_WORKERS = 16
DEVICE = "cuda"
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# Output
OUTPUT_BASE_DIR = "output/train"


# ========== 第三部分：训练主流程 ==========
def _get_distributed_world_size() -> int:
    """返回 torchrun 注入的 world size；未分布式启动时为 1。"""
    return int(os.environ.get("WORLD_SIZE", "1"))


def _get_target_num_gpus() -> int:
    """训练入口期望使用的 GPU 数量。"""
    return int(NUM_GPUS)


def _is_torchrun_process() -> bool:
    """判断当前进程是否已经由 torchrun/elastic 启动。"""
    return "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ


def _detect_visible_cuda_devices() -> int:
    """按 PyTorch 视角检测当前环境可见 GPU 数，检测失败时返回 0。"""
    try:
        import torch
    except ImportError:
        return 0

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _maybe_relaunch_with_torchrun(log_tag: str) -> None:
    """
    支持直接 `python -m custom.train.run_train` 启动多卡训练。

    已在 torchrun 子进程中时只校验 world size；未在 torchrun 中且目标 GPU 数
    大于 1 时，自动重启成等价的 torchrun 命令。这让自定义入口的用法更接近
    原生 RF-DETR 环境：用户只需要保证 CUDA 环境可见，不必手动设置 RANK 等变量。
    """
    target_num_gpus = _get_target_num_gpus()
    if target_num_gpus <= 1:
        return

    if _is_torchrun_process():
        world_size = _get_distributed_world_size()
        if world_size != target_num_gpus:
            raise RuntimeError(
                f"当前训练配置 NUM_GPUS={target_num_gpus}，但 torchrun WORLD_SIZE={world_size}。"
                f"请使用：torchrun --nproc_per_node={target_num_gpus} -m custom.train.run_train"
            )
        return

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={target_num_gpus}",
        "--standalone",
        "-m",
        "custom.train.run_train",
    ]
    print(f"{log_tag} Relaunch with torchrun: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(_PROJECT_ROOT), check=False)
    raise SystemExit(completed.returncode)


def _configure_local_cuda_device() -> None:
    """在 torchrun 启动时，尽早把每个进程绑到自己的本地 GPU。"""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return

    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))


def _resolve_output_dir(output_dir_base: str, resume_path: str, log_tag: str) -> str:
    """让 torchrun 的多个进程使用同一个输出目录。"""
    if resume_path:
        output_dir = str(Path(resume_path).parent)
        print(f"{log_tag} Resume from: {resume_path}")
        print(f"{log_tag} Continue writing to: {output_dir}")
        return output_dir

    explicit_output_dir = os.environ.get("FUSION_OUTPUT_DIR")
    rank = int(os.environ.get("RANK", "0"))
    base_path = Path(output_dir_base)
    base_path.mkdir(parents=True, exist_ok=True)
    run_key = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT", "default")
    marker_path = base_path / f".ddp_run_{run_key}"
    wait_started_at = time.time()

    if rank == 0:
        output_dir = explicit_output_dir or str(base_path / datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        marker_path.write_text(output_dir, encoding="utf-8")
        print(f"{log_tag} Output dir: {output_dir}")
        return output_dir

    for _ in range(300):
        if marker_path.exists() and marker_path.stat().st_mtime >= wait_started_at - 1:
            return marker_path.read_text(encoding="utf-8").strip()
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for DDP output dir marker: {marker_path}")


def _resolve_pretrain_settings(
    projector_scale: list[str],
    use_dinov2_pretrain: bool,
) -> tuple[str | None, bool, str]:
    """
    把训练区的预训练开关解析为 checkpoint 与 DINOv2 backbone 加载策略。

    当前 custom 主线不再加载 RF-DETR 整模型 checkpoint；`pretrain_weights`
    始终保持 None，只通过 `force_no_pretrain` 控制是否跳过 DINOv2 backbone
    预训练权重。
    """
    _ = projector_scale

    if use_dinov2_pretrain:
        return None, False, "dinov2"

    return None, True, "none"


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
    pretrain_weights, force_no_pretrain, pretrain_mode = _resolve_pretrain_settings(
        projector_scale=PROJECTOR_SCALE,
        use_dinov2_pretrain=USE_DINOV2_PRETRAIN,
    )
    output_dir_base = output_base_dir or OUTPUT_BASE_DIR
    log_tag = log_prefix or "[Train]"

    resume_path = RESUME
    _maybe_relaunch_with_torchrun(log_tag)
    _configure_local_cuda_device()
    output_dir = _resolve_output_dir(output_dir_base, resume_path, log_tag)

    import rfdetr.models.backbone.dinov2_with_windowed_attn as dinov2_windowed

    print(f"{log_tag} RF-DETR source: {Path(dinov2_windowed.__file__).resolve()}", flush=True)

    model_cfg = RFDETRBaseConfig(
        num_classes=NUM_CLASSES,
        pretrain_weights=pretrain_weights,
        use_white=use_white,
        fusion_type=fusion_type,
        projector_scale=PROJECTOR_SCALE,
        resolution=RESOLUTION,
        positional_encoding_size=POSITIONAL_ENCODING_SIZE,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )
    model_kwargs = model_cfg.model_dump()
    model_kwargs["dual_modal"] = dual_modal
    model_kwargs["force_no_pretrain"] = force_no_pretrain
    model_kwargs["load_dinov2_weights"] = USE_DINOV2_PRETRAIN
    model_kwargs["dropout"] = DROPOUT
    model_kwargs["drop_path"] = DROP_PATH

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
        "force_no_pretrain": force_no_pretrain,
        "load_dinov2_weights": USE_DINOV2_PRETRAIN,
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
        "expanded_scales": EXPANDED_SCALES,
        "do_random_resize_via_padding": DO_RANDOM_RESIZE_VIA_PADDING,
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
        f"num_gpus={NUM_GPUS}, "
        f"gradient_checkpointing={GRADIENT_CHECKPOINTING}, "
        f"pin_memory={PIN_MEMORY}, persistent_workers={PERSISTENT_WORKERS}, "
        f"multi_scale={MULTI_SCALE}, expanded_scales={EXPANDED_SCALES}, "
        f"batch_resize={not DO_RANDOM_RESIZE_VIA_PADDING}, "
        f"resume={bool(resume_path)}, dual_modal={dual_modal}, "
        f"use_white={use_white}, fusion_type={fusion_type}, "
        f"projector_scale={PROJECTOR_SCALE}, "
        f"pretrain_mode={pretrain_mode}, "
        f"use_dinov2_pretrain={USE_DINOV2_PRETRAIN}, "
        f"pretrain_weights={pretrain_weights or 'none'}, "
        f"force_no_pretrain={force_no_pretrain}"
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
