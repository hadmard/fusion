# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
UV-only training launcher.

This keeps the existing paired dataset layout on disk, but only trains with the
UV image branch.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


# Dataset
DATASET_DIR = "datasets"
CLASS_NAMES = ["NPML", "PML", "PM"]
NUM_CLASSES = 3
UV_ONLY_FROM_DUAL_DATASET = True

# Model
PRETRAIN_WEIGHTS = "rf-detr-base.pth"
DUAL_MODAL = False
USE_WHITE = False
FUSION_TYPE = "none"

# Resume
RESUME = ""

# Training
EPOCHS = 120
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LR = 1e-4
LR_ENCODER = 1.5e-4
WEIGHT_DECAY = 1e-4
CLIP_MAX_NORM = 0.1

# Regularization
DROPOUT = 0.1
DROP_PATH = 0.1

# Strategy
USE_EMA = True
MULTI_SCALE = False
LR_SCHEDULER = "cosine"
WARMUP_EPOCHS = 1
LR_MIN_FACTOR = 0.0
RESUME_LOAD_LR_SCHEDULER = False

# Runtime
EVAL_MAX_DETS = 500
RUN_TEST = False
NUM_WORKERS = 0
DEVICE = "cuda"

# Output
OUTPUT_BASE_DIR = "output/train_uv"


def _find_latest_valid_resume_checkpoint():
    import torch

    base_dir = Path(OUTPUT_BASE_DIR)
    if not base_dir.exists():
        return ""

    candidates = sorted(
        base_dir.glob("*/checkpoint*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    best_path = ""
    best_epoch = -1
    best_mtime = -1.0

    for checkpoint_path in candidates:
        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except Exception:
            continue

        epoch = int(checkpoint.get("epoch", -1))
        mtime = checkpoint_path.stat().st_mtime
        if epoch > best_epoch or (epoch == best_epoch and mtime > best_mtime):
            best_path = str(checkpoint_path)
            best_epoch = epoch
            best_mtime = mtime

    return best_path


def _patch_uv_dataset_builder():
    import rfdetr.main as rfdetr_main
    from custom.uv_dataset import build_uv_dataset

    def _build_dataset_for_uv_only(image_set, args, resolution):
        return build_uv_dataset(
            image_set=image_set,
            dataset_dir=args.dataset_dir,
            resolution=resolution,
            class_names=getattr(args, "class_names", None),
            multi_scale=getattr(args, "multi_scale", False) if image_set == "train" else False,
            expanded_scales=getattr(args, "expanded_scales", False),
            patch_size=getattr(args, "patch_size", 16),
            num_windows=getattr(args, "num_windows", 4),
        )

    rfdetr_main.build_dataset = _build_dataset_for_uv_only


def _patch_drop_path_update():
    import torch
    from rfdetr.models.lwdetr import LWDETR

    def _safe_update_drop_path(self, drop_path_rate, vit_encoder_num_layers):
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, vit_encoder_num_layers)]

        encoder = self.backbone[0].encoder
        block_candidates = [
            getattr(encoder, "blocks", None),
            getattr(getattr(encoder, "trunk", None), "blocks", None),
            getattr(getattr(encoder, "encoder", None), "blocks", None),
            getattr(getattr(getattr(encoder, "encoder", None), "trunk", None), "blocks", None),
            getattr(getattr(getattr(encoder, "model", None), "encoder", None), "layer", None),
        ]
        blocks = next((candidate for candidate in block_candidates if candidate is not None), None)
        if blocks is None:
            return

        for i in range(min(vit_encoder_num_layers, len(blocks))):
            drop_path = getattr(blocks[i], "drop_path", None)
            if hasattr(drop_path, "drop_prob"):
                drop_path.drop_prob = dp_rates[i]

    LWDETR.update_drop_path = _safe_update_drop_path


if __name__ == "__main__":
    _patch_uv_dataset_builder()
    _patch_drop_path_update()

    from rfdetr.config import RFDETRBaseConfig
    from rfdetr.main import Model

    resume_path = RESUME or _find_latest_valid_resume_checkpoint()

    if resume_path:
        output_dir = str(Path(resume_path).parent)
        print(f"[Train-UV] Resume from: {resume_path}")
        print(f"[Train-UV] Continue writing to: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[Train-UV] Output dir: {output_dir}")

    model_cfg = RFDETRBaseConfig(
        num_classes=NUM_CLASSES,
        pretrain_weights=PRETRAIN_WEIGHTS,
    )
    model_kwargs = model_cfg.model_dump()
    model_kwargs["dual_modal"] = DUAL_MODAL

    model = Model(**model_kwargs)
    callbacks = defaultdict(list)

    train_kwargs = {
        "callbacks": callbacks,
        "dataset_dir": DATASET_DIR,
        "dataset_file": "roboflow",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "dual_modal": DUAL_MODAL,
        "use_white": USE_WHITE,
        "fusion_type": FUSION_TYPE,
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
        "square_resize_div_64": True,
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
        f"[Train-UV] epochs={EPOCHS}, batch={BATCH_SIZE}x{GRAD_ACCUM_STEPS}={effective_batch}, "
        f"lr={LR}, scheduler={LR_SCHEDULER}, resume={bool(resume_path)}, "
        f"dual_modal={DUAL_MODAL}, uv_only_from_dual_dataset={UV_ONLY_FROM_DUAL_DATASET}"
    )

    model.train(**train_kwargs)
    print(f"[Train-UV] Done. Outputs saved to: {output_dir}")
