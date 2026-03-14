from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import rfdetr.util.misc as utils
from rfdetr.config import RFDETRBaseConfig
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.engine import evaluate
from rfdetr.main import Model, populate_args
from rfdetr.models import build_criterion_and_postprocessors
from custom.dual_collate import dual_collate_fn
from custom.dual_dataset import DualModalYoloDetection
from custom.dual_transforms import make_dual_transforms
from custom.dataset_auto_coco import _build_coco_dataset_for_split, load_class_names_from_yaml


def ensure_true_test_coco(dataset_root: Path, log_prefix: str = "[TestEval]") -> None:
    auto_coco_root = dataset_root / "_auto_coco"
    test_dir = auto_coco_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_ann_path = test_dir / "_annotations.coco.json"

    classes = load_class_names_from_yaml(dataset_root / "dataset_dual.yaml")
    test_dataset = _build_coco_dataset_for_split(dataset_root, "test", classes)
    with test_ann_path.open("w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)
    print(f"{log_prefix} Rebuilt true test annotation: {test_ann_path}")


def main() -> None:
    project_root = _PROJECT_ROOT
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    output_dir = project_root / "output" / "train" / "2026-03-14_154006"
    dataset_root = project_root / "datasets"
    candidate_paths = [
        output_dir / "checkpoint_best_regular.pth",
        output_dir / "checkpoint_best_ema.pth",
        output_dir / "checkpoint.pth",
    ]
    checkpoint_path = next((p for p in candidate_paths if p.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Missing checkpoint. Tried: " + ", ".join(str(p) for p in candidate_paths)
        )

    ensure_true_test_coco(dataset_root)

    args = populate_args(
        num_classes=3,
        class_names=["NPML", "PML", "PM"],
        dataset_file="roboflow",
        dataset_dir="datasets",
        square_resize_div_64=True,
        dual_modal=True,
        use_white=True,
        fusion_type="uv_queries_white",
        fusion_num_layers=1,
        pretrain_weights=None,
        eval_max_dets=500,
        segmentation_head=False,
        mask_downsample_ratio=4,
        resolution=560,
        batch_size=6,
        num_workers=0,
        device="cuda",
    )

    device = torch.device(args.device)

    # Build model with the same config path as training to avoid missing architecture args.
    model_cfg = RFDETRBaseConfig(
        num_classes=3,
        pretrain_weights=str(checkpoint_path),
        use_white=True,
        fusion_type="uv_queries_white",
        fusion_num_layers=1,
    )
    model_kwargs = model_cfg.model_dump()
    model_kwargs["dual_modal"] = True
    model_kwargs["class_names"] = ["NPML", "PML", "PM"]
    model_kwargs["device"] = args.device

    model_wrapper = Model(**model_kwargs)

    model = model_wrapper.model.to(device)
    model.eval()

    criterion, postprocess = build_criterion_and_postprocessors(args)
    criterion.to(device)
    criterion.eval()

    dataset_test = DualModalYoloDetection(
        dataset_dir=str(dataset_root),
        split="test",
        transforms=make_dual_transforms(
            image_set="test",
            resolution=args.resolution,
            multi_scale=False,
            expanded_scales=False,
            patch_size=getattr(args, "patch_size", 16),
            num_windows=getattr(args, "num_windows", 4),
        ),
        class_names=["NPML", "PML", "PM"],
    )
    if len(dataset_test) == 0:
        raise RuntimeError("Test split is empty. Cannot evaluate.")

    sampler_test = SequentialSampler(dataset_test)
    data_loader_test = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=dual_collate_fn,
        num_workers=args.num_workers,
    )
    base_ds_test = get_coco_api_from_dataset(dataset_test)

    test_stats, _ = evaluate(
        model,
        criterion,
        postprocess,
        data_loader_test,
        base_ds_test,
        device,
        args=args,
    )

    out_json = output_dir / "test_eval_results.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(test_stats, f, ensure_ascii=False, indent=2)

    print("=== Test Evaluation Done ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"results_file: {out_json}")
    coco_bbox = test_stats.get("coco_eval_bbox") or test_stats.get("test_coco_eval_bbox")
    results_json = test_stats.get("results_json") or test_stats.get("test_results_json")

    if coco_bbox is not None:
        print(f"test mAP@50:95: {coco_bbox[0]:.6f}")
        print(f"test mAP@50: {coco_bbox[1]:.6f}")
    if results_json is not None:
        print(f"test precision: {results_json['precision']:.6f}")
        print(f"test recall: {results_json['recall']:.6f}")
        print(f"test f1: {results_json['f1_score']:.6f}")


if __name__ == "__main__":
    main()
