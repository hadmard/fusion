from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from custom.dual_dataset import DualModalYoloDetection
from custom.dual_transforms import make_dual_transforms
from rfdetr.config import RFDETRBaseConfig
from rfdetr.main import Model
from rfdetr.util.benchmark import flop_count, get_shape, measure_time


def _numel(shape):
    n = 1
    for v in shape:
        n *= int(v)
    return int(n)


def gelu_flop_jit(inputs, outputs):
    out_shape = get_shape(outputs[0])
    return Counter({"gelu": _numel(out_shape) * 8})


def silu_flop_jit(inputs, outputs):
    out_shape = get_shape(outputs[0])
    return Counter({"silu": _numel(out_shape) * 4})


def exp_flop_jit(inputs, outputs):
    out_shape = get_shape(outputs[0])
    return Counter({"exp": _numel(out_shape)})


def upsample_bicubic2d_aa_flop_jit(inputs, outputs):
    out_shape = get_shape(outputs[0])
    # Bicubic interpolation is roughly constant work per output pixel.
    return Counter({"upsample_bicubic2d_aa": _numel(out_shape) * 16})


def scaled_dot_product_attention_flop_jit(inputs, outputs):
    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])
    v_shape = get_shape(inputs[2])

    if len(q_shape) < 4 or len(k_shape) < 4 or len(v_shape) < 4:
        return Counter({"scaled_dot_product_attention": 0})

    b = int(q_shape[0])
    h = int(q_shape[1])
    lq = int(q_shape[-2])
    d = int(q_shape[-1])
    lk = int(k_shape[-2])

    # QK^T + AV + softmax approximation.
    qk = b * h * lq * lk * d
    av = b * h * lq * lk * d
    softmax = b * h * lq * lk * 5
    return Counter({"scaled_dot_product_attention": qk + av + softmax})


class DualInferenceWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        samples_uv, samples_white = inputs
        return self.model(samples_uv, samples_white=samples_white)


def pick_checkpoint(output_dir: Path) -> Path:
    candidates = [
        output_dir / "checkpoint_best_regular.pth",
        output_dir / "checkpoint_best_ema.pth",
        output_dir / "checkpoint.pth",
    ]
    ckpt = next((p for p in candidates if p.exists()), None)
    if ckpt is None:
        raise FileNotFoundError(
            "Missing checkpoint. Tried: " + ", ".join(str(p) for p in candidates)
        )
    return ckpt


def main() -> None:
    os.chdir(_PROJECT_ROOT)

    output_dir = _PROJECT_ROOT / "output" / "train" / "2026-03-14_154006"
    dataset_root = _PROJECT_ROOT / "datasets"
    checkpoint_path = pick_checkpoint(output_dir)

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
    model_kwargs["device"] = "cuda"

    model_wrapper = Model(**model_kwargs)
    model = model_wrapper.model.float().cuda().eval()
    wrapped = DualInferenceWrapper(model).cuda().eval()

    dataset_test = DualModalYoloDetection(
        dataset_dir=str(dataset_root),
        split="test",
        transforms=make_dual_transforms(
            image_set="test",
            resolution=560,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=4,
        ),
        class_names=["NPML", "PML", "PM"],
    )
    if len(dataset_test) == 0:
        raise RuntimeError("Test split is empty. Cannot benchmark.")

    warmup_step = 5
    total_step = min(20, len(dataset_test))

    gflops_list = []
    gflops_augmented_list = []
    infer_time_list = []

    customized_ops = {
        "aten::gelu": gelu_flop_jit,
        "aten::silu_": silu_flop_jit,
        "aten::exp": exp_flop_jit,
        "aten::_upsample_bicubic2d_aa": upsample_bicubic2d_aa_flop_jit,
        "aten::scaled_dot_product_attention": scaled_dot_product_attention_flop_jit,
    }

    with torch.no_grad():
        for i in range(total_step):
            sample_uv, sample_white, _ = dataset_test[i]
            inputs = [[sample_uv.cuda(non_blocking=True)], [sample_white.cuda(non_blocking=True)]]

            flops_base = flop_count(wrapped, (inputs,))
            flops_augmented = flop_count(wrapped, (inputs,), customized_ops=customized_ops)

            gflops_list.append(float(sum(flops_base.values())))
            gflops_augmented_list.append(float(sum(flops_augmented.values())))

            t = measure_time(wrapped, inputs, N=10)
            if i >= warmup_step:
                infer_time_list.append(float(t))

    mean_gflops = float(np.mean(gflops_list)) if gflops_list else 0.0
    mean_gflops_augmented = float(np.mean(gflops_augmented_list)) if gflops_augmented_list else 0.0
    mean_time = float(np.mean(infer_time_list)) if infer_time_list else 0.0
    fps = float(1.0 / mean_time) if mean_time > 0 else 0.0
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    result = {
        "checkpoint": str(checkpoint_path),
        "gflops": mean_gflops,
        "gflops_all_ops_est": mean_gflops_augmented,
        "fps": fps,
        "model_size_mb": model_size_mb,
        "samples_used": total_step,
        "timed_samples": len(infer_time_list),
    }

    out_file = output_dir / "test_perf_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
