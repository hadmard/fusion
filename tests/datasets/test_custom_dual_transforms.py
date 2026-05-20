import random

import torch
from PIL import Image

from custom.data.dual_transforms import (
    DualIndustrialPhotometricJitter,
    DualPMLGuidedCrop,
    DualRandomCrop,
    DualResizePad,
)


def _make_rgb_image(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _nonzero_region(image: Image.Image) -> tuple[int, int, int, int]:
    tensor = torch.as_tensor(list(image.getdata()), dtype=torch.uint8).reshape(
        image.height, image.width, 3
    )
    mask = tensor.any(dim=-1)
    ys, xs = torch.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _target_with_boxes() -> dict:
    boxes = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0],
            [60.0, 10.0, 90.0, 30.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    return {
        "boxes": boxes,
        "labels": labels,
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "iscrowd": torch.zeros((2,), dtype=torch.int64),
        "size": torch.tensor([80, 100]),
        "orig_size": torch.tensor([80, 100]),
    }


def test_resize_pad_keeps_uv_white_geometry_synchronized() -> None:
    transform = DualResizePad([64])
    img_uv = _make_rgb_image((40, 20), (255, 0, 0))
    img_white = _make_rgb_image((40, 20), (0, 255, 0))
    target = {
        "boxes": torch.tensor([[5.0, 4.0, 15.0, 14.0]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "area": torch.tensor([100.0], dtype=torch.float32),
        "iscrowd": torch.zeros((1,), dtype=torch.int64),
        "size": torch.tensor([20, 40]),
    }

    out_uv, out_white, out_target = transform(img_uv, img_white, target)

    assert out_uv.size == (64, 64)
    assert out_white.size == (64, 64)
    assert _nonzero_region(out_uv) == _nonzero_region(out_white)
    assert torch.equal(out_target["size"], torch.tensor([64, 64]))
    assert torch.allclose(
        out_target["boxes"],
        torch.tensor([[17.0, 26.0, 27.0, 36.0]], dtype=torch.float32),
    )


def test_pml_guided_crop_uses_label_one_and_preserves_pair_alignment() -> None:
    transform = DualPMLGuidedCrop(label=1, margin_ratio=(0.0, 0.0))
    img_uv = _make_rgb_image((100, 80), (255, 0, 0))
    img_white = _make_rgb_image((100, 80), (0, 255, 0))

    out_uv, out_white, out_target = transform(img_uv, img_white, _target_with_boxes())

    assert out_uv.size == (20, 20)
    assert out_white.size == (20, 20)
    assert _nonzero_region(out_uv) == _nonzero_region(out_white)
    assert out_target["labels"].tolist() == [1]
    assert torch.allclose(
        out_target["boxes"],
        torch.tensor([[0.0, 0.0, 20.0, 20.0]], dtype=torch.float32),
    )


def test_random_crop_preserves_pair_alignment_and_valid_boxes() -> None:
    random.seed(3)
    transform = DualRandomCrop(min_size=30, max_size=50)
    img_uv = _make_rgb_image((100, 80), (255, 0, 0))
    img_white = _make_rgb_image((100, 80), (0, 255, 0))

    out_uv, out_white, out_target = transform(img_uv, img_white, _target_with_boxes())

    assert out_uv.size == out_white.size
    assert _nonzero_region(out_uv) == _nonzero_region(out_white)
    assert out_target["boxes"].shape[1] == 4
    assert torch.all(out_target["boxes"][:, 0::2] >= 0)
    assert torch.all(out_target["boxes"][:, 1::2] >= 0)
    assert torch.all(out_target["boxes"][:, 0::2] <= out_uv.width)
    assert torch.all(out_target["boxes"][:, 1::2] <= out_uv.height)


def test_industrial_photometric_jitter_changes_pixels_only() -> None:
    transform = DualIndustrialPhotometricJitter(
        p=1.0,
        uv_brightness=(1.1, 1.1),
        uv_contrast=(1.0, 1.0),
        white_brightness=(0.9, 0.9),
        white_contrast=(1.0, 1.0),
        white_blur_p=0.0,
        white_noise_p=0.0,
    )
    img_uv = _make_rgb_image((16, 16), (100, 100, 100))
    img_white = _make_rgb_image((16, 16), (120, 120, 120))
    target = _target_with_boxes()

    out_uv, out_white, out_target = transform(img_uv, img_white, target)

    assert out_uv.size == img_uv.size
    assert out_white.size == img_white.size
    assert out_uv.getpixel((0, 0))[0] > img_uv.getpixel((0, 0))[0]
    assert out_white.getpixel((0, 0))[0] < img_white.getpixel((0, 0))[0]
    assert out_target is target
    assert torch.equal(out_target["boxes"], target["boxes"])
