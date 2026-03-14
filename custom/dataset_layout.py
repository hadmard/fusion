"""
文件说明：本文件负责统一解析当前项目支持的几种数据目录结构。
功能说明：兼容原先的 `images/images_white/labels` 成对目录，以及服务器上更接近原始图片的
`train + train_m (+ train_1)` 这类目录，给训练、评估、推理提供一致的路径解析结果。

结构概览：
  第一部分：常量、数据结构与基础工具
  第二部分：split 级目录解析
  第三部分：UV/White 配对文件名解析
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ========== 第一部分：常量、数据结构与基础工具 ==========
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitLayout:
    """表示某个 split 在磁盘上的真实目录布局。"""

    dataset_root: Path
    split: str
    uv_dir: Path
    label_dir: Path | None
    white_dir: Path | None
    layout_name: str


def is_image_file(path: Path) -> bool:
    """判断文件是否是当前项目允许读取的图片。"""
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def has_image_files(directory: Path) -> bool:
    """用来判断目录是否真的是图片目录，而不是空目录或纯标签目录。"""
    if not directory.exists() or not directory.is_dir():
        return False
    return any(is_image_file(path) for path in directory.iterdir())


def list_image_files(directory: Path) -> list[Path]:
    """返回目录下的图片文件列表。"""
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if is_image_file(path))


# ========== 第二部分：split 级目录解析 ==========
def resolve_split_layout(
    dataset_dir: str | Path,
    split: str,
    *,
    require_white: bool,
    require_labels: bool,
) -> SplitLayout:
    """
    自动识别给定 split 的目录布局。

    当前支持：
    1. 标准训练布局：
       `images/<split>` + `images_white/<split>` + `labels/<split>`
    2. 服务器图片布局：
       `<split>` + `<split>_m`，标签优先在 `<split>` 同目录，次选 `<split>_1`
    """
    root = Path(dataset_dir).resolve()

    standard_uv = root / "images" / split
    standard_white = root / "images_white" / split
    standard_label = root / "labels" / split
    if has_image_files(standard_uv):
        if (not require_white or has_image_files(standard_white)) and (
            not require_labels or standard_label.exists()
        ):
            return SplitLayout(
                dataset_root=root,
                split=split,
                uv_dir=standard_uv,
                white_dir=standard_white if standard_white.exists() else None,
                label_dir=standard_label if standard_label.exists() else None,
                layout_name="standard_paired",
            )

    flat_uv_candidates = [root / split, root / f"{split}_1"]
    flat_white_candidates = [root / f"{split}_m", root / f"{split}_white", standard_white]

    for uv_dir in flat_uv_candidates:
        if not has_image_files(uv_dir):
            continue

        label_candidates = [uv_dir, root / f"{split}_1", standard_label]
        white_dir = next((path for path in flat_white_candidates if has_image_files(path)), None)
        label_dir = next((path for path in label_candidates if path.exists()), None)

        if require_white and white_dir is None:
            continue
        if require_labels and label_dir is None:
            continue

        return SplitLayout(
            dataset_root=root,
            split=split,
            uv_dir=uv_dir,
            white_dir=white_dir,
            label_dir=label_dir,
            layout_name="flat_split_dirs",
        )

    raise FileNotFoundError(
        "无法识别数据目录结构。"
        f" dataset_root={root}, split={split}, require_white={require_white},"
        f" require_labels={require_labels}"
    )


# ========== 第三部分：UV/White 配对文件名解析 ==========
def is_probable_uv_image(image_path: Path) -> bool:
    """
    判断一张图片是否应视为 UV 图。

    之所以不强制要求文件名里必须出现 `_uv`，是因为历史数据里存在 UV 图不带 `_uv` 标识、
    但 white 图带 `_white` 标识的批次。
    """
    lower_stem = image_path.stem.lower()
    return "white" not in lower_stem


def resolve_white_path_for_uv(uv_path: Path, white_dir: Path | None) -> Path | None:
    """根据 UV 文件名推断对应 white 图片路径。"""
    if white_dir is None:
        return None

    stem = uv_path.stem
    suffix = uv_path.suffix
    candidate_stems = []

    if "_uv_" in stem.lower():
        candidate_stems.append(stem.replace("_uv_", "_white_"))
        candidate_stems.append(stem.replace("_UV_", "_white_"))
    if stem.lower().endswith("_uv"):
        candidate_stems.append(stem[:-3] + "_white")
    if stem.lower().endswith("uv"):
        candidate_stems.append(stem[:-2] + "white")

    candidate_stems.append(stem)

    seen: set[str] = set()
    for candidate_stem in candidate_stems:
        if candidate_stem in seen:
            continue
        seen.add(candidate_stem)

        candidate_path = white_dir / f"{candidate_stem}{suffix}"
        if candidate_path.exists():
            return candidate_path

    return None
