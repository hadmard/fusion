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

from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import re


# ========== 第一部分：常量、数据结构与基础工具 ==========
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PAIR_STEM_RE = re.compile(
    r"^image_(\d{8}_\d{6})_(uv|white)_(\d+)(?:_aug(\d+))?$",
    re.IGNORECASE,
)
_WHITE_INDEX_CACHE: dict[Path, dict[tuple[int, int | None], list[tuple[datetime, Path]]]] = {}


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


def _parse_pair_stem(stem: str) -> tuple[datetime, str, int, int | None] | None:
    match = PAIR_STEM_RE.match(stem)
    if not match:
        return None

    ts_raw, modality, leaf_raw, aug_raw = match.groups()
    ts = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S")
    return ts, modality.lower(), int(leaf_raw), int(aug_raw) if aug_raw is not None else None


def _build_white_index(white_dir: Path) -> dict[tuple[int, int | None], list[tuple[datetime, Path]]]:
    cached = _WHITE_INDEX_CACHE.get(white_dir)
    if cached is not None:
        return cached

    index: dict[tuple[int, int | None], list[tuple[datetime, Path]]] = {}
    for white_path in list_image_files(white_dir):
        parsed = _parse_pair_stem(white_path.stem)
        if parsed is None:
            continue

        ts, modality, leaf_idx, aug_idx = parsed
        if modality != "white":
            continue

        key = (leaf_idx, aug_idx)
        index.setdefault(key, []).append((ts, white_path))

    for key in index:
        index[key].sort(key=lambda item: item[0])

    _WHITE_INDEX_CACHE[white_dir] = index
    return index


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
     3. 模态分层布局：
         `uv/images/<split>` + `white/images/<split>` + `uv/labels/<split>`
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

    modality_uv = root / "uv" / "images" / split
    modality_white = root / "white" / "images" / split
    modality_label = root / "uv" / "labels" / split
    if has_image_files(modality_uv):
        if (not require_white or has_image_files(modality_white)) and (
            not require_labels or modality_label.exists()
        ):
            return SplitLayout(
                dataset_root=root,
                split=split,
                uv_dir=modality_uv,
                white_dir=modality_white if modality_white.exists() else None,
                label_dir=modality_label if modality_label.exists() else None,
                layout_name="modality_subdirs",
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

    # Fallback for datasets where UV/White timestamps are not identical.
    parsed_uv = _parse_pair_stem(uv_path.stem)
    if parsed_uv is None:
        return None

    uv_ts, _modality, leaf_idx, aug_idx = parsed_uv
    white_index = _build_white_index(white_dir)

    candidates = white_index.get((leaf_idx, aug_idx), [])
    if not candidates:
        for (leaf_key, _aug_key), values in white_index.items():
            if leaf_key == leaf_idx:
                candidates.extend(values)

    if not candidates:
        return None

    _best_ts, best_path = min(
        candidates,
        key=lambda item: abs((item[0] - uv_ts).total_seconds()),
    )
    return best_path

    return None
