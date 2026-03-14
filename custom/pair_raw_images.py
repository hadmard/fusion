"""
文件说明：本文件用于把原始采集图片按时间顺序配对，并整理成训练脚本可直接使用的标准命名。
功能说明：扫描原始目录中的 white/uv 图片，按拍摄时间和曝光编号筛选同一叶片的一对白光与 UV 图片，
再输出为 `pair_0001_white.xxx` / `pair_0001_uv.xxx` 这类命名，同时生成配对索引 CSV。

结构概览：
  第一部分：常量、数据结构与文件名解析
  第二部分：配对规则与索引生成
  第三部分：目录处理与文件输出
  第四部分：命令行入口
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


# ========== 第一部分：常量、数据结构与文件名解析 ==========
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FILENAME_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_(?P<date>\d{8})_(?P<time>\d{6})_(?P<modality>white|uv)_(?P<exposure>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageRecord:
    """表示一张可参与配对的图片记录。"""

    path: Path
    stem: str
    timestamp: datetime
    modality: str
    exposure: int


@dataclass(frozen=True)
class PairedRecord:
    """表示一对白光/UV 图片的输出计划。"""

    pair_index: int
    split: str
    white: ImageRecord
    uv: ImageRecord
    time_diff_seconds: float
    note: str


def parse_image_record(image_path: Path) -> ImageRecord | None:
    """从文件名中提取时间、模态和曝光编号；不符合约定的文件直接跳过。"""
    if image_path.suffix.lower() not in IMAGE_SUFFIXES:
        return None

    match = FILENAME_PATTERN.match(image_path.stem)
    if match is None:
        return None

    timestamp = datetime.strptime(
        f"{match.group('date')}_{match.group('time')}",
        "%Y%m%d_%H%M%S",
    )
    return ImageRecord(
        path=image_path,
        stem=image_path.stem,
        timestamp=timestamp,
        modality=match.group("modality").lower(),
        exposure=int(match.group("exposure")),
    )


def collect_split_records(split_dir: Path) -> list[ImageRecord]:
    """读取单个 split 目录下所有符合命名规则的原始图片。"""
    records: list[ImageRecord] = []
    for image_path in sorted(split_dir.rglob("*")):
        if not image_path.is_file():
            continue
        record = parse_image_record(image_path)
        if record is not None:
            records.append(record)
    return sorted(records, key=lambda item: (item.timestamp, item.modality, item.path.name))


# ========== 第二部分：配对规则与索引生成 ==========
def filter_records_by_exposure(
    records: Iterable[ImageRecord],
    white_exposure: int,
    uv_exposure: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """先按曝光筛选，避免同一叶片的双曝光同时参与后续配对。"""
    white_records = [
        record
        for record in records
        if record.modality == "white" and record.exposure == white_exposure
    ]
    uv_records = [
        record for record in records if record.modality == "uv" and record.exposure == uv_exposure
    ]
    return white_records, uv_records


def pair_split_records(
    split: str,
    white_records: list[ImageRecord],
    uv_records: list[ImageRecord],
    start_index: int,
    max_gap_seconds: float,
    warn_gap_seconds: float,
) -> tuple[list[PairedRecord], list[str], int]:
    """
    按“先 white、后 uv”的采集顺序做贪心配对。

    之所以不用全局最优匹配，是因为采集流程是顺序拍摄，同一叶片的 white/uv 理论上在时间轴上
    邻近且方向固定。贪心顺序匹配更直观，也更容易人工复核。
    """
    paired_records: list[PairedRecord] = []
    warnings: list[str] = []
    uv_cursor = 0
    next_pair_index = start_index

    for white_record in white_records:
        while uv_cursor < len(uv_records) and uv_records[uv_cursor].timestamp < white_record.timestamp:
            warnings.append(
                f"[{split}] 跳过未匹配 UV：{uv_records[uv_cursor].path.name}，原因：时间早于候选 white。"
            )
            uv_cursor += 1

        if uv_cursor >= len(uv_records):
            warnings.append(f"[{split}] 未找到与 {white_record.path.name} 对应的 UV 图片。")
            continue

        uv_record = uv_records[uv_cursor]
        time_diff_seconds = (uv_record.timestamp - white_record.timestamp).total_seconds()
        if time_diff_seconds < 0:
            warnings.append(
                f"[{split}] 配对顺序异常：{white_record.path.name} 晚于 {uv_record.path.name}。"
            )
            uv_cursor += 1
            continue

        if time_diff_seconds > max_gap_seconds:
            warnings.append(
                f"[{split}] 未配对 {white_record.path.name}，原因：最近 UV {uv_record.path.name} "
                f"时间差 {time_diff_seconds:.1f}s 超过阈值 {max_gap_seconds:.1f}s。"
            )
            continue

        note = ""
        if time_diff_seconds > warn_gap_seconds:
            note = f"时间差超过 {warn_gap_seconds:.1f} 秒，请复核"

        paired_records.append(
            PairedRecord(
                pair_index=next_pair_index,
                split=split,
                white=white_record,
                uv=uv_record,
                time_diff_seconds=time_diff_seconds,
                note=note,
            )
        )
        next_pair_index += 1
        uv_cursor += 1

    for remaining_uv in uv_records[uv_cursor:]:
        warnings.append(f"[{split}] 跳过未匹配 UV：{remaining_uv.path.name}，原因：没有后续 white 与之对应。")

    return paired_records, warnings, next_pair_index


# ========== 第三部分：目录处理与文件输出 ==========
def ensure_output_dirs(output_root: Path, split: str) -> tuple[Path, Path]:
    """建立 UV 与 white 的标准输出目录。"""
    uv_dir = output_root / "images" / split
    white_dir = output_root / "images_white" / split
    uv_dir.mkdir(parents=True, exist_ok=True)
    white_dir.mkdir(parents=True, exist_ok=True)
    return uv_dir, white_dir


def copy_or_move_file(source: Path, destination: Path, move_files: bool) -> None:
    """默认复制，避免原始数据被不可逆改动；仅在显式指定时移动。"""
    if move_files:
        shutil.move(str(source), str(destination))
        return
    shutil.copy2(source, destination)


def write_pair_files(
    paired_records: Iterable[PairedRecord],
    output_root: Path,
    move_files: bool,
) -> list[dict[str, str]]:
    """把配对结果写入标准目录，并返回 CSV 索引记录。"""
    rows: list[dict[str, str]] = []

    for paired_record in paired_records:
        uv_dir, white_dir = ensure_output_dirs(output_root, paired_record.split)
        pair_name = f"pair_{paired_record.pair_index:04d}"

        uv_filename = f"{pair_name}_uv{paired_record.uv.path.suffix.lower()}"
        white_filename = f"{pair_name}_white{paired_record.white.path.suffix.lower()}"

        uv_output_path = uv_dir / uv_filename
        white_output_path = white_dir / white_filename

        copy_or_move_file(paired_record.uv.path, uv_output_path, move_files)
        copy_or_move_file(paired_record.white.path, white_output_path, move_files)

        rows.append(
            {
                "pair_index": str(paired_record.pair_index),
                "split": paired_record.split,
                "uv_original": str(paired_record.uv.path),
                "white_original": str(paired_record.white.path),
                "uv_new": str(uv_output_path),
                "white_new": str(white_output_path),
                "time_diff_seconds": f"{paired_record.time_diff_seconds:.1f}",
                "notes": paired_record.note,
            }
        )

    return rows


def write_pair_index_csv(output_root: Path, rows: list[dict[str, str]]) -> Path:
    """输出配对结果索引，方便后续人工核对异常时间差。"""
    csv_path = output_root / "pair_index.csv"
    fieldnames = [
        "pair_index",
        "split",
        "uv_original",
        "white_original",
        "uv_new",
        "white_new",
        "time_diff_seconds",
        "notes",
    ]

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def resolve_split_dirs(input_root: Path, splits: list[str] | None) -> list[Path]:
    """优先按用户指定 split 处理；未指定时默认读取输入根目录下一级子目录。"""
    if splits:
        split_dirs = [input_root / split for split in splits if (input_root / split).exists()]
    else:
        split_dirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]

    if not split_dirs:
        raise FileNotFoundError(f"在 {input_root} 下未找到可处理的 split 目录。")
    return split_dirs


def run_pairing(
    input_root: Path,
    output_root: Path,
    splits: list[str] | None,
    white_exposure: int,
    uv_exposure: int,
    max_gap_seconds: float,
    warn_gap_seconds: float,
    move_files: bool,
) -> tuple[int, Path, list[str]]:
    """执行整批目录的配对与标准命名流程。"""
    split_dirs = resolve_split_dirs(input_root, splits)
    all_rows: list[dict[str, str]] = []
    all_warnings: list[str] = []
    next_pair_index = 1

    for split_dir in split_dirs:
        split = split_dir.name
        records = collect_split_records(split_dir)
        white_records, uv_records = filter_records_by_exposure(records, white_exposure, uv_exposure)
        paired_records, warnings, next_pair_index = pair_split_records(
            split=split,
            white_records=white_records,
            uv_records=uv_records,
            start_index=next_pair_index,
            max_gap_seconds=max_gap_seconds,
            warn_gap_seconds=warn_gap_seconds,
        )
        all_rows.extend(write_pair_files(paired_records, output_root, move_files))
        all_warnings.extend(warnings)

    csv_path = write_pair_index_csv(output_root, all_rows)
    return len(all_rows), csv_path, all_warnings


# ========== 第四部分：命令行入口 ==========
def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数，便于不同批次数据复用同一脚本。"""
    parser = argparse.ArgumentParser(
        description="按时间把原始 white/uv 图片配成对，并整理为 pair_xxxx 命名。"
    )
    parser.add_argument("--input-root", type=Path, required=True, help="原始图片根目录。")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="输出目录，将生成 images/<split> 与 images_white/<split>。",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="要处理的 split 名称，例如 train val test；不填则处理输入根目录下所有子目录。",
    )
    parser.add_argument(
        "--white-exposure",
        type=int,
        default=1,
        help="白光保留的曝光编号，默认 1。",
    )
    parser.add_argument(
        "--uv-exposure",
        type=int,
        default=1,
        help="UV 保留的曝光编号，默认 1。",
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=30.0,
        help="允许 white 到 uv 的最大时间差，超过则不配对，默认 30 秒。",
    )
    parser.add_argument(
        "--warn-gap-seconds",
        type=float,
        default=15.0,
        help="时间差超过该值时只标记到 CSV 备注，默认 15 秒。",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="默认复制文件；显式加上该参数时改为移动原图。",
    )
    return parser


def main() -> int:
    """命令行入口。"""
    args = build_parser().parse_args()

    paired_count, csv_path, warnings = run_pairing(
        input_root=args.input_root.resolve(),
        output_root=args.output_root.resolve(),
        splits=args.splits,
        white_exposure=args.white_exposure,
        uv_exposure=args.uv_exposure,
        max_gap_seconds=args.max_gap_seconds,
        warn_gap_seconds=args.warn_gap_seconds,
        move_files=args.move,
    )

    print(f"配对完成，共生成 {paired_count} 对图片。")
    print(f"索引文件：{csv_path}")
    if warnings:
        print("以下项目需要人工复核：")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
