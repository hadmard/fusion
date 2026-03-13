# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
文件说明：
    提供小型文件下载工具（带进度条），用于下载预训练权重或其它辅助文件。

结构概览：
    第一部分：导入依赖（requests, tqdm）
    第二部分：`download_file` 函数实现，按流方式写入并显示下载进度

注意：此修改仅添加中文注释，不修改下载实现。
"""

import requests
from tqdm import tqdm


def download_file(url: str, filename: str) -> None:
    response = requests.get(url, stream=True)
    total_size = int(response.headers['content-length'])
    with open(filename, "wb") as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
