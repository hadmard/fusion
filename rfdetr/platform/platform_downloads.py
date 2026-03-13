# ------------------------------------------------------------------------
# Platform Model License 1.0 (PML-1.0)
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
#
# Licensed under the Platform Model License 1.0.
# Use, modification, and distribution of code and checkpoints require
# an active Roboflow platform plan or agreement.
#
# See the LICENSE.platform file for full terms and conditions.
# ------------------------------------------------------------------------

"""
文件说明：
    列出平台许可模型（Platform Models）对应的下载 URL。仅在用户接受平台模型许可并且具备相应访问权限时使用这些模型权重。

结构概览：
    - `PLATFORM_MODELS`：字典，key 为权重文件名，value 为下载 URL

注意：此文件仅包含 URL 映射，未实现下载逻辑；已经在 `rfdetr.main` 中使用 `download_file` 进行实际下载。
"""

PLATFORM_MODELS = {
        "rf-detr-xlarge.pth": "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xlarge.pth",
        "rf-detr-xxlarge.pth": "https://storage.googleapis.com/rfdetr/platform-licensed/rf-detr-xxlarge.pth",
}
