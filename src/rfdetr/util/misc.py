# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""旧版 `rfdetr.util.misc` 兼容层。

文件说明：本文件为仍依赖旧 `util.misc` 路径的调用方提供兼容导出。
功能说明：除了把已经迁移到 `utilities` / `models.math` 的符号重新导出之外，
还补回旧训练循环仍会直接用到的 `SmoothedValue`、`MetricLogger` 和
`init_distributed_mode` 等辅助工具，保证 custom 训练入口在新版源码布局下仍可运行。

结构概览：
  第一部分：弃用提示与符号转发
  第二部分：旧训练循环仍依赖的日志/分布式辅助类
"""

from __future__ import annotations

import datetime
import os
import time
from collections import defaultdict, deque
from typing import Any, Generator, Iterable, Optional

from rfdetr.utilities.decorators import _warn_deprecated_module
import torch

_warn_deprecated_module("rfdetr.util.misc", "rfdetr.utilities")

# Re-export symbols that have moved to utilities/.
# Re-export math functions from their canonical location in rfdetr.models.math.
from rfdetr.models.math import accuracy, interpolate, inverse_sigmoid  # noqa: F401, E402
from rfdetr.utilities.distributed import (  # noqa: F401, E402
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from rfdetr.utilities.package import get_sha  # noqa: F401, E402
from rfdetr.utilities.state_dict import strip_checkpoint  # noqa: F401, E402
from rfdetr.utilities.tensors import (  # noqa: E402, F401
    NestedTensor,
    collate_fn,
    nested_tensor_from_tensor_list,
)


# ========== 第二部分：旧训练循环仍依赖的日志/分布式辅助类 ==========
class SmoothedValue:
    """跟踪一串数值，并提供窗口均值与全局均值。"""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None) -> None:
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        # 新版仓库默认单机训练；只有在外部显式启用分布式时才做同步。
        if not is_dist_avail_and_initialized():
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tensor = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        torch.distributed.barrier()
        torch.distributed.all_reduce(tensor)
        self.count = int(tensor[0].item())
        self.total = float(tensor[1].item())

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """旧训练循环使用的轻量日志聚合器。"""

    def __init__(self, delimiter: str = "\t") -> None:
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, (float, int)):
                self.meters[key].update(value)

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {attr!r}")

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable: Iterable[Any], print_freq: int, header: Optional[str] = None) -> Generator[Any, None, None]:
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        size = len(iterable)
        space_fmt = ":" + str(len(str(size))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        mb = 1024.0 * 1024.0
        for index, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if index % print_freq == 0 or index == size - 1:
                eta_seconds = iter_time.global_avg * (size - index)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            index,
                            size,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / mb,
                        )
                    )
                else:
                    print(log_msg.format(index, size, eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)))
            end = time.time()
        total_time = time.time() - start_time
        print(f"{header} Total time: {str(datetime.timedelta(seconds=int(total_time)))} ({total_time / max(size, 1):.4f} s / it)")


def init_distributed_mode(args: Any) -> None:
    """兼容旧接口的最保守分布式初始化。

    当前仓库的 custom 训练默认按单机单进程运行；若环境变量中显式提供
    `RANK/WORLD_SIZE`，则仍按旧约定初始化 torch.distributed。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
        args.dist_backend = "nccl"
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()
        return

    print("Not using distributed mode")
    args.rank = 0
    args.gpu = 0
    args.world_size = 1
    args.distributed = False
