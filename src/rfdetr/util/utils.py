"""
文件说明：本文件保留旧 `rfdetr.util.utils` 常用工具的直接实现。
功能说明：兼容仍在使用旧训练入口的代码，同时避免因为导入 `rfdetr.training`
而额外依赖 `pytorch_lightning`。

结构概览：
  第一部分：导入依赖
  第二部分：EMA 模型
  第三部分：best metric 跟踪
  第四部分：其余兼容转发
"""

import json
import math
from copy import deepcopy
from typing import Callable, Dict, Optional, Union

import torch

from rfdetr.utilities.reproducibility import seed_all  # noqa: F401
from rfdetr.utilities.state_dict import clean_state_dict  # noqa: F401


# ========== 第二部分：EMA 模型 ==========
class ModelEma(torch.nn.Module):
    """对模型权重做指数滑动平均。"""

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9997,
        tau: float = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.tau = tau
        self.updates = 1
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _get_decay(self) -> float:
        if self.tau == 0:
            return self.decay
        return self.decay * (1 - math.exp(-self.updates / self.tau))

    def _update(self, model: torch.nn.Module, update_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> None:
        with torch.no_grad():
            for ema_value, model_value in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_value = model_value.to(device=self.device)
                ema_value.copy_(update_fn(ema_value, model_value))

    def update(self, model: torch.nn.Module) -> None:
        decay = self._get_decay()
        self._update(model, update_fn=lambda ema_value, model_value: decay * ema_value + (1.0 - decay) * model_value)
        self.updates += 1

    def set(self, model: torch.nn.Module) -> None:
        self._update(model, update_fn=lambda _ema_value, model_value: model_value)


# ========== 第三部分：best metric 跟踪 ==========
class BestMetricSingle:
    def __init__(self, init_res: float = 0.0, better: str = "large") -> None:
        self.best_res = init_res
        self.best_ep = -1
        self.better = better
        assert better in {"large", "small"}

    def isbetter(self, new_res: float, old_res: float) -> bool:
        return new_res > old_res if self.better == "large" else new_res < old_res

    def update(self, new_res: float, ep: int) -> bool:
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def summary(self) -> Dict[str, Union[float, int]]:
        return {"best_res": self.best_res, "best_ep": self.best_ep}


class BestMetricHolder:
    def __init__(self, init_res: float = 0.0, better: str = "large", use_ema: bool = False) -> None:
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res: float, epoch: int, is_ema: bool = False) -> bool:
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        if is_ema:
            self.best_ema.update(new_res, epoch)
        else:
            self.best_regular.update(new_res, epoch)
        return self.best_all.update(new_res, epoch)

    def summary(self) -> Dict[str, Union[float, int]]:
        if not self.use_ema:
            return self.best_all.summary()
        result = {}
        result.update({f"all_{key}": value for key, value in self.best_all.summary().items()})
        result.update({f"regular_{key}": value for key, value in self.best_regular.summary().items()})
        result.update({f"ema_{key}": value for key, value in self.best_ema.summary().items()})
        return result

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()
