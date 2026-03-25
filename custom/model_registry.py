"""
文件说明：该文件用于统一管理 `eval/model/` 目录下的模型权重路径解析。
功能说明：负责创建模型目录、处理常用别名，并在评估/检测脚本中提供一致的权重定位逻辑。

结构概览：
  第一部分：导入依赖与路径常量
  第二部分：模型名称归一化
  第三部分：模型路径解析与目录准备
"""

from __future__ import annotations

from pathlib import Path


# ========== 第一部分：导入依赖与路径常量 ==========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval"
MODEL_DIR = EVAL_DIR / "model"

# 常用别名集中放在这里，避免脚本层各自散落硬编码。
MODEL_NAME_ALIASES = {
    "bestema": "checkpoint_best_ema.pth",
    "bestema.pth": "checkpoint_best_ema.pth",
    "kimi": "kimi.pth",
    "menkong": "menkong.pth",
    "uv_single": "uv_single.pth",
}


# ========== 第二部分：模型名称归一化 ==========
def normalize_model_name(model_name: str) -> str:
    """
    统一把常用别名映射回真实权重文件名。

    这里保留“未知名称原样返回”的策略，
    是为了支持用户后续手工放入新权重并直接通过 `--model-name` 指向它。
    """
    clean_name = model_name.strip()
    if not clean_name:
        raise ValueError("`--model-name` 不能为空。")
    return MODEL_NAME_ALIASES.get(clean_name, clean_name)


# ========== 第三部分：模型路径解析与目录准备 ==========
def ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR.resolve()


def resolve_model_checkpoint_path(model_name: str) -> Path:
    """
    优先从 `eval/model/` 读取权重，同时保留旧路径兜底。

    保留 `eval/` 和仓库根目录的兼容读取，
    是为了让本轮目录迁移后仍能兼容少量尚未移动的旧权重。
    """
    normalized_name = normalize_model_name(model_name=model_name)
    candidate_paths = [
        MODEL_DIR / normalized_name,
        EVAL_DIR / normalized_name,
        PROJECT_ROOT / normalized_name,
    ]

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    checked_locations = "、".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(
        f"未找到模型权重 `{normalized_name}`。已检查：{checked_locations}"
    )
