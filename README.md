# RF-DETR 双模态白粉病检测

> 基于 [RF-DETR](https://github.com/roboflow/rf-detr) 的双模态（UV + 白光）植物白粉病目标检测实验项目。

---

## 项目简介

本项目在 RF-DETR 原始框架上扩展了**双模态输入**能力，用于检测植物叶片上的白粉病（Powdery Mildew）区域。

**采集方式：** 对同一株植物同时拍摄白光图像与 UV 图像。
- 白光图像：可清晰看到叶片轮廓和脉络。
- UV 图像：白粉病区域（PM）在 UV 下产生强烈蓝色荧光反应，便于识别。

**检测类别（3类）：**

| ID | 标签 | 含义 |
|----|------|------|
| 0 | NPML | 健康叶片区域 |
| 1 | PML  | 生病叶片（非 PM 区域） |
| 2 | PM   | 白粉病（荧光）区域 |

---

## 模型架构

```
UV 图像  ──→ DINOv2 Encoder ──→ UV 特征 [B,C,H,W]
                                        │
                                   CrossModalFusionStack
                                   (UV ← White 单向 cross-attention)
                                        │
White 图像 ─→ DINOv2 Encoder ──→ White 特征 [B,C,H,W] ─┘
                                        │
                                   Projector → Transformer → 检测头
```

- **主模态**：UV 图像（始终携带检测语义）
- **辅助模态**：白光图像（作为 cross-attention 的 key/value）
- **融合位置**：encoder 输出之后、projector 之前
- **融合机制**：`CrossModalFusionStack`，当前为 6 层的论文式 depth-attention residual 聚合：
  中间层执行 `Attn(q=h_i, k=white, v=white)`，跨层权重由 depth attention 决定，最终只在 stack 末端执行一次 FFN

---

## 目录结构

```
custom/
  cross_modal.py        # 跨模态融合模块（CrossModalFusionBlock / Stack）
  eval_entry_common.py  # 各个 run_eval_*.py 共享的入口转发层
  eval_runtime.py       # 评估内部核心实现（供根目录 eval 入口复用）
  legacy_gate_model.py  # menkong 历史门控结构兼容层
  model_registry.py     # eval/model/ 权重定位与别名解析
  dual_model.py         # 双模态模型封装（DualModalLWDETR）
  dual_dataset.py       # 双模态数据集（UV + White 配对读取）
  dual_transforms.py    # 双模态数据增强流水线
  rfdetr_compat.py      # 自定义训练/验证兼容层（对接 src/rfdetr）
  uv_dataset.py         # 单模态 UV-only 数据集（消融基线用）
  notes/                # 版本记录、实验记录与思路档案
  train/
    run_train.py        # 双模态训练启动脚本
    run_train_uv.py     # UV-only 消融训练启动脚本
detect/
  image_uv/             # 待检测的 UV 图片
  image_white/          # 待检测的 White 图片
  run_detect_kimi.py    # kimi.pth 的双模态检测入口
  run_detect_menkong.py # menkong.pth 的双模态检测入口

eval/
  model/                 # 统一存放评估/检测使用的权重文件
  run_eval_kimi.py      # kimi.pth 的专用评估入口
  run_eval_menkong.py   # menkong.pth 的专用评估入口
  run_eval_bestema.py   # checkpoint_best_ema.pth 的专用评估入口
  run_eval_uv_single.py # uv_single.pth 的专用评估入口

datasets/
  images/               # UV 图像
  images_white/         # 白光图像
  labels/               # YOLO 格式标注
  dataset_dual.yaml     # 数据集配置
```

---

## 数据集格式

采用 YOLO 格式，目录结构如下：

```
datasets/
  images/{train,val}/pair_xxxx_uv.bmp
  images_white/{train,val}/pair_xxxx_white.bmp
  labels/{train,val}/pair_xxxx_uv.txt
  dataset_dual.yaml
```

UV 图与白光图以文件名中的序号配对（`pair_xxxx`）。

训练时会自动检查 COCO 标注：
- 如果数据根目录已经有 `train/_annotations.coco.json` 和 `valid/_annotations.coco.json`，则直接复用。
- 否则会基于当前 `images/ + images_white/ + labels/ + dataset_dual.yaml` 目录自动生成一份缓存到 `datasets/_auto_coco/`。
- `uv_only` 会直接走原始 RF-DETR 的 COCO loader；`dual` 会复用同一份 COCO 标注来对齐 `image_id` 和评估。

---

## 训练

### 环境安装

```bash
# Python >= 3.10
pip install -e .
```

### 双模态训练（UV + White）

```bash
python -m custom.train.run_train
```

主要超参数在脚本顶部的**实验参数区**修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DUAL_MODAL` | `True` | 是否启用双模态 |
| `FUSION_TYPE` | `"uv_queries_white"` | 融合方式 |
| `FUSION_NUM_LAYERS` | `6` | 融合层数 |
| `EPOCHS` | `160` | 训练轮数 |
| `BATCH_SIZE` | `6` | 批大小 |
| `GRAD_ACCUM_STEPS` | `1` | 梯度累积步数 |
| `LR` | `1.2e-4` | 学习率 |

### 单模态消融训练（UV-only 基线）

```bash
python -m custom.train.run_train_uv
```

---

## 推理

```bash
python detect/run_detect_kimi.py
# 或
python detect/run_detect_menkong.py
```

使用前请先把：
- UV 图片放进 `detect/image_uv/`
- White 图片放进 `detect/image_white/`
- 对应权重放进 `eval/model/`

结果保存在 `output/detect/<时间戳>/<模型名>/`，包含：
- 可视化图像（检测框叠加在 UV 图上）
- `summary_report.json`
- `per_image_detections.json`

评估与检测权重现在统一建议放在 `eval/model/` 下。
例如：
- `eval/model/kimi.pth`
- `eval/model/menkong.pth`
- `eval/model/checkpoint_best_ema.pth`

说明：
- `rf-detr-base.pth` 是训练初始化权重，不属于评估模型池，不参与 `eval/` 入口。

---

## 数据增强（训练阶段）

| 步骤 | 增强方式 | 说明 |
|------|----------|------|
| 1 | `DualRandomHorizontalFlip` p=0.5 | UV/White 同步水平翻转 |
| 2A | `DualSquareResize` | 直接缩放（70% 概率） |
| 2B | `DualPMFocusCrop` → Resize | 围绕 PM 框裁剪后缩放（30% 概率） |
| 3 | `DualWhiteLightJitter` p=0.35 | 仅对白光做亮度/对比度扰动 |
| 4 | `DualUVFluorescenceJitter` p=0.4 | 仅对 UV 做荧光强度/gamma 扰动 |
| 5 | `DualGaussianBlur` p=0.08 | 两路独立高斯模糊 |
| 6 | `DualGaussianNoise` p=0.2 | 两路独立高斯噪声 |
| 7 | Normalize | ImageNet 均值/方差标准化 |

---

