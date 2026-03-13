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
- **融合机制**：`CrossModalFusionStack`，使用 zero-init 的 `alpha_attn` / `alpha_ffn` 门控，训练初期对预训练权重零干扰

---

## 目录结构

```
custom/
  cross_modal.py        # 跨模态融合模块（CrossModalFusionBlock / Stack）
  dual_model.py         # 双模态模型封装（DualModalLWDETR）
  dual_dataset.py       # 双模态数据集（UV + White 配对读取）
  dual_transforms.py    # 双模态数据增强流水线
  uv_dataset.py         # 单模态 UV-only 数据集（消融基线用）
  train/
    run_train.py        # 双模态训练启动脚本
    run_train_uv.py     # UV-only 消融训练启动脚本
  detect/
    run_detect.py       # 双模态推理 & 可视化脚本

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

---

## 训练

### 环境安装

```bash
# Python >= 3.10
pip install -e .
```

### 双模态训练（UV + White）

```bash
python custom/train/run_train.py
```

主要超参数在脚本顶部的**实验参数区**修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DUAL_MODAL` | `True` | 是否启用双模态 |
| `FUSION_TYPE` | `"uv_queries_white"` | 融合方式 |
| `FUSION_NUM_LAYERS` | `1` | 融合层数 |
| `EPOCHS` | `120` | 训练轮数 |
| `BATCH_SIZE` | `2` | 批大小 |
| `GRAD_ACCUM_STEPS` | `4` | 梯度累积步数 |
| `LR` | `1e-4` | 学习率 |

### 单模态消融训练（UV-only 基线）

```bash
python custom/train/run_train_uv.py
```

---

## 推理

```bash
python custom/detect/run_detect.py
```

结果保存在 `output/detect/<时间戳>/`，包含：
- 可视化图像（检测框叠加在 UV 图上）
- `detections.json`

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
| 7 | `DualRandomMisalignment` p=0.15 | 白光轻微平移（±2px），模拟配准误差 |
| 8 | Normalize | ImageNet 均值/方差标准化 |

---

