## 三模型统一评估总表

- 额外前置说明：`kimi.pth` 是当前 6 层 `attention_residual` 无门控 cross-attention 融合版本，`menkong.pth` 是早期带门控的 cross-attention 融合版本，`uv_single.pth` 是仅使用紫外光输入的单模态版本。

## 评估规格

- 测试集：`D:\desktop\fusion--\test_uv`、`D:\desktop\fusion--\test_white`、`D:\desktop\fusion--\test_label`
- 测试样本：83 对配对图像
- AP 评估：`pycocotools.COCOeval`，`iouType=bbox`，`maxDets=[1, 10, 100]`
- Precision / Recall / F1：固定 `IoU=0.5`，遍历 confidence threshold 形成 PR 曲线，取总体 F1 最大点对应阈值
- FPS：先 warmup，再用 `torch.cuda.synchronize()` + `time.perf_counter()` 统计整体平均吞吐
- GFLOPs：使用 `thop` 统计 `MACs`，再按 `GFLOPs = 2 * MACs / 1e9` 换算
- Parameters(total)：直接统计模型参数总数
- Model Size：直接读取 `.pth` 权重文件磁盘体积，不是按参数量反推

## 各指标测算方式

- `AP50(%)`
  - 使用 `COCOeval` 的 bbox 指标。
  - 固定 `maxDets=100`。
  - 取 `IoU=0.50` 时的 AP，并乘以 100 转成百分比。
- `AP75(%)`
  - 使用 `COCOeval` 的 bbox 指标。
  - 固定 `maxDets=100`。
  - 取 `IoU=0.75` 时的 AP，并乘以 100 转成百分比。
- `AP50-95(%)`
  - 即 COCO 标准 `mAP`。
  - 在 `IoU=0.50:0.95`、步长 `0.05` 上做平均。
  - 固定 `maxDets=100`。
  - 最终乘以 100 转成百分比。
- `Precision`
  - 固定 `IoU=0.5`。
  - 遍历模型输出中的 confidence threshold，形成整条 PR 曲线。
  - 取总体 `F1` 最大时对应的最优 confidence threshold。
  - 在该阈值下统计 `TP`、`FP`、`FN`，并按 `TP / (TP + FP)` 计算。
- `Recall`
  - 与 `Precision` 使用同一个最优 confidence threshold。
  - 固定 `IoU=0.5`。
  - 按 `TP / (TP + FN)` 计算。
- `F1-Score`
  - 与 `Precision`、`Recall` 使用同一个最优 confidence threshold。
  - 固定 `IoU=0.5`。
  - 按 `2 * Precision * Recall / (Precision + Recall)` 计算。
- `Best Confidence`
  - 指在 `IoU=0.5` 下，遍历 confidence threshold 得到的 PR 曲线中，使总体 `F1` 最大的那个阈值。
- `GFLOPs`
  - 使用 `thop.profile(...)` 对模型单次前向做复杂度统计。
  - 先构造与模型分辨率一致的假输入张量，再跑一遍前向。
  - `thop` 返回 `MACs` 后，按 `GFLOPs = 2 * MACs / 1e9` 换算。
  - 这是理论近似值，某些自定义算子可能带来少量误差。
  - 双模态模型的 `GFLOPs` 表示“一对 UV + White 样本”的单次前向计算量；单模态模型表示“一张 UV 样本”的单次前向计算量。
- `Parameters(total)`
  - 直接统计模型中所有参数张量的元素总数。
  - 计算方式等价于：`sum(parameter.numel() for parameter in model.parameters())`
  - 这是模型真实参数量，不是估算值。
- `FPS`
  - 先对首个样本做 warmup，不计入正式统计。
  - 每次正式计时：
    - 如果使用 CUDA，前向前先 `torch.cuda.synchronize()`
    - 使用 `time.perf_counter()` 记录开始时间
    - 执行一次完整前向推理
    - 如果使用 CUDA，前向后再次 `torch.cuda.synchronize()`
    - 使用 `time.perf_counter()` 记录结束时间
  - 最终按 `总样本数 / 总耗时` 计算。
  - 这里的双模态 `FPS` 是“每秒处理多少对样本”，不是“每秒多少张原始图像”。
- `Model Size`
  - 直接读取 checkpoint 文件的磁盘体积。
  - 计算方式等价于：`checkpoint_path.stat().st_size / (1024 * 1024)`
  - 这个值包含 `.pth` 文件中实际保存的全部内容，通常不等于“参数量 × 4 字节”。

## 总体结果

| Model | Modality | Architecture | AP50(%) | AP75(%) | AP50-95(%) | Precision | Recall | F1-Score | GFLOPs | Parameters(total) | FPS | Model Size | Best Confidence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi.pth | Dual-modal | current | 82.94 | 65.71 | 63.10 | 0.8503 | 0.7997 | 0.8242 | 162.63 | 50,829,568 | 13.17 | 569.11 MB | 0.429199 |
| menkong.pth | Dual-modal | legacy_gate | 82.47 | 65.36 | 63.47 | 0.8493 | 0.8093 | 0.8288 | 162.43 | 38,962,440 | 21.88 | 436.67 MB | 0.503906 |
| uv_single.pth | UV-only | current | 82.84 | 66.25 | 63.64 | 0.8586 | 0.7869 | 0.8212 | 77.79 | 31,861,504 | 32.86 | 355.32 MB | 0.410645 |
| checkpoint_best_ema.pth | Dual-modal | current | 81.59 | 64.85 | 62.31 | 0.8484 | 0.8145 | 0.8311 | 157.88 | 39,380,224 | 18.23 | 438.03 MB | 0.431396 |

## menkong.pth 分类别结果

| Class | AP50(%) | AP75(%) | AP50-95(%) | Precision | Recall | F1-Score | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 92.89 | 86.28 | 79.37 | 0.8869 | 0.8903 | 0.8886 | 1404 | 179 | 173 |
| PML | 87.02 | 82.44 | 78.29 | 0.7882 | 0.8407 | 0.8136 | 227 | 61 | 43 |
| PM | 67.51 | 27.36 | 32.74 | 0.7676 | 0.6009 | 0.6741 | 393 | 119 | 261 |

## kimi.pth 分类别结果

| Class | AP50(%) | AP75(%) | AP50-95(%) | Precision | Recall | F1-Score | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 94.49 | 87.83 | 80.02 | 0.8759 | 0.9042 | 0.8899 | 1426 | 202 | 151 |
| PML | 89.53 | 83.65 | 78.57 | 0.8284 | 0.8222 | 0.8253 | 222 | 46 | 48 |
| PM | 64.82 | 25.64 | 30.70 | 0.7719 | 0.5382 | 0.6342 | 352 | 104 | 302 |

## uv_single.pth 分类别结果

| Class | AP50(%) | AP75(%) | AP50-95(%) | Precision | Recall | F1-Score | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 94.24 | 87.93 | 80.11 | 0.9020 | 0.8694 | 0.8854 | 1371 | 149 | 206 |
| PML | 88.93 | 84.99 | 79.70 | 0.8057 | 0.8444 | 0.8246 | 228 | 55 | 42 |
| PM | 65.37 | 25.83 | 31.10 | 0.7546 | 0.5642 | 0.6457 | 369 | 120 | 285 |

## checkpoint_best_ema.pth 分类别结果

| Class | AP50(%) | AP75(%) | AP50-95(%) | Precision | Recall | F1-Score | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 93.64 | 86.85 | 79.52 | 0.8798 | 0.9004 | 0.8900 | 1420 | 194 | 157 |
| PML | 86.48 | 82.84 | 76.46 | 0.8156 | 0.8519 | 0.8333 | 230 | 52 | 40 |
| PM | 64.63 | 24.86 | 30.96 | 0.7663 | 0.5917 | 0.6678 | 387 | 118 | 267 |

## 简要结论

- `uv_single.pth` 在 `AP75 / AP50-95 / FPS / GFLOPs` 上最有优势。
- `menkong.pth` 在统一口径下的 `Recall / F1-Score` 最优，整体更均衡。
- `kimi.pth` 结果稳定，但在这组三模型统一对比里不再明显领先。
- `checkpoint_best_ema.pth` 是当前根目录一个独立双模态基线，整体风格也偏均衡，`Recall` 和 `F1` 较强。
- 四者共同短板仍然是 `PM` 类别。

## 对应结果文件

- 三模型旧汇总：
  - `output/multi_checkpoint_eval/2026-03-24_174549/combined_summary.md`
  - `output/multi_checkpoint_eval/2026-03-24_174549/combined_summary.json`
- `checkpoint_best_ema.pth`：
  - `output/eval/2026-03-24_224645/summary_report.json`
  - `output/eval/2026-03-24_224645/per_image_detections.json`
