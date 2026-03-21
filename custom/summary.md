# 叠加融合 权重结果以及跟之前的对比

## 1. 评估对象

- 权重文件：`kimi.pth`
- 测试集：`datasets/test_uv` + `datasets/test_white` + `datasets/test_label`
- 评估脚本：`custom/detect/run_test_pair_report.py`
- 输出结果：`output/test_pair_report/2026-03-21_205732/summary_report.json`

## 2. 评估规格

- AP 指标：
  - 使用 `pycocotools.COCOeval`
  - `maxDets = [1, 10, 100]`
  - 输出 `AP50`、`AP75`、`AP50-95`
- Precision / Recall / F1：
  - 定义为 `IoU=0.5, score=0.5, maxDets=100`
  - 输出 `Precision@0.5`、`Recall@0.5`、`F1@0.5`
- FPS：
  - 使用 `torch.cuda.synchronize()` + `time.perf_counter()` 计时
- GFLOPs：
  - 基于 `thop`
  - 采用 `FLOPs ≈ 2 * MACs`
  - 属于近似值

## 3. 叠加融合 总体结果

| Model | Modality | AP50(%) | AP75(%) | AP50-95(%) | Precision@0.5 | Recall@0.5 | F1@0.5 | GFLOPs | Parameters (total) | FPS | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kimi.pth | Dual-modal | 82.94 | 65.71 | 63.10 | 0.8853 | 0.7593 | 0.8175 | 162.63 | 50,829,568 | 10.49 | 569.11 MB |

## 4. 叠加融合 分类别结果

| Class | AP50(%) | AP75(%) | AP50-95(%) | Precision@0.5 | Recall@0.5 | F1@0.5 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 94.49 | 87.83 | 80.02 | 0.9005 | 0.8668 | 0.8834 | 1367 | 151 | 210 |
| PML | 89.53 | 83.65 | 78.57 | 0.8566 | 0.7963 | 0.8253 | 215 | 36 | 55 |
| PM | 64.82 | 25.64 | 30.70 | 0.8431 | 0.4847 | 0.6155 | 317 | 59 | 337 |

## 5. 与 单融合门控 / 单模态 的并排对比

| Model | Modality | AP50(%) | AP75(%) | AP50-95(%) | Precision@0.5 | Recall@0.5 | F1@0.5 | GFLOPs | Parameters (total) | FPS | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| menkong.pth | Dual-modal | 82.27 | 65.22 | 63.38 | 0.8467 | 0.8105 | 0.8282 | 162.56 | 39,275,466 | 16.69 | 436.67 MB |
| uv_single.pth | UV-only | 82.82 | 66.25 | 63.64 | 0.9007 | 0.7469 | 0.8166 | 77.91 | 32,174,530 | 26.20 | 355.32 MB |
| kimi.pth | Dual-modal | 82.94 | 65.71 | 63.10 | 0.8853 | 0.7593 | 0.8175 | 162.63 | 50,829,568 | 10.49 | 569.11 MB |
