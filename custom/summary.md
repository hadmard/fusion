## 七模型统一评估总表

- 口径说明：
  - 本页统一使用仓库根目录 `test/uvtest`、`test/whitetest`、`test/labeltest`
  - AP 指标使用 `pycocotools.COCOeval`，`iouType=bbox`，`maxDets=[1, 10, 100]`
  - Precision / Recall / F1 固定 `IoU=0.5`，遍历 confidence threshold，取整体 F1 最大点
  - FPS 使用 warmup 后的整批推理耗时统计
  - GFLOPs 使用 `thop` 估算，按 `GFLOPs = 2 * MACs / 1e9` 换算
  - `high_resolution.pth`、`multi_feature.pth`、`kimi.pth` 继续沿用历史兼容加载后的可信结果
  - `ema_defor.pth` 为 2026-04-08 新增实测结果，走当前 `current` 结构
  - `defor_best_ema.pth` 为 2026-04-11 新增实测结果，走当前 `custom` 算法对应的 `current` 结构

## 评估规格

- 测试样本：83 对 UV/White 图像
- 统一输出格式：`summary_report.json` + `per_image_detections.json`
- Parameters(total)：直接统计模型参数总量
- Model Size：直接读取 `.pth` 权重文件磁盘大小

## 总体结果

| Model | Modality | Architecture | Resolution | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | PM AP50 | PM mAP | PM F1 | FPS | GFLOPs | Parameters(total) | Model Size | Best Confidence |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high_resolution.pth | Dual-modal | same_grid_readblocks | 672 | 0.833096 | 0.666302 | 0.639425 | 0.880783 | 0.791683 | 0.833860 | 0.663994 | 0.320285 | 0.655014 | 15.218880 | 225.683472 | 39,380,224 | 438.03 MB | 0.488037 |
| ema_defor.pth | Dual-modal | current | 672 | 0.827506 | 0.666831 | 0.639407 | 0.853576 | 0.806477 | 0.829359 | 0.667855 | 0.328558 | 0.667249 | 8.420174 | 257.090421 | 36,078,336 | 400.14 MB | 0.453369 |
| uv_single.pth | UV-only | current | 560 | 0.828435 | 0.662487 | 0.636382 | 0.858639 | 0.786885 | 0.821198 | 0.653682 | 0.311036 | 0.645669 | 28.857690 | 77.786415 | 31,861,504 | 355.32 MB | 0.410645 |
| menkong.pth | Dual-modal | legacy_gate | 560 | 0.824720 | 0.653597 | 0.634656 | 0.849350 | 0.809276 | 0.828829 | 0.675088 | 0.327386 | 0.674099 | 19.238707 | 162.433630 | 38,962,440 | 436.67 MB | 0.503906 |
| kimi.pth | Dual-modal | fusion_layers_attnres | 560 | 0.829438 | 0.657062 | 0.630986 | 0.850340 | 0.799680 | 0.824232 | 0.648180 | 0.307002 | 0.634234 | 13.160060 | 162.630238 | 50,829,568 | 569.11 MB | 0.429199 |
| multi_feature.pth | Dual-modal | same_grid_readblocks | 560 | 0.815875 | 0.648480 | 0.623125 | 0.848397 | 0.814474 | 0.831089 | 0.646341 | 0.309556 | 0.667817 | 14.854178 | 157.878878 | 39,380,224 | 438.03 MB | 0.431396 |
| defor_best_ema.pth | Dual-modal | current | 672 | 0.812928 | 0.644143 | 0.615216 | 0.863616 | 0.784886 | 0.822371 | 0.671817 | 0.326567 | 0.664855 | 8.707108 | 304.320578 | 41,060,512 | 457.41 MB | 0.550293 |

## 分类结果

### high_resolution.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.946665 | 0.884630 | 0.804067 | 0.902581 | 0.887127 | 0.894787 | 1399 | 151 | 178 |
| PML | 0.888628 | 0.849251 | 0.793924 | 0.849057 | 0.833333 | 0.841121 | 225 | 40 | 45 |
| PM | 0.663994 | 0.265024 | 0.320285 | 0.822171 | 0.544343 | 0.655014 | 356 | 77 | 298 |

### ema_defor.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.941278 | 0.878400 | 0.804650 | 0.884085 | 0.894737 | 0.889379 | 1411 | 185 | 166 |
| PML | 0.873385 | 0.828509 | 0.785015 | 0.811594 | 0.829630 | 0.820513 | 224 | 52 | 46 |
| PM | 0.667855 | 0.293583 | 0.328558 | 0.778004 | 0.584098 | 0.667249 | 382 | 109 | 272 |

### uv_single.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.942396 | 0.879283 | 0.801143 | 0.901973 | 0.869372 | 0.885373 | 1371 | 149 | 206 |
| PML | 0.889266 | 0.849881 | 0.796967 | 0.805654 | 0.844444 | 0.824593 | 228 | 55 | 42 |
| PM | 0.653682 | 0.258297 | 0.311036 | 0.754601 | 0.564220 | 0.645669 | 369 | 120 | 285 |

### menkong.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.928928 | 0.862762 | 0.793691 | 0.886860 | 0.890298 | 0.888576 | 1404 | 179 | 173 |
| PML | 0.870142 | 0.824469 | 0.782891 | 0.788194 | 0.840741 | 0.813620 | 227 | 61 | 43 |
| PM | 0.675088 | 0.273560 | 0.327386 | 0.767578 | 0.600917 | 0.674099 | 393 | 119 | 261 |

### kimi.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.944877 | 0.878343 | 0.800220 | 0.875921 | 0.904249 | 0.889860 | 1426 | 202 | 151 |
| PML | 0.895258 | 0.836482 | 0.785735 | 0.828358 | 0.822222 | 0.825279 | 222 | 46 | 48 |
| PM | 0.648180 | 0.256362 | 0.307002 | 0.771930 | 0.538226 | 0.634234 | 352 | 104 | 302 |

### multi_feature.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.936450 | 0.868511 | 0.795237 | 0.879802 | 0.900444 | 0.890003 | 1420 | 194 | 157 |
| PML | 0.864835 | 0.828359 | 0.764583 | 0.815603 | 0.851852 | 0.833333 | 230 | 52 | 40 |
| PM | 0.646341 | 0.248571 | 0.309556 | 0.766337 | 0.591743 | 0.667817 | 387 | 118 | 267 |

### defor_best_ema.pth

| Class | AP50 | AP75 | AP50-95 | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NPML | 0.914522 | 0.834081 | 0.767938 | 0.888817 | 0.871909 | 0.880282 | 1375 | 172 | 202 |
| PML | 0.852445 | 0.806943 | 0.751144 | 0.800725 | 0.818519 | 0.809524 | 221 | 55 | 49 |
| PM | 0.671817 | 0.291406 | 0.326567 | 0.815556 | 0.561162 | 0.664855 | 367 | 83 | 287 |

## 排序与对比

### 整体 AP50-95 排序

1. `high_resolution.pth` = `0.639425`
2. `ema_defor.pth` = `0.639407`
3. `uv_single.pth` = `0.636382`
4. `menkong.pth` = `0.634656`
5. `kimi.pth` = `0.630986`
6. `multi_feature.pth` = `0.623125`
7. `defor_best_ema.pth` = `0.615216`

### 整体 F1 排序

1. `high_resolution.pth` = `0.833860`
2. `multi_feature.pth` = `0.831089`
3. `ema_defor.pth` = `0.829359`
4. `menkong.pth` = `0.828829`
5. `kimi.pth` = `0.824232`
6. `defor_best_ema.pth` = `0.822371`
7. `uv_single.pth` = `0.821198`

### PM 重点排序

按 `PM AP50` 排序：

1. `menkong.pth` = `0.675088`
2. `defor_best_ema.pth` = `0.671817`
3. `ema_defor.pth` = `0.667855`
4. `high_resolution.pth` = `0.663994`
5. `uv_single.pth` = `0.653682`
6. `kimi.pth` = `0.648180`
7. `multi_feature.pth` = `0.646341`

按 `PM mAP` 排序：

1. `ema_defor.pth` = `0.328558`
2. `menkong.pth` = `0.327386`
3. `defor_best_ema.pth` = `0.326567`
4. `high_resolution.pth` = `0.320285`
5. `uv_single.pth` = `0.311036`
6. `multi_feature.pth` = `0.309556`
7. `kimi.pth` = `0.307002`

按 `PM F1` 排序：

1. `menkong.pth` = `0.674099`
2. `multi_feature.pth` = `0.667817`
3. `ema_defor.pth` = `0.667249`
4. `defor_best_ema.pth` = `0.664855`
5. `high_resolution.pth` = `0.655014`
6. `uv_single.pth` = `0.645669`
7. `kimi.pth` = `0.634234`

### 速度与成本

按 `FPS` 排序：

1. `uv_single.pth` = `28.857690`
2. `menkong.pth` = `19.238707`
3. `high_resolution.pth` = `15.218880`
4. `multi_feature.pth` = `14.854178`
5. `kimi.pth` = `13.160060`
6. `defor_best_ema.pth` = `8.707108`
7. `ema_defor.pth` = `8.420174`

按 `GFLOPs` 从低到高：

1. `uv_single.pth` = `77.786415`
2. `multi_feature.pth` = `157.878878`
3. `menkong.pth` = `162.433630`
4. `kimi.pth` = `162.630238`
5. `high_resolution.pth` = `225.683472`
6. `ema_defor.pth` = `257.090421`
7. `defor_best_ema.pth` = `304.320578`

### 保留模型直接对比

`defor_best_ema.pth` 相比 `ema_defor.pth`：

- `AP50-95` 低 `0.024191`
- `AP50` 低 `0.014578`
- `AP75` 低 `0.022687`
- `F1` 低 `0.006987`
- `PM AP50` 高 `0.003963`
- `PM mAP` 低 `0.001991`
- `PM F1` 低 `0.002394`
- `FPS` 高 `0.286935`
- `GFLOPs` 高 `47.230157`

## 简要结论

- 整体主基线仍然是 `high_resolution.pth`
  - 它在 `AP50-95` 上仍是第一
  - 同时保住了所有模型里最高的整体 `F1`

- 当前 deformable EMA 线仍优先保留 `ema_defor.pth`
  - `defor_best_ema.pth` 的 `PM AP50` 更高
  - 但 `AP50-95 / F1 / PM mAP / PM F1` 仍低于 `ema_defor.pth`
  - `defor_best_ema.pth` 的算力成本也更高

- PM 主基线仍然是 `menkong.pth`
  - `PM AP50` 和 `PM F1` 仍是第一
  - `defor_best_ema.pth` 在 `PM AP50` 上已经排到第二

- 效率主基线仍然是 `uv_single.pth`
  - 在速度、算力、模型大小上依然最强
  - 作为低成本对照线继续保留是合理的

## 对应结果文件

- `uv_single.pth`
  - `output/eval/2026-04-03_090543/summary_report.json`
- `menkong.pth`
  - `output/eval/2026-04-03_090624/summary_report.json`
- `high_resolution.pth`
  - `output/eval/high_resolution_fixed_20260403/summary_report.json`
- `multi_feature.pth`
  - `output/eval/multi_feature_fixed_20260403/summary_report.json`
- `kimi.pth`
  - `output/eval/kimi_fixed_20260403/summary_report.json`
- `ema_defor.pth`
  - `output/eval/2026-04-08_125521/summary_report.json`
- `defor_best_ema.pth`
  - `output/eval/2026-04-11_084809/summary_report.json`
