# RF-DETR Training Metrics Report

Date: 2026-03-14

## Data Source

- Log file: fusion/output/train/2026-03-14_154006/log.txt
- Key records used:
	- Line 29: Regular best point (epoch 28)
	- Line 36: EMA and overall best point (epoch 35)

## Best Checkpoints Summary

- Regular best
	- epoch: 28
	- all mAP@50:95: 0.605051
	- all mAP@50: 0.806981
	- all Precision: 0.800742
	- all Recall: 0.762943
	- all F1: 0.778019

- EMA best
	- epoch: 35
	- all mAP@50:95: 0.633379
	- all mAP@50: 0.830350
	- all Precision: 0.827077
	- all Recall: 0.804012
	- all F1: 0.813302

## Class Metrics at Regular Best (epoch 28)

| Class | Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| NPML | 0.827017 | 0.886458 | 0.855706 | 0.905970 | 0.758581 |
| PML  | 0.810811 | 0.806452 | 0.808625 | 0.854336 | 0.758513 |
| PM   | 0.764398 | 0.595918 | 0.669725 | 0.660637 | 0.298060 |
| all  | 0.800742 | 0.762943 | 0.778019 | 0.806981 | 0.605051 |

Regular validation losses:

- test_loss: 4.252869
- test_loss_ce: 0.548828
- test_loss_bbox: 0.115155
- test_loss_giou: 0.357729
- test_class_error: 3.128848

## Class Metrics at EMA Best (epoch 35)

| Class | Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| NPML | 0.884495 | 0.885417 | 0.884956 | 0.929354 | 0.781356 |
| PML  | 0.824121 | 0.881720 | 0.851948 | 0.856882 | 0.775214 |
| PM   | 0.772616 | 0.644898 | 0.703003 | 0.704814 | 0.343568 |
| all  | 0.827077 | 0.804012 | 0.813302 | 0.830350 | 0.633379 |

EMA validation losses:

- ema_test_loss: 3.966042
- ema_test_loss_ce: 0.543403
- ema_test_loss_bbox: 0.098457
- ema_test_loss_giou: 0.318574
- ema_test_class_error: 2.724888

## Regular vs EMA Delta (Best Points)

| Metric | Regular@28 | EMA@35 | Delta (EMA-Regular) |
|---|---:|---:|---:|
| mAP@50:95 (all) | 0.605051 | 0.633379 | +0.028328 |
| mAP@50 (all)    | 0.806981 | 0.830350 | +0.023369 |
| Precision (all) | 0.800742 | 0.827077 | +0.026336 |
| Recall (all)    | 0.762943 | 0.804012 | +0.041069 |
| F1 (all)        | 0.778019 | 0.813302 | +0.035284 |

## Notes

- In this run, EMA clearly outperforms regular weights on all overall metrics.
- PM is still the hardest class, but EMA improves PM across Precision, Recall, and mAP.

## Test Set Summary (checkpoint_best_regular)

Data sources:

- Detection metrics: fusion/output/train/2026-03-14_154006/test_eval_results.json
- Performance metrics: fusion/output/train/2026-03-14_154006/test_perf_results.json

| Metric | Value |
|---|---:|
| AP50 (%) | 77.50 |
| AP75 (%) | 60.26 |
| AP50-95 (%) | 58.80 |
| Precision (%) | 79.09 |
| Recall (%) | 75.39 |
| F1-Score (%) | 76.90 |
| GFLOPs (all forward ops, estimated) | 26.74 |
| FPS | 66.74 |
| Model Size (MB) | 436.68 |

Notes:

- GFLOPs above uses an expanded counting path that includes previously skipped ops (for example attention and activation related ops) as an all-process estimate.
- Baseline count-only GFLOPs (supported ops only) is 3.19.
