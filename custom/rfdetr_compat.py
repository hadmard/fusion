"""
文件说明：兼容性适配用的，不要修改

"""

from __future__ import annotations

import datetime
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, DefaultDict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

import rfdetr.util.misc as utils
from rfdetr._namespace import build_namespace
from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.datasets import build_dataset as _build_dataset_impl
from rfdetr.datasets import get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import PostProcess, build_model
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import BestMetricHolder, ModelEma, clean_state_dict

# ========== 第一部分：导入依赖与可 monkey patch 的数据集构建入口 ==========
# UV-only 支持补丁会在运行时替换这个符号，因此这里保留模块级变量而不是写死局部导入。
build_dataset = _build_dataset_impl


def _resolve_coco_api(dataset: Any) -> Any:
    """
    兼容主库 helper 目前还不认识 custom 双模态数据集的情况。

    为什么这里要单独兜底：
    - `DualModalYoloDetection` 已经显式暴露了 `dataset.coco`；
    - 但 `rfdetr.datasets.get_coco_api_from_dataset()` 只识别官方数据集类型；
    - 训练主体能跑通后，验证阶段会因为拿不到 COCO API 直接报 `NoneType.cats`。
    """
    coco_api = get_coco_api_from_dataset(dataset)
    if coco_api is not None:
        return coco_api

    current = dataset
    for _ in range(10):
        if hasattr(current, "coco"):
            return current.coco
        if hasattr(current, "dataset"):
            current = current.dataset
            continue
        break
    return None


def _limit_dataset_for_smoke(dataset: Any, max_batches: int, batch_size: int) -> Any:
    """
    把数据集截断到有限样本数，方便只做象征性训练/验证。

    这里按 batch 数限制而不是按样本数手写 magic number，
    是为了让 `batch_size` 改动后，smoke 规模仍然保持直观可控。
    """
    if max_batches is None or int(max_batches) <= 0:
        return dataset

    max_samples = int(max_batches) * int(batch_size)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return torch.utils.data.Subset(dataset, list(range(max_samples)))


# ========== 第二部分：参数兼容函数 populate_args ==========
def populate_args(**kwargs: Any) -> SimpleNamespace:
    """
    把当前 custom 入口传入的关键字参数收敛成旧 `args.xxx` 风格 namespace。

    设计原因：
    - 上游已经迁到 `ModelConfig / TrainConfig`。
    - custom 双模态代码仍大量依赖旧 `args` 结构。
    - 这里用“新版配置 + 旧式 namespace”的组合，尽量减少两边重复维护。
    """

    model_field_names = set(RFDETRBaseConfig.model_fields.keys())
    train_field_names = set(TrainConfig.model_fields.keys())

    model_kwargs = {key: value for key, value in kwargs.items() if key in model_field_names}
    train_kwargs = {key: value for key, value in kwargs.items() if key in train_field_names}

    dataset_dir = train_kwargs.get("dataset_dir", kwargs.get("dataset_dir", "."))
    output_dir = train_kwargs.get("output_dir", kwargs.get("output_dir", "output"))

    model_config = RFDETRBaseConfig(**model_kwargs)
    train_kwargs.setdefault("dataset_dir", dataset_dir)
    train_kwargs.setdefault("output_dir", output_dir)
    train_config = TrainConfig(**train_kwargs)

    args = build_namespace(model_config, train_config)

    # 旧 custom 流程仍显式依赖这些字段；它们不应该强行写回上游主配置。
    extra_defaults = {
        "dual_modal": False,
        "class_names": train_config.class_names,
        "resume_load_lr_scheduler": train_config.resume_load_lr_scheduler,
        "distributed": False,
        "gpu": 0,
        "rank": 0,
        "shape": (model_config.resolution, model_config.resolution),
    }
    for key, default_value in extra_defaults.items():
        setattr(args, key, kwargs.get(key, default_value))

    # 最后把其余旧入口特有字段挂上去，避免 custom 代码二次判断缺失。
    for key, value in kwargs.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    return args


# ========== 第三部分：旧式 Model 兼容封装 ==========
class Model:
    """兼容旧 `rfdetr.main.Model` 的轻量封装。"""

    def __init__(self, **kwargs: Any) -> None:
        self.args = populate_args(**kwargs)
        self.resolution = self.args.resolution
        self.stop_early = False

        if self.args.dual_modal:
            from custom.dual_model import build_dual_model

            self.model = build_dual_model(self.args)
        else:
            self.model = build_model(self.args)

        self.device = torch.device(self.args.device)
        self.model = self.model.to(self.device)
        self.postprocess = PostProcess(num_select=self.args.num_select)
        self.class_names = getattr(self.args, "class_names", None)

        if self.args.pretrain_weights is not None:
            self._load_pretrain_weights(self.args.pretrain_weights)

    def _load_pretrain_weights(self, weights_path: str) -> None:
        """沿用旧入口的 checkpoint 加载语义，但使用新版下载/校验工具。"""
        download_pretrain_weights(weights_path)
        validate_pretrain_weights(weights_path, strict=False)

        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        except Exception:
            download_pretrain_weights(weights_path, redownload=True, validate_md5=False)
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if "args" in checkpoint and hasattr(checkpoint["args"], "class_names"):
            self.args.class_names = checkpoint["args"].class_names
            self.class_names = checkpoint["args"].class_names

        checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
        if checkpoint_num_classes != self.args.num_classes + 1:
            self.model.reinitialize_detection_head(checkpoint_num_classes)

        num_desired_queries = self.args.num_queries * self.args.group_detr
        for name in list(checkpoint["model"].keys()):
            if name.endswith("refpoint_embed.weight") or name.endswith("query_feat.weight"):
                checkpoint["model"][name] = checkpoint["model"][name][:num_desired_queries]

        self.model.load_state_dict(checkpoint["model"], strict=False)

        if checkpoint_num_classes != self.args.num_classes + 1:
            self.model.reinitialize_detection_head(self.args.num_classes + 1)

    def reinitialize_detection_head(self, num_classes: int) -> None:
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self) -> None:
        self.stop_early = True

    def train(self, callbacks: DefaultDict[str, List[Any]], **kwargs: Any) -> None:
        """
        沿用旧 `Model.train()` 调用方式执行训练。

        当前实现刻意只保留当前仓库实际在用的主路径：
        - 单机训练
        - 可选 EMA
        - 单模态 / 双模态 dataset 二选一
        - best checkpoint / results.json 产物
        """
        args = populate_args(**kwargs)
        if getattr(args, "class_names", None) is not None:
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        device = torch.device(args.device)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        criterion, postprocess = build_criterion_and_postprocessors(args)
        model = self.model.to(device)
        model_without_ddp = model

        n_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        param_dicts = [group for group in get_param_dict(args, model_without_ddp) if group["params"].requires_grad]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

        if args.dual_modal:
            from custom.dual_collate import dual_collate_fn
            from custom.dual_dataset import build_dual_dataset

            dataset_train = build_dual_dataset(
                image_set="train",
                dataset_dir=args.dataset_dir,
                resolution=args.resolution,
                class_names=getattr(args, "class_names", None),
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            )
            dataset_val = build_dual_dataset(
                image_set="val",
                dataset_dir=args.dataset_dir,
                resolution=args.resolution,
                class_names=getattr(args, "class_names", None),
                multi_scale=False,
                expanded_scales=args.expanded_scales,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            )
            dataset_test = build_dual_dataset(
                image_set="val",
                dataset_dir=args.dataset_dir,
                resolution=args.resolution,
                class_names=getattr(args, "class_names", None),
                multi_scale=False,
                expanded_scales=args.expanded_scales,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            )
            collate_fn = dual_collate_fn
        else:
            dataset_train = build_dataset("train", args=args, resolution=args.resolution)
            dataset_val = build_dataset("val", args=args, resolution=args.resolution)
            dataset_test = build_dataset("test" if args.dataset_file == "roboflow" else "val", args=args, resolution=args.resolution)
            collate_fn = utils.collate_fn

        effective_batch_size = args.batch_size * args.grad_accum_steps
        max_train_batches = int(getattr(args, "max_train_batches", 0) or 0)
        max_val_batches = int(getattr(args, "max_val_batches", 0) or 0)
        max_test_batches = int(getattr(args, "max_test_batches", max_val_batches) or 0)

        dataset_train = _limit_dataset_for_smoke(dataset_train, max_train_batches, effective_batch_size)
        dataset_val = _limit_dataset_for_smoke(dataset_val, max_val_batches, args.batch_size)
        dataset_test = _limit_dataset_for_smoke(dataset_test, max_test_batches, args.batch_size)

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        loader_kwargs = {"collate_fn": collate_fn, "num_workers": args.num_workers}
        if args.device == "cuda":
            loader_kwargs["pin_memory"] = True if args.pin_memory is None else bool(args.pin_memory)
        elif args.pin_memory is not None:
            loader_kwargs["pin_memory"] = bool(args.pin_memory)
        if args.num_workers > 0:
            if args.persistent_workers is not None:
                loader_kwargs["persistent_workers"] = bool(args.persistent_workers)
            if args.prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)

        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        if max_train_batches > 0:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train,
                effective_batch_size,
                drop_last=False,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                **loader_kwargs,
            )
        elif len(dataset_train) < effective_batch_size * 5:
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * 5,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                sampler=sampler,
                **loader_kwargs,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, effective_batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                **loader_kwargs,
            )

        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, **loader_kwargs)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False, **loader_kwargs)

        base_ds = _resolve_coco_api(dataset_val)
        base_ds_test = _resolve_coco_api(dataset_test)

        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None

        steps_per_epoch = max(1, math.ceil(len(dataset_train) / max(effective_batch_size, 1)))
        total_training_steps = steps_per_epoch * args.epochs
        warmup_steps = int(steps_per_epoch * args.warmup_epochs)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if args.lr_scheduler == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
                return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            if current_step < args.lr_drop * steps_per_epoch:
                return 1.0
            return 0.1

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            if args.use_ema and "ema_model" in checkpoint:
                self.ema_m.module.load_state_dict(clean_state_dict(checkpoint["ema_model"]))
            if "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                if args.resume_load_lr_scheduler:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                args.start_epoch = checkpoint["epoch"] + 1

        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_regular_stats = None
        best_ema_stats = None
        start_time = datetime.datetime.now()

        for epoch in range(getattr(args, "start_epoch", 0), args.epochs):
            model.train()
            criterion.train()
            train_stats = train_one_epoch(
                model=model,
                criterion=criterion,
                lr_scheduler=lr_scheduler,
                data_loader=data_loader_train,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                batch_size=effective_batch_size,
                max_norm=args.clip_max_norm,
                ema_m=self.ema_m,
                schedules={},
                num_training_steps_per_epoch=steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers,
                args=args,
                callbacks=callbacks,
            )

            test_stats, coco_evaluator = evaluate(model, criterion, postprocess, data_loader_val, base_ds, device, args=args)
            map_regular = test_stats["coco_eval_bbox"][0 if not args.segmentation_head else 0]
            is_best_regular = best_map_holder.update(map_regular, epoch, is_ema=False)
            if is_best_regular and not args.dont_save_weights:
                best_regular_stats = test_stats
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    output_dir / "checkpoint_best_regular.pth",
                )

            log_stats = {
                **{f"train_{key}": value for key, value in train_stats.items()},
                **{f"test_{key}": value for key, value in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
                "now_time": str(datetime.datetime.now()),
            }

            if args.use_ema:
                ema_test_stats, _ = evaluate(self.ema_m.module, criterion, postprocess, data_loader_val, base_ds, device, args=args)
                log_stats.update({f"ema_test_{key}": value for key, value in ema_test_stats.items()})
                is_best_ema = best_map_holder.update(ema_test_stats["coco_eval_bbox"][0], epoch, is_ema=True)
                if is_best_ema and not args.dont_save_weights:
                    best_ema_stats = ema_test_stats
                    utils.save_on_master(
                        {
                            "model": self.ema_m.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        output_dir / "checkpoint_best_ema.pth",
                    )

            log_stats.update(best_map_holder.summary())
            with (output_dir / "log.txt").open("a", encoding="utf-8") as file:
                file.write(json.dumps(log_stats, ensure_ascii=False) + "\n")

            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                break

        best_is_ema = args.use_ema and best_ema_stats is not None and (
            best_regular_stats is None or best_ema_stats["coco_eval_bbox"][0] >= best_regular_stats["coco_eval_bbox"][0]
        )
        best_checkpoint = output_dir / ("checkpoint_best_ema.pth" if best_is_ema else "checkpoint_best_regular.pth")
        if best_checkpoint.exists():
            shutil.copy2(best_checkpoint, output_dir / "checkpoint_best_total.pth")
            utils.strip_checkpoint(output_dir / "checkpoint_best_total.pth")

        best_results = best_ema_stats if best_is_ema and best_ema_stats is not None else best_regular_stats or test_stats
        with (output_dir / "results.json").open("w", encoding="utf-8") as file:
            json.dump(best_results["results_json"], file, ensure_ascii=False, indent=2)

        if args.run_test and (output_dir / "checkpoint_best_total.pth").exists():
            best_state = torch.load(output_dir / "checkpoint_best_total.pth", map_location="cpu", weights_only=False)["model"]
            model.load_state_dict(best_state, strict=False)
            model.eval()
            test_stats, _ = evaluate(model, criterion, postprocess, data_loader_test, base_ds_test, device, args=args)
            with (output_dir / "results.json").open("r", encoding="utf-8") as file:
                results = json.load(file)
            results["test"] = test_stats["results_json"]
            with (output_dir / "results.json").open("w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)

        for callback in callbacks["on_train_end"]:
            callback()

        print(f"Training started at {start_time} and finished at {datetime.datetime.now()}.")
