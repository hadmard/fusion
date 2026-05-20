import pytest

from custom.train import run_train


def test_pretrain_settings_support_scratch_training() -> None:
    weights, force_no_pretrain, label = run_train._resolve_pretrain_settings(
        projector_scale=["P3", "P4"],
        use_dinov2_pretrain=False,
    )

    assert weights is None
    assert force_no_pretrain is True
    assert label == "none"


def test_pretrain_settings_support_dinov2_only_training() -> None:
    weights, force_no_pretrain, label = run_train._resolve_pretrain_settings(
        projector_scale=["P3", "P4"],
        use_dinov2_pretrain=True,
    )

    assert weights is None
    assert force_no_pretrain is False
    assert label == "dinov2"


def test_run_training_passes_regularization_to_model_kwargs(monkeypatch, tmp_path) -> None:
    recorded = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            recorded["model_kwargs"] = kwargs

        def train(self, **kwargs):
            recorded["train_kwargs"] = kwargs

    import rfdetr.main as rfdetr_main

    monkeypatch.setattr(run_train, "NUM_GPUS", 1)
    monkeypatch.setattr(run_train, "RESUME", "")
    monkeypatch.setattr(run_train, "DROPOUT", 0.23)
    monkeypatch.setattr(run_train, "DROP_PATH", 0.34)
    monkeypatch.setattr(rfdetr_main, "Model", _FakeModel)

    run_train.run_training(
        output_base_dir=str(tmp_path),
        log_prefix="[Test]",
    )

    assert recorded["model_kwargs"]["dropout"] == 0.23
    assert recorded["model_kwargs"]["drop_path"] == 0.34
    assert recorded["train_kwargs"]["dropout"] == 0.23
    assert recorded["train_kwargs"]["drop_path"] == 0.34


def test_torchrun_process_accepts_matching_world_size(monkeypatch) -> None:
    monkeypatch.setattr(run_train, "NUM_GPUS", 2)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    run_train._maybe_relaunch_with_torchrun("[Test]")


def test_torchrun_process_rejects_wrong_world_size(monkeypatch) -> None:
    monkeypatch.setattr(run_train, "NUM_GPUS", 2)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    with pytest.raises(RuntimeError, match="WORLD_SIZE=1"):
        run_train._maybe_relaunch_with_torchrun("[Test]")


def test_direct_python_relaunches_with_torchrun(monkeypatch) -> None:
    recorded = {}

    class _Completed:
        returncode = 0

    def _fake_run(command, cwd, check):
        recorded["command"] = command
        recorded["cwd"] = cwd
        recorded["check"] = check
        return _Completed()

    monkeypatch.setattr(run_train, "NUM_GPUS", 2)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(run_train, "_detect_visible_cuda_devices", lambda: 2)
    monkeypatch.setattr(run_train.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_train._maybe_relaunch_with_torchrun("[Test]")

    assert exc_info.value.code == 0
    assert recorded["command"][1:] == [
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--standalone",
        "-m",
        "custom.train.run_train",
    ]
    assert recorded["check"] is False


def test_direct_python_does_not_precheck_visible_gpus(monkeypatch) -> None:
    recorded = {}

    class _Completed:
        returncode = 0

    def _fake_run(command, cwd, check):
        recorded["command"] = command
        return _Completed()

    monkeypatch.setattr(run_train, "NUM_GPUS", 2)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(run_train, "_detect_visible_cuda_devices", lambda: 0)
    monkeypatch.setattr(run_train.subprocess, "run", _fake_run)

    with pytest.raises(SystemExit) as exc_info:
        run_train._maybe_relaunch_with_torchrun("[Test]")

    assert exc_info.value.code == 0
    assert "--nproc_per_node=2" in recorded["command"]
