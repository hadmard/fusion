import pytest

from custom.train import run_train


def test_pretrain_settings_support_scratch_training() -> None:
    weights, force_no_pretrain, label = run_train._resolve_pretrain_settings(
        projector_scale=["P3", "P4"],
        use_rfdetr_pretrain=False,
        rfdetr_pretrain_weights="rf-detr-base.pth",
        use_dinov2_pretrain=False,
    )

    assert weights is None
    assert force_no_pretrain is True
    assert label == "none"


def test_pretrain_settings_support_dinov2_only_training() -> None:
    weights, force_no_pretrain, label = run_train._resolve_pretrain_settings(
        projector_scale=["P3", "P4"],
        use_rfdetr_pretrain=False,
        rfdetr_pretrain_weights="rf-detr-base.pth",
        use_dinov2_pretrain=True,
    )

    assert weights is None
    assert force_no_pretrain is False
    assert label == "dinov2"


def test_pretrain_settings_support_rfdetr_for_single_p4() -> None:
    weights, force_no_pretrain, label = run_train._resolve_pretrain_settings(
        projector_scale=["P4"],
        use_rfdetr_pretrain=True,
        rfdetr_pretrain_weights="rf-detr-base.pth",
        use_dinov2_pretrain=True,
    )

    assert weights == "rf-detr-base.pth"
    assert force_no_pretrain is False
    assert label == "rfdetr"


def test_pretrain_settings_reject_rfdetr_for_multi_scale_projector() -> None:
    with pytest.raises(ValueError, match=r"requires PROJECTOR_SCALE=\['P4'\]"):
        run_train._resolve_pretrain_settings(
            projector_scale=["P3", "P4"],
            use_rfdetr_pretrain=True,
            rfdetr_pretrain_weights="rf-detr-base.pth",
            use_dinov2_pretrain=False,
        )
