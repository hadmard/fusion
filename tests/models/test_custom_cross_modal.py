import torch

import custom.core.cross_modal as cross_modal
from custom.core.cross_modal import MultiLevelCrossModalFusion, SingleLevelCrossModalFusion


def _make_feature_group(batch_size: int = 1, channels: int = 8) -> list[torch.Tensor]:
    return [torch.randn(batch_size, channels, 2, 2) for _ in range(4)]


def test_same_depth_fusion_removes_depth_attention_residual() -> None:
    assert not hasattr(cross_modal, "DepthAttentionResidual")
    assert not hasattr(cross_modal, "WhiteTextureAdapter")

    fusion = MultiLevelCrossModalFusion(
        input_dims=[8, 8, 8, 8],
        fusion_dim=16,
        num_heads=4,
    )

    assert "DepthAttentionResidual" not in {
        module.__class__.__name__ for module in fusion.modules()
    }
    assert "WhiteTextureAdapter" not in {
        module.__class__.__name__ for module in fusion.modules()
    }
    assert not hasattr(fusion, "requested_num_reads")
    assert not hasattr(fusion, "num_reads")


def test_same_depth_fusion_updates_all_encoder_features_before_projector() -> None:
    torch.manual_seed(0)
    fusion = MultiLevelCrossModalFusion(
        input_dims=[8, 8, 8, 8],
        fusion_dim=16,
        num_heads=4,
    )
    uv_features = _make_feature_group()
    white_features = _make_feature_group()

    with torch.no_grad():
        for fusion_level in fusion.level_fusions:
            fusion_level.output_projector.proj.bias.fill_(0.1)

    with torch.no_grad():
        fused_features = fusion(uv_features, white_features)

    assert len(fused_features) == 4
    for index, fused_feature in enumerate(fused_features):
        assert fused_feature is not uv_features[index]
        assert fused_feature.shape == uv_features[index].shape
        assert not torch.equal(fused_feature, uv_features[index])


def test_single_level_fusion_starts_as_uv_residual_identity() -> None:
    torch.manual_seed(0)
    fusion = SingleLevelCrossModalFusion(
        input_dim=8,
        fusion_dim=16,
        num_heads=4,
    )
    uv = torch.randn(1, 8, 4, 4)
    white = torch.randn(1, 8, 4, 4)

    with torch.no_grad():
        fused = fusion(uv, white)

    assert torch.allclose(fused, uv, atol=1e-6)


def test_multi_level_fusion_keeps_one_read_per_depth() -> None:
    fusion = MultiLevelCrossModalFusion(
        input_dims=[8, 8, 8, 8],
        fusion_dim=16,
        num_heads=4,
    )

    fusion_blocks = [
        module
        for module in fusion.modules()
        if isinstance(module, SingleLevelCrossModalFusion)
    ]

    assert len(fusion.level_fusions) == 4
    assert all(
        isinstance(fusion_level, SingleLevelCrossModalFusion)
        for fusion_level in fusion.level_fusions
    )
    assert len(fusion_blocks) == 4
    assert not hasattr(fusion, "num_layers")


def test_multi_level_fusion_rejects_stack_depth_argument() -> None:
    try:
        MultiLevelCrossModalFusion(
            input_dims=[8, 8, 8, 8],
            fusion_dim=16,
            num_heads=4,
            num_layers=2,
        )
    except TypeError:
        return

    raise AssertionError("MultiLevelCrossModalFusion should not accept num_layers.")
