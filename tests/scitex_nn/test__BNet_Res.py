#!/usr/bin/env python3
"""Test suite for the BNet_Res (residual BNet) architecture.

These tests exercise the real BNet_Res model (no mocks): construction of its
residual blocks and submodules, the z-norm static method, BHead, and the
pooling arithmetic the forward path relies on.

NOTE: BNet_Res.forward is currently non-functional — the source assigns
neither `self.MNet` nor `self.fcs` (both are commented out in __init__),
yet forward calls `self.MNet.forward_bb(x)`. We assert that broken state
explicitly rather than mock around it; completing the forward requires a
design decision about the backbone/head and is out of scope here.
"""

import pytest

# Required for this module
pytest.importorskip("torch")
import torch
import torch.nn as nn
import torch.nn.functional as F

import scitex_nn

BNet = scitex_nn.BNet_Res
BHead = scitex_nn.BHead_Res


@pytest.fixture
def mnet_config():
    """Placeholder MNet config (BNet_Res does not currently build a backbone)."""
    return {"some_param": "value"}


@pytest.fixture
def base_config():
    """Two-modality (MEG + EEG) residual-BNet configuration."""
    return {
        "n_bands": 6,
        "n_virtual_chs": 16,
        "SAMP_RATE": 250,
        "n_fc1": 1024,
        "d_ratio1": 0.85,
        "n_fc2": 256,
        "d_ratio2": 0.85,
        "n_chs_of_modalities": [160, 19],
        "n_classes_of_modalities": [2, 4],
    }


class TestBNetResConstruction:
    """Construction of the real BNet_Res model and its submodules."""

    def test_bnet_res_is_an_nn_module(self, base_config, mnet_config):
        """Test BNet_Res is constructed as an nn.Module."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert isinstance(model, nn.Module)

    def test_bnet_res_has_heads_attribute(self, base_config, mnet_config):
        """Test BNet_Res exposes a `heads` attribute."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert hasattr(model, "heads")

    def test_bnet_res_builds_first_residual_block(self, base_config, mnet_config):
        """Test the first residual block `blk1` is constructed."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert hasattr(model, "blk1")

    def test_bnet_res_builds_seven_residual_blocks(self, base_config, mnet_config):
        """Test all seven residual blocks blk1..blk7 are constructed."""
        # Arrange
        model = BNet(base_config, mnet_config)
        # Act
        present = [hasattr(model, f"blk{i}") for i in range(1, 8)]
        # Assert
        assert all(present)

    def test_bnet_res_builds_one_head_per_modality(self, base_config, mnet_config):
        """Test BNet_Res builds one head per declared modality."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert len(model.heads) == len(base_config["n_chs_of_modalities"])

    def test_bnet_res_builds_one_channel_gain_changer_per_modality(
        self, base_config, mnet_config
    ):
        """Test BNet_Res builds one channel-gain-changer per modality."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert len(model.cgcs) == len(base_config["n_chs_of_modalities"])

    def test_bnet_res_heads_use_configured_virtual_channels(
        self, base_config, mnet_config
    ):
        """Test every head projects to the configured number of virtual channels."""
        # Arrange
        config = {**base_config, "n_virtual_chs": 32}
        model = BNet(config, mnet_config)
        # Act
        out_channels = {head.conv11.out_channels for head in model.heads}
        # Assert
        assert out_channels == {32}

    def test_bnet_res_has_dropout_channels_module(self, base_config, mnet_config):
        """Test BNet_Res exposes a `dc` DropoutChannels submodule."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert isinstance(model.dc, scitex_nn.DropoutChannels)

    def test_bnet_res_has_swap_channels_module(self, base_config, mnet_config):
        """Test BNet_Res exposes a `sc` SwapChannels submodule."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert isinstance(model.sc, scitex_nn.SwapChannels)

    def test_bnet_res_has_freq_gain_changer_for_each_band_count(self, mnet_config):
        """Test BNet_Res builds a FreqGainChanger across band-count configs."""
        # Arrange
        configs = [
            {
                "n_bands": n,
                "n_virtual_chs": 16,
                "SAMP_RATE": 250,
                "n_fc1": 1024,
                "d_ratio1": 0.85,
                "n_fc2": 256,
                "d_ratio2": 0.85,
                "n_chs_of_modalities": [32],
                "n_classes_of_modalities": [2],
            }
            for n in [2, 4, 6, 8, 10]
        ]
        # Act
        have_fgc = [
            isinstance(BNet(c, mnet_config).fgc, scitex_nn.FreqGainChanger)
            for c in configs
        ]
        # Assert
        assert all(have_fgc)

    def test_bnet_res_builds_for_each_sampling_rate(self, mnet_config):
        """Test BNet_Res constructs across a range of sampling rates."""
        # Arrange
        configs = [
            {
                "n_bands": 6,
                "n_virtual_chs": 16,
                "SAMP_RATE": sr,
                "n_fc1": 1024,
                "d_ratio1": 0.85,
                "n_fc2": 256,
                "d_ratio2": 0.85,
                "n_chs_of_modalities": [32],
                "n_classes_of_modalities": [2],
            }
            for sr in [100, 250, 500, 1000, 2000]
        ]
        # Act
        built = [BNet(c, mnet_config).fgc is not None for c in configs]
        # Assert
        assert all(built)

    def test_bnet_res_single_modality_builds_one_head(self, mnet_config):
        """Test a single-modality config produces exactly one head."""
        # Arrange
        config = {
            "n_bands": 4,
            "n_virtual_chs": 16,
            "SAMP_RATE": 1000,
            "n_fc1": 512,
            "d_ratio1": 0.5,
            "n_fc2": 128,
            "d_ratio2": 0.5,
            "n_chs_of_modalities": [32],
            "n_classes_of_modalities": [10],
        }
        # Act
        model = BNet(config, mnet_config)
        # Assert
        assert len(model.heads) == 1

    def test_bnet_res_has_empty_dummy_param(self, base_config, mnet_config):
        """Test the device-tracking dummy parameter has zero elements."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert model.dummy_param.shape == torch.Size([0])

    def test_bnet_res_parameters_start_on_cpu(self, base_config, mnet_config):
        """Test model parameters are created on CPU by default."""
        # Arrange
        # Act
        model = BNet(base_config, mnet_config)
        # Assert
        assert next(model.parameters()).device.type == "cpu"


class TestBNetResModes:
    """Train/eval mode behavior of the real BNet_Res model."""

    def test_bnet_res_train_sets_training_true(self, base_config, mnet_config):
        """Test calling train() flips the model into training mode."""
        # Arrange
        model = BNet(base_config, mnet_config)
        # Act
        model.train()
        # Assert
        assert model.training

    def test_bnet_res_eval_sets_training_false(self, base_config, mnet_config):
        """Test calling eval() flips the model out of training mode."""
        # Arrange
        model = BNet(base_config, mnet_config)
        # Act
        model.eval()
        # Assert
        assert not model.training

    def test_bnet_res_to_cpu_returns_same_model(self, base_config, mnet_config):
        """Test .to('cpu') returns the same in-place model object."""
        # Arrange
        model = BNet(base_config, mnet_config)
        # Act
        model_cpu = model.to("cpu")
        # Assert
        assert model_cpu is model


class TestBNetResForwardIsBroken:
    """The forward path is non-functional (missing MNet/fcs) — documented here."""

    def test_bnet_res_forward_raises_attributeerror_for_missing_backbone(
        self, base_config, mnet_config
    ):
        """Test forward raises AttributeError because self.MNet is never built.

        BNet_Res.__init__ leaves `self.MNet` and `self.fcs` commented out, but
        forward() calls `self.MNet.forward_bb(x)`. Until the architecture is
        completed, forward cannot run; this test pins that contract so the
        bug is visible rather than silently swallowed.
        """
        # Arrange
        model = BNet(base_config, mnet_config)
        x = torch.randn(2, 160, 128)
        # Act
        ctx = pytest.raises(AttributeError)
        # Assert
        with ctx:
            model(x, i_head=0)


class TestBNetResZNorm:
    """The z-normalization static method."""

    def test_znorm_zeroes_the_mean(self):
        """Test z-norm gives each row a near-zero mean along the last dim."""
        # Arrange
        x = torch.randn(8, 32, 512)
        # Act
        x_norm = BNet._znorm_along_the_last_dim(x)
        # Assert
        assert torch.allclose(
            x_norm.mean(dim=-1), torch.zeros_like(x_norm.mean(dim=-1)), atol=1e-06
        )

    def test_znorm_unit_standard_deviation(self):
        """Test z-norm gives each row a near-unit std along the last dim."""
        # Arrange
        x = torch.randn(8, 32, 512)
        # Act
        x_norm = BNet._znorm_along_the_last_dim(x)
        # Assert
        assert torch.allclose(
            x_norm.std(dim=-1), torch.ones_like(x_norm.std(dim=-1)), atol=1e-06
        )


class TestBHeadRes:
    """The BHead_Res attention-head component."""

    def test_bhead_res_forward_maps_to_output_channels(self):
        """Test BHead_Res maps n_chs_in to n_chs_out, preserving seq_len."""
        # Arrange
        head = BHead(32, 16)
        x = torch.randn(4, 32, 256)
        # Act
        output = head(x)
        # Assert
        assert output.shape == (4, 16, 256)

    def test_bhead_res_has_self_attention_attribute(self):
        """Test BHead_Res exposes a self-attention `sa` submodule."""
        # Arrange
        # Act
        head = BHead(32, 16)
        # Assert
        assert hasattr(head, "sa")

    def test_bhead_res_has_pointwise_conv_attribute(self):
        """Test BHead_Res exposes a `conv11` pointwise convolution."""
        # Arrange
        # Act
        head = BHead(32, 16)
        # Assert
        assert hasattr(head, "conv11")


class TestBNetResPoolingArithmetic:
    """The avg-pool arithmetic the residual forward path is built around."""

    def test_transpose_pool_halves_channel_dim(self):
        """Test transpose+avg_pool1d halves the channel dimension."""
        # Arrange
        x = torch.randn(4, 16, 1024)
        # Act
        x_pool = F.avg_pool1d(x.transpose(1, 2), kernel_size=2).transpose(1, 2)
        # Assert
        assert x_pool.shape == (4, 8, 1024)

    def test_plain_pool_halves_sequence_dim(self):
        """Test avg_pool1d halves the sequence dimension."""
        # Arrange
        x = torch.randn(4, 16, 1024)
        # Act
        x_pool = F.avg_pool1d(x, kernel_size=2)
        # Assert
        assert x_pool.shape == (4, 16, 512)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
