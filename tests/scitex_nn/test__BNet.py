#!/usr/bin/env python3
"""Test suite for the BNet neural network architecture.

These tests exercise the real BNet model with its real MNet_1000 backbone
(no mocks — the backbone instantiates and runs end-to-end on CPU). BHead is
the attention-head component; BNet is the multi-modal classifier.
"""

import pytest

# Required for this module
pytest.importorskip("torch")
import torch
import torch.nn as nn

import scitex_nn

BHead = scitex_nn.BHead_v1
BNet = scitex_nn.BNet_v1


@pytest.fixture
def mnet_config():
    """Real MNet_1000 backbone configuration."""
    return dict(scitex_nn.MNet_config)


@pytest.fixture
def base_bnet_config():
    """Two-modality (MEG + EEG) BNet configuration."""
    return {
        "n_bands": 6,
        "SAMP_RATE": 250,
        "n_fc1": 1024,
        "d_ratio1": 0.85,
        "n_fc2": 256,
        "d_ratio2": 0.85,
        "n_chs": [160, 19],
        "n_classes": [2, 4],
    }


class TestBHead:
    """Test suite for the BHead attention-head component."""

    def test_bhead_is_an_nn_module(self):
        """Test BHead is constructed as an nn.Module."""
        # Arrange
        # Act
        head = BHead(32, 64)
        # Assert
        assert isinstance(head, nn.Module)

    def test_bhead_has_self_attention_attribute(self):
        """Test BHead exposes a self-attention `sa` submodule."""
        # Arrange
        # Act
        head = BHead(19, 32)
        # Assert
        assert hasattr(head, "sa")

    def test_bhead_has_pointwise_conv_attribute(self):
        """Test BHead exposes a `conv11` pointwise convolution."""
        # Arrange
        # Act
        head = BHead(160, 32)
        # Assert
        assert hasattr(head, "conv11")

    def test_bhead_forward_maps_channels_to_output_channels(self):
        """Test BHead forward maps n_chs_in to n_chs_out, preserving seq_len."""
        # Arrange
        head = BHead(32, 64)
        x = torch.randn(16, 32, 1000)
        # Act
        output = head(x)
        # Assert
        assert output.shape == (16, 64, 1000)

    def test_bhead_forward_produces_input_gradient(self):
        """Test gradients flow back to the input through BHead."""
        # Arrange
        head = BHead(32, 64)
        x = torch.randn(8, 32, 500, requires_grad=True)
        # Act
        head(x).sum().backward()
        # Assert
        assert x.grad is not None

    def test_bhead_input_gradient_is_finite(self):
        """Test the input gradient through BHead contains no NaNs."""
        # Arrange
        head = BHead(32, 64)
        x = torch.randn(8, 32, 500, requires_grad=True)
        # Act
        head(x).sum().backward()
        # Assert
        assert not torch.isnan(x.grad).any()


class TestBNet:
    """Test suite for the multi-modal BNet architecture (real MNet backbone)."""

    def test_bnet_is_an_nn_module(self, base_bnet_config, mnet_config):
        """Test BNet is constructed as an nn.Module."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert isinstance(model, nn.Module)

    def test_bnet_has_heads_module_list(self, base_bnet_config, mnet_config):
        """Test BNet exposes a `heads` attribute."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert hasattr(model, "heads")

    def test_bnet_has_fc_blocks(self, base_bnet_config, mnet_config):
        """Test BNet exposes an `fcs` attribute."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert hasattr(model, "fcs")

    def test_bnet_builds_one_head_per_modality(self, base_bnet_config, mnet_config):
        """Test BNet builds one head per declared modality."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert len(model.heads) == len(base_bnet_config["n_chs"])

    def test_bnet_builds_one_fc_block_per_modality(self, base_bnet_config, mnet_config):
        """Test BNet builds one FC block per declared modality."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert len(model.fcs) == len(base_bnet_config["n_chs"])

    def test_bnet_single_modality_builds_one_head(self, mnet_config):
        """Test single-modality config produces exactly one head."""
        # Arrange
        config = {
            "n_bands": 4,
            "SAMP_RATE": 1000,
            "n_fc1": 512,
            "d_ratio1": 0.5,
            "n_fc2": 128,
            "d_ratio2": 0.5,
            "n_chs": [32],
            "n_classes": [10],
        }
        # Act
        model = BNet(config, mnet_config)
        # Assert
        assert len(model.heads) == 1

    def test_bnet_single_modality_builds_one_fc_block(self, mnet_config):
        """Test single-modality config produces exactly one FC block."""
        # Arrange
        config = {
            "n_bands": 4,
            "SAMP_RATE": 1000,
            "n_fc1": 512,
            "d_ratio1": 0.5,
            "n_fc2": 128,
            "d_ratio2": 0.5,
            "n_chs": [32],
            "n_classes": [10],
        }
        # Act
        model = BNet(config, mnet_config)
        # Assert
        assert len(model.fcs) == 1

    def test_bnet_five_modalities_build_five_heads(self, mnet_config):
        """Test a five-modality config produces five heads."""
        # Arrange
        config = {
            "n_bands": 6,
            "SAMP_RATE": 250,
            "n_fc1": 1024,
            "d_ratio1": 0.85,
            "n_fc2": 256,
            "d_ratio2": 0.85,
            "n_chs": [160, 19, 64, 32, 128],
            "n_classes": [2, 4, 3, 5, 2],
        }
        # Act
        model = BNet(config, mnet_config)
        # Assert
        assert len(model.heads) == 5

    def test_bnet_five_modalities_build_five_fc_blocks(self, mnet_config):
        """Test a five-modality config produces five FC blocks."""
        # Arrange
        config = {
            "n_bands": 6,
            "SAMP_RATE": 250,
            "n_fc1": 1024,
            "d_ratio1": 0.85,
            "n_fc2": 256,
            "d_ratio2": 0.85,
            "n_chs": [160, 19, 64, 32, 128],
            "n_classes": [2, 4, 3, 5, 2],
        }
        # Act
        model = BNet(config, mnet_config)
        # Assert
        assert len(model.fcs) == 5

    def test_bnet_forward_meg_modality_output_shape(
        self, base_bnet_config, mnet_config
    ):
        """Test forward on the MEG head returns (batch, n_classes[0])."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(16, 160, 1000)
        # Act
        output = model(x, i_head=0)
        # Assert
        assert output.shape == (16, base_bnet_config["n_classes"][0])

    def test_bnet_forward_eeg_modality_output_shape(
        self, base_bnet_config, mnet_config
    ):
        """Test forward on the EEG head returns (batch, n_classes[1])."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(16, 19, 1000)
        # Act
        output = model(x, i_head=1)
        # Assert
        assert output.shape == (16, base_bnet_config["n_classes"][1])

    def test_bnet_forward_handles_smaller_batch(self, base_bnet_config, mnet_config):
        """Test forward works for a batch size of 1."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(1, 160, 1000)
        # Act
        output = model(x, i_head=0)
        # Assert
        assert output.shape == (1, 2)

    def test_bnet_invalid_head_index_raises_indexerror(
        self, base_bnet_config, mnet_config
    ):
        """Test forward with an out-of-range head index raises IndexError."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(16, 160, 1000)
        # Act
        ctx = pytest.raises(IndexError)
        # Assert
        with ctx:
            model(x, i_head=10)

    def test_bnet_forward_output_is_on_cpu(self, base_bnet_config, mnet_config):
        """Test forward output stays on CPU for a CPU input."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(16, 160, 1000)
        # Act
        output = model(x, i_head=0)
        # Assert
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bnet_forward_output_is_on_cuda(self, base_bnet_config, mnet_config):
        """Test forward output is on CUDA for a CUDA input."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config).cuda()
        x = torch.randn(16, 160, 1000).cuda()
        # Act
        output = model(x, i_head=0)
        # Assert
        assert output.device.type == "cuda"

    def test_bnet_forward_produces_input_gradient(self, base_bnet_config, mnet_config):
        """Test gradients flow back to the input through the full BNet."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(4, 160, 1000, requires_grad=True)
        # Act
        model(x, i_head=0).sum().backward()
        # Assert
        assert x.grad is not None

    def test_bnet_input_gradient_is_finite(self, base_bnet_config, mnet_config):
        """Test the input gradient through BNet contains no NaNs."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x = torch.randn(4, 160, 1000, requires_grad=True)
        # Act
        model(x, i_head=0).sum().backward()
        # Assert
        assert not torch.isnan(x.grad).any()

    def test_bnet_znorm_zeroes_the_mean(self):
        """Test the z-norm static method gives each row a near-zero mean."""
        # Arrange
        x = torch.randn(16, 32, 1000)
        # Act
        x_norm = BNet._znorm_along_the_last_dim(x)
        # Assert
        assert torch.allclose(
            x_norm.mean(dim=-1), torch.zeros_like(x_norm.mean(dim=-1)), atol=1e-06
        )

    def test_bnet_znorm_unit_standard_deviation(self):
        """Test the z-norm static method gives each row a near-unit std."""
        # Arrange
        x = torch.randn(16, 32, 1000)
        # Act
        x_norm = BNet._znorm_along_the_last_dim(x)
        # Assert
        assert torch.allclose(
            x_norm.std(dim=-1), torch.ones_like(x_norm.std(dim=-1)), atol=1e-06
        )

    def test_bnet_has_dropout_channels_module(self, base_bnet_config, mnet_config):
        """Test BNet exposes a `dc` DropoutChannels submodule."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert isinstance(model.dc, scitex_nn.DropoutChannels)

    def test_bnet_has_freq_gain_changer_module(self, base_bnet_config, mnet_config):
        """Test BNet exposes a `fgc` FreqGainChanger submodule."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert isinstance(model.fgc, scitex_nn.FreqGainChanger)

    def test_bnet_builds_one_channel_gain_changer_per_modality(
        self, base_bnet_config, mnet_config
    ):
        """Test BNet builds one ChannelGainChanger per modality."""
        # Arrange
        # Act
        model = BNet(base_bnet_config, mnet_config)
        # Assert
        assert len(model.cgcs) == len(base_bnet_config["n_chs"])

    def test_bnet_channel_gain_changers_are_correct_type(
        self, base_bnet_config, mnet_config
    ):
        """Test every channel-gain-changer is a ChannelGainChanger."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        types = {type(cgc) for cgc in model.cgcs}
        # Assert
        assert types == {scitex_nn.ChannelGainChanger}

    def test_bnet_fc_block_has_six_layers(self, base_bnet_config, mnet_config):
        """Test each FC block has six layers."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        lengths = {len(fc_block) for fc_block in model.fcs}
        # Assert
        assert lengths == {6}

    def test_bnet_fc_block_first_layer_is_mish(self, base_bnet_config, mnet_config):
        """Test the first FC-block layer is a Mish activation."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        first_layers = [fc_block[0] for fc_block in model.fcs]
        # Assert
        assert all(isinstance(layer, nn.Mish) for layer in first_layers)

    def test_bnet_fc_block_second_layer_is_dropout(self, base_bnet_config, mnet_config):
        """Test the second FC-block layer is a Dropout."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        second_layers = [fc_block[1] for fc_block in model.fcs]
        # Assert
        assert all(isinstance(layer, nn.Dropout) for layer in second_layers)

    def test_bnet_fc_block_third_layer_is_linear(self, base_bnet_config, mnet_config):
        """Test the third FC-block layer is a Linear."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        third_layers = [fc_block[2] for fc_block in model.fcs]
        # Assert
        assert all(isinstance(layer, nn.Linear) for layer in third_layers)

    def test_bnet_fc_block_final_layer_matches_n_classes(
        self, base_bnet_config, mnet_config
    ):
        """Test each FC block's output features match the head's n_classes."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        out_features = [fc_block[5].out_features for fc_block in model.fcs]
        # Assert
        assert out_features == base_bnet_config["n_classes"]

    def test_bnet_head_uses_thirty_two_virtual_channels(
        self, base_bnet_config, mnet_config
    ):
        """Test every head projects to 32 virtual channels."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        out_channels = {head.conv11.out_channels for head in model.heads}
        # Assert
        assert out_channels == {32}

    def test_bnet_multi_modal_meg_head_output_shape(
        self, base_bnet_config, mnet_config
    ):
        """Test the MEG head output shape in a multi-modal model."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x_meg = torch.randn(8, 160, 1000)
        # Act
        y_meg = model(x_meg, i_head=0)
        # Assert
        assert y_meg.shape == (8, 2)

    def test_bnet_multi_modal_eeg_head_output_shape(
        self, base_bnet_config, mnet_config
    ):
        """Test the EEG head output shape in a multi-modal model."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        x_eeg = torch.randn(8, 19, 1000)
        # Act
        y_eeg = model(x_eeg, i_head=1)
        # Assert
        assert y_eeg.shape == (8, 4)

    def test_bnet_has_a_substantial_parameter_count(
        self, base_bnet_config, mnet_config
    ):
        """Test the model has more than 100k parameters (real backbone)."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        total_params = sum(p.numel() for p in model.parameters())
        # Assert
        assert total_params > 100_000

    def test_bnet_all_parameters_are_trainable(self, base_bnet_config, mnet_config):
        """Test every parameter requires grad (no frozen layers)."""
        # Arrange
        model = BNet(base_bnet_config, mnet_config)
        # Act
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        # Assert
        assert trainable == total


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
