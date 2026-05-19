#!/usr/bin/env python3
"""
Comprehensive test suite for BNet neural network architecture.

This module tests the BNet class and its components including:
- Basic instantiation with different configurations
- Forward pass functionality with various input shapes
- Multi-head architecture capabilities
- Integration with MNet backbone
- Gradient flow and backpropagation
- Device compatibility (CPU/GPU)
"""

import pytest

# Required for this module
pytest.importorskip("torch")
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

import scitex_nn

# Use the standalone-package exports — patches to `scitex_nn._BNet.*`
# above only intercept this package's MNet_1000, not the umbrella's
# separate copy at `scitex_nn._BNet`.
BHead = scitex_nn.BHead_v1
BNet = scitex_nn.BNet_v1
BNet_config = scitex_nn.BNet_config_v1


class TestBHead:
    """Test suite for BHead module - the attention head component of BNet."""

    def test_bhead_instantiation_bnet_behaves_correctly_isinstance(self):
        """Test BHead instantiation with various channel configurations."""
        # Arrange
        configs = [(32, 64), (19, 32), (160, 32), (1, 16)]
        # Act
        # Assert
        for n_chs_in, n_chs_out in configs:
            head = BHead(n_chs_in, n_chs_out)
            assert isinstance(head, nn.Module)
            pass
            pass

    def test_bhead_instantiation_bnet_behaves_correctly_hasattr(self):
        """Test BHead instantiation with various channel configurations."""
        # Arrange
        configs = [(32, 64), (19, 32), (160, 32), (1, 16)]
        # Act
        # Assert
        for n_chs_in, n_chs_out in configs:
            head = BHead(n_chs_in, n_chs_out)
            pass
            assert hasattr(head, 'sa')
            pass

    def test_bhead_instantiation_bnet_behaves_correctly_hasattr_v2(self):
        """Test BHead instantiation with various channel configurations."""
        # Arrange
        configs = [(32, 64), (19, 32), (160, 32), (1, 16)]
        # Act
        # Assert
        for n_chs_in, n_chs_out in configs:
            head = BHead(n_chs_in, n_chs_out)
            pass
            pass
            assert hasattr(head, 'conv11')

    def test_bhead_forward_pass(self):
        """Test BHead forward pass with different input shapes."""
        # Arrange
        batch_size = 16
        seq_len = 1000
        n_chs_in, n_chs_out = (32, 64)
        head = BHead(n_chs_in, n_chs_out)
        x = torch.randn(batch_size, n_chs_in, seq_len)
        # Act
        output = head(x)
        # Assert
        assert output.shape == (batch_size, n_chs_out, seq_len)

    def test_bhead_gradient_flow_grad(self):
        """Test gradient flow through BHead module."""
        # Arrange
        head = BHead(32, 64)
        x = torch.randn(8, 32, 500, requires_grad=True)
        output = head(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_bhead_gradient_flow_any(self):
        """Test gradient flow through BHead module."""
        # Arrange
        head = BHead(32, 64)
        x = torch.randn(8, 32, 500, requires_grad=True)
        output = head(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.isnan(x.grad).any()


class TestBNet:
    """Test suite for BNet - multi-modal neural network architecture."""

    @pytest.fixture
    def base_bnet_config(self):
        """Provide base configuration for BNet."""
        return {
            "n_bands": 6,
            "SAMP_RATE": 250,
            "n_fc1": 1024,
            "d_ratio1": 0.85,
            "n_fc2": 256,
            "d_ratio2": 0.85,
            "n_chs": [160, 19],  # MEG and EEG channels
            "n_classes": [2, 4],  # Binary and 4-class tasks
        }

    @pytest.fixture
    def mock_mnet_config(self):
        """Provide mock MNet configuration."""
        return {
            "n_chs": 32,  # Will be overridden by BNet
            "some_param": "value",
        }

    def test_bnet_instantiation_basic_isinstance(self, base_bnet_config, mock_mnet_config):
        """Test basic BNet instantiation."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            assert isinstance(model, nn.Module)
            pass
            pass
            pass
            pass

    def test_bnet_instantiation_basic_hasattr(self, base_bnet_config, mock_mnet_config):
        """Test basic BNet instantiation."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            assert hasattr(model, 'heads')
            pass
            pass
            pass

    def test_bnet_instantiation_basic_hasattr_v2(self, base_bnet_config, mock_mnet_config):
        """Test basic BNet instantiation."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            pass
            assert hasattr(model, 'fcs')
            pass
            pass

    def test_bnet_instantiation_basic_len(self, base_bnet_config, mock_mnet_config):
        """Test basic BNet instantiation."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            pass
            pass
            assert len(model.heads) == len(base_bnet_config['n_chs'])
            pass

    def test_bnet_instantiation_basic_len_v2(self, base_bnet_config, mock_mnet_config):
        """Test basic BNet instantiation."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            pass
            pass
            pass
            assert len(model.fcs) == len(base_bnet_config['n_chs'])

    def test_bnet_instantiation_single_modality_len(self, mock_mnet_config):
        """Test BNet with single modality configuration."""
        # Arrange
        config = {'n_bands': 4, 'SAMP_RATE': 1000, 'n_fc1': 512, 'd_ratio1': 0.5, 'n_fc2': 128, 'd_ratio2': 0.5, 'n_chs': [32], 'n_classes': [10]}
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(config, mock_mnet_config)
            assert len(model.heads) == 1
            pass

    def test_bnet_instantiation_single_modality_len_v2(self, mock_mnet_config):
        """Test BNet with single modality configuration."""
        # Arrange
        config = {'n_bands': 4, 'SAMP_RATE': 1000, 'n_fc1': 512, 'd_ratio1': 0.5, 'n_fc2': 128, 'd_ratio2': 0.5, 'n_chs': [32], 'n_classes': [10]}
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(config, mock_mnet_config)
            pass
            assert len(model.fcs) == 1

    def test_bnet_instantiation_many_modalities_len(self, mock_mnet_config):
        """Test BNet with many modalities."""
        # Arrange
        config = {'n_bands': 6, 'SAMP_RATE': 250, 'n_fc1': 1024, 'd_ratio1': 0.85, 'n_fc2': 256, 'd_ratio2': 0.85, 'n_chs': [160, 19, 64, 32, 128], 'n_classes': [2, 4, 3, 5, 2]}
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(config, mock_mnet_config)
            assert len(model.heads) == 5
            pass

    def test_bnet_instantiation_many_modalities_len_v2(self, mock_mnet_config):
        """Test BNet with many modalities."""
        # Arrange
        config = {'n_bands': 6, 'SAMP_RATE': 250, 'n_fc1': 1024, 'd_ratio1': 0.85, 'n_fc2': 256, 'd_ratio2': 0.85, 'n_chs': [160, 19, 64, 32, 128], 'n_classes': [2, 4, 3, 5, 2]}
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(config, mock_mnet_config)
            pass
            assert len(model.fcs) == 5

    def test_bnet_forward_meg_modality(self, base_bnet_config, mock_mnet_config):
        """Test forward pass with MEG data (first modality)."""
        # Arrange
        batch_size, n_chs, seq_len = (16, 160, 1000)
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(batch_size, 1024)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x = torch.randn(batch_size, n_chs, seq_len)
            with patch('ipdb.set_trace'):
                output = model(x, i_head=0)
            assert output.shape == (batch_size, base_bnet_config['n_classes'][0])

    def test_bnet_forward_eeg_modality(self, base_bnet_config, mock_mnet_config):
        """Test forward pass with EEG data (second modality)."""
        # Arrange
        batch_size, n_chs, seq_len = (16, 19, 1000)
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(batch_size, 1024)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x = torch.randn(batch_size, n_chs, seq_len)
            with patch('ipdb.set_trace'):
                output = model(x, i_head=1)
            assert output.shape == (batch_size, base_bnet_config['n_classes'][1])

    def test_bnet_forward_different_sequence_lengths(self, base_bnet_config, mock_mnet_config):
        """Test BNet with various sequence lengths."""
        # Arrange
        batch_size = 8
        seq_lengths = [100, 500, 1000, 2000, 5000]
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(batch_size, 1024)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            for seq_len in seq_lengths:
                x = torch.randn(batch_size, 160, seq_len)
                with patch('ipdb.set_trace'):
                    output = model(x, i_head=0)
                assert output.shape == (batch_size, 2)

    def test_bnet_forward_batch_size_variations(self, base_bnet_config, mock_mnet_config):
        """Test BNet with different batch sizes."""
        # Arrange
        batch_sizes = [1, 4, 16, 32, 64]
        seq_len = 1000
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            model = BNet(base_bnet_config, mock_mnet_config)
            for bs in batch_sizes:
                mock_mnet.return_value.forward_bb.return_value = torch.randn(bs, 1024)
                x = torch.randn(bs, 160, seq_len)
                with patch('ipdb.set_trace'):
                    output = model(x, i_head=0)
                assert output.shape == (bs, 2)

    def test_bnet_znorm_functionality_allclose(self):
        """Test z-normalization static method."""
        # Arrange
        x = torch.randn(16, 32, 1000)
        # Act
        x_norm = BNet._znorm_along_the_last_dim(x)
        # Assert
        assert torch.allclose(x_norm.mean(dim=-1), torch.zeros_like(x_norm.mean(dim=-1)), atol=1e-06)
        pass

    def test_bnet_znorm_functionality_allclose_v2(self):
        """Test z-normalization static method."""
        # Arrange
        x = torch.randn(16, 32, 1000)
        # Act
        x_norm = BNet._znorm_along_the_last_dim(x)
        # Assert
        pass
        assert torch.allclose(x_norm.std(dim=-1), torch.ones_like(x_norm.std(dim=-1)), atol=1e-06)

    def test_bnet_dropout_channels_functionality_hasattr(self, base_bnet_config, mock_mnet_config):
        """Test dropout channels are applied during training."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            model.train()
            assert hasattr(model, 'dc')
            pass

    def test_bnet_dropout_channels_functionality_isinstance(self, base_bnet_config, mock_mnet_config):
        """Test dropout channels are applied during training."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            model.train()
            pass
            assert isinstance(model.dc, scitex_nn.DropoutChannels)

    def test_bnet_frequency_gain_changer_hasattr(self, base_bnet_config, mock_mnet_config):
        """Test frequency gain changer configuration."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            assert hasattr(model, 'fgc')
            pass

    def test_bnet_frequency_gain_changer_isinstance(self, base_bnet_config, mock_mnet_config):
        """Test frequency gain changer configuration."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            assert isinstance(model.fgc, scitex_nn.FreqGainChanger)

    def test_bnet_channel_gain_changers_hasattr(self, base_bnet_config, mock_mnet_config):
        """Test channel gain changers for each modality."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            assert hasattr(model, 'cgcs')
            pass
            for i, cgc in enumerate(model.cgcs):
                pass

    def test_bnet_channel_gain_changers_len(self, base_bnet_config, mock_mnet_config):
        """Test channel gain changers for each modality."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            assert len(model.cgcs) == len(base_bnet_config['n_chs'])
            for i, cgc in enumerate(model.cgcs):
                pass

    def test_bnet_channel_gain_changers_isinstance(self, base_bnet_config, mock_mnet_config):
        """Test channel gain changers for each modality."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            pass
            pass
            for i, cgc in enumerate(model.cgcs):
                assert isinstance(cgc, scitex_nn.ChannelGainChanger)

    @pytest.mark.skipif(True, reason='Mocking MNet prevents proper gradient flow testing')
    def test_bnet_gradient_flow_complete_grad(self, base_bnet_config, mock_mnet_config):
        """Test complete gradient flow through BNet."""
        # Arrange
        batch_size = 4
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(batch_size, 1024, requires_grad=True)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x_meg = torch.randn(batch_size, 160, 1000, requires_grad=True)
            with patch('ipdb.set_trace'):
                y_meg = model(x_meg, i_head=0)
            loss = y_meg.sum()
            loss.backward()
            assert x_meg.grad is not None
            pass

    @pytest.mark.skipif(True, reason='Mocking MNet prevents proper gradient flow testing')
    def test_bnet_gradient_flow_complete_any(self, base_bnet_config, mock_mnet_config):
        """Test complete gradient flow through BNet."""
        # Arrange
        batch_size = 4
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(batch_size, 1024, requires_grad=True)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x_meg = torch.randn(batch_size, 160, 1000, requires_grad=True)
            with patch('ipdb.set_trace'):
                y_meg = model(x_meg, i_head=0)
            loss = y_meg.sum()
            loss.backward()
            pass
            assert not torch.isnan(x_meg.grad).any()

    def test_bnet_invalid_head_index(self, base_bnet_config, mock_mnet_config):
        """Test BNet behavior with invalid head index."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            x = torch.randn(16, 160, 1000)
            with pytest.raises(IndexError):
                with patch('ipdb.set_trace'):
                    model(x, i_head=10)

    def test_bnet_device_compatibility_cpu(self, base_bnet_config, mock_mnet_config):
        """Test BNet on CPU."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(16, 1024)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x = torch.randn(16, 160, 1000)
            with patch('ipdb.set_trace'):
                output = model(x, i_head=0)
            assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_bnet_device_compatibility_gpu(self, base_bnet_config, mock_mnet_config):
        """Test BNet on GPU if available."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(16, 1024).cuda()
            mock_instance.cuda.return_value = mock_instance
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config).cuda()
            x = torch.randn(16, 160, 1000).cuda()
            with patch('ipdb.set_trace'):
                output = model(x, i_head=0)
            assert output.device.type == 'cuda'

    def test_bnet_fc_layer_architecture_len(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                assert len(fc_block) == 6
                pass
                pass
                pass
                pass
                pass
                pass
                pass

    def test_bnet_fc_layer_architecture_isinstance(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                assert isinstance(fc_block[0], nn.Mish)
                pass
                pass
                pass
                pass
                pass
                pass

    def test_bnet_fc_layer_architecture_isinstance_v2(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                pass
                assert isinstance(fc_block[1], nn.Dropout)
                pass
                pass
                pass
                pass
                pass

    def test_bnet_fc_layer_architecture_isinstance_v3(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                pass
                pass
                assert isinstance(fc_block[2], nn.Linear)
                pass
                pass
                pass
                pass

    def test_bnet_fc_layer_architecture_isinstance_v4(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                pass
                pass
                pass
                assert isinstance(fc_block[3], nn.Mish)
                pass
                pass
                pass

    def test_bnet_fc_layer_architecture_isinstance_v5(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                pass
                pass
                pass
                pass
                assert isinstance(fc_block[4], nn.Dropout)
                pass
                pass

    def test_bnet_fc_layer_architecture_isinstance_v6(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                pass
                pass
                pass
                pass
                pass
                assert isinstance(fc_block[5], nn.Linear)
                pass

    def test_bnet_fc_layer_architecture_out_features(self, base_bnet_config, mock_mnet_config):
        """Test fully connected layer architecture for each head."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            for i, fc_block in enumerate(model.fcs):
                pass
                pass
                pass
                pass
                pass
                pass
                pass
                assert fc_block[5].out_features == base_bnet_config['n_classes'][i]

    def test_bnet_virtual_channels_consistency_check1(self, base_bnet_config, mock_mnet_config):
        """Test that virtual channels are consistently set to 32."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            model = BNet(base_bnet_config, mock_mnet_config)
            for head in model.heads:
                pass
            assert mock_mnet_config['n_chs'] == 32

    def test_bnet_virtual_channels_consistency_out_channels(self, base_bnet_config, mock_mnet_config):
        """Test that virtual channels are consistently set to 32."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            model = BNet(base_bnet_config, mock_mnet_config)
            for head in model.heads:
                assert head.conv11.out_channels == 32
            pass

    def test_bnet_multi_modal_integration_shape(self, base_bnet_config, mock_mnet_config):
        """Test integration of multiple modalities in single model."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(8, 1024)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x_meg = torch.randn(8, 160, 1000)
            x_eeg = torch.randn(8, 19, 1000)
            with patch('ipdb.set_trace'):
                y_meg = model(x_meg, i_head=0)
                y_eeg = model(x_eeg, i_head=1)
            assert y_meg.shape == (8, 2)
            pass

    def test_bnet_multi_modal_integration_shape_v2(self, base_bnet_config, mock_mnet_config):
        """Test integration of multiple modalities in single model."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000') as mock_mnet:
            mock_instance = Mock()
            mock_instance.forward_bb.return_value = torch.randn(8, 1024)
            mock_mnet.return_value = mock_instance
            model = BNet(base_bnet_config, mock_mnet_config)
            x_meg = torch.randn(8, 160, 1000)
            x_eeg = torch.randn(8, 19, 1000)
            with patch('ipdb.set_trace'):
                y_meg = model(x_meg, i_head=0)
                y_eeg = model(x_eeg, i_head=1)
            pass
            assert y_eeg.shape == (8, 4)

    def test_bnet_parameter_count_total_params(self, base_bnet_config, mock_mnet_config):
        """Test total parameter count is reasonable."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            total_params = sum((p.numel() for p in model.parameters()))
            trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
            assert total_params > 100000
            pass
            pass

    def test_bnet_parameter_count_total_params_v2(self, base_bnet_config, mock_mnet_config):
        """Test total parameter count is reasonable."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            total_params = sum((p.numel() for p in model.parameters()))
            trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
            pass
            assert total_params < 100000000
            pass

    def test_bnet_parameter_count_trainable_params(self, base_bnet_config, mock_mnet_config):
        """Test total parameter count is reasonable."""
        # Arrange
        # Act
        # Assert
        with patch('scitex_nn._BNet.MNet_1000'):
            model = BNet(base_bnet_config, mock_mnet_config)
            total_params = sum((p.numel() for p in model.parameters()))
            trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
            pass
            pass
            assert trainable_params == total_params


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_BNet.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-05-15 16:44:27 (ywatanabe)"
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import numpy as np
#
# # Import specific nn modules to avoid circular imports
# from ._SpatialAttention import SpatialAttention
# from ._SwapChannels import SwapChannels
# from ._DropoutChannels import DropoutChannels
# from ._FreqGainChanger import FreqGainChanger
# from ._ChannelGainChanger import ChannelGainChanger
# from ._MNet_1000 import MNet_1000
#
#
# class BHead(nn.Module):
#     def __init__(self, n_chs_in, n_chs_out):
#         super().__init__()
#         self.sa = SpatialAttention(n_chs_in)
#         self.conv11 = nn.Conv1d(
#             in_channels=n_chs_in, out_channels=n_chs_out, kernel_size=1
#         )
#
#     def forward(self, x):
#         x = self.sa(x)
#         x = self.conv11(x)
#         return x
#
#
# class BNet(nn.Module):
#     def __init__(self, BNet_config, MNet_config):
#         super().__init__()
#         self.dummy_param = nn.Parameter(torch.empty(0))
#         N_VIRTUAL_CHS = 32
#
#         self.sc = SwapChannels()
#         self.dc = DropoutChannels(dropout=0.01)
#         self.fgc = FreqGainChanger(BNet_config["n_bands"], BNet_config["SAMP_RATE"])
#         self.heads = nn.ModuleList(
#             [
#                 BHead(n_ch, N_VIRTUAL_CHS).to(self.dummy_param.device)
#                 for n_ch in BNet_config["n_chs"]
#             ]
#         )
#
#         self.cgcs = [ChannelGainChanger(n_ch) for n_ch in BNet_config["n_chs"]]
#         # self.cgc = ChannelGainChanger(N_VIRTUAL_CHS)
#
#         MNet_config["n_chs"] = N_VIRTUAL_CHS  # BNet_config["n_chs"] # override
#         self.MNet = MNet_1000(MNet_config)
#
#         self.fcs = nn.ModuleList(
#             [
#                 nn.Sequential(
#                     # nn.Linear(N_FC_IN, config["n_fc1"]),
#                     nn.Mish(),
#                     nn.Dropout(BNet_config["d_ratio1"]),
#                     nn.Linear(BNet_config["n_fc1"], BNet_config["n_fc2"]),
#                     nn.Mish(),
#                     nn.Dropout(BNet_config["d_ratio2"]),
#                     nn.Linear(BNet_config["n_fc2"], BNet_config["n_classes"][i_head]),
#                 )
#                 for i_head, _ in enumerate(range(len(BNet_config["n_chs"])))
#             ]
#         )
#
#     @staticmethod
#     def _znorm_along_the_last_dim(x):
#         return (x - x.mean(dim=-1, keepdims=True)) / x.std(dim=-1, keepdims=True)
#
#     def forward(self, x, i_head):
#         x = self._znorm_along_the_last_dim(x)
#         # x = self.sc(x)
#         x = self.dc(x)
#         x = self.fgc(x)
#         x = self.cgcs[i_head](x)
#         x = self.heads[i_head](x)
#         import ipdb
#
#         ipdb.set_trace()
#         # x = self.cgc(x)
#         x = self.MNet.forward_bb(x)
#         x = self.fcs[i_head](x)
#         return x
#
#
# # BNet_config = {
# #     "n_chs": 32,
# #     "n_bands": 6,
# #     "SAMP_RATE": 1000,
# # }
# BNet_config = {
#     "n_bands": 6,
#     "SAMP_RATE": 250,
#     # "n_chs": 270,
#     "n_fc1": 1024,
#     "d_ratio1": 0.85,
#     "n_fc2": 256,
#     "d_ratio2": 0.85,
# }
#
#
# if __name__ == "__main__":
#     ## Demo data
#     # MEG
#     BS, N_CHS, SEQ_LEN = 16, 160, 1000
#     x_MEG = torch.rand(BS, N_CHS, SEQ_LEN).cuda()
#     # EEG
#     BS, N_CHS, SEQ_LEN = 16, 19, 1000
#     x_EEG = torch.rand(BS, N_CHS, SEQ_LEN).cuda()
#
#     # model = MNetBackBorn(scitex_nn.MNet_config).cuda()
#     # model(x_MEG)
#     # Model
#     BNet_config["n_chs"] = [160, 19]
#     BNet_config["n_classes"] = [2, 4]
#     model = BNet(BNet_config, scitex_nn.MNet_config).cuda()
#
#     # MEG
#     y = model(x_MEG, 0)
#     y = model(x_EEG, 1)
#
#     # # EEG
#     # y = model(x_EEG)
#
#     y.sum().backward()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_BNet.py
# --------------------------------------------------------------------------------
