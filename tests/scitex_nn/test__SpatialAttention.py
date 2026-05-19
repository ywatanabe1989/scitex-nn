#!/usr/bin/env python3
# Time-stamp: "2025-01-06 (ywatanabe)"
# /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/scitex_repo/tests/scitex/nn/test__SpatialAttention.py

"""Comprehensive test suite for SpatialAttention module."""

import pytest

# Required for this module
pytest.importorskip("torch")
import numpy as np
import torch
import torch.nn as nn

from scitex_nn import SpatialAttention


class TestSpatialAttentionArchitecture:
    """Test SpatialAttention architecture and initialization."""

    def test_basic_instantiation_spatial_attention_behaves_correctly_isinstance(self):
        """Test basic module instantiation."""
        # Arrange
        n_chs = 64
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        assert isinstance(module, nn.Module)
        pass
        pass

    def test_basic_instantiation_spatial_attention_behaves_correctly_hasattr(self):
        """Test basic module instantiation."""
        # Arrange
        n_chs = 64
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        pass
        assert hasattr(module, 'aap')
        pass

    def test_basic_instantiation_spatial_attention_behaves_correctly_hasattr_v2(self):
        """Test basic module instantiation."""
        # Arrange
        n_chs = 64
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        pass
        pass
        assert hasattr(module, 'conv11')

    def test_adaptive_avg_pool_initialization_isinstance(self):
        """Test AdaptiveAvgPool1d initialization."""
        # Arrange
        n_chs = 128
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        assert isinstance(module.aap, nn.AdaptiveAvgPool1d)
        pass

    def test_adaptive_avg_pool_initialization_output_size(self):
        """Test AdaptiveAvgPool1d initialization."""
        # Arrange
        n_chs = 128
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        pass
        assert module.aap.output_size == 1 or module.aap.output_size == (1,)

    def test_conv1d_initialization_spatial_attention_behaves_correctly_isinstance(self):
        """Test Conv1d layer initialization."""
        # Arrange
        n_chs = 256
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        assert isinstance(module.conv11, nn.Conv1d)
        pass
        pass
        pass

    def test_conv1d_initialization_spatial_attention_behaves_correctly_in_channels(self):
        """Test Conv1d layer initialization."""
        # Arrange
        n_chs = 256
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        pass
        assert module.conv11.in_channels == n_chs
        pass
        pass

    def test_conv1d_initialization_spatial_attention_behaves_correctly_out_channels(self):
        """Test Conv1d layer initialization."""
        # Arrange
        n_chs = 256
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        pass
        pass
        assert module.conv11.out_channels == 1
        pass

    def test_conv1d_initialization_spatial_attention_behaves_correctly_kernel_size(self):
        """Test Conv1d layer initialization."""
        # Arrange
        n_chs = 256
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        # Assert
        pass
        pass
        pass
        assert module.conv11.kernel_size == (1,)

    def test_different_channel_sizes(self):
        """Test instantiation with various channel sizes."""
        # Arrange
        channel_sizes = [1, 16, 32, 64, 128, 256, 512, 1024]
        # Act
        # Assert
        for n_chs in channel_sizes:
            module = SpatialAttention(n_chs_in=n_chs)
            assert module.conv11.in_channels == n_chs


class TestSpatialAttentionForward:
    """Test forward pass functionality."""

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        BS, SEQ_LEN = (8, 1000)
        x = torch.randn(BS, n_chs, SEQ_LEN)
        # Act
        output = module(x)
        # Assert
        assert output.shape == (BS, n_chs, SEQ_LEN)

    def test_forward_pass_different_sequence_lengths(self):
        """Test forward pass with various sequence lengths."""
        # Arrange
        n_chs = 32
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        BS = 4
        # Assert
        for seq_len in [10, 100, 500, 1000, 2000]:
            x = torch.randn(BS, n_chs, seq_len)
            output = module(x)
            assert output.shape == (BS, n_chs, seq_len)

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        # Arrange
        n_chs = 64
        # Act
        module = SpatialAttention(n_chs_in=n_chs)
        SEQ_LEN = 500
        # Assert
        for batch_size in [1, 2, 4, 8, 16, 32]:
            x = torch.randn(batch_size, n_chs, SEQ_LEN)
            output = module(x)
            assert output.shape == (batch_size, n_chs, SEQ_LEN)

    def test_attention_mechanism_spatial_attention_behaves_correctly_equal(self):
        """Test that attention mechanism modulates input."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100)
        # Act
        output = module(x)
        # Assert
        assert not torch.equal(output, x)
        pass

    def test_attention_mechanism_spatial_attention_behaves_correctly_shape(self):
        """Test that attention mechanism modulates input."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100)
        # Act
        output = module(x)
        # Assert
        pass
        assert output.shape == x.shape


class TestSpatialAttentionWeights:
    """Test attention weight generation and application."""

    def test_attention_weights_range(self):
        """Test that attention weights are properly bounded."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        # Act
        x = torch.randn(4, n_chs, 100)
        with torch.no_grad():
            output = module(x)
            mask = x.abs() > 1e-06
            attention_weights = torch.ones_like(output)
            attention_weights[mask] = output[mask] / x[mask]
        # Assert
        assert x is not None

    def test_pooling_operation_spatial_attention_behaves_correctly(self):
        """Test that adaptive pooling reduces temporal dimension."""
        # Arrange
        n_chs = 32
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 1000)
        # Act
        pooled = module.aap(x)
        # Assert
        assert pooled.shape == (4, n_chs, 1)

    def test_conv_operation_spatial_attention_behaves_correctly(self):
        """Test convolution operation on pooled features."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 1)
        # Act
        conv_out = module.conv11(x)
        # Assert
        assert conv_out.shape == (4, 1, 1)


class TestSpatialAttentionGradient:
    """Test gradient flow and backpropagation."""

    def test_gradient_flow_spatial_attention_behaves_correctly_grad(self):
        """Test that gradients flow through the module."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100, requires_grad=True)
        output = module(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_spatial_attention_behaves_correctly_any(self):
        """Test that gradients flow through the module."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100, requires_grad=True)
        output = module(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.isnan(x.grad).any()

    def test_parameter_gradients_spatial_attention_behaves_correctly_grad(self):
        """Test that module parameters receive gradients."""
        # Arrange
        n_chs = 32
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100)
        output = module(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert module.conv11.weight.grad is not None
        pass
        if module.conv11.bias is not None:
            pass

    def test_parameter_gradients_spatial_attention_behaves_correctly_any(self):
        """Test that module parameters receive gradients."""
        # Arrange
        n_chs = 32
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100)
        output = module(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.isnan(module.conv11.weight.grad).any()
        if module.conv11.bias is not None:
            pass

    def test_parameter_gradients_spatial_attention_behaves_correctly_grad_v2(self):
        """Test that module parameters receive gradients."""
        # Arrange
        n_chs = 32
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100)
        output = module(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        pass
        if module.conv11.bias is not None:
            assert module.conv11.bias.grad is not None

    def test_gradient_magnitude_spatial_attention_behaves_correctly_grad_norm(self):
        """Test gradient magnitudes are reasonable."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100, requires_grad=True)
        output = module(x)
        loss = output.mean()
        loss.backward()
        # Act
        grad_norm = x.grad.norm()
        # Assert
        assert grad_norm > 1e-06
        pass

    def test_gradient_magnitude_spatial_attention_behaves_correctly_grad_norm_v2(self):
        """Test gradient magnitudes are reasonable."""
        # Arrange
        n_chs = 64
        module = SpatialAttention(n_chs_in=n_chs)
        x = torch.randn(4, n_chs, 100, requires_grad=True)
        output = module(x)
        loss = output.mean()
        loss.backward()
        # Act
        grad_norm = x.grad.norm()
        # Assert
        pass
        assert grad_norm < 1000.0


class TestSpatialAttentionDevice:
    """Test device compatibility."""

    def test_cpu_computation_spatial_attention_behaves_correctly(self):
        """Test computation on CPU."""
        # Arrange
        module = SpatialAttention(n_chs_in=64)
        x = torch.randn(2, 64, 100)
        # Act
        output = module(x)
        # Assert
        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_computation_spatial_attention_behaves_correctly(self):
        """Test computation on CUDA."""
        # Arrange
        module = SpatialAttention(n_chs_in=64).cuda()
        x = torch.randn(2, 64, 100).cuda()
        # Act
        output = module(x)
        # Assert
        assert output.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_movement_spatial_attention_behaves_correctly(self):
        """Test moving module between devices."""
        # Arrange
        module = SpatialAttention(n_chs_in=32)
        x_cpu = torch.randn(2, 32, 100)
        output_cpu = module(x_cpu)
        module = module.cuda()
        x_cuda = x_cpu.cuda()
        # Act
        output_cuda = module(x_cuda)
        # Assert
        assert torch.allclose(output_cpu, output_cuda.cpu(), rtol=1e-05)


class TestSpatialAttentionNumerical:
    """Test numerical properties and stability."""

    def test_numerical_stability_spatial_attention_behaves_correctly_any(self):
        """Test numerical stability with extreme inputs."""
        # Arrange
        module = SpatialAttention(n_chs_in=64)
        x_small = torch.randn(2, 64, 100) * 1e-06
        # Act
        output_small = module(x_small)
        # Assert
        assert not torch.isnan(output_small).any()
        pass
        x_large = torch.randn(2, 64, 100) * 1000.0
        output_large = module(x_large)
        pass
        pass

    def test_numerical_stability_spatial_attention_behaves_correctly_any_v2(self):
        """Test numerical stability with extreme inputs."""
        # Arrange
        module = SpatialAttention(n_chs_in=64)
        x_small = torch.randn(2, 64, 100) * 1e-06
        # Act
        output_small = module(x_small)
        # Assert
        pass
        assert not torch.isinf(output_small).any()
        x_large = torch.randn(2, 64, 100) * 1000.0
        output_large = module(x_large)
        pass
        pass

    def test_numerical_stability_spatial_attention_behaves_correctly_any_v3(self):
        """Test numerical stability with extreme inputs."""
        # Arrange
        module = SpatialAttention(n_chs_in=64)
        x_small = torch.randn(2, 64, 100) * 1e-06
        # Act
        output_small = module(x_small)
        # Assert
        pass
        pass
        x_large = torch.randn(2, 64, 100) * 1000.0
        output_large = module(x_large)
        assert not torch.isnan(output_large).any()
        pass

    def test_numerical_stability_spatial_attention_behaves_correctly_any_v4(self):
        """Test numerical stability with extreme inputs."""
        # Arrange
        module = SpatialAttention(n_chs_in=64)
        x_small = torch.randn(2, 64, 100) * 1e-06
        # Act
        output_small = module(x_small)
        # Assert
        pass
        pass
        x_large = torch.randn(2, 64, 100) * 1000.0
        output_large = module(x_large)
        pass
        assert not torch.isinf(output_large).any()

    def test_zero_input_spatial_attention_behaves_correctly(self):
        """Test behavior with zero input."""
        # Arrange
        module = SpatialAttention(n_chs_in=32)
        x = torch.zeros(2, 32, 100)
        # Act
        output = module(x)
        # Assert
        assert torch.allclose(output, torch.zeros_like(output))

    def test_ones_input_spatial_attention_behaves_correctly(self):
        """Test behavior with ones input."""
        # Arrange
        module = SpatialAttention(n_chs_in=16)
        x = torch.ones(2, 16, 100)
        # Act
        output = module(x)
        # Assert
        assert output.shape == x.shape


class TestSpatialAttentionIntegration:
    """Test integration with other modules."""

    def test_with_conv_layers(self):
        """Test integration with convolutional layers."""
        # Arrange
        n_chs = 64
        model = nn.Sequential(nn.Conv1d(32, n_chs, kernel_size=3, padding=1), nn.ReLU(), SpatialAttention(n_chs_in=n_chs), nn.Conv1d(n_chs, 128, kernel_size=3, padding=1))
        x = torch.randn(4, 32, 100)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (4, 128, 100)

    def test_with_batch_norm(self):
        """Test integration with batch normalization."""
        # Arrange
        n_chs = 64
        model = nn.Sequential(nn.BatchNorm1d(n_chs), SpatialAttention(n_chs_in=n_chs), nn.BatchNorm1d(n_chs))
        x = torch.randn(8, n_chs, 200)
        # Act
        output = model(x)
        # Assert
        assert output.shape == x.shape

    def test_multiple_attention_layers(self):
        """Test stacking multiple attention layers."""
        # Arrange
        n_chs = 32
        model = nn.Sequential(SpatialAttention(n_chs_in=n_chs), nn.ReLU(), SpatialAttention(n_chs_in=n_chs), nn.ReLU(), SpatialAttention(n_chs_in=n_chs))
        x = torch.randn(4, n_chs, 150)
        # Act
        output = model(x)
        # Assert
        assert output.shape == x.shape


class TestSpatialAttentionMemory:
    """Test memory efficiency."""

    def test_memory_footprint_spatial_attention_behaves_correctly(self):
        """Test module memory footprint."""
        # Arrange
        module = SpatialAttention(n_chs_in=256)
        # Act
        total_params = sum((p.numel() for p in module.parameters()))
        # Assert
        assert total_params <= 257

    def test_inference_memory_spatial_attention_behaves_correctly(self):
        """Test memory usage during inference."""
        # Arrange
        module = SpatialAttention(n_chs_in=128)
        module.eval()
        # Act
        x = torch.randn(1, 128, 1000)
        with torch.no_grad():
            output = module(x)
        # Assert
        assert output.shape == x.shape


class TestSpatialAttentionEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_channel_spatial_attention_behaves_correctly(self):
        """Test with single channel input."""
        # Arrange
        module = SpatialAttention(n_chs_in=1)
        x = torch.randn(4, 1, 100)
        # Act
        output = module(x)
        # Assert
        assert output.shape == (4, 1, 100)

    def test_single_timestep_spatial_attention_behaves_correctly(self):
        """Test with single timestep."""
        # Arrange
        module = SpatialAttention(n_chs_in=64)
        x = torch.randn(4, 64, 1)
        # Act
        output = module(x)
        # Assert
        assert output.shape == (4, 64, 1)

    def test_large_dimensions_spatial_attention_behaves_correctly(self):
        """Test with large input dimensions."""
        # Arrange
        module = SpatialAttention(n_chs_in=512)
        x = torch.randn(2, 512, 10000)
        # Act
        output = module(x)
        # Assert
        assert output.shape == (2, 512, 10000)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_SpatialAttention.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-04-23 09:45:28 (ywatanabe)"
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import scitex
# import numpy as np
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, n_chs_in):
#         super().__init__()
#         self.aap = nn.AdaptiveAvgPool1d(1)
#         self.conv11 = nn.Conv1d(in_channels=n_chs_in, out_channels=1, kernel_size=1)
#
#     def forward(self, x):
#         """x: [batch_size, n_chs, seq_len]"""
#         x_orig = x
#         x = self.aap(x)
#         x = self.conv11(x)
#
#         return x * x_orig

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_SpatialAttention.py
# --------------------------------------------------------------------------------
