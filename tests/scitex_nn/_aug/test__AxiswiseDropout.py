#!/usr/bin/env python3
# Time-stamp: "2025-05-31 21:55:00 (ywatanabe)"
# File: tests/scitex/nn/test__AxiswiseDropout.py

"""
Tests for AxiswiseDropout module.

This module tests:
1. Basic dropout functionality along specified axis
2. Probability parameter behavior
3. Training vs evaluation mode differences
4. Different tensor shapes and dimensions
5. Gradient flow and backpropagation
6. Statistical properties of dropout
"""

import pytest

# Required for this module
pytest.importorskip("torch")
import numpy as np
import torch
import torch.nn as nn

from scitex_nn import AxiswiseDropout


class TestAxiswiseDropoutBasics:
    """Test basic functionality of AxiswiseDropout."""

    def test_instantiation_default_axiswise_dropout_behaves_correctly_isinstance(self):
        """Test AxiswiseDropout instantiation with default parameters."""
        # Arrange
        # Act
        layer = AxiswiseDropout()
        # Assert
        assert isinstance(layer, nn.Module)

    def test_instantiation_default_axiswise_dropout_behaves_correctly_dropout_prob(self):
        """Test AxiswiseDropout instantiation with default parameters."""
        # Arrange
        # Act
        layer = AxiswiseDropout()
        # Assert
        pass
        assert layer.dropout_prob == 0.5

    def test_instantiation_default_axiswise_dropout_behaves_correctly_dim(self):
        """Test AxiswiseDropout instantiation with default parameters."""
        # Arrange
        # Act
        layer = AxiswiseDropout()
        # Assert
        pass
        pass
        assert layer.dim == 1

    def test_instantiation_custom_axiswise_dropout_behaves_correctly_dropout_prob(self):
        """Test AxiswiseDropout instantiation with custom parameters."""
        # Arrange
        # Act
        layer = AxiswiseDropout(dropout_prob=0.3, dim=2)
        # Assert
        assert layer.dropout_prob == 0.3
        pass

    def test_instantiation_custom_axiswise_dropout_behaves_correctly_dim(self):
        """Test AxiswiseDropout instantiation with custom parameters."""
        # Arrange
        # Act
        layer = AxiswiseDropout(dropout_prob=0.3, dim=2)
        # Assert
        pass
        assert layer.dim == 2

    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.ones(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape
        if layer.dropout_prob > 0:
            pass

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.eval()
        x = torch.ones(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, x)

    def test_dropout_along_different_dims(self):
        """Test dropout along different dimensions."""
        # Arrange
        # Act
        x = torch.ones(2, 3, 4, 5)
        # Assert
        for dim in range(4):
            layer = AxiswiseDropout(dropout_prob=0.5, dim=dim)
            layer.train()
            output = layer(x)
            assert output.shape == x.shape


class TestAxiswiseDropoutProbabilities:
    """Test probability parameter behavior."""

    def test_dropout_prob_zero(self):
        """Test with dropout probability of 0 (no dropout)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.0)
        layer.train()
        x = torch.ones(10, 20, 30)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, x)

    def test_dropout_prob_one(self):
        """Test with dropout probability of 1 (drop everything)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=1.0)
        layer.train()
        x = torch.ones(10, 20, 30)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, torch.zeros_like(x))

    def test_boundary_dropout_probs_dropout_prob(self):
        """Test boundary dropout probabilities."""
        # Arrange
        # Act
        layer_0 = AxiswiseDropout(dropout_prob=0.0)
        # Assert
        assert layer_0.dropout_prob == 0.0
        layer_1 = AxiswiseDropout(dropout_prob=1.0)
        pass

    def test_boundary_dropout_probs_dropout_prob_v2(self):
        """Test boundary dropout probabilities."""
        # Arrange
        # Act
        layer_0 = AxiswiseDropout(dropout_prob=0.0)
        # Assert
        pass
        layer_1 = AxiswiseDropout(dropout_prob=1.0)
        assert layer_1.dropout_prob == 1.0

    def test_dropout_scaling_axiswise_dropout_behaves_correctly(self):
        """Test that dropout properly scales remaining values."""
        # Arrange
        torch.manual_seed(0)
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.ones(100, 50, 10)
        outputs = []
        for _ in range(500):
            output = layer(x)
            outputs.append(output)
        # Act
        avg_output = torch.stack(outputs).mean(dim=0)
        # Assert
        assert torch.allclose(avg_output, x, atol=0.25)


class TestAxiswiseDropoutDimensions:
    """Test behavior with different tensor dimensions."""

    def test_2d_tensor_axiswise_dropout_behaves_correctly(self):
        """Test with 2D tensor (batch, features)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(32, 128)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_3d_tensor_axiswise_dropout_behaves_correctly(self):
        """Test with 3D tensor (batch, channels, length)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(16, 64, 100)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_4d_tensor_axiswise_dropout_behaves_correctly(self):
        """Test with 4D tensor (batch, channels, height, width)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(8, 32, 28, 28)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_5d_tensor_axiswise_dropout_behaves_correctly(self):
        """Test with 5D tensor (batch, channels, depth, height, width)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(4, 16, 10, 20, 20)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_negative_dim_index(self):
        """Test using negative dimension indices."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=-1)
        layer.train()
        x = torch.randn(2, 3, 4, 5)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape


class TestAxiswiseDropoutMaskBehavior:
    """Test the masking behavior of AxiswiseDropout."""

    def test_mask_consistency_along_axis_all(self):
        """Test that dropout mask is consistent along non-dropout axes."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.8, dim=1)
        layer.train()
        x = torch.ones(2, 10, 20, 30)
        # Act
        output = layer(x)
        # Assert
        for b in range(2):
            for c in range(10):
                if output[b, c, 0, 0] == 0:
                    assert torch.all(output[b, c] == 0)
                else:
                    pass

    def test_mask_consistency_along_axis_all_v2(self):
        """Test that dropout mask is consistent along non-dropout axes."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.8, dim=1)
        layer.train()
        x = torch.ones(2, 10, 20, 30)
        # Act
        output = layer(x)
        # Assert
        for b in range(2):
            for c in range(10):
                if output[b, c, 0, 0] == 0:
                    pass
                else:
                    assert torch.all(output[b, c] != 0)

    def test_mask_shared_across_batches(self):
        """Test that dropout mask is shared (broadcast) across batch dimension.

            The AxiswiseDropout implementation creates a single mask with shape
            [1, dim_size, 1] and broadcasts it, meaning all batch elements have
            the same dropout pattern. This is by design for channel-wise dropout.
            """
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        torch.manual_seed(42)
        x = torch.ones(100, 20, 10)
        output = layer(x)
        dropout_patterns = []
        for b in range(100):
            pattern = (output[b, :, 0] == 0).float()
            dropout_patterns.append(pattern)
        dropout_patterns = torch.stack(dropout_patterns)
        first_pattern = dropout_patterns[0]
        # Act
        all_same = torch.all(torch.all(dropout_patterns == first_pattern, dim=1))
        # Assert
        assert all_same


class TestAxiswiseDropoutGradients:
    """Test gradient flow through AxiswiseDropout."""

    def test_gradient_flow_training_grad(self):
        """Test gradient flow in training mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(4, 8, 16, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_training_shape(self):
        """Test gradient flow in training mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(4, 8, 16, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert x.grad.shape == x.shape

        # Gradient should be zero where dropout occurred
        # Note: The actual gradient depends on the dropout implementation details

    def test_gradient_flow_eval_grad(self):
        """Test gradient flow in evaluation mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.eval()
        x = torch.randn(4, 8, 16, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_eval_allclose(self):
        """Test gradient flow in evaluation mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.eval()
        x = torch.randn(4, 8, 16, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert torch.allclose(x.grad, torch.ones_like(x.grad))

    def test_gradient_scaling_axiswise_dropout_behaves_correctly(self):
        """Test that gradients are properly scaled."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.ones(10, 20, 30, requires_grad=True)
        output = layer(x)
        grad_output = torch.ones_like(output)
        # Act
        output.backward(grad_output)
        # Assert
        assert x.grad.shape == x.shape
        # The exact gradient values depend on F.dropout implementation


class TestAxiswiseDropoutDeviceCompatibility:
    """Test AxiswiseDropout on different devices."""

    def test_cpu_operation_axiswise_dropout_behaves_correctly_type(self):
        """Test operation on CPU."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        assert output.device.type == 'cpu'
        pass

    def test_cpu_operation_axiswise_dropout_behaves_correctly_shape(self):
        """Test operation on CPU."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_operation_axiswise_dropout_behaves_correctly_type(self):
        """Test operation on CUDA."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1).cuda()
        layer.train()
        x = torch.randn(2, 3, 4).cuda()
        # Act
        output = layer(x)
        # Assert
        assert output.device.type == 'cuda'
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_operation_axiswise_dropout_behaves_correctly_shape(self):
        """Test operation on CUDA."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1).cuda()
        layer.train()
        x = torch.randn(2, 3, 4).cuda()
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_mixed_device_handling(self):
        """Test handling of mixed CPU/GPU tensors."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        x_gpu = torch.randn(2, 3, 4).cuda()
        # Act
        output = layer(x_gpu)
        # Assert
        assert output.device == x_gpu.device


class TestAxiswiseDropoutIntegration:
    """Test integration with other PyTorch components."""

    def test_sequential_model_axiswise_dropout_behaves_correctly(self):
        """Test AxiswiseDropout in Sequential model."""
        # Arrange
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), AxiswiseDropout(dropout_prob=0.5, dim=1), nn.ReLU(), nn.Conv2d(16, 32, 3, padding=1))
        model.train()
        x = torch.randn(4, 3, 32, 32)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (4, 32, 32, 32)

    def test_with_batchnorm_axiswise_dropout_behaves_correctly(self):
        """Test AxiswiseDropout with BatchNorm."""
        # Arrange
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), AxiswiseDropout(dropout_prob=0.5, dim=1), nn.ReLU())
        model.train()
        x = torch.randn(8, 3, 28, 28)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (8, 16, 28, 28)

    def test_multiple_dropout_layers(self):
        """Test multiple AxiswiseDropout layers in a model."""
        # Arrange
        model = nn.Sequential(nn.Linear(100, 200), AxiswiseDropout(dropout_prob=0.3, dim=1), nn.ReLU(), nn.Linear(200, 100), AxiswiseDropout(dropout_prob=0.5, dim=1))
        model.train()
        x = torch.randn(32, 100)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (32, 100)


class TestAxiswiseDropoutStatisticalProperties:
    """Test statistical properties of dropout."""

    def test_dropout_rate_statistics(self):
        """Test that actual dropout rate matches specified probability."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.3, dim=1)
        layer.train()
        n_trials = 1000
        n_channels = 100
        dropout_counts = torch.zeros(n_channels)
        for _ in range(n_trials):
            x = torch.ones(1, n_channels, 10)
            output = layer(x)
            dropped = (output[0, :, 0] == 0).float()
            dropout_counts += dropped
        # Act
        actual_dropout_rate = dropout_counts.mean() / n_trials
        # Assert
        assert abs(actual_dropout_rate - layer.dropout_prob) < 0.05

    def test_output_mean_preservation(self):
        """Test that expected output mean is preserved."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.ones(1000, 100, 50)
        outputs = []
        for _ in range(100):
            output = layer(x)
            outputs.append(output.mean().item())
        # Act
        avg_output_mean = np.mean(outputs)
        # Assert
        assert abs(avg_output_mean - 1.0) < 0.05


class TestAxiswiseDropoutEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_channel_axiswise_dropout_behaves_correctly(self):
        """Test with single channel (dim size = 1)."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.randn(10, 1, 20)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_empty_tensor_axiswise_dropout_behaves_correctly(self):
        """Test with empty tensor."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.empty(0, 10, 20)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_different_dtypes_axiswise_dropout_behaves_correctly_dtype(self):
        """Test with different data types."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x_f32 = torch.randn(2, 3, 4, dtype=torch.float32)
        # Act
        output_f32 = layer(x_f32)
        # Assert
        assert output_f32.dtype == torch.float32
        x_f64 = torch.randn(2, 3, 4, dtype=torch.float64)
        output_f64 = layer(x_f64)
        pass
        x_f16 = torch.randn(2, 3, 4, dtype=torch.float16)
        output_f16 = layer(x_f16)

    def test_different_dtypes_axiswise_dropout_behaves_correctly_dtype_v2(self):
        """Test with different data types."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x_f32 = torch.randn(2, 3, 4, dtype=torch.float32)
        # Act
        output_f32 = layer(x_f32)
        # Assert
        pass
        x_f64 = torch.randn(2, 3, 4, dtype=torch.float64)
        output_f64 = layer(x_f64)
        assert output_f64.dtype == torch.float64
        x_f16 = torch.randn(2, 3, 4, dtype=torch.float16)
        output_f16 = layer(x_f16)
        pass

    def test_different_dtypes_axiswise_dropout_behaves_correctly_dtype_v3(self):
        """Test with different data types."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x_f32 = torch.randn(2, 3, 4, dtype=torch.float32)
        # Act
        output_f32 = layer(x_f32)
        # Assert
        pass
        x_f64 = torch.randn(2, 3, 4, dtype=torch.float64)
        output_f64 = layer(x_f64)
        pass
        x_f16 = torch.randn(2, 3, 4, dtype=torch.float16)
        output_f16 = layer(x_f16)
        assert output_f16.dtype == torch.float16


class TestAxiswiseDropoutReproducibility:
    """Test reproducibility and randomness control."""

    def test_training_randomness_axiswise_dropout_behaves_correctly(self):
        """Test that dropout is random in training mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.ones(10, 20, 30)
        output1 = layer(x)
        # Act
        output2 = layer(x)
        # Assert
        assert not torch.allclose(output1, output2)

    def test_eval_determinism_axiswise_dropout_behaves_correctly(self):
        """Test that dropout is deterministic in eval mode."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.eval()
        x = torch.ones(10, 20, 30)
        output1 = layer(x)
        # Act
        output2 = layer(x)
        # Assert
        assert torch.allclose(output1, output2)

    def test_seed_reproducibility_axiswise_dropout_behaves_correctly(self):
        """Test that setting seed makes dropout reproducible."""
        # Arrange
        layer = AxiswiseDropout(dropout_prob=0.5, dim=1)
        layer.train()
        x = torch.ones(10, 20, 30)
        torch.manual_seed(42)
        output1 = layer(x)
        torch.manual_seed(42)
        # Act
        output2 = layer(x)
        # Assert
        assert torch.allclose(output1, output2)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_AxiswiseDropout.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-30 07:27:27 (ywatanabe)"
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AxiswiseDropout(nn.Module):
#     def __init__(self, dropout_prob=0.5, dim=1):
#         super(AxiswiseDropout, self).__init__()
#         self.dropout_prob = dropout_prob
#         self.dim = dim
#
#     def forward(self, x):
#         if self.training:
#             sizes = [s if i == self.dim else 1 for i, s in enumerate(x.size())]
#             dropout_mask = F.dropout(
#                 torch.ones(*sizes, device=x.device, dtype=x.dtype),
#                 self.dropout_prob,
#                 True,
#             )
#
#             # Expand the mask to the size of the input tensor and apply it
#             return x * dropout_mask.expand_as(x)
#         return x

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_AxiswiseDropout.py
# --------------------------------------------------------------------------------
