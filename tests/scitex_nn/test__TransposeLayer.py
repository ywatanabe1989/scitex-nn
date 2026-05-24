#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-05-31 21:50:00 (ywatanabe)"
# File: tests/scitex/nn/test__TransposeLayer.py

"""
Tests for TransposeLayer module.

This module tests:
1. Basic transpose functionality
2. Different axis combinations
3. Various tensor shapes (2D, 3D, 4D, 5D)
4. Error handling and edge cases
5. Integration with PyTorch autograd
6. Device compatibility (CPU/GPU)
"""

import pytest

# Required for this module
pytest.importorskip("torch")
import numpy as np
import torch
import torch.nn as nn

from scitex_nn import TransposeLayer


class TestTransposeLayerBasics:
    """Test basic functionality of TransposeLayer."""

    def test_instantiation_transpose_layer_behaves_correctly_isinstance(self):
        """Test that TransposeLayer can be instantiated with valid parameters."""
        # Arrange
        # Act
        layer = TransposeLayer(0, 1)
        # Assert
        assert isinstance(layer, nn.Module)

    def test_instantiation_transpose_layer_behaves_correctly_axis1(self):
        """Test that TransposeLayer can be instantiated with valid parameters."""
        # Arrange
        # Act
        layer = TransposeLayer(0, 1)
        # Assert
        pass
        assert layer.axis1 == 0

    def test_instantiation_transpose_layer_behaves_correctly_axis2(self):
        """Test that TransposeLayer can be instantiated with valid parameters."""
        # Arrange
        # Act
        layer = TransposeLayer(0, 1)
        # Assert
        pass
        pass
        assert layer.axis2 == 1

    def test_forward_2d_tensor_shape(self):
        """Test transpose operation on 2D tensors."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(3, 4)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (4, 3)
        pass

    def test_forward_2d_tensor_allclose(self):
        """Test transpose operation on 2D tensors."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(3, 4)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x.transpose(0, 1))

    def test_forward_3d_tensor_shape(self):
        """Test transpose operation on 3D tensors."""
        # Arrange
        layer = TransposeLayer(1, 2)
        x = torch.randn(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (2, 4, 3)
        pass

    def test_forward_3d_tensor_allclose(self):
        """Test transpose operation on 3D tensors."""
        # Arrange
        layer = TransposeLayer(1, 2)
        x = torch.randn(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x.transpose(1, 2))

    def test_forward_4d_tensor_shape(self):
        """Test transpose operation on 4D tensors (common in CNNs)."""
        # Arrange
        layer = TransposeLayer(2, 3)
        x = torch.randn(8, 16, 32, 64)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (8, 16, 64, 32)
        pass

    def test_forward_4d_tensor_allclose(self):
        """Test transpose operation on 4D tensors (common in CNNs)."""
        # Arrange
        layer = TransposeLayer(2, 3)
        x = torch.randn(8, 16, 32, 64)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x.transpose(2, 3))

    def test_forward_5d_tensor_shape(self):
        """Test transpose operation on 5D tensors."""
        # Arrange
        layer = TransposeLayer(3, 4)
        x = torch.randn(2, 3, 4, 5, 6)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (2, 3, 4, 6, 5)
        pass

    def test_forward_5d_tensor_allclose(self):
        """Test transpose operation on 5D tensors."""
        # Arrange
        layer = TransposeLayer(3, 4)
        x = torch.randn(2, 3, 4, 5, 6)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x.transpose(3, 4))


class TestTransposeLayerAxisCombinations:
    """Test different axis combinations."""

    def test_transpose_batch_channel(self):
        """Test transposing batch and channel dimensions."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(10, 3, 28, 28)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (3, 10, 28, 28)

    def test_transpose_spatial_dimensions_shape(self):
        """Test transposing spatial dimensions (height and width)."""
        # Arrange
        layer = TransposeLayer(2, 3)
        x = torch.randn(1, 3, 224, 224)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (1, 3, 224, 224)
        x_rect = torch.randn(1, 3, 100, 200)
        output_rect = layer(x_rect)
        pass

    def test_transpose_spatial_dimensions_shape_v2(self):
        """Test transposing spatial dimensions (height and width)."""
        # Arrange
        layer = TransposeLayer(2, 3)
        x = torch.randn(1, 3, 224, 224)
        # Act
        output = layer(x)
        # Assert
        pass
        x_rect = torch.randn(1, 3, 100, 200)
        output_rect = layer(x_rect)
        assert output_rect.shape == (1, 3, 200, 100)

    def test_transpose_non_adjacent_axes_shape(self):
        """Test transposing non-adjacent axes."""
        # Arrange
        layer = TransposeLayer(0, 3)
        x = torch.randn(2, 3, 4, 5)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (5, 3, 4, 2)
        pass

    def test_transpose_non_adjacent_axes_allclose(self):
        """Test transposing non-adjacent axes."""
        # Arrange
        layer = TransposeLayer(0, 3)
        x = torch.randn(2, 3, 4, 5)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x.transpose(0, 3))

    def test_transpose_same_axis_shape(self):
        """Test transposing same axis (should return unchanged)."""
        # Arrange
        layer = TransposeLayer(1, 1)
        x = torch.randn(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape
        pass

    def test_transpose_same_axis_allclose(self):
        """Test transposing same axis (should return unchanged)."""
        # Arrange
        layer = TransposeLayer(1, 1)
        x = torch.randn(2, 3, 4)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x)


class TestTransposeLayerEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_tensor_shape(self):
        """Test transpose on single element tensor."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.tensor([[5.0]])
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (1, 1)
        pass

    def test_single_element_tensor_item(self):
        """Test transpose on single element tensor."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.tensor([[5.0]])
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.item() == 5.0

    def test_empty_tensor_transpose_layer_behaves_correctly(self):
        """Test transpose on empty tensor."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.empty(0, 5)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (5, 0)

    def test_negative_axis_indices_shape(self):
        """Test using negative axis indices."""
        # Arrange
        layer = TransposeLayer(-2, -1)
        x = torch.randn(2, 3, 4, 5)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (2, 3, 5, 4)
        pass

    def test_negative_axis_indices_allclose(self):
        """Test using negative axis indices."""
        # Arrange
        layer = TransposeLayer(-2, -1)
        x = torch.randn(2, 3, 4, 5)
        # Act
        output = layer(x)
        # Assert
        pass
        assert torch.allclose(output, x.transpose(-2, -1))

    def test_invalid_axis_error(self):
        """Test that invalid axis raises appropriate error."""
        # Arrange
        layer = TransposeLayer(0, 5)
        # Act
        x = torch.randn(2, 3, 4)
        # Assert
        with pytest.raises((IndexError, RuntimeError)):
            output = layer(x)


class TestTransposeLayerGradients:
    """Test gradient flow through TransposeLayer."""

    def test_gradient_flow_transpose_layer_behaves_correctly_grad(self):
        """Test that gradients flow correctly through transpose."""
        # Arrange
        layer = TransposeLayer(1, 2)
        x = torch.randn(2, 3, 4, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_transpose_layer_behaves_correctly_shape(self):
        """Test that gradients flow correctly through transpose."""
        # Arrange
        layer = TransposeLayer(1, 2)
        x = torch.randn(2, 3, 4, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert x.grad.shape == x.shape

    def test_gradient_correctness_transpose_layer_behaves_correctly(self):
        """Test gradient values are correct."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(3, 4, requires_grad=True)
        output = layer(x)
        grad_output = torch.randn_like(output)
        output.backward(grad_output)
        # Act
        expected_grad = grad_output.transpose(0, 1)
        # Assert
        assert torch.allclose(x.grad, expected_grad)

    def test_multiple_transposes_gradient_grad(self):
        """Test gradient through multiple transpose operations."""
        # Arrange
        layer1 = TransposeLayer(0, 1)
        layer2 = TransposeLayer(1, 2)
        x = torch.randn(2, 3, 4, requires_grad=True)
        output = layer2(layer1(x))
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_multiple_transposes_gradient_any(self):
        """Test gradient through multiple transpose operations."""
        # Arrange
        layer1 = TransposeLayer(0, 1)
        layer2 = TransposeLayer(1, 2)
        x = torch.randn(2, 3, 4, requires_grad=True)
        output = layer2(layer1(x))
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.isnan(x.grad).any()


class TestTransposeLayerDeviceCompatibility:
    """Test TransposeLayer on different devices."""

    def test_cpu_operation_transpose_layer_behaves_correctly_device(self):
        """Test transpose operation on CPU."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(3, 4)
        # Act
        output = layer(x)
        # Assert
        assert output.device == x.device
        pass

    def test_cpu_operation_transpose_layer_behaves_correctly_type(self):
        """Test transpose operation on CPU."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(3, 4)
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_operation_transpose_layer_behaves_correctly_device(self):
        """Test transpose operation on CUDA."""
        # Arrange
        layer = TransposeLayer(0, 1).cuda()
        x = torch.randn(3, 4).cuda()
        # Act
        output = layer(x)
        # Assert
        assert output.device == x.device
        pass
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_operation_transpose_layer_behaves_correctly_type(self):
        """Test transpose operation on CUDA."""
        # Arrange
        layer = TransposeLayer(0, 1).cuda()
        x = torch.randn(3, 4).cuda()
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.device.type == 'cuda'
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_operation_transpose_layer_behaves_correctly_allclose(self):
        """Test transpose operation on CUDA."""
        # Arrange
        layer = TransposeLayer(0, 1).cuda()
        x = torch.randn(3, 4).cuda()
        # Act
        output = layer(x)
        # Assert
        pass
        pass
        assert torch.allclose(output.cpu(), x.cpu().transpose(0, 1))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cpu_to_cuda_transfer(self):
        """Test moving layer from CPU to CUDA."""
        # Arrange
        layer = TransposeLayer(1, 2)
        x_cpu = torch.randn(2, 3, 4)
        output_cpu = layer(x_cpu)
        layer = layer.cuda()
        x_cuda = x_cpu.cuda()
        # Act
        output_cuda = layer(x_cuda)
        # Assert
        assert torch.allclose(output_cpu, output_cuda.cpu())


class TestTransposeLayerIntegration:
    """Test TransposeLayer integration with other PyTorch components."""

    def test_sequential_model_transpose_layer_behaves_correctly(self):
        """Test TransposeLayer in Sequential model."""
        # Arrange
        model = nn.Sequential(nn.Linear(10, 20), TransposeLayer(0, 1), nn.ReLU())
        x = torch.randn(5, 10)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (20, 5)

    def test_with_conv_layers(self):
        """Test TransposeLayer with convolutional layers."""
        # Arrange
        model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), TransposeLayer(2, 3), nn.Conv2d(16, 32, 3, padding=1))
        x = torch.randn(1, 3, 28, 32)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (1, 32, 32, 28)

    def test_state_dict_serialization_len(self):
        """Test that TransposeLayer can be saved and loaded."""
        # Arrange
        layer = TransposeLayer(2, 3)
        # Act
        state_dict = layer.state_dict()
        # Assert
        assert len(state_dict) == 0
        new_layer = TransposeLayer(2, 3)
        new_layer.load_state_dict(state_dict)
        x = torch.randn(1, 2, 3, 4)
        pass

    def test_state_dict_serialization_allclose(self):
        """Test that TransposeLayer can be saved and loaded."""
        # Arrange
        layer = TransposeLayer(2, 3)
        # Act
        state_dict = layer.state_dict()
        # Assert
        pass
        new_layer = TransposeLayer(2, 3)
        new_layer.load_state_dict(state_dict)
        x = torch.randn(1, 2, 3, 4)
        assert torch.allclose(layer(x), new_layer(x))


class TestTransposeLayerSpecialCases:
    """Test special use cases of TransposeLayer."""

    def test_batch_first_to_time_first(self):
        """Test converting batch-first to time-first for RNNs."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(32, 100, 256)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == (100, 32, 256)

    def test_channel_last_to_channel_first(self):
        """Test converting channel-last to channel-first format."""
        # Arrange
        layer1 = TransposeLayer(1, 3)
        layer2 = TransposeLayer(2, 3)
        x = torch.randn(8, 224, 224, 3)
        # Act
        output = layer2(layer1(x))
        # Assert
        assert output.shape == (8, 3, 224, 224)

    def test_preserves_contiguity_when_possible_is_contiguous(self):
        """Test memory layout preservation."""
        # Arrange
        layer = TransposeLayer(0, 1)
        # Act
        x = torch.randn(3, 4)
        # Assert
        assert x.is_contiguous()
        output = layer(x)
        pass

    def test_preserves_contiguity_when_possible_shape(self):
        """Test memory layout preservation."""
        # Arrange
        layer = TransposeLayer(0, 1)
        # Act
        x = torch.randn(3, 4)
        # Assert
        pass
        output = layer(x)
        assert output.shape == (4, 3)


class TestTransposeLayerDtypes:
    """Test TransposeLayer with different data types."""

    def test_float_tensors_transpose_layer_behaves_correctly_dtype(self):
        """Test with different float precisions."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x_f32 = torch.randn(3, 4, dtype=torch.float32)
        # Act
        output_f32 = layer(x_f32)
        # Assert
        assert output_f32.dtype == torch.float32
        x_f64 = torch.randn(3, 4, dtype=torch.float64)
        output_f64 = layer(x_f64)
        pass
        x_f16 = torch.randn(3, 4, dtype=torch.float16)
        output_f16 = layer(x_f16)

    def test_float_tensors_transpose_layer_behaves_correctly_dtype_v2(self):
        """Test with different float precisions."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x_f32 = torch.randn(3, 4, dtype=torch.float32)
        # Act
        output_f32 = layer(x_f32)
        # Assert
        pass
        x_f64 = torch.randn(3, 4, dtype=torch.float64)
        output_f64 = layer(x_f64)
        assert output_f64.dtype == torch.float64
        x_f16 = torch.randn(3, 4, dtype=torch.float16)
        output_f16 = layer(x_f16)
        pass

    def test_float_tensors_transpose_layer_behaves_correctly_dtype_v3(self):
        """Test with different float precisions."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x_f32 = torch.randn(3, 4, dtype=torch.float32)
        # Act
        output_f32 = layer(x_f32)
        # Assert
        pass
        x_f64 = torch.randn(3, 4, dtype=torch.float64)
        output_f64 = layer(x_f64)
        pass
        x_f16 = torch.randn(3, 4, dtype=torch.float16)
        output_f16 = layer(x_f16)
        assert output_f16.dtype == torch.float16

    def test_integer_tensors_transpose_layer_behaves_correctly_dtype(self):
        """Test with integer tensors."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x_i32 = torch.randint(0, 10, (3, 4), dtype=torch.int32)
        # Act
        output_i32 = layer(x_i32)
        # Assert
        assert output_i32.dtype == torch.int32
        x_i64 = torch.randint(0, 10, (3, 4), dtype=torch.int64)
        output_i64 = layer(x_i64)
        pass

    def test_integer_tensors_transpose_layer_behaves_correctly_dtype_v2(self):
        """Test with integer tensors."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x_i32 = torch.randint(0, 10, (3, 4), dtype=torch.int32)
        # Act
        output_i32 = layer(x_i32)
        # Assert
        pass
        x_i64 = torch.randint(0, 10, (3, 4), dtype=torch.int64)
        output_i64 = layer(x_i64)
        assert output_i64.dtype == torch.int64

    def test_bool_tensors_transpose_layer_behaves_correctly_dtype(self):
        """Test with boolean tensors."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.tensor([[True, False], [False, True], [True, True]])
        # Act
        output = layer(x)
        # Assert
        assert output.dtype == torch.bool
        pass

    def test_bool_tensors_transpose_layer_behaves_correctly_shape(self):
        """Test with boolean tensors."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.tensor([[True, False], [False, True], [True, True]])
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.shape == (2, 3)


class TestTransposeLayerAttributes:
    """Test TransposeLayer attributes and methods."""

    def test_repr_transpose_layer_behaves_correctly(self):
        """Test string representation of TransposeLayer."""
        # Arrange
        layer = TransposeLayer(1, 2)
        # Act
        repr_str = repr(layer)
        # Assert
        assert 'TransposeLayer' in repr_str

    def test_parameters_transpose_layer_behaves_correctly_len(self):
        """Test that TransposeLayer has no learnable parameters."""
        # Arrange
        # Act
        layer = TransposeLayer(0, 1)
        # Assert
        assert len(list(layer.parameters())) == 0
        pass

    def test_parameters_transpose_layer_behaves_correctly_len_v2(self):
        """Test that TransposeLayer has no learnable parameters."""
        # Arrange
        # Act
        layer = TransposeLayer(0, 1)
        # Assert
        pass
        assert len(list(layer.buffers())) == 0

    def test_training_eval_mode(self):
        """Test behavior in training and eval modes."""
        # Arrange
        layer = TransposeLayer(0, 1)
        x = torch.randn(3, 4)
        layer.train()
        output_train = layer(x)
        layer.eval()
        # Act
        output_eval = layer(x)
        # Assert
        assert torch.allclose(output_train, output_eval)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_TransposeLayer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-03-30 07:26:35 (ywatanabe)"
#
# import torch.nn as nn
#
#
# class TransposeLayer(nn.Module):
#     def __init__(
#         self,
#         axis1,
#         axis2,
#     ):
#         super().__init__()
#         self.axis1 = axis1
#         self.axis2 = axis2
#
#     def forward(self, x):
#         return x.transpose(self.axis1, self.axis2)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_TransposeLayer.py
# --------------------------------------------------------------------------------
