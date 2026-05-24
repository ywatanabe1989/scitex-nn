#!/usr/bin/env python3
"""
Comprehensive test suite for ResNet1D neural network architecture.

This module tests the ResNet1D class and ResNetBasicBlock including:
- Basic instantiation with different configurations
- Forward pass functionality with various input shapes
- Residual connections and skip connections
- Multi-kernel convolutions (3, 5, 7)
- Gradient flow and vanishing gradient prevention
- Different network depths
- Channel expansion handling
- BatchNorm and activation functions
"""

import pytest

# Required for this module
pytest.importorskip("torch")

import numpy as np
import torch
import torch.nn as nn

import scitex_nn


class TestResNetBasicBlock:
    """Test suite for ResNetBasicBlock - the fundamental building block of ResNet1D."""

    def test_basic_block_instantiation_isinstance(self):
        """Test ResNetBasicBlock instantiation with various channel configurations."""
        # Arrange
        configs = [(32, 64), (64, 64), (19, 76), (128, 32)]
        # Act
        # Assert
        for in_chs, out_chs in configs:
            block = scitex_nn.ResNetBasicBlock(in_chs, out_chs)
            assert isinstance(block, nn.Module)
            pass
            pass

    def test_basic_block_instantiation_in_chs(self):
        """Test ResNetBasicBlock instantiation with various channel configurations."""
        # Arrange
        configs = [(32, 64), (64, 64), (19, 76), (128, 32)]
        # Act
        # Assert
        for in_chs, out_chs in configs:
            block = scitex_nn.ResNetBasicBlock(in_chs, out_chs)
            pass
            assert block.in_chs == in_chs
            pass

    def test_basic_block_instantiation_out_chs(self):
        """Test ResNetBasicBlock instantiation with various channel configurations."""
        # Arrange
        configs = [(32, 64), (64, 64), (19, 76), (128, 32)]
        # Act
        # Assert
        for in_chs, out_chs in configs:
            block = scitex_nn.ResNetBasicBlock(in_chs, out_chs)
            pass
            pass
            assert block.out_chs == out_chs

    def test_basic_block_layers_structure_hasattr(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        assert hasattr(block, "conv7")

    def test_basic_block_layers_structure_hasattr_v2(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        assert hasattr(block, "conv5")
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass

    def test_basic_block_layers_structure_hasattr_v3(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        assert hasattr(block, "conv3")
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass

    def test_basic_block_layers_structure_hasattr_v4(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        assert hasattr(block, "expansion_conv")
        pass
        pass
        pass
        pass

    def test_basic_block_layers_structure_hasattr_v5(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        assert hasattr(block, "bn7")
        pass

    def test_basic_block_layers_structure_hasattr_v6(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        pass
        pass
        assert hasattr(block, "bn5")
        pass
        pass
        pass
        pass

    def test_basic_block_layers_structure_hasattr_v7(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        assert hasattr(block, "bn3")
        pass
        pass
        pass
        pass
        pass

    def test_basic_block_layers_structure_hasattr_v8(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        assert hasattr(block, "bn")
        pass
        pass
        pass
        pass

    def test_basic_block_layers_structure_hasattr_v9(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        assert hasattr(block, "activation7")

    def test_basic_block_layers_structure_hasattr_v10(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        pass
        assert hasattr(block, "activation5")
        pass

    def test_basic_block_layers_structure_hasattr_v11(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        pass
        pass
        pass
        assert hasattr(block, "activation3")
        pass

    def test_basic_block_layers_structure_hasattr_v12(self):
        """Test that ResNetBasicBlock has all required layers."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        assert hasattr(block, "activation")

    def test_basic_block_forward_same_channels(self):
        """Test forward pass when input and output channels are the same."""
        # Arrange
        batch_size, in_chs, seq_len = (16, 64, 1000)
        block = scitex_nn.ResNetBasicBlock(in_chs, in_chs)
        x = torch.randn(batch_size, in_chs, seq_len)
        # Act
        output = block(x)
        # Assert
        assert output.shape == (batch_size, in_chs, seq_len)

    def test_basic_block_forward_different_channels(self):
        """Test forward pass with channel expansion."""
        # Arrange
        batch_size, in_chs, out_chs, seq_len = (16, 32, 128, 500)
        block = scitex_nn.ResNetBasicBlock(in_chs, out_chs)
        x = torch.randn(batch_size, in_chs, seq_len)
        # Act
        output = block(x)
        # Assert
        assert output.shape == (batch_size, out_chs, seq_len)

    def test_basic_block_conv_k_static_method_isinstance(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        assert isinstance(conv, nn.Conv1d)
        pass

    def test_basic_block_conv_k_static_method_in_channels(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        pass
        assert conv.in_channels == 32
        pass
        pass
        pass
        pass
        pass

    def test_basic_block_conv_k_static_method_out_channels(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        assert conv.out_channels == 64

    def test_basic_block_conv_k_static_method_kernel_size(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        pass
        pass
        pass
        assert conv.kernel_size == (3,)
        pass
        pass
        pass

    def test_basic_block_conv_k_static_method_stride(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        pass
        pass
        pass
        pass
        assert conv.stride == (1,)
        pass
        pass

    def test_basic_block_conv_k_static_method_padding(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        pass
        pass
        pass
        pass
        assert conv.padding == (1,)

    def test_basic_block_conv_k_static_method_bias(self):
        """Test the static conv_k method for creating convolution layers."""
        # Arrange
        # Act
        conv = scitex_nn.ResNetBasicBlock.conv_k(32, 64, k=3, s=1, p=1)
        # Assert
        pass
        pass
        pass
        pass
        pass
        assert conv.bias is None

    def test_basic_block_residual_connection_allclose(self):
        """Test that residual connection is properly applied."""
        # Arrange
        block = scitex_nn.ResNetBasicBlock(64, 64)
        x = torch.randn(8, 64, 256, requires_grad=True)
        # Act
        output = block(x)
        # Assert
        assert not torch.allclose(output, x)
        loss = output.sum()
        loss.backward()
        pass

    def test_basic_block_residual_connection_grad(self):
        """Test that residual connection is properly applied."""
        # Arrange
        block = scitex_nn.ResNetBasicBlock(64, 64)
        x = torch.randn(8, 64, 256, requires_grad=True)
        # Act
        output = block(x)
        # Assert
        pass
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_basic_block_expansion_conv_usage_expansion_conv(self):
        """Test expansion convolution is used when channels differ."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        assert block.expansion_conv is not None
        pass
        pass
        pass
        x = torch.randn(4, 32, 128)
        output = block(x)
        pass

    def test_basic_block_expansion_conv_usage_isinstance(self):
        """Test expansion convolution is used when channels differ."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        assert isinstance(block.expansion_conv, nn.Conv1d)
        pass
        pass
        x = torch.randn(4, 32, 128)
        output = block(x)
        pass

    def test_basic_block_expansion_conv_usage_in_channels(self):
        """Test expansion convolution is used when channels differ."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        assert block.expansion_conv.in_channels == 32
        pass
        x = torch.randn(4, 32, 128)
        output = block(x)
        pass

    def test_basic_block_expansion_conv_usage_out_channels(self):
        """Test expansion convolution is used when channels differ."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        assert block.expansion_conv.out_channels == 64
        x = torch.randn(4, 32, 128)
        output = block(x)
        pass

    def test_basic_block_expansion_conv_usage_check5(self):
        """Test expansion convolution is used when channels differ."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        pass
        x = torch.randn(4, 32, 128)
        output = block(x)
        assert output.shape[1] == 64

    def test_basic_block_kernel_sizes_kernel_size(self):
        """Test different kernel sizes in the block."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        assert block.conv7.kernel_size == (7,)
        pass
        pass
        pass

    def test_basic_block_kernel_sizes_kernel_size_v2(self):
        """Test different kernel sizes in the block."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        assert block.conv5.kernel_size == (5,)

    def test_basic_block_kernel_sizes_kernel_size_v3(self):
        """Test different kernel sizes in the block."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        assert block.conv3.kernel_size == (3,)
        pass

    def test_basic_block_kernel_sizes_kernel_size_v4(self):
        """Test different kernel sizes in the block."""
        # Arrange
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        pass
        pass
        assert block.expansion_conv.kernel_size == (1,)

    def test_basic_block_padding_preservation(self):
        """Test that padding preserves sequence length."""
        # Arrange
        seq_lengths = [100, 256, 512, 1000, 2048]
        # Act
        block = scitex_nn.ResNetBasicBlock(32, 64)
        # Assert
        for seq_len in seq_lengths:
            x = torch.randn(2, 32, seq_len)
            output = block(x)
            assert output.shape[-1] == seq_len

    def test_basic_block_gradient_flow_grad(self):
        """Test gradient flows properly through the block."""
        # Arrange
        block = scitex_nn.ResNetBasicBlock(32, 64)
        x = torch.randn(4, 32, 128, requires_grad=True)
        output = block(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass
        pass

    def test_basic_block_gradient_flow_allclose(self):
        """Test gradient flows properly through the block."""
        # Arrange
        block = scitex_nn.ResNetBasicBlock(32, 64)
        x = torch.randn(4, 32, 128, requires_grad=True)
        output = block(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        pass
        pass

    def test_basic_block_gradient_flow_any(self):
        """Test gradient flows properly through the block."""
        # Arrange
        block = scitex_nn.ResNetBasicBlock(32, 64)
        x = torch.randn(4, 32, 128, requires_grad=True)
        output = block(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        pass
        assert not torch.isnan(x.grad).any()
        pass

    def test_basic_block_gradient_flow_any_v2(self):
        """Test gradient flows properly through the block."""
        # Arrange
        block = scitex_nn.ResNetBasicBlock(32, 64)
        x = torch.randn(4, 32, 128, requires_grad=True)
        output = block(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        pass
        assert not torch.isinf(x.grad).any()


class TestResNet1D:
    """Test suite for ResNet1D - 1D ResNet for signal classification."""

    def test_resnet1d_instantiation_default_isinstance(self):
        """Test ResNet1D instantiation with default parameters."""
        # Arrange
        # Act
        model = scitex_nn.ResNet1D()
        # Assert
        assert isinstance(model, nn.Module)

    def test_resnet1d_instantiation_default_hasattr(self):
        """Test ResNet1D instantiation with default parameters."""
        # Arrange
        # Act
        model = scitex_nn.ResNet1D()
        # Assert
        pass
        assert hasattr(model, "res_conv_blk_layers")

    def test_resnet1d_instantiation_custom(self):
        """Test ResNet1D instantiation with custom parameters."""
        # Arrange
        configs = [(19, 10, 5), (32, 4, 3), (64, 2, 10), (160, 5, 7)]
        # Act
        # Assert
        for n_chs, n_out, n_blks in configs:
            model = scitex_nn.ResNet1D(n_chs=n_chs, n_out=n_out, n_blks=n_blks)
            assert isinstance(model, nn.Module)

    def test_resnet1d_blocks_structure_isinstance(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        assert isinstance(model.res_conv_blk_layers, nn.Sequential)
        pass
        first_block = model.res_conv_blk_layers[0]
        pass
        pass
        pass
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]
            pass
            pass

    def test_resnet1d_blocks_structure_len(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        pass
        assert len(model.res_conv_blk_layers) == n_blks
        first_block = model.res_conv_blk_layers[0]
        pass
        pass
        pass
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]

    def test_resnet1d_blocks_structure_isinstance_v2(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        pass
        pass
        first_block = model.res_conv_blk_layers[0]
        assert isinstance(first_block, scitex_nn.ResNetBasicBlock)
        pass
        pass
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]
            pass

    def test_resnet1d_blocks_structure_in_chs(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        pass
        pass
        first_block = model.res_conv_blk_layers[0]
        pass
        assert first_block.in_chs == n_chs
        pass
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]
            pass
            pass

    def test_resnet1d_blocks_structure_out_chs(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        pass
        pass
        first_block = model.res_conv_blk_layers[0]
        pass
        pass
        assert first_block.out_chs == n_chs * 4
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]
            pass
            pass

    def test_resnet1d_blocks_structure_in_chs_v2(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        pass
        first_block = model.res_conv_blk_layers[0]
        pass
        pass
        pass
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]
            assert block.in_chs == n_chs * 4
            pass

    def test_resnet1d_blocks_structure_out_chs_v2(self):
        """Test the structure of residual blocks in ResNet1D."""
        # Arrange
        n_blks = 5
        n_chs = 32
        # Act
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_blks=n_blks)
        # Assert
        first_block = model.res_conv_blk_layers[0]
        pass
        pass
        pass
        for i in range(1, n_blks):
            block = model.res_conv_blk_layers[i]
            pass
            assert block.out_chs == n_chs * 4

    def test_resnet1d_forward_pass_basic(self):
        """Test basic forward pass through ResNet1D."""
        # Arrange
        batch_size, n_chs, seq_len = (16, 32, 1000)
        model = scitex_nn.ResNet1D(n_chs=n_chs, n_out=10)
        x = torch.randn(batch_size, n_chs, seq_len)
        # Act
        output = model(x)
        expected_out_chs = n_chs * 4
        # Assert
        assert output.shape == (batch_size, expected_out_chs, seq_len)

    def test_resnet1d_forward_various_sequence_lengths(self):
        """Test ResNet1D with various sequence lengths."""
        # Arrange
        # Act
        model = scitex_nn.ResNet1D(n_chs=19, n_out=4)
        batch_size = 8
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        # Assert
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, 19, seq_len)
            output = model(x)
            assert output.shape == (batch_size, 76, seq_len)

    def test_resnet1d_forward_various_batch_sizes(self):
        """Test ResNet1D with various batch sizes."""
        # Arrange
        # Act
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10)
        seq_len = 1000
        batch_sizes = [1, 4, 16, 32, 64, 128]
        # Assert
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 32, seq_len)
            output = model(x)
            assert output.shape == (batch_size, 128, seq_len)

    def test_resnet1d_different_depths(self):
        """Test ResNet1D with different network depths."""
        # Arrange
        depths = [1, 3, 5, 10, 20, 50]
        # Act
        # Assert
        for n_blks in depths:
            model = scitex_nn.ResNet1D(n_chs=16, n_out=10, n_blks=n_blks)
            x = torch.randn(2, 16, 256)
            output = model(x)
            assert output.shape == (2, 64, 256)

    def test_resnet1d_gradient_flow_shallow_grad(self):
        """Test gradient flow in shallow ResNet1D."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10, n_blks=3)
        x = torch.randn(4, 32, 512, requires_grad=True)
        output = model(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass
        pass

    def test_resnet1d_gradient_flow_shallow_any(self):
        """Test gradient flow in shallow ResNet1D."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10, n_blks=3)
        x = torch.randn(4, 32, 512, requires_grad=True)
        output = model(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.isnan(x.grad).any()
        pass

    def test_resnet1d_gradient_flow_shallow_any_v2(self):
        """Test gradient flow in shallow ResNet1D."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10, n_blks=3)
        x = torch.randn(4, 32, 512, requires_grad=True)
        output = model(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        pass
        assert not torch.isinf(x.grad).any()

    def test_resnet1d_gradient_flow_deep_grad(self):
        """Test gradient flow in deep ResNet1D (vanishing gradient test)."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10, n_blks=20)
        x = torch.randn(2, 32, 256, requires_grad=True)
        output = model(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass
        pass

    def test_resnet1d_gradient_flow_deep_allclose(self):
        """Test gradient flow in deep ResNet1D (vanishing gradient test)."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10, n_blks=20)
        x = torch.randn(2, 32, 256, requires_grad=True)
        output = model(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        pass

    def test_resnet1d_gradient_flow_deep_any(self):
        """Test gradient flow in deep ResNet1D (vanishing gradient test)."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10, n_blks=20)
        x = torch.randn(2, 32, 256, requires_grad=True)
        output = model(x)
        loss = output.mean()
        # Act
        loss.backward()
        # Assert
        pass
        pass
        assert not torch.isnan(x.grad).any()

    def test_resnet1d_device_compatibility_cpu(self):
        """Test ResNet1D on CPU."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=19, n_out=4)
        x = torch.randn(8, 19, 1000)
        # Act
        output = model(x)
        # Assert
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_resnet1d_device_compatibility_gpu(self):
        """Test ResNet1D on GPU if available."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=19, n_out=4).cuda()
        x = torch.randn(8, 19, 1000).cuda()
        # Act
        output = model(x)
        # Assert
        assert output.device.type == "cuda"

    def test_resnet1d_parameter_count_total_params(self):
        """Test parameter count for different configurations."""
        # Arrange
        configs = [(19, 10, 5), (32, 4, 10), (64, 2, 20)]
        # Act
        # Assert
        for n_chs, n_out, n_blks in configs:
            model = scitex_nn.ResNet1D(n_chs=n_chs, n_out=n_out, n_blks=n_blks)
            total_params = sum((p.numel() for p in model.parameters()))
            trainable_params = sum(
                (p.numel() for p in model.parameters() if p.requires_grad)
            )
            assert total_params > 10000
            pass

    def test_resnet1d_parameter_count_total_params_v2(self):
        """Test parameter count for different configurations."""
        # Arrange
        configs = [(19, 10, 5), (32, 4, 10), (64, 2, 20)]
        # Act
        # Assert
        for n_chs, n_out, n_blks in configs:
            model = scitex_nn.ResNet1D(n_chs=n_chs, n_out=n_out, n_blks=n_blks)
            total_params = sum((p.numel() for p in model.parameters()))
            trainable_params = sum(
                (p.numel() for p in model.parameters() if p.requires_grad)
            )
            assert total_params < 100000000
            pass

    def test_resnet1d_parameter_count_trainable_params(self):
        """Test parameter count for different configurations."""
        # Arrange
        configs = [(19, 10, 5), (32, 4, 10), (64, 2, 20)]
        # Act
        # Assert
        for n_chs, n_out, n_blks in configs:
            model = scitex_nn.ResNet1D(n_chs=n_chs, n_out=n_out, n_blks=n_blks)
            total_params = sum((p.numel() for p in model.parameters()))
            trainable_params = sum(
                (p.numel() for p in model.parameters() if p.requires_grad)
            )
            pass
            assert trainable_params == total_params

    def test_resnet1d_eval_train_modes_training(self):
        """Test behavior in training vs evaluation modes."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10)
        x = torch.randn(4, 32, 256)
        # Act
        model.train()
        # Assert
        assert model.training
        output_train = model(x)
        model.eval()
        output_eval = model(x)
        pass

    def test_resnet1d_eval_train_modes_training_v2(self):
        """Test behavior in training vs evaluation modes."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10)
        x = torch.randn(4, 32, 256)
        # Act
        model.train()
        # Assert
        pass
        output_train = model(x)
        model.eval()
        assert not model.training
        output_eval = model(x)
        pass

    def test_resnet1d_eval_train_modes_shape(self):
        """Test behavior in training vs evaluation modes."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10)
        x = torch.randn(4, 32, 256)
        # Act
        model.train()
        # Assert
        pass
        output_train = model(x)
        model.eval()
        pass
        output_eval = model(x)
        assert output_train.shape == output_eval.shape

    def test_resnet1d_multiple_forward_consistency(self):
        """Test consistency of multiple forward passes in eval mode."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=19, n_out=4)
        model.eval()
        x = torch.randn(8, 19, 512)
        out1 = model(x)
        # Act
        out2 = model(x)
        # Assert
        assert torch.allclose(out1, out2)

    def test_resnet1d_channels_per_filter_ratio(self):
        """Test the 4x channel expansion ratio."""
        # Arrange
        n_chs_list = [8, 16, 19, 32, 64]
        # Act
        # Assert
        for n_chs in n_chs_list:
            model = scitex_nn.ResNet1D(n_chs=n_chs)
            x = torch.randn(2, n_chs, 128)
            output = model(x)
            assert output.shape[1] == n_chs * 4

    def test_resnet1d_with_single_channel(self):
        """Test ResNet1D with single channel input."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=1, n_out=2, n_blks=3)
        x = torch.randn(16, 1, 1000)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (16, 4, 1000)

    def test_resnet1d_with_many_channels(self):
        """Test ResNet1D with many input channels."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=256, n_out=10, n_blks=5)
        x = torch.randn(2, 256, 512)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (2, 1024, 512)

    def test_resnet1d_output_features(self):
        """Test that output maintains spatial resolution."""
        # Arrange
        model = scitex_nn.ResNet1D(n_chs=32, n_out=10)
        x = torch.randn(4, 32, 1000)
        # Act
        output = model(x)
        # Assert
        assert output.shape[-1] == x.shape[-1]

    def test_resnet1d_integration_with_classifier(self):
        """Test ResNet1D can be integrated with a classifier head."""
        # Arrange

        class ResNet1DClassifier(nn.Module):
            def __init__(self, n_chs, n_out):
                super().__init__()
                self.feature_extractor = scitex_nn.ResNet1D(n_chs, n_out)
                self.classifier = nn.Linear(n_chs * 4, n_out)

            def forward(self, x):
                features = self.feature_extractor(x)
                pooled = features.mean(dim=-1)
                return self.classifier(pooled)

        model = ResNet1DClassifier(n_chs=19, n_out=4)
        x = torch.randn(8, 19, 1000)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (8, 4)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_ResNet1D.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-05-15 16:46:54 (ywatanabe)"
#
# import torch
# import torch.nn as nn
# from torchsummary import summary
#
#
# class ResNet1D(nn.Module):
#     """
#     A representative convolutional neural network for signal classification tasks.
#     """
#
#     def __init__(self, n_chs=19, n_out=10, n_blks=5):
#         super().__init__()
#
#         # Parameters
#         N_CHS = n_chs
#         _N_FILTS_PER_CH = 4
#         N_FILTS = N_CHS * _N_FILTS_PER_CH
#         N_BLKS = n_blks
#
#         # Convolutional layers
#         self.res_conv_blk_layers = nn.Sequential(
#             ResNetBasicBlock(N_CHS, N_FILTS),
#             *[ResNetBasicBlock(N_FILTS, N_FILTS) for _ in range(N_BLKS - 1)],
#         )
#
#         # ## FC layer
#         # self.fc = nn.Sequential(
#         #     nn.Linear(N_FILTS, 64),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.5),
#         #     nn.Linear(64, 32),
#         #     nn.ReLU(),
#         #     nn.Dropout(p=0.5),
#         #     nn.Linear(32, n_out),
#         # )
#
#     def forward(self, x):
#         x = self.res_conv_blk_layers(x)
#         # x = x.mean(axis=-1)
#         # x = self.fc(x)
#         return x
#
#
# class ResNetBasicBlock(nn.Module):
#     """The basic block of the ResNet1D model"""
#
#     def __init__(self, in_chs, out_chs):
#         super(ResNetBasicBlock, self).__init__()
#         self.in_chs = in_chs
#         self.out_chs = out_chs
#
#         self.conv7 = self.conv_k(in_chs, out_chs, k=7, p=3)
#         self.bn7 = nn.BatchNorm1d(out_chs)
#         self.activation7 = nn.ReLU()
#
#         self.conv5 = self.conv_k(out_chs, out_chs, k=5, p=2)
#         self.bn5 = nn.BatchNorm1d(out_chs)
#         self.activation5 = nn.ReLU()
#
#         self.conv3 = self.conv_k(out_chs, out_chs, k=3, p=1)
#         self.bn3 = nn.BatchNorm1d(out_chs)
#         self.activation3 = nn.ReLU()
#
#         self.expansion_conv = self.conv_k(in_chs, out_chs, k=1, p=0)
#
#         self.bn = nn.BatchNorm1d(out_chs)
#         self.activation = nn.ReLU()
#
#     @staticmethod
#     def conv_k(in_chs, out_chs, k=1, s=1, p=1):
#         """Build size k kernel's convolution layer with padding"""
#         return nn.Conv1d(
#             in_chs, out_chs, kernel_size=k, stride=s, padding=p, bias=False
#         )
#
#     def forward(self, x):
#         residual = x
#
#         x = self.conv7(x)
#         x = self.bn7(x)
#         x = self.activation7(x)
#
#         x = self.conv5(x)
#         x = self.bn5(x)
#         x = self.activation5(x)
#
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.activation3(x)
#
#         if self.in_chs != self.out_chs:
#             residual = self.expansion_conv(residual)
#         residual = self.bn(residual)
#
#         x = x + residual
#         x = self.activation(x)
#
#         return x
#
#
# if __name__ == "__main__":
#     import sys
#
#     sys.path.append("./DEAP/")
#     import utils
#
#     # Demo data
#     bs, n_chs, seq_len = 16, 32, 8064
#     Xb = torch.rand(bs, n_chs, seq_len)
#
#     model = ResNet1D(
#         n_chs=n_chs,
#         n_out=4,
#     )  # utils.load_yaml("./config/global.yaml")["EMOTIONS"]
#     y = model(Xb)  # 16,4
#     summary(model, Xb)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_ResNet1D.py
# --------------------------------------------------------------------------------
