import pytest

# Required for this module
pytest.importorskip("torch")
import random
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

# Import the module to test
from scitex_nn import SwapChannels


class TestSwapChannels:
    """Comprehensive test suite for SwapChannels layer."""

    def test_basic_instantiation_swap_channels_behaves_correctly_p(self):
        """Test basic instantiation with default parameters."""
        # Arrange
        # Act
        layer = SwapChannels()
        # Assert
        assert layer.dropout.p == 0.5
        pass

    def test_basic_instantiation_swap_channels_behaves_correctly_isinstance(self):
        """Test basic instantiation with default parameters."""
        # Arrange
        # Act
        layer = SwapChannels()
        # Assert
        pass
        assert isinstance(layer, nn.Module)

    def test_custom_dropout_rate(self):
        """Test instantiation with custom dropout probability."""
        # Arrange
        # Act
        layer = SwapChannels(dropout=0.3)
        # Assert
        assert layer.dropout.p == 0.3

    def test_forward_shape_preservation(self):
        """Test that output shape matches input shape."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        x = torch.randn(4, 10, 100)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_eval_mode_no_swapping(self):
        """Test that no channel swapping occurs in evaluation mode."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.eval()
        x = torch.randn(4, 10, 100)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, x)

    def test_train_mode_applies_swapping(self):
        """Test that channel swapping is applied in training mode."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100)
        torch.manual_seed(42)
        random.seed(42)
        # Act
        output = layer(x)
        channels_same = 0
        for ch in range(10):
            if torch.allclose(x[:, ch, :], output[:, ch, :]):
                channels_same += 1
        # Assert
        assert channels_same < 10

    def test_different_batch_sizes(self):
        """Test layer works with different batch sizes."""
        # Arrange
        # Act
        layer = SwapChannels(dropout=0.5)
        # Assert
        for batch_size in [1, 2, 8, 16, 32]:
            x = torch.randn(batch_size, 10, 100)
            output = layer(x)
            assert output.shape == x.shape

    def test_different_channel_counts(self):
        """Test layer works with different channel counts."""
        # Arrange
        # Act
        layer = SwapChannels(dropout=0.5)
        # Assert
        for n_channels in [1, 5, 20, 64, 128]:
            x = torch.randn(4, n_channels, 100)
            output = layer(x)
            assert output.shape == x.shape

    def test_different_sequence_lengths(self):
        """Test layer works with different sequence lengths."""
        # Arrange
        # Act
        layer = SwapChannels(dropout=0.5)
        # Assert
        for seq_len in [10, 50, 100, 500, 1000]:
            x = torch.randn(4, 10, seq_len)
            output = layer(x)
            assert output.shape == x.shape

    def test_dropout_rate_zero(self):
        """Test layer with dropout rate of 0 (no swapping)."""
        # Arrange
        layer = SwapChannels(dropout=0.0)
        layer.train()
        x = torch.randn(4, 10, 100)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, x)

    def test_dropout_rate_one(self):
        """Test layer with dropout rate of 1.0 (all channels eligible for swapping)."""
        # Arrange
        layer = SwapChannels(dropout=1.0)
        layer.train()
        x = torch.zeros(4, 10, 100)
        for i in range(10):
            x[:, i, :] = i
        torch.manual_seed(42)
        random.seed(42)
        # Act
        output = layer(x)
        channels_same = 0
        for ch in range(10):
            if torch.allclose(x[:, ch, :], output[:, ch, :]):
                channels_same += 1
        # Assert
        assert channels_same <= 3

    def test_gradient_flow_swap_channels_behaves_correctly_grad(self):
        """Test that gradients flow through the layer."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_swap_channels_behaves_correctly_all(self):
        """Test that gradients flow through the layer."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.all(x.grad == 0)

    def test_device_compatibility_cpu(self):
        """Test layer works on CPU."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        x = torch.randn(4, 10, 100)
        # Act
        output = layer(x)
        # Assert
        assert output.device == x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_compatibility_cuda_device(self):
        """Test layer works on CUDA."""
        # Arrange
        layer = SwapChannels(dropout=0.5).cuda()
        x = torch.randn(4, 10, 100).cuda()
        # Act
        output = layer(x)
        # Assert
        assert output.device == x.device
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_compatibility_cuda_is_cuda(self):
        """Test layer works on CUDA."""
        # Arrange
        layer = SwapChannels(dropout=0.5).cuda()
        x = torch.randn(4, 10, 100).cuda()
        # Act
        output = layer(x)
        # Assert
        pass
        assert output.is_cuda

    def test_reproducibility_with_seed(self):
        """Test reproducible results with same random seed."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100)
        torch.manual_seed(42)
        random.seed(42)
        output1 = layer(x)
        torch.manual_seed(42)
        random.seed(42)
        # Act
        output2 = layer(x)
        # Assert
        assert torch.allclose(output1, output2)

    def test_channel_permutation_properties(self):
        """Test that swapping is a permutation (no data loss)."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.zeros(4, 10, 100)
        for i in range(10):
            x[:, i, :] = i
        torch.manual_seed(42)
        random.seed(42)
        output = layer(x)
        original_values = set(range(10))
        # Act
        output_values = set(output[:, :, 0].flatten().tolist())
        # Assert
        assert original_values == output_values

    def test_swap_maintains_channel_integrity(self):
        """Test that entire channels are swapped, not mixed."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.zeros(2, 5, 10)
        for ch in range(5):
            x[:, ch, :] = ch * torch.ones(2, 10)
        # Act
        output = layer(x)
        # Assert
        for ch in range(5):
            channel_data = output[:, ch, :]
            assert torch.all(channel_data == channel_data[0, 0])

    def test_integration_with_sequential_check1(self):
        """Test integration in nn.Sequential."""
        # Arrange
        model = nn.Sequential(nn.Conv1d(10, 20, 3), SwapChannels(dropout=0.5), nn.Conv1d(20, 10, 3))
        x = torch.randn(4, 10, 100)
        # Act
        output = model(x)
        # Assert
        assert output.shape[0] == 4
        pass

    def test_integration_with_sequential_check2(self):
        """Test integration in nn.Sequential."""
        # Arrange
        model = nn.Sequential(nn.Conv1d(10, 20, 3), SwapChannels(dropout=0.5), nn.Conv1d(20, 10, 3))
        x = torch.randn(4, 10, 100)
        # Act
        output = model(x)
        # Assert
        pass
        assert output.shape[1] == 10

    def test_state_dict_save_load(self):
        """Test saving and loading state dict."""
        # Arrange
        layer1 = SwapChannels(dropout=0.3)
        layer2 = SwapChannels(dropout=0.7)
        # Act
        layer2.load_state_dict(layer1.state_dict())
        # Assert
        assert layer2.dropout.p == 0.7

    def test_memory_efficiency_swap_channels_behaves_correctly(self):
        """Test memory efficiency with large tensors."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(32, 256, 1000)
        # Act
        output = layer(x)
        # Assert
        assert output.shape == x.shape

    def test_numerical_stability_swap_channels_behaves_correctly_any(self):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100) * 1000000.0
        # Act
        output = layer(x)
        # Assert
        assert not torch.any(torch.isnan(output))
        pass
        x = torch.randn(4, 10, 100) * 1e-06
        output = layer(x)
        pass
        pass

    def test_numerical_stability_swap_channels_behaves_correctly_any_v2(self):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100) * 1000000.0
        # Act
        output = layer(x)
        # Assert
        pass
        assert not torch.any(torch.isinf(output))
        x = torch.randn(4, 10, 100) * 1e-06
        output = layer(x)
        pass
        pass

    def test_numerical_stability_swap_channels_behaves_correctly_any_v3(self):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100) * 1000000.0
        # Act
        output = layer(x)
        # Assert
        pass
        pass
        x = torch.randn(4, 10, 100) * 1e-06
        output = layer(x)
        assert not torch.any(torch.isnan(output))
        pass

    def test_numerical_stability_swap_channels_behaves_correctly_any_v4(self):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 10, 100) * 1000000.0
        # Act
        output = layer(x)
        # Assert
        pass
        pass
        x = torch.randn(4, 10, 100) * 1e-06
        output = layer(x)
        pass
        assert not torch.any(torch.isinf(output))

    def test_training_flag_inheritance_training(self):
        """Test that training flag is properly inherited from parent module."""
        # Arrange
        parent = nn.Sequential(SwapChannels(dropout=0.5))
        # Act
        parent.train()
        # Assert
        assert parent[0].training
        parent.eval()
        pass

    def test_training_flag_inheritance_training_v2(self):
        """Test that training flag is properly inherited from parent module."""
        # Arrange
        parent = nn.Sequential(SwapChannels(dropout=0.5))
        # Act
        parent.train()
        # Assert
        pass
        parent.eval()
        assert not parent[0].training

    def test_partial_channel_swapping(self):
        """Test that only some channels are swapped based on dropout rate."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        # Act
        layer.train()
        total_swaps = 0
        n_trials = 100
        n_channels = 20
        for trial in range(n_trials):
            x = torch.arange(n_channels).float().unsqueeze(0).unsqueeze(-1).expand(1, -1, 10)
            torch.manual_seed(trial)
            output = layer(x)
            for ch in range(n_channels):
                if output[0, ch, 0] != ch:
                    total_swaps += 1
        avg_swap_rate = total_swaps / (n_trials * n_channels)
        # Assert
        assert 0.4 <= avg_swap_rate <= 0.6

    def test_single_channel_no_swap(self):
        """Test that single channel input is unchanged."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.randn(4, 1, 100)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, x)

    def test_zero_input_preservation(self):
        """Test that zero inputs remain zero after swapping."""
        # Arrange
        layer = SwapChannels(dropout=0.5)
        layer.train()
        x = torch.zeros(4, 10, 100)
        # Act
        output = layer(x)
        # Assert
        assert torch.all(output == 0)

    def test_channel_swap_symmetry_sorted(self):
        """Test that channel swapping is symmetric (bijective mapping)."""
        # Arrange
        layer = SwapChannels(dropout=1.0)
        layer.train()
        n_channels = 10
        x = torch.arange(n_channels).float().view(1, n_channels, 1).expand(1, -1, 10)
        torch.manual_seed(42)
        random.seed(42)
        # Act
        output = layer(x)
        permutation = []
        for ch in range(n_channels):
            value = output[0, ch, 0].item()
            permutation.append(int(value))
        # Assert
        assert sorted(permutation) == list(range(n_channels))
        pass

    def test_channel_swap_symmetry_len(self):
        """Test that channel swapping is symmetric (bijective mapping)."""
        # Arrange
        layer = SwapChannels(dropout=1.0)
        layer.train()
        n_channels = 10
        x = torch.arange(n_channels).float().view(1, n_channels, 1).expand(1, -1, 10)
        torch.manual_seed(42)
        random.seed(42)
        # Act
        output = layer(x)
        permutation = []
        for ch in range(n_channels):
            value = output[0, ch, 0].item()
            permutation.append(int(value))
        # Assert
        pass
        assert len(set(permutation)) == n_channels


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_SwapChannels.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2023-05-04 21:21:19 (ywatanabe)"
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchsummary import summary
# import scitex
# import numpy as np
# import random
#
#
# class SwapChannels(nn.Module):
#     def __init__(self, dropout=0.5):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         """x: [batch_size, n_chs, seq_len]"""
#         if self.training:
#             orig_chs = torch.arange(x.shape[1])
#
#             indi_orig = self.dropout(torch.ones(x.shape[1])).bool()
#             chs_to_shuffle = orig_chs[~indi_orig]
#
#             rand_chs = random.sample(
#                 list(np.array(chs_to_shuffle)), len(chs_to_shuffle)
#             )
#
#             swapped_chs = orig_chs.clone()
#             swapped_chs[~indi_orig] = torch.LongTensor(rand_chs)
#
#             x = x[:, swapped_chs.long(), :]
#
#         return x
#
#
# if __name__ == "__main__":
#     ## Demo data
#     bs, n_chs, seq_len = 16, 360, 1000
#     x = torch.rand(bs, n_chs, seq_len)
#
#     sc = SwapChannels()
#     print(sc(x).shape)  # [16, 19, 1000]
#
#     # sb = SubjectBlock(n_chs=n_chs)
#     # print(sb(x, s).shape) # [16, 270, 1000]
#
#     # summary(sb, x, s)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_SwapChannels.py
# --------------------------------------------------------------------------------
