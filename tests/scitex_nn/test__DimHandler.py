#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31 20:25:00 (Claude)"
# File: /tests/scitex/gen/test__DimHandler.py

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from scitex_nn import DimHandler


class TestDimHandler:
    """Test cases for DimHandler class."""

    @pytest.fixture
    def dim_handler(self):
        """Create a DimHandler instance."""
        return DimHandler()

    def test_init_dim_handler_is_dimhandler(self, dim_handler):
        """Test DimHandler initialization."""
        # Arrange
        # Act
        # Assert
        assert isinstance(dim_handler, DimHandler)

    def test_fit_numpy_basic_result_shape_equals_n_40_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert result.shape == (40, 1, 3, 6)

    def test_fit_numpy_basic_dim_handler_shape_fit_equals_n_1_2_3_4_5_6(self, dim_handler):
        # Arrange
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert dim_handler.shape_fit == (1, 2, 3, 4, 5, 6)

    def test_fit_numpy_basic_dim_handler_n_non_keepdims_equals_n_2_4_5(self, dim_handler):
        # Arrange
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert dim_handler.n_non_keepdims == [2, 4, 5]

    def test_fit_numpy_basic_dim_handler_n_keepdims_equals_n_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert dim_handler.n_keepdims == [1, 3, 6]


    def test_fit_torch_basic_result_shape_equals_torch_size_40_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert result.shape == torch.Size([40, 1, 3, 6])

    def test_fit_torch_basic_dim_handler_shape_fit_equals_torch_size_1_2_3_4_5_6(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert dim_handler.shape_fit == torch.Size([1, 2, 3, 4, 5, 6])

    def test_fit_torch_basic_dim_handler_n_non_keepdims_equals_n_2_4_5(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert dim_handler.n_non_keepdims == [2, 4, 5]

    def test_fit_torch_basic_dim_handler_n_keepdims_equals_n_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert dim_handler.n_keepdims == [1, 3, 6]


    def test_fit_negative_indices(self, dim_handler):
        """Test fit with negative dimension indices."""
        # Arrange
        x = np.random.rand(2, 3, 4, 5)
        keepdims = [-1, -2]  # Keep last two dimensions

        # Act
        result = dim_handler.fit(x, keepdims=keepdims)

        # Negative indices [-1, -2] should be [3, 2] -> sorted [2, 3]
        # Non-kept: [0, 1] -> sizes [2, 3] -> product = 6
        # Result should be (6, 4, 5)
        # Assert
        assert result.shape == (6, 4, 5)

    def test_fit_duplicate_indices(self, dim_handler):
        """Test fit with duplicate dimension indices."""
        # Arrange
        x = np.random.rand(2, 3, 4)
        keepdims = [1, 1, 2]  # Duplicate index

        # Act
        result = dim_handler.fit(x, keepdims=keepdims)

        # Duplicates should be removed, so keepdims = [1, 2]
        # Non-kept: [0] -> size [2]
        # Result should be (2, 3, 4)
        # Assert
        assert result.shape == (2, 3, 4)

    def test_fit_empty_keepdims(self, dim_handler):
        """Test fit with empty keepdims list."""
        # Arrange
        x = np.random.rand(2, 3, 4)

        # Act
        result = dim_handler.fit(x, keepdims=[])

        # All dims are flattened
        # Assert
        assert result.shape == (24,)

    def test_fit_all_keepdims(self, dim_handler):
        """Test fit keeping all dimensions."""
        # Arrange
        x = np.random.rand(2, 3, 4)
        keepdims = [0, 1, 2]

        # Act
        result = dim_handler.fit(x, keepdims=keepdims)

        # No flattening occurs
        # Assert
        assert result.shape == (1, 2, 3, 4)

    def test_unfit_basic_x_fitted_shape_equals_n_40_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Fit
        # Act
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert x_fitted.shape == (40, 1, 3, 6)

    def test_unfit_basic_x_restored_shape_equals_n_2_4_5_1_3_6_split_1(self, dim_handler):
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        assert x_fitted.shape == (40, 1, 3, 6)

    def test_unfit_basic_x_restored_shape_equals_n_2_4_5_1_3_6_split_2(self, dim_handler):
        # Arrange
        x = np.random.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        x_fitted.shape == (40, 1, 3, 6)
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        assert x_restored.shape == (2, 4, 5, 1, 3, 6)


    def test_unfit_after_reduction_x_fitted_shape_equals_torch_size_40_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        # Fit
        # Act
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_unfit_after_reduction_y_shape_equals_torch_size_40_1_6_split_1(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_unfit_after_reduction_y_shape_equals_torch_size_40_1_6_split_2(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        y = x_fitted.mean(dim=-2)
        # Act
        # Assert
        assert y.shape == torch.Size([40, 1, 6])

    def test_unfit_after_reduction_y_restored_shape_equals_torch_size_2_4_5_1_6_split_1(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_unfit_after_reduction_y_restored_shape_equals_torch_size_2_4_5_1_6_split_2(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        y = x_fitted.mean(dim=-2)
        # Act
        # Assert
        assert y.shape == torch.Size([40, 1, 6])

    def test_unfit_after_reduction_y_restored_shape_equals_torch_size_2_4_5_1_6_split_3(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        keepdims = [0, 2, 5]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        y = x_fitted.mean(dim=-2)
        y.shape == torch.Size([40, 1, 6])
        y_restored = dim_handler.unfit(y)
        # Act
        # Assert
        assert y_restored.shape == torch.Size([2, 4, 5, 1, 6])


    def test_fit_invalid_keepdims(self, dim_handler):
        """Test fit with invalid keepdims."""
        # Arrange
        # Act
        x = np.random.rand(2, 3, 4)

        # Too many dimensions
        # Assert
        with pytest.raises(AssertionError):
            dim_handler.fit(x, keepdims=[0, 1, 2, 3])

    def test_fit_preserves_data_numpy_x_restored_shape_equals_n_2_4_3(self, dim_handler):
        # Arrange
        # Arrange
        x = np.arange(24).reshape(2, 3, 4)
        keepdims = [1]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        # Assert
        assert x_restored.shape == (2, 4, 3)

    def test_fit_preserves_data_numpy_set_x_flatten_set_x_restored_flatten(self, dim_handler):
        # Arrange
        # Arrange
        x = np.arange(24).reshape(2, 3, 4)
        keepdims = [1]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        # Assert
        assert set(x.flatten()) == set(x_restored.flatten())

    def test_fit_preserves_data_numpy_x_size_equals_x_restored_size(self, dim_handler):
        # Arrange
        # Arrange
        x = np.arange(24).reshape(2, 3, 4)
        keepdims = [1]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        # Assert
        assert x.size == x_restored.size


    def test_fit_preserves_data_torch_x_restored_shape_equals_torch_size_2_4_3(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.arange(24).reshape(2, 3, 4).float()
        keepdims = [1]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        # Assert
        assert x_restored.shape == torch.Size([2, 4, 3])

    def test_fit_preserves_data_torch_set_x_flatten_tolist_set_x_restored_flatten_tolist(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.arange(24).reshape(2, 3, 4).float()
        keepdims = [1]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        # Assert
        assert set(x.flatten().tolist()) == set(x_restored.flatten().tolist())

    def test_fit_preserves_data_torch_x_numel_x_restored_numel(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.arange(24).reshape(2, 3, 4).float()
        keepdims = [1]
        x_fitted = dim_handler.fit(x, keepdims=keepdims)
        # Act
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        # Assert
        assert x.numel() == x_restored.numel()


    def test_example1_from_docstring_x_fitted_shape_equals_torch_size_40_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        # Fit
        # Act
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        # Act
        # Assert
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_example1_from_docstring_x_restored_shape_equals_torch_size_2_4_5_1_3_6_split_1(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        # Act
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_example1_from_docstring_x_restored_shape_equals_torch_size_2_4_5_1_3_6_split_2(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        x_restored = dim_handler.unfit(x_fitted)
        # Act
        # Assert
        assert x_restored.shape == torch.Size([2, 4, 5, 1, 3, 6])


    def test_example2_from_docstring_x_fitted_shape_equals_torch_size_40_1_3_6(self, dim_handler):
        # Arrange
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        # Fit
        # Act
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        # Act
        # Assert
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_example2_from_docstring_y_shape_equals_torch_size_40_1_6_split_1(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        # Act
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_example2_from_docstring_y_shape_equals_torch_size_40_1_6_split_2(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        y = x_fitted.mean(axis=-2)
        # Act
        # Assert
        assert y.shape == torch.Size([40, 1, 6])

    def test_example2_from_docstring_y_restored_shape_equals_torch_size_2_4_5_1_6_split_1(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        # Act
        # Assert
        assert x_fitted.shape == torch.Size([40, 1, 3, 6])

    def test_example2_from_docstring_y_restored_shape_equals_torch_size_2_4_5_1_6_split_2(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        y = x_fitted.mean(axis=-2)
        # Act
        # Assert
        assert y.shape == torch.Size([40, 1, 6])

    def test_example2_from_docstring_y_restored_shape_equals_torch_size_2_4_5_1_6_split_3(self, dim_handler):
        # Arrange
        x = torch.rand(1, 2, 3, 4, 5, 6)
        x_fitted = dim_handler.fit(x, keepdims=[0, 2, 5])
        x_fitted.shape == torch.Size([40, 1, 3, 6])
        y = x_fitted.mean(axis=-2)
        y.shape == torch.Size([40, 1, 6])
        y_restored = dim_handler.unfit(y)
        # Act
        # Assert
        assert y_restored.shape == torch.Size([2, 4, 5, 1, 6])


    def test_multiple_fit_unfit_cycles_x1_shape_equals_n_15_2_4(self, dim_handler):
        # Arrange
        # Arrange
        x = np.random.rand(2, 3, 4, 5)
        # First cycle
        keepdims1 = [0, 2]
        # Act
        x1 = dim_handler.fit(x, keepdims=keepdims1)
        # Act
        # Assert
        # Assert
        assert x1.shape == (15, 2, 4)

    def test_multiple_fit_unfit_cycles_x2_shape_equals_n_40_3_split_1(self, dim_handler):
        # Arrange
        x = np.random.rand(2, 3, 4, 5)
        keepdims1 = [0, 2]
        x1 = dim_handler.fit(x, keepdims=keepdims1)
        # Act
        # Assert
        assert x1.shape == (15, 2, 4)

    def test_multiple_fit_unfit_cycles_x2_shape_equals_n_40_3_split_2(self, dim_handler):
        # Arrange
        x = np.random.rand(2, 3, 4, 5)
        keepdims1 = [0, 2]
        x1 = dim_handler.fit(x, keepdims=keepdims1)
        x1.shape == (15, 2, 4)
        dim_handler2 = DimHandler()
        keepdims2 = [1]
        x2 = dim_handler2.fit(x, keepdims=keepdims2)
        # Act
        # Assert
        assert x2.shape == (40, 3)


    @pytest.mark.parametrize(
        "shape,keepdims,expected_fitted_shape",
        [
            ((2, 3), [0], (3, 2)),
            ((2, 3), [1], (2, 3)),
            ((2, 3, 4), [0, 1], (4, 2, 3)),
            ((2, 3, 4), [1, 2], (2, 3, 4)),
            ((2, 3, 4, 5), [0, 3], (12, 2, 5)),
        ],
    )
    def test_parametrized_shapes_result_shape_equals_expected_fitted_shap(
        self, dim_handler, shape, keepdims, expected_fitted_shape
    ):
        """Test various shape and keepdims combinations."""
        # Arrange
        x = np.random.rand(*shape)
        # Act
        result = dim_handler.fit(x, keepdims=keepdims)
        # Assert
        assert result.shape == expected_fitted_shape

    def test_gradient_preservation_torch(self, dim_handler):
        """Test that gradients are preserved through fit/unfit for torch tensors."""
        # Arrange
        x = torch.rand(2, 3, 4, requires_grad=True)
        keepdims = [1]

        # Fit
        x_fitted = dim_handler.fit(x, keepdims=keepdims)

        # Some operation
        y = x_fitted.sum()

        # Check gradients can flow
        # Act
        y.backward()
        # Assert
        assert x.grad is not None


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_DimHandler.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-05 00:39:26 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_DimHandler.py
#
# """
# This script demonstrates DimHandler, which:
# 1) Keeps designated dimensions,
# 2) Permutes the kept dimensions to the last while maintaining their relative order,
# 3) Reshapes the remaining dimensions to the first, batch dimension,
# 4) (Performs calculations),
# 5) Restores the summarized dimensions to their original shapes.
# """
#
# # Imports
# import sys
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
#
# # Functions
# class DimHandler:
#     """
#     A utility class for handling dimension manipulations on tensors or arrays, including reshaping and permuting dimensions.
#
#     Attributes:
#         orig_shape (tuple): The original shape of the input tensor or array before any manipulation.
#         keepdims (list): The list of dimensions to be kept and moved to the end.
#         n_non_keepdims (list): The sizes of the dimensions not kept, used for reshaping back to the original shape.
#         n_keepdims (list): The sizes of the kept dimensions, used for reshaping.
#
#     Example1:
#         import torch
#
#         dh = DimHandler()
#         x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#         print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#         x = dh.fit(x, keepdims=[0, 2, 5])
#         print(x.shape)  # torch.Size([40, 1, 3, 6])
#         x = dh.unfit(x)
#         print(x.shape)  # torch.Size([2, 4, 5, 1, 3, 6])
#
#     Example 2:
#         import torch
#
#         dh = DimHandler()
#         x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#         print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#         x = dh.fit(x, keepdims=[0, 2, 5])
#         print(x.shape)  # torch.Size([40, 1, 3, 6])
#         y = x.mean(axis=-2) # calculation on the kept dims
#         print(y.shape) # torch.Size([40, 1, 6])
#         y = dh.unfit(y)
#         print(y.shape) # torch.Size([2, 4, 5, 1, 6])
#     """
#
#     def __init__(self):
#         pass
#         # self.orig_shape = None
#         # self.keepdims = None
#
#     def fit(self, x, keepdims=[]):
#         if isinstance(x, np.ndarray):
#             return self._fit_numpy(x, keepdims=keepdims)
#         elif isinstance(x, torch.Tensor):
#             return self._fit_torch(x, keepdims=keepdims)
#
#     def _fit_numpy(self, x, keepdims=[]):
#         """
#         Reshapes the input NumPy array by flattening the dimensions not in `keepdims` and moving the kept dimensions to the end.
#
#         Arguments:
#             x (numpy.ndarray): The input array to be reshaped.
#             keepdims (list of int): The indices of the dimensions to keep.
#
#         Returns:
#             x_flattened (numpy.ndarray): The reshaped array with kept dimensions moved to the end.
#         """
#         assert len(keepdims) <= len(x.shape), (
#             "keepdims cannot have more dimensions than the array itself."
#         )
#
#         # Normalize negative indices to positive indices
#         total_dims = len(x.shape)
#         keepdims = [dim if dim >= 0 else total_dims + dim for dim in keepdims]
#         keepdims = sorted(set(keepdims))
#
#         self.shape_fit = x.shape
#
#         non_keepdims = [ii for ii in range(len(self.shape_fit)) if ii not in keepdims]
#
#         self.n_non_keepdims = [self.shape_fit[nkd] for nkd in non_keepdims]
#         self.n_keepdims = [self.shape_fit[kd] for kd in keepdims]
#
#         # Permute the array dimensions so that the non-kept dimensions come first
#         new_order = non_keepdims + keepdims
#         x_permuted = np.transpose(x, axes=new_order)
#
#         # Flatten the non-kept dimensions
#         x_flattened = x_permuted.reshape(-1, *self.n_keepdims)
#
#         return x_flattened
#
#     def _fit_torch(self, x, keepdims=[]):
#         """
#         Reshapes the input tensor or array by flattening the dimensions not in `keepdims` and moving the kept dimensions to the end.
#
#         Arguments:
#             x (torch.Tensor): The input tensor or array to be reshaped.
#             keepdims (list of int): The indices of the dimensions to keep.
#
#         Returns:
#             x_flattend (torch.Tensor): The reshaped tensor or array with kept dimensions moved to the end.
#
#         Note:
#             This method modifies the `orig_shape`, `keepdims`, `n_non_keepdims`, and `n_keepdims` attributes based on the input.
#         """
#         assert len(keepdims) <= len(x.shape), (
#             "keepdims cannot have more dimensions than the tensor itself."
#         )
#
#         keepdims = torch.tensor(keepdims).clone().detach().cpu().int()
#         # Normalize negative indices to positive indices
#         total_dims = len(x.shape)
#         keepdims = [dim if dim >= 0 else total_dims + dim for dim in keepdims]
#         keepdims = sorted(set(keepdims))
#
#         self.shape_fit = x.shape
#
#         non_keepdims = [
#             int(ii) for ii in torch.arange(len(self.shape_fit)) if ii not in keepdims
#         ]
#
#         self.n_non_keepdims = [self.shape_fit[nkd] for nkd in non_keepdims]
#         self.n_keepdims = [self.shape_fit[kd] for kd in keepdims]
#
#         x_permuted = x.permute(*non_keepdims, *keepdims)
#         x_flattend = x_permuted.reshape(-1, *self.n_keepdims)
#
#         return x_flattend
#
#     def unfit(self, y):
#         """
#         Restores the first dimension of reshaped tensor or array back to its original shape before the `fit` operation.
#
#         Arguments:
#             y (torch.Tensor or numpy.array): The tensor or array to be restored to its original shape.
#
#         Returns:
#             y_restored (torch.Tensor or numpy.array): The tensor or array restored to its original shape.
#         """
#         self.shape_unfit = y.shape
#         return y.reshape(*self.n_non_keepdims, *self.shape_unfit[1:])
#
#
# if __name__ == "__main__":
#     import scitex
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt)
#
#     # Example1:
#     scitex_gen.printc("Example 1")
#     dh = DimHandler()
#     x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#     print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#     x = dh.fit(x, keepdims=[0, 2, 5])
#     print(x.shape)  # torch.Size([40, 1, 3, 6])
#     x = dh.unfit(x)
#     print(x.shape)  # torch.Size([2, 4, 5, 1, 3, 6])
#
#     # Example 2:
#     scitex_gen.printc("Example 2")
#     dh = DimHandler()
#     x = torch.rand(1, 2, 3, 4, 5, 6)  # Example tensor
#     print(x.shape)  # torch.Size([1, 2, 3, 4, 5, 6])
#     x = dh.fit(x, keepdims=[0, 2, 5])
#     print(x.shape)  # torch.Size([40, 1, 3, 6])
#     y = x.mean(axis=-2)  # calculation on the kept dims
#     print(y.shape)  # torch.Size([40, 1, 6])
#     y = dh.unfit(y)
#     print(y.shape)  # torch.Size([2, 4, 5, 1, 6])
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/scitex/gen/_DimHandler.py
# """
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_DimHandler.py
# --------------------------------------------------------------------------------
