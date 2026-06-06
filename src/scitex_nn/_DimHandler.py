#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:39:26 (ywatanabe)"
# File: ./src/scitex_nn/_DimHandler.py
"""DimHandler — dimension-manipulation utility for tensors / arrays.

Ported from scitex_gen._introspect._DimHandler (Phase B retirement wave).

This script demonstrates DimHandler, which:
  1) Keeps designated dimensions,
  2) Permutes the kept dimensions to the last while maintaining their relative order,
  3) Reshapes the remaining dimensions to the first, batch dimension,
  4) (Performs calculations),
  5) Restores the summarized dimensions to their original shapes.
"""

import numpy as np
import torch


class DimHandler:
    """
    A utility class for handling dimension manipulations on tensors or
    arrays, including reshaping and permuting dimensions.

    Attributes
    ----------
    shape_fit : tuple
        Original shape of the input before fitting.
    n_non_keepdims : list[int]
        Sizes of the dimensions not kept.
    n_keepdims : list[int]
        Sizes of the dimensions kept.

    Example
    -------
    >>> import torch
    >>> dh = DimHandler()
    >>> x = torch.rand(1, 2, 3, 4, 5, 6)
    >>> x.shape
    torch.Size([1, 2, 3, 4, 5, 6])
    >>> x = dh.fit(x, keepdims=[0, 2, 5])
    >>> x.shape
    torch.Size([40, 1, 3, 6])
    >>> x = dh.unfit(x)
    >>> x.shape
    torch.Size([2, 4, 5, 1, 3, 6])
    """

    def __init__(self):
        pass

    def fit(self, x, keepdims=None):
        if keepdims is None:
            keepdims = []
        if isinstance(x, np.ndarray):
            return self._fit_numpy(x, keepdims=keepdims)
        elif isinstance(x, torch.Tensor):
            return self._fit_torch(x, keepdims=keepdims)

    def _fit_numpy(self, x, keepdims=None):
        """
        Reshapes the input NumPy array by flattening the dimensions not
        in ``keepdims`` and moving the kept dimensions to the end.
        """
        if keepdims is None:
            keepdims = []
        assert len(keepdims) <= len(
            x.shape
        ), "keepdims cannot have more dimensions than the array itself."

        # Normalize negative indices to positive indices
        total_dims = len(x.shape)
        keepdims = [dim if dim >= 0 else total_dims + dim for dim in keepdims]
        keepdims = sorted(set(keepdims))

        self.shape_fit = x.shape

        non_keepdims = [
            ii for ii in range(len(self.shape_fit)) if ii not in keepdims
        ]

        self.n_non_keepdims = [self.shape_fit[nkd] for nkd in non_keepdims]
        self.n_keepdims = [self.shape_fit[kd] for kd in keepdims]

        # Permute the array dimensions so that the non-kept dimensions come first
        new_order = non_keepdims + keepdims
        x_permuted = np.transpose(x, axes=new_order)

        # Flatten the non-kept dimensions
        x_flattened = x_permuted.reshape(-1, *self.n_keepdims)

        return x_flattened

    def _fit_torch(self, x, keepdims=None):
        """
        Reshapes the input tensor by flattening the dimensions not in
        ``keepdims`` and moving the kept dimensions to the end.
        """
        if keepdims is None:
            keepdims = []
        assert len(keepdims) <= len(
            x.shape
        ), "keepdims cannot have more dimensions than the tensor itself."

        keepdims = torch.tensor(keepdims).clone().detach().cpu().int()
        # Normalize negative indices to positive indices
        total_dims = len(x.shape)
        keepdims = [dim if dim >= 0 else total_dims + dim for dim in keepdims]
        keepdims = sorted(set(keepdims))

        self.shape_fit = x.shape

        non_keepdims = [
            int(ii)
            for ii in torch.arange(len(self.shape_fit))
            if ii not in keepdims
        ]

        self.n_non_keepdims = [self.shape_fit[nkd] for nkd in non_keepdims]
        self.n_keepdims = [self.shape_fit[kd] for kd in keepdims]

        x_permuted = x.permute(*non_keepdims, *keepdims)
        x_flattend = x_permuted.reshape(-1, *self.n_keepdims)

        return x_flattend

    def unfit(self, y):
        """
        Restores the first dimension of reshaped tensor or array back to
        its original shape before the ``fit`` operation.
        """
        self.shape_unfit = y.shape
        return y.reshape(*self.n_non_keepdims, *self.shape_unfit[1:])


# EOF
