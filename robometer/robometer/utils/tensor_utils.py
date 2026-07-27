#!/usr/bin/env python3
"""
Tensor utility functions for common tensor operations.
"""

import torch
import numpy as np


def t2n(tensor, dtype=torch.float32):
    """
    Convert tensor to numpy array.

    Args:
        tensor: PyTorch tensor to convert
        dtype: Target dtype for conversion (default: torch.float32)

    Returns:
        numpy.ndarray: Converted numpy array
    """
    if tensor is None:
        return None
    if isinstance(tensor, np.ndarray):
        return tensor
    if not torch.is_tensor(tensor):
        return np.array(tensor)
    return tensor.detach().to(dtype=dtype).cpu().numpy()
