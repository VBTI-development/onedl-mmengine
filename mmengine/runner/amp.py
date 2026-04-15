# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import Optional

import torch

from mmengine.device import get_device
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


@contextmanager
def autocast(device_type: Optional[str] = None,
             dtype: Optional[torch.dtype] = None,
             enabled: bool = True,
             cache_enabled: Optional[bool] = None):
    """A wrapper of ``torch.autocast`` and ``toch.cuda.amp.autocast``.

    This function provides a unified interface by wrapping
    ``torch.autocast`` and ``torch.cuda.amp.autocast``, which resolves the
    compatibility issues that ``torch.cuda.amp.autocast`` does not support
    running mixed precision with cpu, and both contexts have different
    arguments. We suggest users using this function in the code
    to achieve maximized compatibility of different PyTorch versions.

    Note:
        ``autocast`` requires pytorch version >= 1.10.0. If pytorch version
        <= 1.10.0 it will raise an error.

    Examples:
         >>> # case1: Pytorch version >= 1.10.0
         >>> with autocast():
         >>>    # default cuda mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu'):
         >>>    # cpu mixed precision context
         >>>    pass
         >>> with autocast(
         >>>     device_type='cuda', enabled=True, cache_enabled=True):
         >>>    # enable precision context with more specific arguments.
         >>>    pass

    Args:
        device_type (str, required):  Whether to use 'cuda' or 'cpu' device.
        enabled(bool):  Whether autocasting should be enabled in the region.
            Defaults to True
        dtype (torch_dtype, optional):  Whether to use ``torch.float16`` or
            ``torch.bfloat16``.
        cache_enabled(bool, optional):  Whether the weight cache inside
            autocast should be enabled.
    """
    # If `enabled` is True, enable an empty context and all calculations
    # are performed under fp32.
    assert digit_version(TORCH_VERSION) >= digit_version('1.10.0'), (
        'The minimum pytorch version requirements of mmengine is 1.10.0, but '
        f'got {TORCH_VERSION}')

    # Modified from https://github.com/pytorch/pytorch/blob/master/torch/amp/autocast_mode.py # noqa: E501
    # This code should update with the `torch.autocast`.
    if cache_enabled is None:
        cache_enabled = torch.is_autocast_cache_enabled()
    device = get_device()
    device_type = device if device_type is None else device_type

    if device_type == 'cuda':
        if dtype is None:
            dtype = torch.get_autocast_dtype('cuda')

        if dtype == torch.bfloat16 and not \
                torch.cuda.is_bf16_supported():
            raise RuntimeError(
                'Current CUDA Device does not support bfloat16. Please '
                'switch dtype to float16.')

    elif device_type == 'cpu':
        if dtype is None:
            dtype = torch.bfloat16
        assert dtype == torch.bfloat16, (
            'In CPU autocast, only support `torch.bfloat16` dtype')

    elif device_type == 'mlu':
        pass

    elif device_type == 'npu':
        pass
    elif device_type == 'musa':
        if dtype is None:
            dtype = torch.get_autocast_dtype('cuda')
        with torch.musa.amp.autocast(
                enabled=enabled, dtype=dtype, cache_enabled=cache_enabled):
            yield
            return
    else:
        # Device like MPS does not support fp16 training or testing.
        # If an inappropriate device is set and fp16 is enabled, an error
        # will be thrown.
        if enabled is False:
            yield
            return
        else:
            raise ValueError('User specified autocast device_type must be '
                             f'cuda or cpu, but got {device_type}')

    with torch.autocast(
            device_type=device_type,
            enabled=enabled,
            dtype=dtype,
            cache_enabled=cache_enabled):
        yield
