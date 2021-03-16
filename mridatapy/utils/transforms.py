#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import numpy as np


def fft_centered(
    input: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    dim: Optional[Tuple[int, ...]] = None,
    norm: Optional[str] = None
) -> np.ndarray:
    """Computes the centered N dimensional discrete Fourier transform (FFT) of
    input.

    Args:
        input: Input NumPy array, can be complex.
        shape (Optional): Shape of output, truncated or zero-padded.
        dim (Optional): Dimensions along which to apply FFT.
        norm (Optional): Normalization mode:
            - "forward": normalized by 1 / n;
            - "backward" (Default): no normalization;
            - "ortho": normalized by 1 / sqrt(n), making the FFT orthonormal.

    Returns:
        The FFT of input. Output has the same shape as input.
    """
    input = np.fft.ifftshift(input, axes=dim)
    input = np.fft.fftn(input, s=shape, axes=dim, norm=norm)
    input = np.fft.fftshift(input, axes=dim)
    return input


def ifft_centered(
    input: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    dim: Optional[Tuple[int, ...]] = None,
    norm: Optional[str] = None
) -> np.ndarray:
    """Computes the centered N dimensional inverse discrete Fourier transform
    (IFFT) of input.

    Args:
        input: Input NumPy array, can be complex.
        shape (Optional): Shape of output, truncated or zero-padded.
        dim (Optional): Dimensions along which to apply IFFT.
        norm (Optional): Normalization mode:
            - "forward": normalized by 1 / n;
            - "backward" (Default): no normalization;
            - "ortho": normalized by 1 / sqrt(n), making the IFFT orthonormal.

    Returns:
        The IFFT of input. Output has the same shape as input.
    """
    input = np.fft.ifftshift(input, axes=dim)
    input = np.fft.ifftn(input, s=shape, axes=dim, norm=norm)
    input = np.fft.fftshift(input, axes=dim)
    return input
