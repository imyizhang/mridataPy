#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy


def _asfloat64(gt: np.ndarray, pred: np.ndarray) -> (np.ndarray, np.ndarray):
    """Copies two images, and casts to np.float64.

    Args:
        gt: Ground-truth image.
        pred: Predicted image.

    Returns:
        The tuple of two copies.
    """
    return gt.astype(np.float64), pred.astype(np.float64)


def _assert_shape(gt: np.ndarray, pred: np.ndarray):
    """Raise an error if shapes of two images do not match.

    Args:
        gt: Ground-truth image.
        pred: Predicted image.
    """
    if not gt.shape == pred.shape:
        raise ValueError('Input images must have the same shape.')


def mean_squared_error(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Computes the Mean Squared Error (MSE) between two images.

    Args:
        gt: Ground-truth image.
        pred: Predicted image with the same shape as ground-truth image.

    Returns:
        The Mean Squared Error (MSE).
    """
    _assert_shape(gt, pred)
    gt, pred = _asfloat64(gt, pred)
    return np.mean(np.square(gt - pred))


def normalized_mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Computes the Normalized Mean Squared Error (NMSE) between two images.

    Args:
        gt: Ground-truth image.
        pred: Predicted image with the same shape as ground-truth image.

    Returns:
        The Normalized Mean Squared Error (NMSE).
    """
    _assert_shape(gt, pred)
    gt, pred = _asfloat64(gt, pred)
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def peak_signal_noise_ratio(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: Optional[float] = None
) -> np.ndarray:
    """Computes the Peak Signal to Noise Ratio (PSNR) between two images.

    Args:
        gt: Ground-truth image.
        pred: Predicted image with the same shape as ground-truth image.

    Returns:
        The Peak Signal to Noise Ratio (PSNR).
    """
    _assert_shape(gt, pred)
    gt, pred = _asfloat64(gt, pred)
    peak_signal = data_range if data_range else (gt.max() - gt.min())
    noise = mean_squared_error(gt, pred)
    return 10 * np.log10((peak_signal ** 2) / noise)


def structural_similarity(
    gt: np.ndarray,
    pred: np.ndarray,
    data_range: Optional[float] = None
) -> np.ndarray:
    """Computes the Structural Similarity Index (SSIM) between two images.

    Args:
        gt: Ground-truth 2 dimensional image.
        pred: Predicted image with the same shape as ground-truth image.

    Returns:
        The Structural Similarity Index (SSIM).

    References:
        [1] https://github.com/scikit-image/scikit-image/blob/v0.18.x/skimage/metrics/_structural_similarity.py
    """
    ndim = gt.ndim
    if ndim != 2:
        raise ValueError
    _assert_shape(gt, pred)

    # parameters
    filter = scipy.ndimage.uniform_filter
    size = 7
    K1 = 0.01
    K2 = 0.03
    L = data_range if data_range else (gt.max() - gt.min())
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # scipy.ndimage filters need floating point data
    gt, pred = _asfloat64(gt, pred)

    # compute (weighted) means
    ux = filter(gt, size=size)
    uy = filter(pred, size=size)

    # compute (weighted) variances and covariances
    uxx = filter(gt * gt, size=size)
    uyy = filter(pred * pred, size=size)
    uxy = filter(gt * pred, size=size)

    # filter has already normalized by NP
    NP = size ** ndim
    cov_norm = NP / (NP - 1)

    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    A1, A2, B1, B2 = (2 * ux * uy + C1,
                      2 * vxy + C2,
                      ux ** 2 + uy ** 2 + C1,
                      vx + vy + C2)

    S = (A1 * A2) / (B1 * B2)

    # to avoid edge effects will ignore filter radius strip around edges
    edge = (size - 1) // 2

    # compute (weighted) mean of ssim
    return _crop(S, edge).mean()


def _crop(a: np.ndarray, edge: int) -> np.ndarray:
    """Crop Numpy array by the given edge size along each dimension.

    Args:
        a: Input NumPy array.
        edge: Edge size to be cropped along each dimension.

    Returns:
        Cropped NumPy array, a sliced view of input array.

    References:
        [1] https://github.com/scikit-image/scikit-image/blob/v0.18.x/skimage/util/arraycrop.py
    """
    edges = [[edge, edge]] * a.ndim
    slices = tuple([slice(e1, a.shape[i] - e2) for i, (e1, e2) in enumerate(edges)])
    return a[slices]
