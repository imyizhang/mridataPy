#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def root_sum_squares(
    input: np.ndarray,
    dim: int = 1,
    complex: bool = False
) -> np.ndarray:
    """Computes the Root Sum of Squares (RSS) along the a given dimension (coil
    dimension).

    Args:
        input: Input NumPy array with multiple channels (coils), can be complex.
        dim: Dimension along which to apply RSS transform.
        complex: Complex toggle, to handle the case where the real and imaginary
            parts are stacked along the last dimension of input.
                - False (Default): Input has the shape of (Nslices, Ncoils, Nx, Ny)
                or (Ncoils, Nx, Ny).
                - True: Input has the shape of (Nslices, Ncoils, Nx, Ny, 2)
                or (Ncoils, Nx, Ny, 2). Note that the last dimension of input
                is 2 since the real and imaginary parts are stacked along it.

    Returns:
        The RSS of input. Output has the shape of (Nslices, Nx, Ny) or (Nx, Ny).
    """
    if complex:
        input = np.sum(np.square(input), axis=-1)
    else:
        input = np.square(np.abs(input))
    return np.sqrt(np.sum(input, axis=dim))
