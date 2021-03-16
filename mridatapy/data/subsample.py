#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
import contextlib
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import numba as nb


@contextlib.contextmanager
def _temp_seed(
    random_generator: np.random.RandomState,
    seed: Optional[Union[int, Sequence[int]]] = None
):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = random_generator.get_state()
        random_generator.seed(seed)
        try:
            yield
        finally:
            random_generator.set_state(state)


class Mask(object):
    """A sampling Mask object that can densely sample the center region in
    k-space while subsample the outer region based on acceleration factor.

    References:
        [1] https://github.com/facebookresearch/fastMRI/blob/master/fastmri/data/subsample.py
    """
    __metaclass__ = abc.ABCMeta

    def __init__(
        self,
        acceleration_factor: int,
        center_fraction: float
    ):
        """
        Args:
            acceleration_factor: Amount of subsampling, leading to n-fold
                acceleration over the fully sampled k-space.
            center_fraction: Fraction of the center region to be retained for
                auto-calibration corresponding to low spatial frequencies.
        """
        self.acceleration_factor = acceleration_factor
        self.center_fraction = center_fraction
        self.random_generator = np.random.RandomState()

    def __call__(
        self,
        shape: Tuple[int, ...],
        max_attempts: int = 30,
        tolerance: float = 0.1,
        seed: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        raise NotImplementedError


class RandomLine(Mask):
    """RandomLine generates a sampling mask of the given shape. The mask randomly
    selects a subset of columns from input k-space data.

    Assuming the k-space data has N columns, the mask picks out:
        1. N_center = N * center_fraction columns in the center region
        corresponding to low spatial frequencies.
        2. The remaining columns are randomly selected uniformly from the outer
        region with a probability = (N / acceleration_factor - N_center) /
        (N - N_center).
    Note that the expected number of columns to be selected is equivalent to
    N / acceleration.
    """

    def __call__(
        self,
        shape: Tuple[int, ...],
        max_attempts: int = 30,
        tolerance: float = 0.1,
        seed: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Args:
            shape: Shape of the mask to be generated, which should contain
                exactly 2 dimensions.
            max_attempts: Maximum attempts to perform sampling in order to
                obtain satisfactory acceleration.
            tolerance: Tolerance for how much the resulting acceleration can
                deviate from desired one.
            seed (Optional): Seed for the random number generator.

        Returns:
            A sampling mask of the given shape.
        """
        if not len(shape) == 2:
            raise ValueError('Shape should contain exactly 2 dimensions.')

        # parameters
        num_columns = shape[1]
        num_center = round(num_columns * self.center_fraction)
        center_start = (num_columns - num_center + 1) // 2
        center_end = center_start + num_center
        probability = ((num_columns / self.acceleration_factor - num_center) /
                       (num_columns - num_center))

        with _temp_seed(self.random_generator, seed):
            current_accel = 0
            k = 0
            while k < max_attempts:
                #print('attempt %d'%k)
                # create the mask
                mask = self.random_generator.uniform(size=num_columns) < probability
                # retain the center region
                mask[center_start:center_end] = True
                current_accel = num_columns / np.sum(mask)
                if abs(self.acceleration_factor - current_accel) < tolerance:
                    break
                k +=1

            if abs(self.acceleration_factor - current_accel) >= tolerance:
                raise ValueError

            # reshape the mask
            mask = np.ones(shape) * mask

        return mask  # mask.dtype = np.float64


class EquispacedLine(Mask):
    """
    EquispacedLine generates a sampling mask of the given shape. The mask
    selects a roughly equispaced subset of columns from input k-space data.

    Assuming the k-space data has N columns, the mask picks out:
        1. N_center = (N * center_fraction) columns in the center region
           corresponding to low spatial frequencies.
        2. The remaining columns are selected with equal spacing at a proportion
           that meets the desired acceleration factor taking into consideration
           N_center columns.
    Note that the expected number of columns to be selected is equivalent to
    (N / acceleration).
    """

    def __call__(
        self,
        shape: Tuple[int, ...],
        max_attempts: int = 30,
        tolerance: float = 0.1,
        seed: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Args:
            shape: Shape of the mask to be generated, which should contain
                exactly 2 dimensions.
            max_attempts: Maximum attempts to perform sampling in order to
                obtain satisfactory acceleration.
            tolerance: Tolerance for how much the resulting acceleration can
                deviate from desired one.
            seed (Optional): Seed for the random number generator.

        Returns:
            A sampling mask of the given shape.
        """
        if not len(shape) == 2:
            raise ValueError('Shape should contain exactly 2 dimensions.')

        # parameters
        num_columns = shape[1]
        num_center = round(num_columns * self.center_fraction)
        center_start = (num_columns - num_center + 1) // 2
        center_end = center_start + num_center
        adjusted_accel = ((num_columns - num_center) /
                          (num_columns / self.acceleration_factor - num_center))

        with _temp_seed(self.random_generator, seed):
            # initialiaze the mask
            mask = np.zeros(num_columns)

            current_accel = 0
            k = 0
            while k < max_attempts:
                #print('attempt %d'%k)
                # create the mask
                offset = self.random_generator.randint(0, round(adjusted_accel))
                selection = np.arange(offset, num_columns - 1, adjusted_accel)
                # it may not give exactly equispaced samples since we use `np.around`
                selection = np.around(selection).astype(np.uint)
                mask[selection] = 1
                # retain the center region
                mask[center_start:center_end] = 1
                current_accel = num_columns / np.sum(mask)
                if abs(self.acceleration_factor - current_accel) < tolerance:
                    break
                k += 1

            if abs(self.acceleration_factor - current_accel) >= tolerance:
                raise ValueError

            # reshape the mask
            mask = np.ones(shape) * mask

        return mask  # mask.dtype = np.float64


class PoissonDisk(Mask):
    """
    PoissonDisk generates a sampling mask of the given shape. The mask selects
    a subset of points from input k-space data, characterized by the Poisson
    disk sampling pattern.

    Assuming the k-space radius is r, the Poisson disk mask samples with the
    density proportional to:
        1 / (1 + r * slope)

    References:
        [1] Bridson, Robert. Fast Poisson disk sampling in arbitrary dimensions.
        SIGGRAPH sketches. 2007.
        [2] https://github.com/mikgroup/sigpy/blob/master/sigpy/mri/samp.py
    """

    def __call__(
        self,
        shape: Tuple[int, ...],
        max_attempts: int = 30,
        tolerance: float = 0.1,
        seed: Optional[Union[int, Sequence[int]]] = None
    ) -> np.ndarray:
        """
        Args:
            shape: Shape of the mask to be generated, which should contain
                exactly 2 dimensions.
            max_attempts: Maximum number of samples to choose before rejection
                in Poisson disk sampling.
            tolerance: Tolerance for how much the resulting acceleration can
                deviate from desired factor.
            seed (Optional): Seed for the random number generator.

        Returns:
            A sampling mask of the given shape.
        """
        if not len(shape) == 2:
            raise ValueError('Shape should contain exactly 2 dimensions.')

        # parameters
        nx, ny = shape
        nx_prime = ny_prime = round(math.sqrt(nx * ny * self.center_fraction))
        center_shape = (nx_prime, ny_prime)

        # determine the k-space radius
        x, y = np.mgrid[0:nx, 0:ny]
        x = np.maximum(np.abs(x - nx / 2) - nx_prime / 2, 0)
        x /= x.max()
        y = np.maximum(np.abs(y - ny / 2) - ny_prime / 2, 0)
        y /= y.max()
        r = np.sqrt(np.square(x) + np.square(y))

        # perform a binary search on the slope s.t. leading to a satisfactory
        # acceleration factor
        current_accel = 0
        slope_min = 0
        slope_max = max(nx, ny)
        #i = 0
        while slope_min < slope_max:
            #print('binary search %d'%i)
            slope = (slope_min + slope_max) / 2
            radius_x = np.clip((1 + r * slope) * nx / max(nx, ny), 1, None)
            radius_y = np.clip((1 + r * slope) * ny / max(nx, ny), 1, None)
            # create the mask
            mask = _sample(shape, center_shape, radius_x, radius_y, max_attempts, seed)
            # crop corners of the k-space
            mask *= (r < 1)
            current_accel = nx * ny / np.sum(mask)
            if abs(self.acceleration_factor - current_accel) < tolerance:
                break
            if current_accel < self.acceleration_factor:
                slope_min = slope
            else:
                slope_max = slope

        if abs(self.acceleration_factor - current_accel) >= tolerance:
            raise ValueError

        return mask  # mask.dtype = np.float64


@nb.jit(nopython=True, cache=True)
def _sample(
    shape: Tuple[int, ...],
    center_shape: Tuple[int, ...],
    radius_x: np.ndarray,
    radius_y: np.ndarray,
    max_attempts: int,
    seed: Optional[Union[int, Sequence[int]]] = None
) -> np.ndarray:
    # Parameters
    nx, ny = shape
    nx_prime, ny_prime = center_shape
    if seed is not None:
        np.random.seed(int(seed))

    #with local_seed(random_generator, seed):
    # Step 0. Initialize the active list and the mask.
    pxs = np.empty(nx * ny, dtype=np.int64)
    pys = np.empty(nx * ny, dtype=np.int64)
    mask = np.zeros(shape)

    # Step 1. Select the initial sample, x0, randomly chosen uniformly
    # from the grid, and insert it to the active list with index 0.
    pxs[0] = np.random.randint(0, nx)
    pys[0] = np.random.randint(0, ny)
    num_actives = 1

    # Step 2. While the active list is not empty,
    while num_actives > 0:
        # choose a random index from the active list, i,
        i = np.random.randint(0, num_actives)
        # xi from the active list
        px, py = pxs[i], pys[i]
        rx = radius_x[px, py]
        ry = radius_y[px, py]

        # Generate up to `max_attempts` points
        found = False
        k = 0
        while not found and k < max_attempts:
            # Generate a point, q, randomly chosen uniformly from the
            # spherical annulus between radius r and 2r around xi
            v = np.sqrt(np.random.random() * 3 + 1)
            t = 2 * np.pi * np.random.random()
            qx = px + v * rx * np.cos(t)
            qy = py + v * ry * np.sin(t)

            # Reject it if outside the grid,
            if (qx >= 0 and qx < nx) and (qy >= 0 and qy < ny):
                found = True
                # or if not adequately far from existing samples
                startx = max(int(qx - rx), 0)
                endx = min(int(qx + rx + 1), nx)
                starty = max(int(qy - ry), 0)
                endy = min(int(qy + ry + 1), ny)
                for x in range(startx, endx):
                    for y in range(starty, endy):
                        if (mask[x, y] == 1) and (((qx - x) / radius_x[x, y]) ** 2 + ((qy - y) / (radius_y[x, y])) ** 2 < 1):
                            found = False
                            break

            # Then perform next attempt.
            k += 1

        # Add the point to the active list if found,
        if found:
            pxs[num_actives] = qx
            pys[num_actives] = qy
            mask[int(qx), int(qy)] = 1
            num_actives += 1
        # otherwise remove it from the active list.
        else:
            pxs[i] = pxs[num_actives - 1]
            pys[i] = pys[num_actives - 1]
            num_actives -= 1

    # Retain the center region.
    mask[int(nx / 2 - nx_prime / 2):int(nx / 2 + nx_prime / 2),
         int(ny / 2 - ny_prime / 2):int(ny / 2 + ny_prime / 2)] = 1

    return mask  # mask.dtype = np.float64
