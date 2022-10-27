# -*- coding: utf-8 -*-
from dipy.data import get_sphere
import numpy as np
from numba import njit


class AsymmetricFilter():
    def __init__(self, sh_order, sh_basis, full_basis, sphere_str,
                 sigma_spatial, sigma_align, sigma_angle,
                 sigma_range):
        self.sh_order = sh_order
        self.basis = sh_basis
        self.full_basis = full_basis
        self.sphere = get_sphere(sphere_str)
        self.sigma_spatial = sigma_spatial
        self.sigma_align = sigma_align
        self.sigma_angle = sigma_angle
        self.sigma_range = sigma_range

        nx_filter, uv_filter = _build_filters(self.sphere.vertices,
                                              sigma_spatial,
                                              sigma_align,
                                              sigma_angle)

    def __call__(self, data):
        print('Called.')


@njit(cache=True)
def _build_filters(directions, sigma_spatial, sigma_align, sigma_angle):
    directions = np.ascontiguousarray(directions.astype(np.float32))

    half_width = int(round(3 * sigma_spatial))
    nx_weights = np.zeros((2*half_width+1, 2*half_width+1, 2*half_width+1),
                          dtype=np.float32)
    uv_weights = np.zeros((len(directions), len(directions)), dtype=np.float32)

    # 1. precompute weights on angle
    # c'est toujours les mêmes peu importe le voxel en question
    for u_i, u in enumerate(directions):
        uvec = np.reshape(np.ascontiguousarray(u), (1, 3))
        weights = np.arccos(np.clip(np.dot(uvec, directions.T), -1.0, 1.0))
        weights = _evaluate_gaussian(sigma_angle, weights)
        weights /= np.sum(weights)
        uv_weights[u_i] = weights  # each column sums to 1.

    # 2. orientation weights
    # dans chaque 

    # 3. spatial weights
    for i in range(-half_width, half_width+1):
        for j in range(-half_width, half_width+1):
            for k in range(-half_width, half_width+1):
                dxy = np.array([[i, j, k]], dtype=np.float32)
                length = np.sqrt(dxy[0, 0]**2 + dxy[0, 1]**2 + dxy[0, 2]**2)
                dxy /= length
                print(dxy, length)

    return nx_weights, uv_weights


@njit(cache=True)
def _evaluate_gaussian(sigma, x):
    # gaussian is not normalized
    return np.exp(-x**2/(2*sigma**2))
