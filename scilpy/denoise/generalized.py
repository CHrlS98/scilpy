# -*- coding: utf-8 -*-
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf_matrix
import numpy as np
from numba import njit, prange
from scilpy.gpuparallel.opencl_utils import CLManager, CLKernel


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

        # shape (n_dirs, n_dirs)
        self.uv_filter = _build_uv_filter(self.sphere.vertices,
                                          self.sigma_angle)
        # shape (hx, hy, hz, n_dirs)
        self.nx_filter = _build_nx_filter(self.sphere.vertices, sigma_spatial,
                                          sigma_align)

        self.B = sh_to_sf_matrix(self.sphere, self.sh_order,
                                 self.basis, self.full_basis,
                                 return_inv=False)
        _, self.B_inv = sh_to_sf_matrix(self.sphere, self.sh_order,
                                        self.basis, True, return_inv=True)

        # initialize gpu
        self.cl_kernel = None
        self.cl_manager = None
        self._prepare_gpu()

    def _prepare_gpu(self):
        self.cl_kernel = CLKernel('filter', 'denoise',
                                  'generalized_filter.cl')

        self.cl_kernel.set_define("WIN_WIDTH", self.nx_filter.shape[0])
        self.cl_kernel.set_define("SIGMA_RANGE",
                                  "{}f".format(self.sigma_range))
        self.cl_kernel.set_define("N_DIRS", len(self.sphere.vertices))
        self.cl_manager = CLManager(self.cl_kernel)

    def __call__(self, sh_data, patch_size=None):
        # organiser une mecanique de patches
        # copy to gpu
        sf_data = np.dot(sh_data, self.B)
        output_shape = sf_data.shape
        # pad input
        win_hwidth = self.nx_filter.shape[0] // 2
        sf_data = np.pad(sf_data, ((win_hwidth, win_hwidth),
                                   (win_hwidth, win_hwidth),
                                   (win_hwidth, win_hwidth),
                                   (0, 0)))
        print(sf_data.shape)
        self.cl_manager.add_input_buffer("sf_data", sf_data)
        self.cl_manager.add_input_buffer("nx_filter", self.nx_filter)
        print(self.nx_filter.shape)
        print(self.uv_filter[0])
        self.cl_manager.add_input_buffer("uv_filter", self.uv_filter)

        # init output buffer
        print(output_shape)
        self.cl_manager.add_output_buffer('out_sf', output_shape)

        out_sf = self.cl_manager.run(output_shape[:-1])[0]
        print(out_sf.shape)

        # out_sf = _correlate_sh(sf_data, self.nx_filter,
        #                        self.uv_filter, self.sigma_range)

        print(np.sum(sf_data), np.sum(out_sf))
        out_sh = np.dot(out_sf, self.B_inv)
        return out_sh


@njit(cache=True)
def _build_uv_filter(directions, sigma_angle):
    directions = np.ascontiguousarray(directions.astype(np.float32))
    uv_weights = np.zeros((len(directions), len(directions)), dtype=np.float32)

    # 1. precompute weights on angle
    # c'est toujours les mêmes peu importe le voxel en question
    for u_i, u in enumerate(directions):
        uvec = np.reshape(np.ascontiguousarray(u), (1, 3))
        weights = np.arccos(np.clip(np.dot(uvec, directions.T), -1.0, 1.0))
        weights = _evaluate_gaussian(sigma_angle, weights)
        weights /= np.sum(weights)
        uv_weights[u_i] = weights  # each line sums to 1.

    return uv_weights


@njit(cache=True)
def _build_nx_filter(directions, sigma_spatial, sigma_align):
    directions = np.ascontiguousarray(directions.astype(np.float32))

    half_width = int(round(3 * sigma_spatial))
    nx_weights = np.zeros((2*half_width+1, 2*half_width+1,
                           2*half_width+1, len(directions)),
                          dtype=np.float32)

    for i in range(-half_width, half_width+1):
        for j in range(-half_width, half_width+1):
            for k in range(-half_width, half_width+1):
                dxy = np.array([[i, j, k]], dtype=np.float32)
                len_xy = np.sqrt(dxy[0, 0]**2 + dxy[0, 1]**2 + dxy[0, 2]**2)

                # the length controls spatial weight
                w_spatial = _evaluate_gaussian(sigma_spatial, len_xy)

                # the direction controls the align weight
                if i == j == k == 0:
                    w_align = np.zeros((1, len(directions)), dtype=np.float32)
                else:
                    dxy /= len_xy
                    w_align = np.arccos(np.clip(np.dot(dxy, directions.T),
                                                -1.0, 1.0))  # 1, N
                w_align = _evaluate_gaussian(sigma_align, w_align)
                nx_weights[half_width + i,
                           half_width + j,
                           half_width + k] = w_align * w_spatial

    # sur chaque u, le filtre doit sommer à 1
    for ui in range(len(directions)):
        w_sum = np.sum(nx_weights[..., ui])
        nx_weights /= w_sum

    return nx_weights


@njit(cache=True)
def _evaluate_gaussian(sigma, x):
    # gaussian is not normalized
    return np.exp(-x**2/(2*sigma**2))


@njit(cache=True)
def _correlate_sh(sf_data, nx_filter, uv_filter, sigma_range):
    """
    nx_filter: ndarray(hx, hy, hz, n_dirs)
    uv_filter: ndarray(n_dirs, n_dirs)
    """
    sf_shape = sf_data.shape
    halfwidth_f = nx_filter.shape[0] // 2
    out_sf = np.zeros_like(sf_data, dtype=np.float32)
    for i in range(sf_shape[0]):
        range_i = (i - halfwidth_f, i + halfwidth_f + 1)
        range_hi = _get_win_boundary(range_i, sf_shape[0])
        range_i = (max(range_i[0], 0), min(sf_shape[0], range_i[1]))
        for j in range(sf_shape[1]):
            range_j = (j - halfwidth_f, j + halfwidth_f + 1)
            range_hj = _get_win_boundary(range_j, sf_shape[1])
            range_j = (max(range_j[0], 0), min(sf_shape[1], range_j[1]))
            for k in range(sf_shape[2]):
                range_k = (k - halfwidth_f, k + halfwidth_f + 1)
                range_hk = _get_win_boundary(range_k, sf_shape[2])
                range_k = (max(range_k[0], 0), min(sf_shape[2], range_k[1]))

                # extract window from image and convert to SF
                win_sf = np.zeros(nx_filter.shape, dtype=np.float32)
                win_sf[range_hi[0]:range_hi[1], range_hj[0]:range_hj[1],
                       range_hk[0]:range_hk[1]] = (
                    sf_data[range_i[0]:range_i[1], range_j[0]:range_j[1],
                            range_k[0]:range_k[1]])

                # if all is 0 we can actually skip it, return is 0
                if win_sf.max() > 0:
                    print(i, j, k)
                    # apply weigths
                    for ui in range(sf_shape[-1]):
                        # scalar value
                        psi_ui = win_sf[halfwidth_f, halfwidth_f, halfwidth_f, ui]
                        # hi, hy, hz, n_dirs
                        range_weights = _evaluate_gaussian(sigma_range,
                                                           np.abs(win_sf - psi_ui))
                        # 1, n_dirs
                        align_weights = uv_filter[ui]
                        # hi, hj, hk
                        spat_ori_weigths = nx_filter[..., ui:ui+1]
                        # combine weights
                        weights = range_weights * align_weights\
                            * spat_ori_weigths
                        # resulting combined weights *must sum to 1*
                        weights /= np.sum(weights)
                        psi_ui_tilde = np.sum(weights * win_sf)
                        out_sf[i, j, k, ui] = psi_ui_tilde

    return out_sf


@njit(cache=True)
def _get_win_boundary(win_bounds_in_im, im_max_dim):
    h_start = -win_bounds_in_im[0] if win_bounds_in_im[0] < 0 else 0
    h_end = (im_max_dim - win_bounds_in_im[0]
             if win_bounds_in_im[1] > im_max_dim
             else win_bounds_in_im[1] - win_bounds_in_im[0])
    return (h_start, h_end)


@njit(cache=True)
def _apply_filter(win_sf, nx_weights, uv_weights, sigma_range):
    halfwidth_f = nx_filter.shape[0] // 2
    psi_x = win_sf[halfwidth_f, halfwidth_f, halfwidth_f]
