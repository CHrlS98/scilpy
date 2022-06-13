# -*- coding: utf-8 -*-
import numpy as np
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager
from scilpy.reconst.utils import get_sh_order_and_fullness
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere


def compute_ftd_gpu(fodf, mask, n_seeds_per_vox=500,
                    step_size=0.5, theta=20.0,
                    min_nbr_points=10.0, max_nbr_points=20.0,
                    sh_basis='descoteaux07'):
    """
    Compute fiber trajectory distribution from FODF image.

    Parameters
    ----------
    fodf: array_like
        FODF field expressed as SH coefficients.
    mask: array_like
        Tracking mask.
    n_seeds_per_vox: int
        Maximum number of tracks per voxel.
    step_size: float, optional
        Step size for path integration in voxel space.
    theta: float, optional
        Maximum angle (degrees) between two consecutive streamlines.
    min_length: float, optional
        Minimum number of points of reconstructed paths.
    max_length: float, optional
        Maximum number of points of reconstructed paths.
    sh_basis: str, optional
        SH basis used for FODF representation.
    """
    # runs on GPU
    # method
    # 1. Foreach voxel inside the mask
    #     1.2 For i from 0 to max_tracks_per_vox
    #         1.2.1 Generate seed at random position inside voxel
    #         1.2.2 Track seed in both directions
    #         1.2.3 Cluster tracks using Quickbundle
    #     1.3 Foreach QB cluster
    #         1.3.1 Compute fiber trajectory distribution

    sh_order, full_basis = get_sh_order_and_fullness(fodf.shape[-1])
    sphere = get_sphere('symmetric724')
    vertices = sphere.vertices
    min_cos_theta = np.cos(np.deg2rad(theta))
    min_n_points = int(min_length / step_size) + 1
    max_n_points = int(max_length / step_size) + 1
    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis,
                            full_basis, return_inv=False)

    # position of voxels inside the mask
    # we flatten in order to feed as uint3 to the GPU.
    voxel_ids = np.argwhere(mask).astype(np.uint32).flatten()

    cl_kernel = CLKernel('compute_ftd', 'reconst', 'ftd.cl')
    # image dimensions
    cl_kernel.set_define('IM_X_DIM', fodf.shape[0])
    cl_kernel.set_define('IM_Y_DIM', fodf.shape[1])
    cl_kernel.set_define('IM_Z_DIM', fodf.shape[2])
    cl_kernel.set_define('IM_N_COEFFS', fodf.shape[3])
    cl_kernel.set_define('N_DIRS')
    cl_kernel.set_define('N_SEEDS_PER_VOX', f'{n_seeds_per_vox}')
    cl_kernel.set_define('MIN_COS_THETA', '{0:.6f}f'.format(min_cos_theta))
    cl_kernel.set_define('STEP_SIZE', '{0:.6f}f'.format(step_size))
    cl_kernel.set_define('MIN_LENGTH', '{0:.6f}f'.format(min_length))
    cl_kernel.set_define('MAX_LENGTH', '{0:.6f}f'.format(max_length))

    N_INPUTS = 3
    N_OUTPUTS = 1
    cl_manager = CLManager(cl_kernel, N_INPUTS, N_OUTPUTS)
    cl_manager.add_input_buffer(0, fodf, np.float32)
    cl_manager.add_input_buffer(1, voxel_ids, np.uint32)
    return 0
