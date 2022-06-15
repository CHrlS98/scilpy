# -*- coding: utf-8 -*-
import numpy as np
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager
from scilpy.reconst.utils import get_sh_order_and_fullness
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere


def compute_ftd_gpu(fodf, seeds, mask, n_seeds_per_vox,
                    step_size, theta, min_nb_points,
                    max_nb_points, sh_basis='descoteaux07'):
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
    nb_vertices = len(sphere.vertices)
    min_cos_theta = np.cos(np.deg2rad(theta))
    B_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis,
                            full_basis, return_inv=False)

    # position of voxels inside the mask
    voxel_ids = np.argwhere(seeds).astype(np.float32)
    nb_voxels = len(voxel_ids)
    # we flatten in order to feed as float3 to the GPU.
    voxel_ids = np.column_stack((voxel_ids, np.ones(nb_voxels)))\
        .astype(np.float32).flatten()

    # we will flatten the sphere vertices for the same reason.
    # float3 on the GPU.
    vertices = np.column_stack((sphere.vertices, np.ones(nb_vertices)))\
        .astype(np.float32).flatten()

    cl_kernel = CLKernel('main', 'reconst', 'ftd.cl')
    # image dimensions
    cl_kernel.set_define('IM_X_DIM', fodf.shape[0])
    cl_kernel.set_define('IM_Y_DIM', fodf.shape[1])
    cl_kernel.set_define('IM_Z_DIM', fodf.shape[2])
    cl_kernel.set_define('IM_N_COEFFS', fodf.shape[3])
    cl_kernel.set_define('N_DIRS', f'{nb_vertices}')
    cl_kernel.set_define('N_VOX', f'{nb_voxels}')
    cl_kernel.set_define('N_SEEDS_PER_VOX', f'{n_seeds_per_vox}')
    cl_kernel.set_define('MIN_COS_THETA', '{0:.6f}f'.format(min_cos_theta))
    cl_kernel.set_define('STEP_SIZE', f'{step_size}f')
    cl_kernel.set_define('MIN_LENGTH', f'{min_nb_points}')
    cl_kernel.set_define('MAX_LENGTH', f'{max_nb_points}')
    cl_kernel.set_define('FORWARD_ONLY', 'true')

    N_INPUTS = 5
    N_OUTPUTS = 2
    cl_manager = CLManager(cl_kernel, N_INPUTS, N_OUTPUTS)
    cl_manager.add_input_buffer(0, voxel_ids, np.float32)
    cl_manager.add_input_buffer(1, fodf, np.float32)
    cl_manager.add_input_buffer(2, mask, np.float32)
    cl_manager.add_input_buffer(3, B_mat, np.float32)
    cl_manager.add_input_buffer(4, vertices, np.float32)

    # output buffer 0, at most 5 FTD matrices of shape
    # 3 x 10 per voxel in voxel_ids
    # cl_manager.add_output_buffer(0, (nb_voxels, 3, 10), np.float32)
    # output buffer 1, one integer between 0 and 5 per voxel
    # in voxel_ids
    # cl_manager.add_output_buffer(1, (nb_voxels, 1), np.uint32)

    # Test that tracking works properly
    cl_manager.add_output_buffer(0, (nb_voxels*n_seeds_per_vox,
                                     max_nb_points, 3),
                                 dtype=np.float32)  # out tracks
    cl_manager.add_output_buffer(1, (nb_voxels*n_seeds_per_vox, 1),
                                 dtype=np.uint32)  # nb tracks
    tracks, n_points = cl_manager.run((nb_voxels, 1, 1))

    # unpack valid tracks
    streamlines = []
    n_points = n_points.flatten()
    for i in range(nb_voxels*n_seeds_per_vox):
        if n_points[i] > min_nb_points:
            streamlines.append(tracks[i, :n_points[i]])

    return streamlines
