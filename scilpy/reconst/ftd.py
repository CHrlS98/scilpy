# -*- coding: utf-8 -*-
import numpy as np
import itertools
from scilpy.gpuparallel.opencl_utils import CLKernel, CLManager
from scilpy.reconst.utils import get_sh_order_and_fullness
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere


class FTDFitter(object):
    """
    Fits the fiber trajectory distribution to a FODF field.

    Parameters
    ----------
    fodf : array_like
        FODF field.
    seeds : array_like
        Seed mask.
    mask : array_like
        Input mask.
    n_seeds_per_vox : int
        Number of seeds per voxel.
    step_size : float
        Integration step size in voxel space.
    theta : float
        Maximum curvature angle in degrees.
    min_nb_points : int
        Minimum number of points of sampled trajectories.
    max_nb_points : int
        Maximum number of points of sampled trajectories.
    qb_mdf_th : float
        Quickbundle minimum average direct-flip (MDF) distance threshold
        for clustering.
    qb_mdf_merge_th : float
        Quickbundle MDF distance threshold for merging bundles.
    qb_n_tracks_rel_th : float
        Quickbundle relative number of tracks threshold for fitting a FTD.
    forward_only : bool
        If True, only forward tracking is performed.
    sh_basis : str
        SH basis used for FODF representation.
    """
    def __init__(self, fodf, seeds, mask, n_seeds_per_vox,
                 step_size, theta, min_nb_points, max_nb_points,
                 qb_mdf_th=1.0, qb_mdf_merge_th=1.5, batch_size=1000,
                 qb_n_tracks_rel_th=0.05, forward_only=False,
                 sh_basis='descoteaux07'):
        self.sphere = get_sphere('symmetric724')
        sh_order, full_basis = get_sh_order_and_fullness(fodf.shape[-1])
        self.min_cos_theta = np.cos(np.deg2rad(theta))

        self.fodf = fodf
        self.b_mat = sh_to_sf_matrix(self.sphere, sh_order, sh_basis,
                                     full_basis, return_inv=False)
        sf_max = np.array([sh.dot(self.b_mat) for sh in fodf])
        self.sf_max = np.max(sf_max, axis=-1)
        self.n_batches = int(np.ceil(np.count_nonzero(seeds) // batch_size))
        self.seed_voxel_ids = np.array_split(np.argwhere(seeds),
                                             self.n_batches)
        self.mask = mask
        self.n_seeds_per_vox = n_seeds_per_vox
        self.step_size = step_size
        self.theta = theta
        self.min_nb_points = min_nb_points
        self.max_nb_points = max_nb_points
        self.qb_mdf_th = qb_mdf_th
        self.qb_mdf_merge_th = qb_mdf_merge_th
        self.qb_n_tracks_rel_th = qb_n_tracks_rel_th
        self.sh_basis = sh_basis
        self.forward_only = forward_only

    def fit(self):
        """
        Fits the fiber trajectory distribution to a FODF field.
        """
        # number of voxels and vertices
        nb_vertices = len(self.sphere.vertices)

        # instanciate opencl kernel
        cl_kernel = CLKernel('main', 'reconst', 'ftd.cl')

        # image dimensions
        cl_kernel.set_define('IM_X_DIM', self.fodf.shape[0])
        cl_kernel.set_define('IM_Y_DIM', self.fodf.shape[1])
        cl_kernel.set_define('IM_Z_DIM', self.fodf.shape[2])
        cl_kernel.set_define('IM_N_COEFFS', self.fodf.shape[3])

        # number of directions on the sphere
        cl_kernel.set_define('N_DIRS', f'{nb_vertices}')

        # number of seeds per voxel
        cl_kernel.set_define('N_SEEDS_PER_VOX', f'{self.n_seeds_per_vox}')

        # tracking parameters
        cl_kernel.set_define('STEP_SIZE', f'{self.step_size}f')
        cl_kernel.set_define('MIN_COS_THETA', '{0:.6f}f'
                             .format(self.min_cos_theta))
        cl_kernel.set_define('MIN_LENGTH', f'{self.min_nb_points}')
        cl_kernel.set_define('MAX_LENGTH', f'{self.max_nb_points}')
        cl_kernel.set_define('FORWARD_ONLY',
                             'true' if self.forward_only else 'false')

        # QB clustering parameters
        cl_kernel.set_define('QB_MDF_THRESHOLD', f'{self.qb_mdf_th}f')
        cl_kernel.set_define('QB_MDF_MERGE_THRESHOLD',
                             f'{self.qb_mdf_merge_th}f')

        N_INPUTS = 6
        N_OUTPUTS = 3
        cl_manager = CLManager(cl_kernel, N_INPUTS, N_OUTPUTS)

        # input buffers
        # we will flatten the sphere vertices to pass it as float4.
        vertices_gpu = np.column_stack((self.sphere.vertices,
                                        np.ones(nb_vertices)))\
            .astype(np.float32).flatten()

        cl_manager.add_input_buffer(1, self.fodf, np.float32)
        cl_manager.add_input_buffer(2, self.mask, np.float32)
        cl_manager.add_input_buffer(3, self.b_mat, np.float32)
        cl_manager.add_input_buffer(4, vertices_gpu, np.float32)
        cl_manager.add_input_buffer(5, self.sf_max, np.float32)

        streamlines = []
        cluster_ids = []
        ftds = {}
        for batch_i, batch_vox_ids in enumerate(self.seed_voxel_ids):
            nb_voxels = len(batch_vox_ids)
            print("batch [{}/{}]: {} voxels"
                  .format(batch_i + 1, self.n_batches, nb_voxels))
            # update voxel ids for batch
            # we flatten in order to feed as float4 to the GPU.
            voxel_ids_gpu = np.column_stack((batch_vox_ids,
                                             np.ones(len(batch_vox_ids))))\
                .astype(np.float32).flatten()
            cl_manager.add_input_buffer(0, voxel_ids_gpu, np.float32)

            # update output buffers
            cl_manager.add_output_buffer(0, (len(batch_vox_ids)
                                             * self.n_seeds_per_vox,
                                             self.max_nb_points, 3),
                                         dtype=np.float32)
            cl_manager.add_output_buffer(1, (len(batch_vox_ids)
                                             * self.n_seeds_per_vox, 1),
                                         dtype=np.uint32)
            cl_manager.add_output_buffer(2, (len(batch_vox_ids)
                                             * self.n_seeds_per_vox, 1),
                                         dtype=np.int32)

            # run the kernel
            tracks, n_points, clusters = cl_manager.run((len(batch_vox_ids),))

            # reshape the outputs
            n_points = n_points.reshape((nb_voxels, self.n_seeds_per_vox))
            clusters = clusters.reshape((nb_voxels, self.n_seeds_per_vox))
            tracks = tracks.reshape((nb_voxels, self.n_seeds_per_vox, -1, 3))

            for vox_i, vox_pos in enumerate(batch_vox_ids):
                # for each voxel, extract the streamlines
                vox_tracks = tracks[vox_i, :, :, :]
                vox_n_points = n_points[vox_i, :]
                vox_clusters = clusters[vox_i, :]
                n_tracks_per_cluster =\
                    np.bincount(vox_clusters[vox_clusters != -1])
                n_tracks_for_voxel = np.sum(n_tracks_per_cluster)
                n_tracks_abs_th = n_tracks_for_voxel * self.qb_n_tracks_rel_th

                clusters_importance = np.argsort(n_tracks_per_cluster)[::-1]

                # initialize the FTD of current voxel
                ftds[tuple(vox_pos.astype(int))] = {}

                # fit each bundle
                for cluster_key, cluster_i in enumerate(clusters_importance):
                    n_tracks_cluster_i = n_tracks_per_cluster[cluster_i]
                    if (n_tracks_cluster_i > n_tracks_abs_th):
                        cluster_tracks = vox_tracks[vox_clusters == cluster_i]
                        cluster_n_points = vox_n_points[vox_clusters == cluster_i]

                        # cluster tracks
                        cluster_tracks = [cluster_tracks[i, :cluster_n_points[i]]
                                          for i in range(len(cluster_tracks))]

                        ftd = self.compute_ftd_for_bundle(cluster_tracks,
                                                          vox_pos.astype(float))
                        streamlines.extend(cluster_tracks)
                        cluster_ids.extend([cluster_i]*len(cluster_tracks))
                        ftds[tuple(vox_pos.astype(int))][cluster_key] = ftd

        return ftds, streamlines, cluster_ids

    def compute_ftd_for_bundle(self, streamlines, vox_pos):
        """
        Compute the fiber trajectory distribution for a bundle.
        """
        # 2nd compute derivatives
        V = np.concatenate([s[1:] - s[:-1] for s in streamlines], axis=0)
        V = V / np.linalg.norm(V, axis=-1, keepdims=True)

        # 3rd compute polynomial representation of the streamlines
        C = np.concatenate([s[:-1] for s in streamlines], axis=0)
        centroid = np.mean(C, axis=0)

        # center points around voxel position
        C = C - vox_pos
        min_bounds, max_bounds = np.min(C, axis=0), np.max(C, axis=0)

        C = _project_to_polynomial(C)

        # 4th Solve the least-squares problem to find the FTD
        FTD = np.linalg.lstsq(C, V, rcond=None)[0]

        return FTD


def compute_ftd_gpu(fodf, seeds, mask, n_seeds_per_vox,
                    step_size, theta, min_nb_points,
                    max_nb_points, qb_mdf_th=1.2,
                    qb_mdf_merge_th=2.0,
                    qb_n_tracks_rel_th=0.1,
                    sh_basis='descoteaux07'):
    """
    Compute fiber trajectory distribution from FODF image.

    Parameters
    ----------
    fodf: array_like
        FODF field expressed as SH coefficients.
    seeds: array_like
        Seeding mask.
    mask: array_like
        Tracking mask.
    n_seeds_per_vox: int
        Maximum number of tracks per voxel.
    step_size: float, optional
        Step size for path integration in voxel space.
    theta: float, optional
        Maximum angle (degrees) between two consecutive streamlines.
    min_nb_points: float, optional
        Minimum number of points of reconstructed paths.
    max_nb_points: float, optional
        Maximum number of points of reconstructed paths.
    sh_basis: str, optional
        SH basis used for FODF representation.
    """
    sh_order, full_basis = get_sh_order_and_fullness(fodf.shape[-1])
    sphere = get_sphere('symmetric724')
    nb_vertices = len(sphere.vertices)
    min_cos_theta = np.cos(np.deg2rad(theta))
    b_mat = sh_to_sf_matrix(sphere, sh_order, sh_basis,
                            full_basis, return_inv=False)
    sf_max = np.array([sh.dot(b_mat) for sh in fodf])
    sf_max = np.max(sf_max, axis=-1)

    # position of voxels inside the mask
    voxel_ids = np.argwhere(seeds).astype(np.float32)
    nb_voxels = len(voxel_ids)

    # we flatten in order to feed as float4 to the GPU.
    voxel_ids = np.column_stack((voxel_ids, np.ones(nb_voxels)))\
        .astype(np.float32).flatten()

    # we will flatten the sphere vertices for the same reason.
    vertices = np.column_stack((sphere.vertices, np.ones(nb_vertices)))\
        .astype(np.float32).flatten()

    cl_kernel = CLKernel('main', 'reconst', 'ftd.cl')

    # image dimensions
    cl_kernel.set_define('IM_X_DIM', fodf.shape[0])
    cl_kernel.set_define('IM_Y_DIM', fodf.shape[1])
    cl_kernel.set_define('IM_Z_DIM', fodf.shape[2])
    cl_kernel.set_define('IM_N_COEFFS', fodf.shape[3])

    # number of directions on the sphere
    cl_kernel.set_define('N_DIRS', f'{nb_vertices}')

    # number of voxels to seed and number of seeds per voxel
    cl_kernel.set_define('N_VOX', f'{nb_voxels}')  # TODO: Batch
    cl_kernel.set_define('N_SEEDS_PER_VOX', f'{n_seeds_per_vox}')

    # tracking parameters
    cl_kernel.set_define('STEP_SIZE', f'{step_size}f')
    cl_kernel.set_define('MIN_COS_THETA', '{0:.6f}f'.format(min_cos_theta))
    cl_kernel.set_define('MIN_LENGTH', f'{min_nb_points}')
    cl_kernel.set_define('MAX_LENGTH', f'{max_nb_points}')
    cl_kernel.set_define('QB_MDF_THRESHOLD', f'{qb_mdf_th}f')
    cl_kernel.set_define('QB_MDF_MERGE_THRESHOLD', f'{qb_mdf_merge_th}f')

    qb_n_tracks_abs_th = int(n_seeds_per_vox*qb_n_tracks_rel_th)
    cl_kernel.set_define('QB_N_TRACKS_THRESHOLD', f'{qb_n_tracks_abs_th}')
    cl_kernel.set_define('FORWARD_ONLY', 'false')

    N_INPUTS = 6
    N_OUTPUTS = 3
    cl_manager = CLManager(cl_kernel, N_INPUTS, N_OUTPUTS)
    cl_manager.add_input_buffer(0, voxel_ids, np.float32)  # TODO: Batch
    cl_manager.add_input_buffer(1, fodf, np.float32)
    cl_manager.add_input_buffer(2, mask, np.float32)
    cl_manager.add_input_buffer(3, b_mat, np.float32)
    cl_manager.add_input_buffer(4, vertices, np.float32)
    cl_manager.add_input_buffer(5, sf_max, np.float32)

    # Test that tracking works properly
    cl_manager.add_output_buffer(0, (nb_voxels*n_seeds_per_vox,
                                     max_nb_points, 3),
                                 dtype=np.float32)  # TODO: Batch
    cl_manager.add_output_buffer(1, (nb_voxels*n_seeds_per_vox, 1),
                                 dtype=np.uint32)  # TODO: Batch
    cl_manager.add_output_buffer(2, (nb_voxels*n_seeds_per_vox, 1),
                                 dtype=np.int32)  # TODO: Batch
    tracks, n_points, clusters = cl_manager.run((nb_voxels, 1, 1))

    # unpack valid tracks
    streamlines = []
    cluster_ids = []
    ftds = {}

    n_points = n_points.reshape((nb_voxels, n_seeds_per_vox))
    clusters = clusters.reshape((nb_voxels, n_seeds_per_vox))
    tracks = tracks.reshape((nb_voxels, n_seeds_per_vox, -1, 3))
    for vox_i, vox_pos in enumerate(voxel_ids.reshape((nb_voxels, 4))[:, :3]):
        # for each voxel, extract the streamlines
        vox_tracks = tracks[vox_i, :, :, :]
        vox_n_points = n_points[vox_i, :]
        vox_clusters = clusters[vox_i, :]
        n_tracks_per_cluster = np.bincount(vox_clusters[vox_clusters != -1])
        clusters_importance = np.argsort(n_tracks_per_cluster)[::-1]

        # initialize the FTD of current voxel
        ftds[tuple(vox_pos.astype(int))] = {}
        for cluster_key, cluster_i in enumerate(clusters_importance):
            if n_tracks_per_cluster[cluster_i] > qb_n_tracks_abs_th:
                cluster_tracks = vox_tracks[vox_clusters == cluster_i]
                cluster_n_points = vox_n_points[vox_clusters == cluster_i]

                # cluster tracks
                cluster_tracks = [cluster_tracks[i, :cluster_n_points[i], :]
                                  for i in range(len(cluster_tracks))]

                ftd = compute_ftd_for_bundle(cluster_tracks, vox_pos)
                streamlines.extend(cluster_tracks)
                cluster_ids.extend([cluster_i]*len(cluster_tracks))
                ftds[tuple(vox_pos.astype(int))][cluster_key] = ftd

    return ftds, streamlines, cluster_ids


def _project_to_polynomial(P):
    if P.ndim == 1:
        P = P[None, :]
    c = np.column_stack([P[:, 0]**2,
                         P[:, 1]**2,
                         P[:, 2]**2,
                         P[:, 0]*P[:, 1],
                         P[:, 0]*P[:, 2],
                         P[:, 1]*P[:, 2],
                         P[:, 0],
                         P[:, 1],
                         P[:, 2],
                         np.ones(len(P))])
    return c

