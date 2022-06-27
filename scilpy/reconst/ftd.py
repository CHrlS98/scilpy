# -*- coding: utf-8 -*-
import numpy as np
import json

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
                 qb_mdf_th=-0.9, qb_mdf_merge_th=-0.9, batch_size=500,
                 qb_n_tracks_rel_th=0.05, forward_only=False,
                 sh_basis='descoteaux07', return_tracks=False):
        self.sphere = get_sphere('symmetric724')
        sh_order, full_basis = get_sh_order_and_fullness(fodf.shape[-1])
        self.min_cos_theta = np.cos(np.deg2rad(theta))

        self.fodf = fodf
        self.b_mat = sh_to_sf_matrix(self.sphere, sh_order, sh_basis,
                                     full_basis, return_inv=False)
        self.sf_max = np.array([np.max(sh.dot(self.b_mat), axis=-1)
                                for sh in fodf])
        self.n_batches = int(np.ceil(np.count_nonzero(seeds) / batch_size))
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
        self.return_tracks = return_tracks

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
                ftds[tuple(vox_pos.astype(int))] = []

                # fit each bundle
                for cluster_i in clusters_importance:
                    n_tracks_cluster_i = n_tracks_per_cluster[cluster_i]
                    if (n_tracks_cluster_i > n_tracks_abs_th):
                        cluster_tracks = vox_tracks[vox_clusters == cluster_i]
                        cluster_n_points =\
                            vox_n_points[vox_clusters == cluster_i]

                        # cluster tracks
                        cluster_tracks =\
                            [cluster_tracks[i, :cluster_n_points[i]]
                             for i in range(len(cluster_tracks))]

                        ftd =\
                            self._compute_ftd_for_bundle(cluster_tracks,
                                                         vox_pos.astype(float))
                        if self.return_tracks:
                            streamlines.extend(cluster_tracks)
                        ftds[tuple(vox_pos.astype(int))].append(ftd)

                # remove element if no valid clusters are created
                if len(ftds[tuple(vox_pos.astype(int))]) == 0:
                    ftds.pop(tuple(vox_pos.astype(int)))

        if self.return_tracks:
            return FTDFit(ftds), streamlines
        return FTDFit(ftds)

    def _compute_ftd_for_bundle(self, streamlines, vox_pos, reg=0.01):
        """
        Compute the fiber trajectory distribution for a bundle.
        """
        # Compute derivatives to use as target (dependent variable)
        T = np.concatenate([s[1:] - s[:-1] for s in streamlines], axis=0)
        T = T / np.linalg.norm(T, axis=-1, keepdims=True)

        # Compute independent variable
        phi_x =\
            np.concatenate([s[:-1] for s in streamlines], axis=0) - vox_pos

        # project to polynomial
        phi_x = project_to_polynomial(phi_x)  # (N, 10)

        # maximum likelihood
        W = np.linalg.inv(phi_x.dot(phi_x.T) + reg).dot(phi_x).dot(T)

        # we are supposed to get near-unit vectors. Is it the case?
        # TODO: Evaluate goodness of fit.

        return W.astype(np.float32)


class FTDFitDeluxe(object):
    """
    Deluxe FTD field container.
    """
    def __init__(self, dimensions):
        self.dims = dimensions
        self.ftd_dict = {"dimensions": self.dims,
                         "voxels": {}}

    def _hash_pos(self, position):
        pos_int = position.astype(int)
        hash_n_smash = pos_int[2] * self.dims[1] * self.dims[0]\
            + pos_int[1] * self.dims[0] + pos_int[0]
        return hash_n_smash

    def add_element(self, position, ftd, weight):
        pos_key = self._hash_pos(position)
        ftd_dict_voxels = self.ftd_dict["voxels"]
        if pos_key not in ftd_dict_voxels:
            ftd_dict_voxels[pos_key] = {}
        cluster_key = "cluster {}".format(len(ftd_dict_voxels[pos_key]))
        ftd_dict_voxels[pos_key][cluster_key] = {}
        ftd_dict_voxels[pos_key][cluster_key]["ftd"] = ftd.tolist()
        ftd_dict_voxels[pos_key][cluster_key]["weight"] = weight

    def to_json(self):
        return json.dumps(self.ftd_dict, indent=2)


class FTDFit(object):
    """
    Fiber trajectory distribution.

    Parameters
    ----------
    ftd_dict: dictionary
        Dictionary of fitted FTDs. Maps each nonzero voxel by its ID
        to a list of coefficients.
    """
    def __init__(self, ftd_dict):
        self.ftd_dict = ftd_dict

    @staticmethod
    def from_json(filename):
        # le json a des stringified tuples en keys
        # et ftds transformées en list
        f = open(filename, 'r')
        json_dict = json.load(f)
        ftd_dict = {}
        for k, v in json_dict.items():
            key = tuple(int(i) for i in k.replace('(', '')
                                         .replace(')', '')
                                         .split(', '))
            value = [np.asarray(nested_v) for nested_v in v]
            ftd_dict[key] = value
        return FTDFit(ftd_dict)

    def __getitem__(self, pos):
        """
        Get vector field at position `pos`.
        """
        pos_key = tuple(pos.astype(int))
        ftd_xyz = pos - pos.astype(int)
        polynomial = project_to_polynomial(ftd_xyz).reshape((1, 10))

        vec_f = []
        if pos_key in self.ftd_dict:
            for ftd in self.ftd_dict[pos_key]:
                vec_f.append(np.squeeze(polynomial.dot(ftd)))
        return vec_f

    def defined_voxels(self):
        return self.ftd_dict.keys()

    def save_to_json(self, filename):
        file = open(filename, mode='w')
        serializable_dict = {str(k): np.asarray(v).tolist()
                             for k, v in self.ftd_dict.items()}
        json.dump((serializable_dict), file, indent=2)


def project_to_polynomial(P):
    if P.ndim == 1:
        P = P[None, :]
    c = np.vstack([P[:, 0]**2,
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


class ClusterForFTD(object):
    def __init__(self, sft, interface_roi):
        # sft in voxel space with origin corner
        sft.to_vox()
        sft.to_corner()

        self.streamlines = sft.streamlines
        self.start_status = sft.data_per_streamline['start_status']
        self.end_status = sft.data_per_streamline['end_status']
        self.seeds = sft.data_per_streamline['seeds'] + 0.5
        self.interface_vox = np.argwhere(interface_roi)
        self.voxel_groups = []
        self._bundle_by_seeding_vox()

    def _bundle_by_seeding_vox(self):
        seed_vox = self.seeds.astype(np.int32)
        unique_vox = np.unique(seed_vox, axis=0)

        self.voxel_groups.clear()
        for uvox in unique_vox:
            vox_mask = np.all(seed_vox == uvox, axis=1)
            self.voxel_groups.append(np.nonzero(vox_mask)[0])

    def _filter_endpoint_rois(self, start_pos, end_pos):
        """
        Bundle-endpoint streamlines should be excluded if they don't
        have at least one valid endpoint.
        """
        start_indices = start_pos.astype(int)
        end_indices = end_pos.astype(int)

        # FIXME: Does not work properly!
        start_at_endpoint = np.array([idx in self.interface_vox
                                      for idx in start_indices]
                                     ).reshape((-1, 1))
        end_at_endpoint = np.array([idx in self.interface_vox
                                    for idx in end_indices]
                                   ).reshape((-1, 1))

        # If a strl starts and ends in endpoint roi it should not be discarded.
        both_ends_to_keep = np.logical_and(start_at_endpoint, end_at_endpoint)\
            .reshape((-1))

        # the other extremity of a strl in endpoint roi can't be invalid!
        start_to_keep = np.logical_and(start_at_endpoint, self.end_status == 0)
        end_to_keep = np.logical_and(end_at_endpoint, self.start_status == 0)

        to_keep = np.logical_or(start_to_keep, end_to_keep).reshape((-1))
        return np.logical_or(to_keep, both_ends_to_keep)

    def _filter_unexpected_termination(self):
        statuses = np.column_stack([self.start_status, self.end_status])
        valid = np.all(statuses == 0, axis=-1)
        return valid

    def filter_streamlines(self):
        start_pos = np.array([s[0] for s in self.streamlines])
        end_pos = np.array([s[-1] for s in self.streamlines])

        # 1. Streamlines with one endpoint in interface rois are only valid
        # if their second endpoint is valid.
        valid_endpoint_in_roi = self._filter_endpoint_rois(start_pos, end_pos)

        # 2. If streamline has two valid endpoints, it is valid
        valid_both_endpoints = self._filter_unexpected_termination()

        # 3. valid streamlines are the union of both sets
        all_valid = np.logical_or(valid_endpoint_in_roi, valid_both_endpoints)

        self.streamlines = self.streamlines[all_valid]
        self.seeds = self.seeds[all_valid]

        # 4. generate bundles based on seeding position
        self._bundle_by_seeding_vox()

    def cluster_gpu(self):
        strl_gpu = []
        strl_lengths = []
        group_lengths = []
        for group in self.voxel_groups:
            strl_group = self.streamlines[group]
            lengths = self.streamlines._lengths[group]
            nb_strl = len(lengths)

            strl_gpu.extend(np.concatenate(strl_group, axis=0))
            strl_lengths.extend(lengths)
            group_lengths.append(nb_strl)

        strl_offsets = np.append([0], np.cumsum(strl_lengths))
        group_offsets = np.append([0], np.cumsum(group_lengths))

        # TODO: Input resampled or compressed tractogram
        # Peut-être que mes streamlines devraient être resamplees
        # avant d'aller sur GPU

        # TODO: 1er bundling sur l'orientation par voxel
        # TODO: 2e bundling sur la distance entre les clusters, intervoxel

        # Return each streamlines cluster id

        return 0