# -*- coding: utf-8 -*-

import logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from dipy.reconst.shm import (sh_to_sf, sf_to_sh,
                              sph_harm_full_ind_list)
from dipy.core.sphere import HemiSphere, Sphere, hemi_icosahedron
from dipy.core.ndindex import ndindex
from dipy.direction.peaks import peak_directions


def ncoef_from_order(sh_order, sh_basis):
    """
    Get the number of SH coefficients up to a given order
    for a given basis

    Parameters
    ==========
    sh_order: int
        The maximum order of the spherical harmonics basis
    sh_basis: str
        Name of the spherical harmonics basis
    """
    if '_full' in sh_basis:
        return (sh_order + 1)**2
    return 1/2 * (sh_order + 1) * (sh_order + 2)


class AFiberOrientationDistribution(object):
    def __init__(self, fodf, affine, sh_basis, sh_order):
        """
        Build the AFiberOrientationDistribution object

        Parameters
        ==========
        fodf: numpy array
            array containing the fodf
        affine: numpy array
            voxel to world 4x4 transformation matrix
        sh_basis: str
            SH basis of the fodf data. One of 'descoteaux07',
            'descoteaux07_full', 'tournier07', 'tournier07_full'
        sh_order: int
            The maximum order of the SH basis
        """
        self.fodf = fodf
        self.mask = np.linalg.norm(self.fodf, axis=-1) > 0
        self.affine = affine
        self.sh_basis = sh_basis
        self.sh_order = sh_order

    def get_affine(self):
        return self.affine

    def average(self, sphere, sh_order=8, sh_basis='descoteaux07_full',
                dot_sharpness=1.0, sigma=1.0, batch_size=10, mask=False):
        """
        Average the FODF

        Parameters
        ==========
        sphere: Sphere
            Sphere to use to project the spherical functions
        sh_order: int
            Maximum order of the SH series (default: 8)
        sh_basis: str
            SH basis of the averaged data. One of {'descoteaux07',
            'descoteaux07_full', 'tournier07', 'tournier07_full'}
            (default: 'descoteaux07_full')
        dot_sharpness: float
            Exponent of the dot product. When set to 0.0, directions
            are not weighted by the dot product (default: 1.0)
        sigma: float
            Variance of the gaussian (default: 1.0)
        batch_size: int
            Number of volume slices processed at a same time. The
            last batch to be processed can be of a size up to
            (batch_size * 2.0 - 1) (default: 10)
        mask: bool
            Remove FODF in voxels where there were initially no FODF
            (default: False)
        """
        # Convert to spherical function
        # TODO: Use B matrix for more efficient conversion
        sf = np.array([sh_to_sf(i, sphere, self.sh_order, self.sh_basis)
                       for i in self.fodf],
                      dtype='float32')

        # Initialize array for mean SF with current voxel value
        mean_sf = np.copy(sf)

        # Zero pad sf data
        sf = np.pad(sf, ((1, 1), (1, 1), (1, 1), (0, 0)),
                    mode='constant', constant_values=0.0)

        # Prepare batch
        batch_indices = self._get_batches_indices(batch_size)
        w_by_dir, norm_w = self._get_weights(sphere, dot_sharpness, sigma)

        # Compute average in batches
        for index in batch_indices:
            batch = sf[index]
            dim = tuple(np.array(batch.shape[:-1]) - np.array([2, 2, 2]))

            for w in w_by_dir:
                direction = np.array([w])
                weight = w_by_dir[w]

                i, j, k = int(w[0] + 1), int(w[1] + 1), int(w[2] + 1)
                mean_sf[index[0]:index[0] + dim[0]] +=\
                    np.multiply(batch[i:dim[0]+i, j:dim[1]+j, k:dim[2]+k],
                                weight)

            mean_sf[index[0]:index[0] + dim[0]] =\
                np.multiply(
                    mean_sf[index[0]:index[0] + dim[0]],
                    1.0 / (norm_w + 1.0))

        print('fodf/avfodf energy: ', np.sum(sf) / np.sum(mean_sf))

        self.fodf = np.zeros(
            np.append(
                self.fodf.shape[:-1],
                [ncoef_from_order(sh_order, sh_basis)]),
            dtype='float32')

        if mask:
            self.fodf[self.mask] =\
                np.array([sf_to_sh(i, sphere, sh_order, sh_basis)
                          for i in mean_sf])[self.mask]
        else:
            self.fodf =\
                np.array([sf_to_sh(i, sphere, sh_order, sh_basis)
                          for i in mean_sf])

        self.sh_basis = sh_basis
        self.sh_order = sh_order

    def save_to_file(self, filename):
        """
        Save the FODF to file under given filename

        Parameters
        ==========
        filename: str
            name of the file to save with extension
        """
        image = nib.Nifti1Image(self.fodf.astype(np.float32), self.affine)
        image.to_filename(filename)

    def compute_odd_on_full_coeffs_ratio(self):
        """
        Measure the asymmetry of the FODF per voxel.

        Returns
        =======
        asym_measure: numpy array
            array containing the asymmetry measure of each voxel

        Note
        ====
        The asymmetry measure corresponds to the ratio of the norm of odd
        order SH coefficients on the norm of full order SH order coefficients
        """
        if '_full' not in self.sh_basis:
            logging.error('Can\'t compute asymmetry measure ' +
                          'from symmetric SH basis')
            import sys
            sys.exit('Aborting program')

        _, l_list = sph_harm_full_ind_list(self.sh_order)
        odd_order_coeffs = self.fodf[..., l_list % 2 == 1]
        odd_order_norms = np.linalg.norm(odd_order_coeffs, axis=-1)
        full_norms = np.linalg.norm(self.fodf, axis=-1)

        asymmetry_measure = np.zeros_like(full_norms)
        mask = full_norms > 0

        asymmetry_measure[mask] = odd_order_norms[mask] / full_norms[mask]

        return asymmetry_measure

    def compute_mean_antipodal_distance(self, lod=2):
        """
        Compute the normalized distance between a vertex and its antipod
        """
        hemisphere = hemi_icosahedron.subdivide(lod)
        n_pts_per_hemisphere = hemisphere.vertices.shape[0]
        sphere =\
            Sphere(xyz=np.vstack((hemisphere.vertices, -hemisphere.vertices)))
        verts = np.arange(n_pts_per_hemisphere)
        inv_verts = np.arange(n_pts_per_hemisphere, sphere.vertices.shape[0])

        sf = np.array([sh_to_sf(fodf, sphere, self.sh_order, self.sh_basis)
                       for fodf in self.fodf])
        sf = np.array([sf[..., verts], sf[..., inv_verts]])
        sf[sf < 0] = 0

        d = np.abs(sf[0] - sf[1])
        max_sf = np.max(sf, axis=0)
        d[np.nonzero(max_sf)] =\
            d[np.nonzero(max_sf)] / max_sf[np.nonzero(max_sf)]

        return np.sum(d, axis=-1) / n_pts_per_hemisphere

    def clean_false_pos(self, epsilon):
        """
        Remove false positives by forcing to zero values smaller
        than epsilon and recompute mask
        """
        self.fodf[self.fodf[..., 0] < epsilon] = 0.0
        self.mask = np.linalg.norm(self.fodf, axis=-1) > 0

    def extract_peaks(self, sphere, npeaks=10, a_threshold=0.0):
        """
        Extract peaks on FODF without any asumption of symmetry

        Parameters
        ----------
        sphere: Sphere
            sphere to use for peak extraction
        npeaks: int
            max number of peaks to extract per odf
        a_threshold: float
            absolute threshold. FODF directions under this threshold
            won't be considered in peak extraction

        Returns
        -------
        peaks: APeaks
            APeaks object containing extracted peaks
        """
        # TODO: Use B matrix for more efficient conversion
        sf = np.array([sh_to_sf(i, sphere, self.sh_order, self.sh_basis)
                       for i in self.fodf])
        sf[sf < a_threshold] = 0.

        peaks_dirs = np.zeros(list(sf.shape[0:3]) + [npeaks, 3])
        for index in ndindex(sf.shape[:-1]):
            directions, _, _ =\
                peak_directions(sf[index], sphere, is_symmetric=False)
            n = min(npeaks, directions.shape[0])
            peaks_dirs[index][:n] = directions[:n]

        return APeaks(peaks_dirs, self.affine)

    def _get_batches_indices(self, batch_size):
        """
        Get the index of slices of data set along the first axis

        Parameters
        ==========
        batch_size: int
            Number of slices per batch

        Returns
        =======
        split_indices: list
            List of indices for each batch
        """
        nb_slices = self.fodf.shape[0]
        pad_width = 2

        number_of_batches = int(nb_slices / batch_size + 1.0)
        indices = np.arange(number_of_batches * batch_size + pad_width)
        split_indices = []
        for batch_id in range(number_of_batches):
            start = batch_size * batch_id
            split_indices.append(indices[start:start + batch_size + pad_width])

        split_indices[-1] =\
            split_indices[-1][split_indices[-1] < nb_slices + pad_width]

        return split_indices

    def _get_directions_to_voxels(self):
        """
        Get the vectors to neighboor voxels

        Returns
        =======
        directions: array 26x3
            array of directions from center of voxel to neighboors
        """
        # center directions around current voxel
        directions = np.transpose(np.indices((3, 3, 3)) - np.ones((3, 3, 3)))
        directions = np.reshape(directions, (27, 3))

        # remove direction (0, 0, 0)
        directions = np.delete(directions, 13, 0)

        return directions

    def _get_weights(self, sphere, dot_sharpness, sigma):
        """
        Get neighbors weight in respect to the direction to a voxel

        Parameters
        ----------
        sphere: Sphere
            sphere used for SF reconstruction
        dot_sharpness: float
            dot product exponent
        sigma: float
            variance of the gaussian used for weighting neighbors

        Returns
        -------
        weights: dictionary
            vertices weights in respect to directions
        norm: array
            per vertex norm of weights
        """
        directions = self._get_directions_to_voxels()
        dir_norms = np.linalg.norm(directions, axis=-1, keepdims=True)
        normalized_dir = directions/dir_norms

        g_weights = np.exp(-dir_norms**2 / (2 * sigma**2))
        d_weights = np.dot(sphere.vertices, normalized_dir.T)
        d_weights = np.where(d_weights > 0.0, d_weights**dot_sharpness, 0.0)

        weights = np.multiply(d_weights, g_weights.T)
        norm = np.sum(weights, axis=-1)

        dir_keys = list(map(tuple, directions))
        weights_by_dir = dict(zip(dir_keys, list(weights.T)))

        return weights_by_dir, norm

    def _get_ratio_asym_on_total_voxels(self, asym_measure):
        """
        Get the proportion of asymmetric voxels in the data set for
        different asymmetry thresholds

        Parameters
        ----------
        asym_measure: numpy array
            array containing voxel asymmetry measure (in range [0, 1])

        Returns
        -------
        ratios : numpy array(2, N)
            pairs of threshold-ratio for equally spaced threshold values
        """
        asym_thresholds = np.arange(0.0, 1.0, 0.001)
        asym_ratios = np.array(
            [np.count_nonzero(asym_measure[self.mask] > dat)
             for dat in asym_thresholds])
        asym_ratios = asym_ratios / asym_measure[self.mask].size

        return np.array([asym_thresholds, asym_ratios])


class APeaks(object):
    def __init__(self, data, affine):
        """
        Build the APeaks object representing peaks
        extracted from asymmetric FODF
        """
        self.peaks = data
        self.affine = affine
        self.npeaks = data.shape[-2]
        self.peaks_count =\
            np.cumsum(np.linalg.norm(self.peaks, axis=-1) > 0.,
                      axis=-1)[..., -1]
        self.labels = None

    def get_peaks(self):
        return self.peaks

    def get_affine(self):
        return self.affine

    def get_peaks_count(self):
        return self.peaks_count

    def save_to_file(self, filename):
        """
        Save peaks to file 'filename'
        """
        nib.save(nib.Nifti1Image(self.peaks, self.affine), filename)


class AFODMetricsPopper(object):
    def __init__(self, aFOD=None, aPeaks=None, ofr=None, mad=None, nufo=None):
        """
        Build the AFODMetricsPopper object

        The AFODMetricsPopper is a class containing metrics with
        respect to asymmetric fiber orientation distribution functions

        Parameters
        ----------
        aFOD: AFOD
        """
        self.aFOD = aFOD
        self.aPeaks = aPeaks
        self.odd_on_full_coeffs_ratio = ofr
        self.mean_antipodal_distance = mad
        self.nufo = nufo
        self.labels = None

    def save_odf_on_full_coeffs_ratio(self, filename):
        """
        Save odd on full coefficients ratios to file ``filename``

        Parameters
        ----------
        filename: str
            Name of the file to save
        """
        if self.odd_on_full_coeffs_ratio is not None:
            nib.Nifti1Image(self.odd_on_full_coeffs_ratio.astype(np.float32),
                            self.aFOD.get_affine()).to_filename(filename)

    def save_mean_antipodal_distance(self, filename):
        """
        Save mean antipodal distances to file ``filename``

        Parameters
        ----------
        filename: str
            Name of the file to save
        """
        if self.mean_antipodal_distance is not None:
            nib.Nifti1Image(self.mean_antipodal_distance.astype(np.float32),
                            self.aFOD.get_affine()).to_filename(filename)

    def save_fiber_config_labels(self, fname):
        """
        Save fiber configurations labels to file ``filename``

        Parameters
        ----------
        fname: str
            Name of the file to save
        """
        if self.labels is not None:
            nib.Nifti1Image(self.labels.astype(np.uint8),
                            self.aPeaks.get_affine()).to_filename(fname)
        if self.nufo is not None:
            n_classes = int(self.nufo.max() + 1)
            for i in range(n_classes):
                mask = self.nufo == i
                tmp = np.zeros_like(self.labels)
                tmp[mask] = self.labels[mask]
                tmp_fname = fname[:fname.find('.')] +\
                    '_nufo_{0}'.format(i) + fname[fname.find('.'):]
                nib.Nifti1Image(tmp.astype(np.uint8),
                                self.aPeaks.get_affine()).to_filename(fname)

    def compute_odd_on_full_coeffs_ratio(self):
        """
        Compute the ratio of the norm of odd order SH coefficients on the
        norm of full order SH order coefficients per voxel
        """
        if self.aFOD:
            self.odd_on_full_coeffs_ratio =\
                self.aFOD.compute_odd_on_full_coeffs_ratio()
        else:
            logger.warning('No FODF supplied for computing\
                            odd/full coefficients ratio')

    def compute_mean_antipodal_distance(self, precision=2):
        """
        Compute the normalized distance between a vertex and its antipod

        Parameters
        ----------
        precision: int
            number of subdivisions of the icosahedron used for estimating SF
        """
        if self.aFOD:
            self.mean_antipodal_distance =\
                self.aFOD.compute_mean_antipodal_distance(precision)
        else:
            logger.warning('No FODF supplied for computing\
                            mean antipodal distance')

    def compute_configs_labels_from_mad(self, mad_th=0.2):
        """
        Label configuration of fiber orientations for each voxel

        Parameters
        ----------
        mad_th: float
            threshold applied on mean antipodal symmetry map to
            classify FODF as symmetric or not
        """
        if self.aPeaks is None:
            logger.warning('Can\'t label fiber orientations configurations' +
                           ' without peaks data.')
            return
        if self.mean_antipodal_distance is None:
            logger.warning('Can\'t label fiber configurations without' +
                           ' mean antipodal distance map')
            return

        peaks_count = self.aPeaks.get_peaks_count()
        mad = self.mean_antipodal_distance
        labels = np.zeros_like(peaks_count, dtype='int32')

        # half-orientations are labelled '1'
        labels[peaks_count == 1] = 1
        # straight single fiber orientations are labelled '2' whereas
        # bending single fiber orientations are labelled '3'
        labels[np.logical_and(peaks_count == 2, mad < mad_th)] = 2
        labels[np.logical_and(peaks_count == 2, mad > mad_th)] = 3
        # 3-directions FODF (such as 'T''s and 'Y''s) are labelled '4'
        labels[peaks_count == 3] = 4
        # symmetric 2-fibers crossings ('X''s) are labeled '5'
        labels[np.logical_and(peaks_count == 4, mad < mad_th)] = 5
        # symmetric 3-fibers crossings are labeled '6'
        labels[np.logical_and(peaks_count == 6, mad < mad_th)] = 6
        # other complex fiber orientations are labeled '7'
        labels[np.logical_and(labels == 0, peaks_count > 0)] = 7

        self.labels = labels

    def compute_configs_labels_from_ofr(self, ofr_th=0.3):
        """
        Label configuration of fiber orientations for each voxel

        Parameters
        ----------
        ofr_th: float
            threshold applied on odd/full ratio map to
            classify FODF as symmetric or not
        """
        if self.aPeaks is None:
            logger.warning('Can\'t label fiber orientations configurations' +
                           ' without peaks data.')
            return
        if self.odd_on_full_coeffs_ratio is None:
            logger.warning('Can\'t label fiber configurations without' +
                           ' OFR map')
            return

        peaks_count = self.aPeaks.get_peaks_count()
        ofr = self.odd_on_full_coeffs_ratio
        labels = np.zeros_like(peaks_count, dtype='int32')

        # half-orientations are labelled '1'
        labels[peaks_count == 1] = 1
        # straight single fiber orientations are labelled '2' whereas
        # bending single fiber orientations are labelled '3'
        labels[np.logical_and(peaks_count == 2, ofr < ofr_th)] = 2
        labels[np.logical_and(peaks_count == 2, ofr > ofr_th)] = 3
        # 3-directions FODF (such as 'T''s and 'Y''s) are labelled '4'
        labels[peaks_count == 3] = 4
        # symmetric 2-fibers crossings ('X''s) are labeled '5'
        labels[np.logical_and(peaks_count == 4, ofr < ofr_th)] = 5
        # symmetric 3-fibers crossings are labeled '6'
        labels[np.logical_and(peaks_count == 6, ofr < ofr_th)] = 6
        # other complex fiber orientations are labeled '7'
        labels[np.logical_and(labels == 0, peaks_count > 0)] = 7

        self.labels = labels
