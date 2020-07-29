# -*- coding: utf-8 -*-

import logging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from dipy.reconst.shm import (sh_to_sf, sf_to_sh, sh_to_sf_matrix,
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
        # TODO: Apply hemisphere to opposite hemisphere
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

    def extract_peaks(self, sphere, npeaks=10, a_threshold=0.0,
                      r_threshold=0.5):
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
        r_threshold: float
            relative threshold. Peaks less than r_threshold times
            the biggest peak are discarded.

        Returns
        -------
        peaks: APeaks
            APeaks object containing extracted peaks
        """
        B = sh_to_sf_matrix(sphere, self.sh_order, self.sh_basis, False)

        peaks_dirs = np.zeros(list(self.fodf.shape[0:3]) + [npeaks, 3])
        peaks_count = np.zeros(self.fodf.shape[:-1], dtype=np.uint8)
        for index in ndindex(self.fodf.shape[:-1]):
            sf = np.dot(self.fodf[index], B)
            sf[sf < a_threshold] = 0.
            directions, _, _ =\
                peak_directions(sf, sphere, is_symmetric=False,
                                relative_peak_threshold=r_threshold)
            n = min(npeaks, directions.shape[0])
            peaks_count[index] = n
            peaks_dirs[index][:n] = directions[:n]

        return APeaks(peaks_dirs, self.affine, peaks_count)

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
    def __init__(self, data, affine, nupeaks=None):
        """
        Build the APeaks object representing peaks
        extracted from asymmetric FODF
        """
        self.peaks = data
        self.affine = affine
        self.npeaks = data.shape[-2]
        if nupeaks is not None:
            self.nupeaks = nupeaks
        else:
            logger.warning('NuPeaks not provided. Computing from peaks. Could '
                           'result in false positives from rounded 0 values.')
            self.nupeaks =\
                np.cumsum(np.sum(self.peaks**2, axis=-1) > 0.,
                          axis=-1)[..., -1]
        self.labels = None

    def get_peaks(self):
        return self.peaks

    def get_affine(self):
        return self.affine

    def save_to_file(self, filename):
        """
        Save peaks to file 'filename'
        """
        nib.save(nib.Nifti1Image(self.peaks, self.affine), filename)

    def save_nupeaks(self, filename):
        """
        Save nupeaks to file 'filename'
        """
        nib.save(nib.Nifti1Image(self.nupeaks.astype(np.uint8), self.affine),
                 filename)

    def get_nupeaks(self):
        """
        Get NuPeaks map, where the value for a voxel is the number
        of peaks at this voxel. For a symmetric input, the NuPeaks map is
        equivalent to the NuFO map times 2.

        Returns
        -------
        nupeaks: array
            number of peaks per voxel for each voxel
        """
        return self.nupeaks


class AFODMetricsPopper(object):
    def __init__(self, aFOD=None, aPeaks=None, ofr=None, vf=None):
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
        self.ofr = ofr
        self.vf = vf
        self.labels = None

    def save_odf_on_full_coeffs_ratio(self, filename):
        """
        Save odd on full coefficients ratios to file ``filename``

        Parameters
        ----------
        filename: str
            Name of the file to save
        """
        if self.ofr is not None:
            nib.Nifti1Image(self.ofr.astype(np.float32),
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

    def compute_odd_on_full_coeffs_ratio(self):
        """
        Compute the ratio of the norm of odd order SH coefficients on the
        norm of full order SH order coefficients per voxel
        """
        if self.aFOD:
            self.ofr =\
                self.aFOD.compute_odd_on_full_coeffs_ratio()
        else:
            logger.warning('No FODF supplied for computing\
                            odd/full coefficients ratio')

    def compute_configs_labels(self, ofr_th=0.3):
        """
        Label configuration of fiber orientations for each voxel

        Parameters
        ----------
        ofr_th: float
            threshold applied on odd/full ratio map to
            classify FODF as symmetric or not

        Note
        ----
        * half-orientations are labelled '1'
        * straight single fiber orientations are labelled '2' whereas
          bending single fiber orientations are labelled '3'
        * 3-directions FODF (such as 'T''s and 'Y''s) are labelled '4'
        * symmetric 2-fibers crossings ('X''s) are labeled '5'
        * symmetric 3-fibers crossings are labeled '6'
        * other complex fiber orientations are labeled '7'
        """
        if self.aPeaks is None:
            logger.warning('Can\'t label fiber orientations configurations' +
                           ' without peaks data.')
            return
        if self.ofr is None:
            logger.warning('Can\'t label fiber configurations without' +
                           ' OFR map')
            return

        peaks_count = self.aPeaks.get_nupeaks()
        ofr = self.ofr
        labels = np.zeros_like(peaks_count, dtype='int32')

        labels[peaks_count == 1] = 1
        labels[np.logical_and(peaks_count == 2, ofr < ofr_th)] = 2
        labels[np.logical_and(peaks_count == 2, ofr > ofr_th)] = 3
        labels[peaks_count == 3] = 4
        labels[np.logical_and(peaks_count == 4, ofr < ofr_th)] = 5
        labels[np.logical_and(peaks_count == 6, ofr < ofr_th)] = 6
        labels[np.logical_and(labels == 0, peaks_count > 0)] = 7

        self.labels = labels

    def get_crossing_fibers_proportions(self, tissue, threshold):
        """
        Compute proportion of crossing fibers in the region defined by
        `mask` if specified.

        Parameters
        ----------
        tissue: {'csf', 'gm', 'wm'}
            tissue to consider in volume fractions map
        threshold: float
            threshold to apply to volume fractions map for chosen tissue

        Returns
        -------
        ratio: float
            ratio of crossing fibers on total number of voxels with fiber

        Note
        ----
        A crossing fiber is any fiber with more than 2 peaks.
        """
        # Pre-processing checks
        if self.aPeaks is None:
            logger.warning('Peaks data missing. Can\'t compute crossing'
                           ' fibers proportions')
            return
        if self.vf is None:
            logger.warning('Volume fractions data missing. Can\'t compute '
                           'crossing fibers proportions')
            return

        if tissue == 'csf':
            mask = self.vf[..., 0] > threshold
        elif tissue == 'gm':
            mask = self.vf[..., 1] > threshold
        elif tissue == 'wm':
            mask = self.vf[..., 2] > threshold
        else:
            logger.warning('Invalid tissue name.')
            return

        nupeaks = self.aPeaks.get_nupeaks()
        if nupeaks[mask].size > 0:
            ratio = np.count_nonzero(nupeaks[mask] > 2) / nupeaks[mask].size
        else:
            ratio = 0.0

        return ratio


def compare_nupeaks(sym_nupeaks, asym_nupeaks, npeaks):
    """
    Compare the NuPeaks map for symmetric fODF with the NuPeaks map for
    asymmetric fODF
    """
    output = ''
    for i in range(npeaks + 1):
        mask = sym_nupeaks == i
        masked_asym_nupeaks = asym_nupeaks[mask]
        size = masked_asym_nupeaks.size
        # Number of occurence of each number of peaks
        bincount = np.bincount(masked_asym_nupeaks)
        output += 'For sym_nupeaks == {0}\n'.format(i)
        for j in range(bincount.size):
            prop = bincount[j] / size * 100.
            output +=\
                ' Proportion of asym_nupeaks == {0}: {1} %\n'.format(j, prop)
    return output
