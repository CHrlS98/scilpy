# -*- coding: utf-8 -*-

import logging
import numpy as np
import nibabel as nib

from dipy.reconst.shm import (sh_to_sf, sf_to_sh,
                              sph_harm_full_ind_list)
from dipy.core.sphere import HemiSphere
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


class FiberOrientationDistribution(object):
    def __init__(self, fodf, affine, sh_basis, sh_order):
        """
        Build the FiberOrientationDistribution object

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
        sf = np.array([sh_to_sf(i, sphere, self.sh_order, self.sh_basis)
                       for i in self.fodf],
                      dtype='float32')

        # Initialize array for mean SF with current voxel value
        mean_sf = np.copy(sf)

        # Zero pad sf data
        sf = np.pad(sf, ((1, 1),(1, 1),(1, 1),(0, 0)),
                    mode='constant', constant_values=0.0)

        # Prepare batch
        batch_indices = self._get_batches_indices(batch_size)
        hemis_by_dir, sum_of_dot_weights =\
            self._get_dot_weights(sphere, dot_sharpness)
        gauss_weights_by_dir, sum_of_gauss_weights =\
            self._get_gaussian_weights(sigma)

        # Compute average in batches
        for batch_index in batch_indices:
            batch = sf[batch_index]
            dim = (batch.shape[0] - 2, 
                   batch.shape[1] - 2,
                   batch.shape[2] - 2, 
                   batch.shape[3])

            for key in hemis_by_dir:
                direction = np.array([key])
                hemisphere = hemis_by_dir[key]
                gauss_weight = gauss_weights_by_dir[key]

                i, j, k = int(key[0] + 1), int(key[1] + 1), int(key[2] + 1)
                mean_sf[batch_index[0]:batch_index[0] + dim[0]] += gauss_weight *\
                        np.multiply(batch[i:dim[0]+i, j:dim[1]+j, k:dim[2]+k],
                                    hemisphere)

            mean_sf[batch_index[0]:batch_index[0] + dim[0]] = \
                np.multiply(mean_sf[batch_index[0]:batch_index[0] + dim[0]],
                            1.0 / (sum_of_dot_weights * gauss_weight + 1.0))

        self.fodf = np.zeros((self.fodf.shape[0],
                             self.fodf.shape[1],
                             self.fodf.shape[2],
                             ncoef_from_order(sh_order,
                                              sh_basis)),
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


    def normalize_by_voxel(self):
        """
        Perform voxel-wise normalization of FODF (in place)
        """
        norm = np.linalg.norm(self.fodf, axis=-1, keepdims=False)
        normalized_sh = np.zeros_like(self.fodf)
        mask = norm > 0
        masked_norm = np.reshape(norm[mask],
                                (norm[mask].shape[0], 1))
        normalized_sh[mask] = self.fodf[mask] / masked_norm

        self.fodf = normalized_sh


    def compute_asymmetry_measure(self):
        """
        Measure the asymmetry of the FODF per voxel.

        Note
        ====
        The asymmetry measure corresponds to the ratio of the norm of odd
        order SH coefficients on the norm of full order SH order coefficients
        """
        if not '_full' in self.sh_basis:
            logging.error('Can\'t compute asymmetry measure ' +\
                          'from symmetric SH basis')
            import sys
            sys.exit('Aborting program')

        _, l_list = sph_harm_full_ind_list(self.sh_order)
        odd_order_coeffs = self.fodf[..., l_list % 2==1]
        odd_order_norms = np.linalg.norm(odd_order_coeffs, axis=-1)
        full_norms = np.linalg.norm(self.fodf, axis=-1)
        
        asymmetry_measure = np.zeros_like(full_norms)
        mask = full_norms > 0

        asymmetry_measure[mask] = odd_order_norms[mask] / full_norms[mask]
        return asymmetry_measure


    def clean_false_pos(self, epsilon):
        """
        Remove false positives by forcing to zero values smaller 
        than epsilon and recompute mask
        """
        self.fodf[self.fodf[..., 0] < epsilon] = 0.0
        self.mask = np.linalg.norm(self.fodf, axis=-1) > 0

    
    def extract_peaks(self, sphere, npeaks=5):
        """
        Extract peaks on FODF without any asumption of symmetry

        Parameters
        ==========
        sphere: Sphere
            sphere to use for peak extraction
        """
        sf = np.array([sh_to_sf(i, sphere, self.sh_order, self.sh_basis) 
                       for i in self.fodf])

        peaks_dirs = np.zeros(list(sf.shape[0:3]) + [2*npeaks, 3])
        for index in ndindex(sf.shape[:-1]):
            directions, _, _ = peak_directions(sf[index], sphere, is_symmetric=False)
            n = min(2*npeaks, directions.shape[0])
            peaks_dirs[index][:n] = directions[:n]

        return peaks_dirs


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


    def _get_dot_weights(self, sphere, sharpness):
        """
        Calculate the dot product (cos(theta)**sharpness) between
        vertices on the sphere and directions to adjacent voxels

        Parameters
        ==========
        sphere: Sphere
            sphere on which the SF is approximated
        sharpness: float
            exponent of the dot product

        Returns
        =======
        hemis_by_dir: dictionary
            dictionary mapping voxel directions to dot products with
            sphere vertices in this direction for each direction
        sum_of_weights: numpy array
            Normalization factor for each averaged direction on the sphere
        """
        directions = self._get_directions_to_voxels()

        dir_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        hemispheres = np.dot(sphere.vertices, dir_norm.T)
        hemispheres = np.where(hemispheres > 0.0, hemispheres**sharpness, 0.0)

        dir_keys = list(map(tuple, directions))
        hemis_by_dir = dict(zip(dir_keys, list(hemispheres.T)))
        sum_of_dot_weights = np.sum(hemispheres, axis=-1)

        return hemis_by_dir, sum_of_dot_weights


    def _get_gaussian_weights(self, sigma):
        """
        Get weights for neighboors using a gaussian of variance sigma
        (not normalized)

        Parameters
        ==========
        sigma: float
            Variance of the gaussian

        Returns
        =======

        """
        directions = self._get_directions_to_voxels()
        dir_norms = np.linalg.norm(directions, axis=-1, keepdims=True)

        weights = np.exp(-dir_norms**2 / (2 * sigma**2))
        sum_of_gauss_weights = np.sum(weights)

        dir_keys = list(map(tuple, directions))
        gaus_weight_by_dir = dict(zip(dir_keys, list(weights)))

        return gaus_weight_by_dir, sum_of_gauss_weights


    def _get_hemisphere_around_dir(self, dir, data, sphere):
        """
        Get the indices of vertices on the same hemisphere than dir

        Parameters
        ==========
        dir: numpy array(N,3)
            direction of hemisphere
        sphere: Sphere
            sphere to use for hemisphere

        Returns
        =======
        indices: array
            Indices of vertices on hemisphere in direction dir
        """
        mask = np.dot(dir, sphere.vertices.T) >= 0
        hemispheres = []
        sf_on_hemispheres = []
        for i in range(mask.shape[0]):
            verts = sphere.vertices[mask[i]]
            sf = data[i][mask[i], None]
            hemispheres.append(HemiSphere(xyz=verts))
            sf_on_hemispheres.append(sf)

        return hemispheres, sf_on_hemispheres


    def _get_opposite_indices(self, sphere):
        """
        Get index of opposites vertices for each index on the sphere
        """
        return np.array([sphere.find_closest(vertex)
                        for vertex in -sphere.vertices])


    def _mirror_sf_through_plane(self, sf, sphere, normal):
        """
        Mirror the spherical function on the sphere through the
        plane going through the center of the sphere defined by
        normal by assigning the SF value on a sphere vertice to
        the sphere opposite vertice
        """
        dot_val = np.dot(normal, sphere.vertices.T)
        n = int(sphere.vertices.shape[0] / 2)

        hemisphere_index = np.argsort(dot_val)[..., n:]
        opposites = self._get_opposite_indices(sphere)
        opposites_index =opposites[hemisphere_index]

        mirrored_sf = np.copy(sf)
        for index in ndindex(mirrored_sf.shape[:-1]):
            mirrored_sf[index][opposites_index[index]] =\
                mirrored_sf[index][hemisphere_index[index]]

        return mirrored_sf, hemisphere_index