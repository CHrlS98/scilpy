# -*- coding: utf-8 -*-

import logging
import numpy as np

from dipy.reconst.shm import sh_to_sf, sf_to_sh


def get_hemisphere_from_direction(direction, sphere):
    """
    Get indices of vertices of sphere on the same hemisphere as 'direction'
    """
    if np.nonzero(direction) == 0:
        return np.arange(sphere.vertices.size)

    direction.reshape((3,1))
    dotprod = np.dot(sphere.vertices, direction)

    return np.nonzero(dotprod >= 0)[0]


def compute_local_mean(data, sphere, antipods_table):
    """
    Compute local mean on the array 'data' (3x3x3xD array)
    """
    out_value = np.zeros_like(data[0, 0, 0])
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                direction = np.array([x-1, y-1, z-1])
                hemisphere = get_hemisphere_from_direction(direction, sphere)
                out_value[hemisphere] = out_value[hemisphere]\
                    + data[x, y, z, antipods_table[hemisphere]]
    return out_value / (data.shape[0] * data.shape[1] * data.shape[2])


def compute_avg_fodf_batch(data, sphere, sh_order=8,
                           input_sh_basis='descoteaux07'):
    """
    Compute the average of fodf in data with
    its 26 neighbors by batches
    """
    # Default batch size (10 slices)
    batch_size = 10

    # If the volume is smaller than batch_size, compute it
    # in one pass with compute_avg_fodf
    if data.shape[0] + 2 < batch_size:
        return compute_avg_fodf(data, sphere, sh_order, input_sh_basis)

    # Table of correspondance between a segment and its invert on the sphere
    antipods_table = np.array([sphere.find_closest(xyz) for xyz in -sphere.vertices])

    # Convert to spherical function
    sf = np.array([sh_to_sf(i, sphere, sh_order, input_sh_basis) for i in data])

    # Zero-initialize array for mean SF
    mean_sf = np.zeros_like(sf)
    dim = mean_sf.shape

    # Zero pad sf data
    pad_width = ((1, 1),(1, 1),(1, 1),(0, 0))
    sf = np.pad(sf, pad_width, mode='constant', constant_values=0.0)
    augm_dim = sf.shape

    # Last batch can be bigger than the others
    number_of_batches = int((augm_dim[0] - 2) / (batch_size - 2))

    # TODO: Compute directions and hemisphere once before loops here

    for num_batch in range(number_of_batches):
        # Select batch to process
        start = int(num_batch * (batch_size - 2))
        stop = int(start + batch_size)
        if (num_batch + 1) * (batch_size - 2) + batch_size > augm_dim[0]:
            stop = augm_dim[0]

        batch = sf[start:stop]
        dim = (batch.shape[0] - 2,
               batch.shape[1] - 2,
               batch.shape[2] - 2,
               batch.shape[3])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    direction = np.array([i-1, j-1, k-1])
                    hemisphere = get_hemisphere_from_direction(direction, sphere)
                    # too computationnaly intense
                    # won't work on big datasets
                    mean_sf[start:start + dim[0]][..., hemisphere] += \
                        batch[i:dim[0]+i, j:dim[1]+j, k:dim[2]+k, antipods_table[hemisphere]]

    mean_sf = mean_sf / 27.0
    mean_sh = np.array([sf_to_sh(i, sphere, sh_order, 'descoteaux07_full') for i in mean_sf])

    return mean_sh


def compute_avg_fodf(data, sphere, sh_order=8,
                     input_sh_basis='descoteaux07'):
    """
    Compute the average of fodf in data with its 26 neighbors.
    """
    # Table of correspondance between a segment and its invert on the sphere
    antipods_table = np.array([sphere.find_closest(xyz) for xyz in -sphere.vertices])

    # Convert to spherical function
    sf = np.array([sh_to_sf(i, sphere, sh_order, input_sh_basis) for i in data])

    # Zero-initialize array for mean SF
    mean_sf = np.zeros_like(sf)
    dim = mean_sf.shape

    # Zero pad sf data
    pad_width = ((1, 1),(1, 1),(1, 1),(0, 0))
    sf = np.pad(sf, pad_width, mode='constant', constant_values=0.0)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                direction = np.array([i-1, j-1, k-1])
                hemisphere = get_hemisphere_from_direction(direction, sphere)
                # too computationnaly intense
                # won't work on big datasets
                mean_sf[..., hemisphere] += \
                    sf[i:dim[0]+i, j:dim[1]+j, k:dim[2]+k, antipods_table[hemisphere]]

    mean_sf = mean_sf / 27.0
    mean_sh = np.array([sf_to_sh(i, sphere, sh_order, 'descoteaux07_full') for i in mean_sf])

    return mean_sh


def compute_naive_avg_fodf(data, sphere, sh_order=8,
                           input_sh_basis='descoteaux07'):
    """
    Naive implementation of neighbors average using for loops. Not optimized.
    """
    # Table of correspondance between a segment and its invert on the sphere
    antipods_table = np.array([sphere.find_closest(xyz) for xyz in -sphere.vertices])
    output_sh_basis = 'descoteaux07_full'

    # naive implementation for ground truth
    sf = np.array([sh_to_sf(slice_i, sphere, sh_order=sh_order, basis_type=input_sh_basis) for slice_i in data])
    mean_sf = np.zeros_like(sf)
    for x in range(1, data.shape[0] - 1):
        for y in range(1, data.shape[1] - 1):
            for z in range(1, data.shape[2] - 1):
                mean_sf[x, y, z] = compute_local_mean(sf[x-1:x+2,y-1:y+2, z-1:z+2], sphere, antipods_table)

    averaged_data = np.array([sf_to_sh(slice_i, sphere, sh_order=8, basis_type=output_sh_basis) for slice_i in mean_sf])

    return averaged_data


def compute_diff_fodf(input_data, averaged_data, sh_order, symmetric_basis, full_sh_basis, sphere):
    """
    Compute the fodf resulting from the subtraction of the original 
    symmetric signal to the averaged full signal
    """
    input_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, symmetric_basis) for slice_i in input_data])
    mean_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, full_sh_basis) for slice_i in averaged_data])

    resulting_sf = mean_sf - input_sf
    output_sh = np.array([sf_to_sh(slice_i, sphere, sh_order, full_sh_basis) for slice_i in resulting_sf])

    return output_sh


def compute_error(input_data, averaged_data, sh_order, symmetric_basis, full_sh_basis, sphere):
    """
    Compute mean square error between signal recovered from our symmetric
    basis and the signal recovered from a full SH basis
    """
    input_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, symmetric_basis) for slice_i in input_data])
    mean_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, full_sh_basis) for slice_i in averaged_data])

    error = np.sum((mean_sf - input_sf)**2, axis=-1)
    return error / error.max()


def compute_reconst_error(averaged_data, sh_order, symmetric_basis, full_sh_basis, sphere):
    """
    Compute the reconstruction error when using a symmetric basis to average the signal 
    obtained from a full spherical harmonics basis
    """
    full_sf = np.array([sh_to_sf(i, sphere, sh_order, full_sh_basis) for i in averaged_data])
    sym_sh = np.array([sf_to_sh(i, sphere, sh_order, symmetric_basis) for i in full_sf])

    sym_sf = np.array([sh_to_sf(i, sphere, sh_order, symmetric_basis) for i in sym_sh])

    error = np.sum((full_sf - sym_sf)**2, axis=-1)
    return error / error.max()


def compute_diff_mask(original_data, averaged_data):
    """
    Compute a mask showing highlighting the zones where a FODF is only
    present in one of the images
    """
    mask_for_original = (np.sum(np.abs(original_data), axis=-1)) > 0
    mask_for_averaged = (np.sum(np.abs(averaged_data), axis=-1)) > 0

    return (mask_for_averaged != mask_for_original)