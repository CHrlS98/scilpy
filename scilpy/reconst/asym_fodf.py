# -*- coding: utf-8 -*-

import logging
import numpy as np

from dipy.reconst.shm import sh_to_sf, sf_to_sh


def prepare_batch(batch_size, augm_dim):
    """
    Compute number of batches and resulting batch size.
    Resulting batch size will be modified if there is
    not more than one batch to process
    """
    # Last batch can be bigger than the others
    number_of_batches = int((augm_dim[0] - 2) / (batch_size - 2))

    # Corner case: When the dimension of the
    # data is smaller than the batch size
    if number_of_batches == 0:
        number_of_batches = 1
        batch_size = augm_dim[0]

    return number_of_batches, batch_size


def get_weights_table(sphere, sharpness):
    """
    Calculate the dot product (cos(theta)) between vertices
    on the sphere and directions to adjacent voxels

    Returns dictionary mapping voxel directions to dot products
    with sphere vertices in this direction for each direction
    """
    # center directions around current voxel
    directions = np.transpose(np.indices((3, 3, 3)) - np.ones((3, 3, 3)))
    directions = np.reshape(directions, (27, 3))

    # remove direction (0, 0, 0)
    directions = np.delete(directions, 13, 0)

    dir_norm = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    hemispheres = np.dot(sphere.vertices, dir_norm.T)
    hemispheres = np.where(hemispheres > 0.0, hemispheres**sharpness, 0.0)

    keys = list(map(tuple, directions))
    hemis_by_dir = dict(zip(keys, list(hemispheres.T)))
    sum_of_weights = np.sum(hemispheres, axis=-1)

    return hemis_by_dir, sum_of_weights


def compute_avg_fodf_no_weight(data, sphere, sh_order=8, batch_size=10,
                           input_sh_basis='descoteaux07'):
    """
    Compute the average of fodf in data with
    its 26 neighbors by batches
    """
    # Convert to spherical function
    sf = np.array([sh_to_sf(i, sphere, sh_order, input_sh_basis) for i in data],
                  dtype='float32')

    # Initialize array for mean SF with current voxel value
    mean_sf = np.copy(sf)

    # Zero pad sf data
    pad_width = ((1, 1),(1, 1),(1, 1),(0, 0))
    sf = np.pad(sf, pad_width, mode='constant', constant_values=0.0)

    # Prepare batch
    number_of_batches, batch_size = prepare_batch(batch_size, sf.shape)
    hemis_by_dir, sum_of_weights = get_weights_table(sphere, 0.0)

    # Compute average in batches
    for num_batch in range(number_of_batches):
        # Select batch to process
        start = int(num_batch * (batch_size - 2))
        stop = int(start + batch_size)
        if (num_batch + 1) * (batch_size - 2) + batch_size > sf.shape[0]:
            stop = sf.shape[0]

        batch = sf[start:stop]
        dim = (batch.shape[0] - 2,
               batch.shape[1] - 2,
               batch.shape[2] - 2,
               batch.shape[3])

        for key in hemis_by_dir:
            i, j, k = int(key[0] + 1), int(key[1] + 1), int(key[2] + 1)
            hemisphere = hemis_by_dir[key]
            mean_sf[start:start + dim[0]] += \
                np.multiply(batch[i:dim[0]+i, j:dim[1]+j, k:dim[2]+k],
                            hemisphere)

        mean_sf[start:start + dim[0]] = \
            np.multiply(mean_sf[start:start + dim[0]],
                        1.0 / (sum_of_weights + 1.0))

    mean_sh = np.array([sf_to_sh(i, sphere, sh_order, 'descoteaux07_full') for i in mean_sf])

    # DEBUG
    #a = np.mean(mean_sf, axis=(0, 1, 2))
    #b = np.mean(sf[1:-1, 1:-1, 1:-1], axis=(0, 1, 2))
    #mean_error = np.mean(np.abs(b - a))
    #print('Mean error: ', mean_error)
    # DEBUG

    return mean_sh


def compute_avg_fodf_with_weights(data, sphere, sh_order=8, sharpness=1.0,
                              batch_size=10, input_sh_basis='descoteaux07'):
    """
    Compute the average of fodf in data with
    its 26 neighbors by batches
    """
    # Convert to spherical function
    sf = np.array([sh_to_sf(i, sphere, sh_order, input_sh_basis) for i in data],
                  dtype='float32')

    # Initialize array for mean SF with current voxel value
    mean_sf = np.copy(sf)

    # Zero pad sf data
    pad_width = ((1, 1),(1, 1),(1, 1),(0, 0))
    sf = np.pad(sf, pad_width, mode='constant', constant_values=0.0)

    # Prepare batch
    number_of_batches, batch_size = prepare_batch(batch_size, sf.shape)
    hemis_by_dir, sum_of_weights = get_weights_table(sphere, sharpness)

    # Compute average in batches
    for num_batch in range(number_of_batches):
        # Select batch to process
        start = int(num_batch * (batch_size - 2))
        stop = int(start + batch_size)
        if (num_batch + 1) * (batch_size - 2) + batch_size > sf.shape[0]:
            stop = sf.shape[0]

        batch = sf[start:stop]
        dim = (batch.shape[0] - 2, batch.shape[1] - 2,
               batch.shape[2] - 2, batch.shape[3])

        for key in hemis_by_dir:
            direction = np.array([key])
            hemisphere = hemis_by_dir[key]
            i, j, k = int(key[0] + 1), int(key[1] + 1), int(key[2] + 1)
            mean_sf[start:start + dim[0]] += \
                np.multiply(batch[i:dim[0]+i, j:dim[1]+j, k:dim[2]+k], 
                            hemisphere)

        mean_sf[start:start + dim[0]] = \
            np.multiply(mean_sf[start:start + dim[0]],
                        1.0 / (sum_of_weights + 1.0))

    # DEBUG
    #a = np.mean(mean_sf, axis=(0, 1, 2))
    #b = np.mean(sf[1:-1, 1:-1, 1:-1], axis=(0, 1, 2))
    #mean_error = np.mean(np.abs(b - a))
    #print('Mean error: ', mean_error)
    # DEBUG

    mean_sh = np.array([sf_to_sh(i, sphere, sh_order, 'descoteaux07_full') for i in mean_sf])

    return mean_sh


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