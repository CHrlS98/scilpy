# -*- coding: utf-8 -*-

import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere


def average_fodf_asymmetrically(fodf,  sh_order=8, sh_basis='descoteaux07',
                                sphere_str='symmetric724', in_full_basis=False,
                                out_full_basis=True, dot_sharpness=1.0,
                                sigma=1.0, mask=None, batch_size=10):
    """Average the fODF projected on a sphere using a first-neighbor gaussian
    blur and a dot product weight between sphere directions and the direction
    to neighborhood voxels, forcing to 0 negative values and thus performing
    asymmetric hemisphere-aware filtering.

    Parameters
    ----------
    fodf: ndarray (x, y, z, n_coeffs)
        Input fODF array
    sh_order: int, optional
        Maximum order of the SH series. Default: 8
    sh_basis: {'descoteaux07', 'tournier07'}, optional
        SH basis of the fODF. Default: 'descoteaux07'
    sphere_str: str
        Name of the Sphere to use to project SH coefficients to SF.
        Default: 'symmetric724'
    in_full_basis: bool, optional
        True if input SH coefficients are in a full SH basis. Default: False
    out_full_basis: bool, optional
        True if output SH coefficients are in a full SH basis. Default: True
    dot_sharpness: float, optional
        Exponent of the dot product. When set to 0.0, directions
        are not weighted by the dot product. Default: 1.0
    sigma: float, optional
        Variance of the gaussian. Default: 1.0
    batch_size: int, optional
        Number of volume slices processed at a same time. The
        last batch to be processed can be of a size up to
        (batch_size * 2.0 - 1). Default: 10
    mask: ndarray, optional
        If supplied, forces to 0 fODF in voxels outside mask. Default: None

    Returns
    -------
    avafodf: ndarray (x, y, z, n_coeffs)
        Asymmetric averaged fODF represented with the SH basis `sh_basis`.
        If `out_full_basis`, a full SH basis is used for reconstructing the
        output fODF.
    """
    # Load the sphere used for projection of SH
    sphere = get_sphere(sphere_str)

    # Convert to spherical function
    B = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=sh_basis,
                        use_full_basis=in_full_basis, return_inv=False)
    sf = np.array([np.dot(i, B) for i in fodf], dtype='float32')

    # Initialize array for mean SF with current voxel value
    mean_sf = np.copy(sf)

    # Zero-pad sf data
    sf = np.pad(sf, ((1, 1), (1, 1), (1, 1), (0, 0)),
                mode='constant', constant_values=0.0)

    # Prepare batch
    batch_indices = _get_batches_indices(fodf.shape, batch_size)
    w_by_dir, norm_w = _get_weights(sphere, dot_sharpness, sigma)

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

    # Release sf array from memory before instantiating output fODF array
    del sf

    _, B_inv = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=sh_basis,
                               use_full_basis=out_full_basis)
    if mask is not None:
        avafodf = np.zeros(
            np.append(fodf.shape[:-1], [B_inv.shape[-1]]),
            dtype='float32')
        avafodf[mask] = np.array(
            [np.dot(i, B_inv) for i in mean_sf])[mask]
    else:
        avafodf = np.array([np.dot(i, B_inv) for i in mean_sf])

    return avafodf


def _get_directions_to_voxels():
    """
    Get the vectors to neighboor voxels

    Returns
    -------
    directions: array (26, 3)
        array of directions from center of voxel to neighboors
    """
    # center directions around current voxel
    directions = np.transpose(np.indices((3, 3, 3)) - np.ones((3, 3, 3)))
    directions = np.reshape(directions, (27, 3))

    # remove direction (0, 0, 0)
    directions = np.delete(directions, 13, 0)

    return directions


def _get_weights(sphere, dot_sharpness, sigma):
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
    directions = _get_directions_to_voxels()
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


def _get_batches_indices(fodf_shape, batch_size):
    """
    Get the index of slices of data set along the first axis

    Parameters
    ----------
    fodf_shape: tuple
        Shape of fODF array
    batch_size: int
        Number of slices per batch

    Returns
    ------
    split_indices: list
        List of indices for each batch
    """
    nb_slices = fodf_shape[0]
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
