# -*- coding: utf-8 -*-

import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere
from scipy.ndimage import correlate


def average_fodf_asymmetrically(fodf,  sh_order=8, sh_basis='descoteaux07',
                                sphere_str='repulsion724', dot_sharpness=1.0,
                                sigma=1.0, mask=None):
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
    dot_sharpness: float, optional
        Exponent of the dot product. When set to 0.0, directions
        are not weighted by the dot product. Default: 1.0
    sigma: float, optional
        Variance of the gaussian. Default: 1.0
    mask: ndarray, optional
        If supplied, forces to 0 fODF in voxels outside mask. Default: None

    Returns
    -------
    avafodf: ndarray (x, y, z, n_coeffs)
        Asymmetric averaged fODF represented with the SH basis `sh_basis`.
        The output fODF in returned in a full basis
    """
    # Load the sphere used for projection of SH
    sphere = get_sphere(sphere_str)
    # Normalized filter for each sf direction
    weigths = _get_weights(sphere, dot_sharpness, sigma)
    c_w_per_sf, n_w_per_sf = weigths

    # Detect if the basis is full based on its order
    # and the number of coefficients of the SH
    in_sh_basis = sh_basis
    if fodf.shape[-1] == (sh_order + 1)**2:
        in_sh_basis += '_full'

    img_shape = fodf.shape[:-1]
    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros((img_shape[0], img_shape[1], img_shape[2], nb_sf))
    B = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=in_sh_basis,
                        return_inv=False)

    # We want a B matrix to project on an inverse sphere to have the sf on
    # the opposite hemisphere for a given vertice
    neg_B = sh_to_sf_matrix(Sphere(xyz=-sphere.vertices), sh_order=sh_order,
                            basis_type=in_sh_basis, return_inv=False)

    for sf_i in range(nb_sf):
        # first pass: filtering opposite hemispheres
        current_sf = np.dot(fodf, neg_B[:, sf_i])
        w_filter = n_w_per_sf[sf_i]
        mean_sf[..., sf_i] = correlate(current_sf, w_filter, mode="constant")

        # second pass: apply weight of center sf
        current_sf = np.dot(fodf, B[:, sf_i])
        w_filter = c_w_per_sf[sf_i]
        mean_sf[..., sf_i] += correlate(current_sf, w_filter, mode="constant")

    # Convert back to SH coefficients
    out_sh_basis = sh_basis + '_full'
    _, B_inv = sh_to_sf_matrix(sphere, sh_order=sh_order,
                               basis_type=out_sh_basis)
    if mask is not None:
        avafodf = np.zeros(
            np.append(fodf.shape[:-1], [B_inv.shape[-1]]),
            dtype='float32')
        avafodf[mask] = np.array(
            [np.dot(i, B_inv) for i in mean_sf])[mask]
    else:
        avafodf = np.array([np.dot(i, B_inv) for i in mean_sf])

    return avafodf


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
    directions = np.transpose(np.indices((3, 3, 3)) - np.ones((3, 3, 3)))
    directions = np.reshape(directions, (27, 3))
    non_zero_dir = np.ones([27], dtype=bool)
    non_zero_dir[13] = False

    # normalize dir
    dir_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions[non_zero_dir] /= dir_norm[non_zero_dir]

    g_weights = np.exp(-dir_norm**2 / (2 * sigma**2))
    d_weights = np.dot(sphere.vertices, directions.T)
    d_weights = np.where(d_weights > 0.0, d_weights**dot_sharpness, 0.0)
    weights = d_weights * g_weights.T
    weights[:, 13] = 1.0
    # normalize filter right here
    weights /= weights.sum(axis=-1, keepdims=True)

    # Filter for center voxel
    c_weights = np.zeros_like(weights)
    c_weights[:, 13] = weights[:, 13]
    c_weights = c_weights.reshape((len(sphere.vertices), 3, 3, 3))

    # Filter for neighbors
    n_weights = np.copy(weights)
    n_weights[:, 13] = 0.0
    n_weights = n_weights.reshape((len(sphere.vertices), 3, 3, 3))

    return c_weights, n_weights
