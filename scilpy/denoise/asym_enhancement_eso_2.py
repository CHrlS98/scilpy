# -*- coding: utf-8 -*-

import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from scipy.ndimage import correlate


def average_fodf_asymmetrically(fodf,  sh_order=8, sh_basis='descoteaux07',
                                sphere_str='repulsion724', in_full_basis=False,
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
    w_per_sf = _get_weights2(sphere, dot_sharpness, sigma)

    # Convert to spherical function
    in_sh_basis = sh_basis
    if in_full_basis:
        in_sh_basis += '_full'
    B = sh_to_sf_matrix(sphere, sh_order=sh_order, basis_type=in_sh_basis,
                        return_inv=False)

    img_shape = fodf.shape[:-1]
    nb_sf = len(sphere.vertices)
    mean_sf = np.zeros((img_shape[0], img_shape[1], img_shape[2], nb_sf))

    for sf_i in range(nb_sf):
        current_sf = np.array([np.dot(i, B[:, sf_i]) for i in fodf], dtype='float64')
        # we could adapt the w_filter locally for the mask and weighted
        w_filter = w_per_sf[sf_i]
        w_filter /= w_filter.sum()
        mean_sf[..., sf_i] = correlate(current_sf, w_filter, mode="constant", cval=0)

    # Convert back to SH coefficients
    out_sh_basis = sh_basis
    if out_full_basis:
        out_sh_basis += '_full'
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


def _get_weights2(sphere, dot_sharpness, sigma):
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

    weights = weights.reshape((len(sphere.vertices), 3, 3, 3))
    return weights
