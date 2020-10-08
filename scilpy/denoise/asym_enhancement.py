# -*- coding: utf-8 -*-

import numpy as np
from dipy.reconst.shm import sh_to_sf_matrix
from dipy.data import get_sphere
from dipy.core.sphere import Sphere, HemiSphere
from scipy.ndimage import correlate

import nibabel as nib


def _get_hemisphere_repulsion(n_pts, nb_iter=5000):
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, nb_iter)

    return hsph_updated


def infer_asymmetries_from_neighbors(fodf, sh_basis='descoteaux07',
                                     sh_order=8, sphere_name='repulsion724'):
    sphere = get_sphere(sphere_name)
    sphere_mirror = Sphere(xyz=-sphere.vertices)

    # fODF tendencies from neighborhood information
    global_fodf = average_fodf_asymmetrically(fodf, exclude_center=True)

    B_full = sh_to_sf_matrix(sphere, basis_type=sh_basis + '_full',
                             sh_order=8, return_inv=False)
    B_full_mirror = sh_to_sf_matrix(sphere_mirror,
                                    basis_type=sh_basis + '_full',
                                    sh_order=8, return_inv=False)

    if fodf.shape[-1] == (sh_order + 1)**2:
        sh_basis += '_full'
        B = B_full
        B_mirror = B_full_mirror
    else:
        B = sh_to_sf_matrix(sphere, basis_type=sh_basis,
                            sh_order=8,  return_inv=False)
        B_mirror = sh_to_sf_matrix(sphere_mirror, basis_type=sh_basis,
                                sh_order=8, return_inv=False)

    nb_sf = len(sphere.vertices)
    out_sf = np.zeros(np.append(fodf.shape[:-1], [nb_sf]))

    for sf_i in range(nb_sf):
        v = np.dot(fodf, B[..., sf_i])
        v_minus = np.dot(fodf, B_mirror[..., sf_i])
        v[v < 0] = 0
        v_minus[v_minus < 0] = 0
        sym_energy = v + v_minus

        v = np.dot(global_fodf, B_full[..., sf_i])
        v_minus = np.dot(global_fodf, B_full_mirror[..., sf_i])
        v[v < 0] = 0
        v_minus[v_minus < 0] = 0
        asym_energy = v + v_minus

        non_zero_sf = asym_energy > 0
        ratios = np.zeros_like(asym_energy)
        ratios[non_zero_sf] = v[non_zero_sf] / asym_energy[non_zero_sf]
        out_sf[..., sf_i] = sym_energy * ratios

    _, B_inv = sh_to_sf_matrix(sphere, sh_order=8,
                               basis_type='descoteaux07_full')

    asym_fodf = np.array([np.dot(i, B_inv) for i in out_sf])
    return asym_fodf


def average_fodf_asymmetrically(fodf,  sh_order=8, sh_basis='descoteaux07',
                                sphere_str='repulsion724', dot_sharpness=1.0,
                                sigma=1.0, mask=None, exclude_center=False):
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
        Default: 'repulsion724'
    dot_sharpness: float, optional
        Exponent of the dot product. When set to 0.0, directions
        are not weighted by the dot product. Default: 1.0
    sigma: float, optional
        Sigma for the gaussian. Default: 1.0
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
    weights = _get_weights(sphere, dot_sharpness, sigma, exclude_center)

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
        w_filter = weights[..., sf_i]

        # Calculate contribution of center voxel
        current_sf = np.dot(fodf, B[:, sf_i])
        mean_sf[..., sf_i] = w_filter[1, 1, 1] * current_sf

        # Add contributions of neighbors using opposite hemispheres
        current_sf = np.dot(fodf, neg_B[:, sf_i])
        w_filter[1, 1, 1] = 0.0
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


def _get_weights(sphere, dot_sharpness, sigma, exclude_center=False):
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
    directions = np.zeros((3, 3, 3, 3))
    for x in range(3):
        for y in range(3):
            for z in range(3):
                directions[x, y, z, 0] = x - 1
                directions[x, y, z, 1] = y - 1
                directions[x, y, z, 2] = z - 1

    non_zero_dir = np.ones((3, 3, 3), dtype=bool)
    non_zero_dir[1, 1, 1] = False

    # normalize dir
    dir_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions[non_zero_dir] /= dir_norm[non_zero_dir]

    g_weights = np.exp(-dir_norm**2 / (2 * sigma**2))
    d_weights = np.dot(directions, sphere.vertices.T)

    d_weights = np.where(d_weights > 0.0, d_weights**dot_sharpness, 0.0)
    weights = d_weights * g_weights
    weights[1, 1, 1, :] = 0.0 if exclude_center else 1.0

    # Normalize filters so that all sphere directions weights sum to 1
    weights /= weights.reshape((-1, weights.shape[-1])).sum(axis=0)

    return weights
