# -*- coding: utf-8 -*-

import logging
import numpy as np

from dipy.reconst.shm import sh_to_sf, sf_to_sh


def get_hemisphere_from_direction(direction, sphere):
    """
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

    """
    if np.nonzero(direction) == 0:
        return np.arange(sphere.vertices.size)

    direction.reshape((3,1))
    dotprod = np.dot(sphere.vertices, direction)

    return np.nonzero(dotprod >= 0)[0]


def compute_local_mean(data, sphere, antipods_table):
    out_value = np.zeros_like(data[0, 0, 0])
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            for z in range(data.shape[2]):
                direction = np.array([x-1, y-1, z-1])
                hemisphere = get_hemisphere_from_direction(direction, sphere)
                out_value[hemisphere] = out_value[hemisphere]\
                    + data[x, y, z, antipods_table[hemisphere]]
    return out_value / (data.shape[0] * data.shape[1] * data.shape[2])


def compute_avg_fodf(data, affine, sphere, mask = None, 
                     sh_order=8, input_sh_basis='descoteaux07'):
    """
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

    """
    # TODO: safety checks

    # Table of correspondance between a segment and its invert on the sphere
    segments_table = np.array([sphere.find_closest(xyz) for xyz in -sphere.vertices])

    # Out of memory on big data sets
    # Besoin de la tranche precedente et suivante
    sf = sh_to_sf(data, sphere, sh_order=sh_order, basis_type=input_sh_basis)

    # Computing average of fODFs
    half_width = 1
    padding = np.full(len(sf.shape), 2 * half_width)
    padding[-1] = 0
    augm_dim = tuple(np.array(sf.shape) + padding)
    padded_sf = np.zeros(augm_dim)

    for x in range(-half_width, half_width + 1):
        for y in range(-half_width, half_width + 1):
            for z in range(-half_width, half_width + 1):
                # TODO: learn how to use affine information
                direction = np.matmul(affine[:3,:3], np.array([x, y, z]))
                hemisphere = get_hemisphere_from_direction(direction, sphere)
                opposite_hemisphere = segments_table[hemisphere]
                padded_sf[\
                    half_width + x:augm_dim[0] - half_width + x,\
                    half_width + y:augm_dim[1] - half_width + y,\
                    half_width + z:augm_dim[2] - half_width + z,\
                    hemisphere
                ] = padded_sf[\
                        half_width + x:augm_dim[0] - half_width + x,\
                        half_width + y:augm_dim[1] - half_width + y,\
                        half_width + z:augm_dim[2] - half_width + z,\
                        hemisphere
                    ] + sf[:, :, :, opposite_hemisphere]

    padded_sf = padded_sf / 27.0
    sf = padded_sf[half_width:-half_width, half_width:-half_width, half_width:-half_width]

    # convert back to sh using a full basis
    sh_data = sf_to_sh(sf, sphere, sh_order=sh_order, basis_type=output_sh_basis)

    if mask is not None:
        bin_mask = mask > 0
        # TODO: Replace for loops by more efficient alternative
        for x in range(sh_data.shape[0]):
            for y in range(sh_data.shape[1]):
                for z in range(sh_data.shape[2]):
                    if not bin_mask[x, y, z]:
                        sh_data[x, y, z] = np.zeros_like(sh_data[x, y, z])

    return sh_data


def compute_naive_avg_fodf(data, sphere, sh_order=8,
                           input_sh_basis='descoteaux07'):
    """
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

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
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

    """
    input_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, symmetric_basis) for slice_i in input_data])
    mean_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, full_sh_basis) for slice_i in averaged_data])

    resulting_sf = mean_sf - input_sf
    output_sh = np.array([sf_to_sh(slice_i, sphere, sh_order, full_sh_basis) for slice_i in resulting_sf])

    return output_sh


def compute_error(input_data, averaged_data, sh_order, symmetric_basis, full_sh_basis, sphere):
    """
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

    """
    input_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, symmetric_basis) for slice_i in input_data])
    mean_sf = np.array([sh_to_sf(slice_i, sphere, sh_order, full_sh_basis) for slice_i in averaged_data])

    input_sf_norm = np.linalg.norm(input_sf, axis=-1)
    mean_sf_norm = np.linalg.norm(mean_sf, axis=-1)

    error = np.sum((mean_sf - input_sf)**2, axis=-1)
    return error / error.max()


def compute_reconst_error(averaged_data, sh_order, symmetric_basis, full_sh_basis, sphere):
    """
     DESCRIPTION

    Parameters
    ----------
    PARAM1: PARAM DESCRIPTION

    Returns
    -------
    RET1: RETURN VALUE DESCRIPTION

    """
    full_sf = np.array([sh_to_sf(i, sphere, sh_order, full_sh_basis) for i in averaged_data])
    sym_sh = np.array([sf_to_sh(i, sphere, sh_order, symmetric_basis) for i in full_sf])

    sym_sf = np.array([sh_to_sf(i, sphere, sh_order, symmetric_basis) for i in sym_sh])

    error = np.sum((full_sf - sym_sf)**2, axis=-1)
    return error/error.max()

