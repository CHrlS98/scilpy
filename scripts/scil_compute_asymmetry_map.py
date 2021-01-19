#!/usr/bin/env python3

"""
Compute odd-power map, peaks directions and values
and nupeaks maps for ava-fodf.
"""

import argparse
import nibabel as nib
import numpy as np
from dipy.data import get_sphere
from dipy.reconst.shm import order_from_ncoef, sph_harm_full_ind_list
from scilpy.io.utils import (add_overwrite_arg,
                             add_sh_basis_args,
                             save_matrix_in_any_format,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.multi_processes import peaks_from_sh


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_sh', help='Input SH file.')
    p.add_argument('out_asymmetry', help='Output asymmerty map.')

    add_overwrite_arg(p)
    add_sh_basis_args(p)
    return p


def compute_asymmetry_map(data):
    """
    Compute asymmetry measure as defined in Cetin Karayumak et al, 2018
    """
    order = order_from_ncoef(data.shape[-1], is_full_basis=True)
    _, l_list = sph_harm_full_ind_list(order)

    sign = np.power(-1.0, l_list)
    sign = np.reshape(sign, (1, 1, 1, len(l_list)))
    data_squared = data**2
    mask = data_squared.sum(axis=-1) > 0.

    asym_map = np.zeros(data.shape[:-1])
    asym_map[mask] = np.sum(data_squared * sign, axis=-1)[mask] / \
        np.sum(data_squared, axis=-1)[mask]

    asym_map = np.sqrt(1 - asym_map**2) * mask

    return asym_map


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Safety checks for inputs
    assert_inputs_exist(parser, args.in_sh)

    # Safety checks for outputs
    assert_outputs_exist(parser, args, args.out_asymmetry)

    # Load inputs
    sh_img = nib.load(args.in_sh)
    sh_data = sh_img.get_fdata(dtype=np.float)

    # Compute asymmetry map
    asym_map = compute_asymmetry_map(sh_data)
    nib.save(nib.Nifti1Image(asym_map.astype(np.float32),
                             sh_img.affine), args.out_asymmetry)


if __name__ == '__main__':
    main()
