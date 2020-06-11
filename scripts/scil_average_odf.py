#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute neighbors average from fODF
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.data import get_sphere

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_force_b0_arg,
                             add_sh_basis_args)

from scilpy.reconst.average_fodf import compute_avg_fodf


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input fODF file (.nii or .nii.gz format)')
    
    p.add_argument(
        'output',
        help='Output filename for the averaged fiber ODF coefficients.')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    p.add_argument(
        '--sphere', default='symmetric724',
        help='sphere used for the SH reprojection'
    )

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Checking args
    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output)

    # Prepare data
    sphere = get_sphere(args.sphere)
    img = nib.nifti1.load(args.input)
    data = img.get_fdata()

    # Computing neighbors average of fODFs
    avg_fodf = compute_avg_fodf(
        data, sphere, sh_order=args.sh_order, input_sh_basis=args.sh_basis)

    # Saving results
    # nib.save(nib.Nifti1Image(
    #     reshape_peaks_for_visualization(peaks_csd), vol.affine), args.peaks)


if __name__ == "__main__":
    main()
