#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute neighbors average from fODF
"""

import time
import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)

from scilpy.denoise.asym_enhancement import (average_fodf_asymmetrically)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_fodf',
                   help='Path to the input file')

    p.add_argument('out_avafodf',
                   help='Output path of averaged fODF')

    p.add_argument('--mask',
                   help='Path to a mask to apply on output')

    p.add_argument(
        '--sh_order', default=8, type=int,
        help='SH order of the input [%(default)s]')

    p.add_argument(
        '--sphere', default='symmetric724', type=str,
        help='Sphere used for the SH reprojection [%(default)s]'
    )

    p.add_argument(
        '--sharpness', default=1.0, type=float,
        help='Specify sharpness factor to use for weighted average'
        ' [%(default)s]'
    )

    p.add_argument(
        '--sigma', default=1.0, type=float,
        help='Sigma of the gaussian to use [%(default)s]'
    )

    p.add_argument(
        '--in_full_basis', default='False',
        choices=['True', 'False'], type=str,
        help='True if input fODF is in full SH basis [%(default)s]'
    )

    p.add_argument(
        '--out_full_basis', default='True',
        choices=['True', 'False'], type=str,
        help='True if output fODF is in full SH basis [%(default)s]'
    )

    p.add_argument(
        '--batch_size', default=10, type=int,
        help='Size of batches when computing average [%(default)s]'
    )

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    inputs = []
    inputs.append(args.in_fodf)
    if args.mask:
        inputs.append(args.mask)

    # Checking args
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_avafodf,
                         check_dir_exists=True)

    # Prepare data
    fodf_img = nib.nifti1.load(args.in_fodf)
    fodf_data = fodf_img.get_fdata().astype(np.float32)

    mask_data = None
    if args.mask:
        mask_data = nib.nifti1.load(args.mask).get_fdata().astype(np.bool)

    # Computing neighbors asymmetric average of fODFs
    t0 = time.perf_counter()
    logging.info('Computing asymmetric averaged fODF')
    in_full_basis = args.in_full_basis == 'True'
    out_full_basis = args.out_full_basis == 'True'
    avafodf =\
        average_fodf_asymmetrically(fodf_data,
                                    sh_order=args.sh_order,
                                    sh_basis=args.sh_basis,
                                    sphere_str=args.sphere,
                                    in_full_basis=in_full_basis,
                                    out_full_basis=out_full_basis,
                                    dot_sharpness=args.sharpness,
                                    sigma=args.sigma, mask=mask_data,
                                    batch_size=args.batch_size)
    nib.save(nib.Nifti1Image(avafodf.astype(np.float), fodf_img.affine),
             args.out_avafodf)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
