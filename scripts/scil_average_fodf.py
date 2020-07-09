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

from dipy.data import get_sphere

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)

from scilpy.reconst.asym_fodf import (compute_avg_fodf_with_weights,
                                      compute_avg_fodf_no_weight)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input fODF file')
    
    p.add_argument('output',
                   help='Output path with extension')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    p.add_argument(
        '--sphere', default='symmetric724',
        help='Sphere used for the SH reprojection'
    )

    p.add_argument(
        '--weighted', default=False, action='store_true',
        help='Use weights from dot product in average'
    )

    p.add_argument(
        '--sharpness', default=1.0, type=float,
        help='Specify sharpness factor to use for weighted average'
    )

    p.add_argument(
        '--batch_size', default=10, type=int,
        help='Size of batches when computing average (at least 3)'
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
    assert_outputs_exist(parser, args, args.output, check_dir_exists=True)

    if args.batch_size < 3:
        parser.error('Batch size must be of at least 3.')

    # Prepare data
    sphere = get_sphere(args.sphere)
    img = nib.nifti1.load(args.input)

    img_data = img.get_fdata()
    affine = img.affine

    # Computing neighbors average of fODFs
    t0 = time.perf_counter()
    avg_img = None
    if args.weighted:
        logging.info('Computing average fodf (weighted)')
        avg_fodf = compute_avg_fodf_with_weights(img_data, sphere, args.sh_order,
                                             args.sharpness, args.batch_size,
                                             args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)
    else:
        logging.info('Computing average fodf (not weighted)')
        avg_fodf = compute_avg_fodf_no_weight(img_data, sphere, args.sh_order,
                                          args.batch_size, args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)

    logging.info('Saving output')
    avg_img.to_filename(args.output)


if __name__ == "__main__":
    main()
