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

from scilpy.reconst.asym_fodf import (compute_naive_avg_fodf,
                                      compute_avg_fodf_weighted,
                                      compute_avg_fodf_batch)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input fODF file (.nii or .nii.gz format)')
    
    p.add_argument('output',
        help='Output path with extension (.nii or .nii.gz)')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    p.add_argument(
        '--sphere', default='symmetric724',
        help='Sphere used for the SH reprojection'
    )

    p.add_argument(
        '--naive', default=False, action='store_true',
        help='Use naive implementation with for loops (for ground truth)'
    )

    p.add_argument(
        '--weighted', default=False, action='store_true',
        help='Use weights from dot product in average'
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

    # Prepare data
    sphere = get_sphere(args.sphere)
    img = nib.nifti1.load(args.input)
    #mask_data = None
    #if args.mask != None:
    #    mask = nib.nifti1.load(args.mask)
    #    mask_data = mask.get_fdata()

    img_data = img.get_fdata()
    affine = img.affine

    # Computing neighbors average of fODFs
    t0 = time.perf_counter()
    avg_img = None
    if args.weighted:
        logging.info('Computing average fodf (weighted)')
        avg_fodf = compute_avg_fodf_weighted(img_data, sphere, args.sh_order,
                                          args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)
    elif args.naive:
        logging.info('Computing average fodf (naive implementation)')
        avg_fodf = compute_naive_avg_fodf(img_data, sphere, args.sh_order,
                                          args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)
    else:
        logging.info('Computing average fodf (fast implementation)')
        avg_fodf = compute_avg_fodf_batch(img_data, sphere,
                                    args.sh_order, args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)

    logging.info('Saving output')
    avg_img.to_filename(args.output)


if __name__ == "__main__":
    main()
