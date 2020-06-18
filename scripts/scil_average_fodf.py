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
                             assert_outputs_exist, add_sh_basis_args)

from scilpy.reconst.average_fodf import (compute_avg_fodf, 
                                         compute_naive_avg_fodf, 
                                         compute_diff_fodf, 
                                         compute_error, 
                                         compute_reconst_error)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input fODF file (.nii or .nii.gz format)')
    
    p.add_argument('output',
        help='Output filename (without extension).')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    p.add_argument(
        '--sphere', default='symmetric724',
        help='Sphere used for the SH reprojection'
    )

    p.add_argument(
        '--mask', default=None,
        help='Mask to use for computing the average fodf'
    )

    p.add_argument(
        '--naive', default=False, action='store_true',
        help='Use naive implementation with for loops (for ground truth)'
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
    mask_data = None
    if args.mask != None:
        mask = nib.nifti1.load(args.mask)
        mask_data = mask.get_fdata()

    img_data = img.get_fdata()
    affine = img.affine

    # Computing neighbors average of fODFs
    avg_fodf = None
    avg_img = None
    if args.naive:
        avg_fodf = compute_naive_avg_fodf(img_data, sphere, args.sh_order,
                                          args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)
    else:
        avg_fodf = compute_avg_fodf(img_data, affine, sphere, mask_data,
                                    args.sh_order, args.sh_basis)
        avg_img = nib.Nifti1Image(avg_fodf.astype(np.float32), affine)

    # Computing difference between symmetric and asymmetric images
    diff_fodf = compute_diff_fodf(img_data, avg_fodf, args.sh_order, args.sh_basis, 'descoteaux07_full', sphere)
    diff_img = nib.Nifti1Image(diff_fodf.astype(np.float32), affine)

    # Computing mean-squared error
    ms_error = compute_error(img_data, avg_fodf, args.sh_order, args.sh_basis, 'descoteaux07_full', sphere)
    ms_error_img = nib.Nifti1Image(ms_error.astype(np.float32), affine)

    # Computing reconstruction error
    reconst_error = compute_reconst_error(avg_fodf, args.sh_order, args.sh_basis, 'descoteaux07_full', sphere)
    reconst_error_img = nib.Nifti1Image(reconst_error.astype(np.float32), affine)

    avg_img.to_filename(args.output + '_fodf.nii.gz')
    diff_img.to_filename(args.output + '_diff.nii.gz')
    ms_error_img.to_filename(args.output + '_ms_error.nii.gz')
    reconst_error_img.to_filename(args.output + '_reconst_error.nii.gz')


if __name__ == "__main__":
    main()
