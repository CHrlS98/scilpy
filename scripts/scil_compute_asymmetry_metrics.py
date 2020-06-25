#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compute asymmetry metrics on FODF using full basis
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.data import get_sphere

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

from scilpy.reconst.asym_fodf import (compute_diff_fodf, 
                                         compute_error, 
                                         compute_reconst_error,
                                         compute_diff_mask)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    sh_basis_choices = [
        'descoteaux07',
        'tournier07',
        'descoteaux07_full',
        'tournier07_full'
    ]

    p.add_argument('mean_fodf_input',
                   help='Path to the input mean fODF file (.nii or .nii.gz)')

    p.add_argument('mean_fodf_sh_basis', choices=sh_basis_choices, 
                   default='descoteaux07_full',
                   help='Name of the SH basis used with the mean fodf image')

    p.add_argument('symm_fodf_input',
                   help='Path to the input symmetric fodf file (.nii or .nii.gz)')

    p.add_argument('symm_fodf_sh_basis', choices=sh_basis_choices,
                   default='descoteaux07',
                   help='Name of the SH basis used with the symmetric fodf image')

    p.add_argument(
        '--diff_out',
        help='Path to the diff output image (.nii or .nii.gz)'
    )

    p.add_argument(
        '--mean_square_error_out',
        help='Path to the mean square error output image (.nii or .nii.gz)'
    )

    p.add_argument(
        '--reconst_error_out',
        help='Path to the reconstruction error output image (.nii or .nii.gz)'
    )

    p.add_argument(
        '--diff_mask_out',
        help='Map displaying presence of FODF in one of both images'
    )

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    p.add_argument(
        '--sphere', default='symmetric724',
        help='Sphere used for the SH reprojection (Default: \'symmetric724\')'
    )

    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Checking args
    outputs = []
    if args.diff_out:
        outputs.append(args.diff_out)
    if args.mean_square_error_out:
        outputs.append(args.mean_square_error_out)
    if args.reconst_error_out:
        outputs.append(args.reconst_error_out)
    if args.diff_mask_out:
        outputs.append(args.diff_mask_out)
    if not outputs:
        parser.error('No output to be done')

    inputs = [ args.mean_fodf_input, args.symm_fodf_input]

    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, outputs, check_dir_exists=True)

    # Prepare data
    sphere = get_sphere(args.sphere)
    mean_fodf_img = nib.nifti1.load(args.mean_fodf_input)
    symm_fodf_img = nib.nifti1.load(args.symm_fodf_input)

    mean_fodf = mean_fodf_img.get_fdata()
    symm_fodf = symm_fodf_img.get_fdata()

    if args.diff_out:
        logging.info('Computing difference fodf between input and output')
        diff_fodf = compute_diff_fodf(symm_fodf, mean_fodf, args.sh_order,
                                      args.symm_fodf_sh_basis,
                                      args.mean_fodf_sh_basis, sphere)
        diff_img = nib.Nifti1Image(diff_fodf.astype(np.float32),
                                   mean_fodf_img.affine)
        diff_img.to_filename(args.diff_out)

    if args.mean_square_error_out:
        logging.info('Computing mean squared error')
        ms_error = compute_error(symm_fodf, mean_fodf, args.sh_order,
                                 args.symm_fodf_sh_basis,
                                 args.mean_fodf_sh_basis, sphere)
        ms_error_img = nib.Nifti1Image(ms_error.astype(np.float32),
                                       mean_fodf_img.affine)
        ms_error_img.to_filename(args.mean_square_error_out)

    if args.reconst_error_out:
        logging.info('Computing reconstruction error')
        reconst_error = compute_reconst_error(mean_fodf, args.sh_order,
                                              args.symm_fodf_sh_basis,
                                              args.mean_fodf_sh_basis, sphere)
        reconst_error_img = nib.Nifti1Image(reconst_error.astype(np.float32),
                                            mean_fodf_img.affine)
        reconst_error_img.to_filename(args.reconst_error_out)

    if args.diff_mask_out:
        logging.info('Compute diff mask')
        diff_mask = compute_diff_mask(symm_fodf, mean_fodf)
        diff_mask_img = nib.Nifti1Image(diff_mask.astype(np.float32),
                                        mean_fodf_img.affine)
        diff_mask_img.to_filename(args.diff_mask_out)

    logging.info('Done.')


if __name__ == "__main__":
    main()
