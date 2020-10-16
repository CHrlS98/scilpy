#!/usr/bin/env python3
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

from scilpy.denoise.asym_enhancement import (infer_asymmetries_from_neighbors)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_fodf',
                   help='Path to the input file.')

    p.add_argument('out_avafodf',
                   help='Output path of averaged fODF.')

    p.add_argument('--nb_it', type=int, default=1,
                   help='Number of iterations.')

    add_overwrite_arg(p)

    return p


def get_file_prefix_and_extension(avafodf_file):
    extension_index = avafodf_file.find('.nii')
    if extension_index != -1:
        extension = avafodf_file[extension_index:]
        prefix = avafodf_file[:extension_index]
    else:
        extension = '.nii.gz'
        prefix = avafodf_file
    return prefix, extension


def filter_iterative(fodf_data, args, affine):
    f_prefix, f_extension = get_file_prefix_and_extension(args.out_avafodf)
    data = fodf_data
    for i in range(args.nb_it):
        logging.info('Iteration {0}'.format(i))
        out_fodf = infer_asymmetries_from_neighbors(data)
        outfile = f_prefix + '_{0}'.format(i) + f_extension
        nib.save(nib.Nifti1Image(out_fodf.astype(np.float), affine),
                 outfile)
        data = out_fodf


def filter_one_shot(fodf_data, args, affine):
    f_prefix, f_extension = get_file_prefix_and_extension(args.out_avafodf)
    asym_fodf = infer_asymmetries_from_neighbors(fodf_data)

    outfile = f_prefix + f_extension
    nib.save(nib.Nifti1Image(asym_fodf.astype(np.float), affine),
             outfile)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    inputs = []
    inputs.append(args.in_fodf)

    # Checking args
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_avafodf,
                         check_dir_exists=True)

    # Prepare data
    fodf_img = nib.nifti1.load(args.in_fodf)
    fodf_data = fodf_img.get_fdata().astype(np.float32)

    # Computing neighbors asymmetric average of fODFs
    t0 = time.perf_counter()
    logging.info('Computing asymmetric fODF')
    if args.nb_it > 1:
        filter_iterative(fodf_data, args, fodf_img.affine)
    else:
        filter_one_shot(fodf_data, args, fodf_img.affine)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
