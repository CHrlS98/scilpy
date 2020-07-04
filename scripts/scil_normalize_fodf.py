#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to normalize fODF
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
                                         compute_avg_fodf_batch)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input fODF file (.nii or .nii.gz format)')
    
    p.add_argument('output',
        help='Output path with extension (.nii or .nii.gz)')

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
    img = nib.nifti1.load(args.input)

    img_data = img.get_fdata()
    affine = img.affine

    logging.info('Normalizing FODFs')
    norm = np.linalg.norm(img_data, axis=-1, keepdims=False)
    normalized_sh = np.zeros_like(img_data)
    mask = norm > 0
    masked_norm = np.reshape(norm[mask],
                             (norm[mask].shape[0], 1))
    normalized_sh[mask] = img_data[mask] / masked_norm

    logging.info('Saving output')
    normalized_img = nib.Nifti1Image(normalized_sh.astype(np.float32), affine)
    normalized_img.to_filename(args.output)


if __name__ == "__main__":
    main()
