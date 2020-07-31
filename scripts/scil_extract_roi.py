#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to extract a region of interest from a dataset
"""

import argparse
import logging

import nibabel as nib
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input', help='Path to the input file')
    p.add_argument('x_min', type=int, help='Minimum X index')
    p.add_argument('x_max', type=int, help='Maximum X index')
    p.add_argument('y_min', type=int, help='Minimum Y index')
    p.add_argument('y_max', type=int, help='Maximum Y index')
    p.add_argument('z_min', type=int, help='Minimum Z index')
    p.add_argument('z_max', type=int, help='Maximum Z index')
    p.add_argument('output', help='Path to the output file')
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Checking args
    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output, check_dir_exists=True)

    in_image = nib.nifti1.load(args.input)
    data = in_image.get_fdata()
    roi = data[args.x_min:args.x_max,
               args.y_min:args.y_max,
               args.z_min:args.z_max]
    out_image = nib.Nifti1Image(roi.astype(in_image.get_data_dtype()),
                                in_image.affine)
    out_image.to_filename(args.output)


if __name__ == "__main__":
    main()
