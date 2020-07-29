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
import matplotlib.pyplot as plt

from dipy.data import get_sphere
from dipy.direction.peaks import reshape_peaks_for_visualization

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)

from scilpy.reconst.asym_utils import (AFiberOrientationDistribution,
                                       APeaks)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input file')

    p.add_argument('--avfodf',
                   help='Output path of averaged fODF')

    p.add_argument('--rm_false_pos',
                   help='Output path of cleaned fODF file.')

    p.add_argument('--peaks',
                   help='Output path of peak directions file')

    p.add_argument('--nupeaks',
                   help='Output path of NuPeaks file')

    p.add_argument('--epsilon', default=1e-16, type=float,
                   help='Float epsilon for removing false positives')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    p.add_argument(
        '--sphere', default='symmetric724',
        help='Sphere used for the SH reprojection'
    )

    p.add_argument(
        '--sharpness', default=1.0, type=float,
        help='Specify sharpness factor to use for weighted average'
    )

    p.add_argument(
        '--batch_size', default=10, type=int,
        help='Size of batches when computing average (at least 3)'
    )

    p.add_argument(
        '--sigma', default=1.0, type=float,
        help='Sigma of the gaussian to use'
    )

    p.add_argument(
        '--mask', default=False, action='store_true',
        help='Mask null fodf'
    )

    p.add_argument(
        '--npeaks', default=10, type=int,
        help='Number of peaks for peak extraction'
    )

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    outputs = []
    if args.avfodf:
        outputs.append(args.avfodf)
    if args.rm_false_pos:
        outputs.append(args.rm_false_pos)
    if args.peaks:
        outputs.append(args.peaks)
    if args.nupeaks:
        outputs.append(args.nupeaks)
    if not outputs:
        parser.error('No output to be done.')

    # Checking args
    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, outputs, check_dir_exists=True)

    # Prepare data
    sphere = get_sphere(args.sphere)
    img = nib.nifti1.load(args.input)

    img_data = img.get_fdata()
    affine = img.affine

    FOD = AFiberOrientationDistribution(img_data,
                                        affine,
                                        args.sh_basis,
                                        args.sh_order)

    # Computing neighbors average of fODFs
    t0 = time.perf_counter()
    if args.rm_false_pos:
        logging.info('Cleaning FODF')
        FOD.clean_false_pos(args.epsilon)
        FOD.save_to_file(args.rm_false_pos)
    if args.avfodf:
        logging.info('Average FODF')
        FOD.average(sphere, dot_sharpness=args.sharpness, sigma=args.sigma,
                    batch_size=args.batch_size, mask=args.mask)
        FOD.save_to_file(args.avfodf)
    if args.peaks:
        logging.info('Extract peaks')
        peaks = FOD.extract_peaks(sphere, args.npeaks, 0.0, 0.3)
        peaks.save_to_file(args.peaks)
    if args.nupeaks:
        if not args.peaks:
            peaks = FOD.extract_peaks(sphere, args.npeaks, 0.0, 0.3)
        peaks.save_nupeaks(args.nupeaks)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
