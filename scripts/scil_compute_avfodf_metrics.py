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
                                       APeaks, AFODMetricsPopper)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('--in_fodf',
                   help='Path to the input FODF file')

    p.add_argument('--in_peaks',
                   help='Path to the input peaks file')

    p.add_argument('--odd_full_ratio',
                   help='Output path of odd on full coefficients ratio file')

    p.add_argument('--mad',
                   help='Output path of mean antipodal distance file')

    p.add_argument('--labels',
                   help='Output path of the labeled image')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    inputs = []
    if args.in_fodf:
        inputs.append(args.in_fodf)
    if args.in_peaks:
        inputs.append(args.in_peaks)
    if not inputs:
        parser.error('No input: Please supply at least one input')

    outputs = []
    if args.odd_full_ratio:
        if not args.in_fodf:
            parser.error('Can\'t compute odd/full coefficients\
                          ratio without FODF file')
        outputs.append(args.odd_full_ratio)
    if args.mad:
        if not args.in_fodf:
            parser.error('Can\'t compute mean antipodal distance\
                          without FODF file')
        outputs.append(args.mad)
    if args.labels:
        if not args.in_peaks:
            parser.error('Can\'t clean fodf from peaks file')
        outputs.append(args.labels)
    if not outputs:
        parser.error('No output to be done.')

    # Checking args
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, outputs, check_dir_exists=True)

    FOD = None
    peaks = None
    if args.in_fodf:
        fodf_img = nib.nifti1.load(args.in_fodf)
        fodf_data = fodf_img.get_fdata()
        fodf_affine = fodf_img.affine
        FOD = AFiberOrientationDistribution(fodf_data,
                                            fodf_affine,
                                            args.sh_basis,
                                            args.sh_order)
    if args.in_peaks:
        peaks_img = nib.nifti1.load(args.in_peaks)
        peaks_data = peaks_img.get_fdata()
        peaks_affine = peaks_img.affine
        peaks = APeaks(peaks_data, peaks_affine)

    metrics_popper = AFODMetricsPopper(FOD, peaks)

    # Computing neighbors average of fODFs
    t0 = time.perf_counter()
    if args.odd_full_ratio:
        logging.info('Compute odd/full coefficients ratio')
        metrics_popper.compute_odd_on_full_coeffs_ratio()
        metrics_popper.save_odf_on_full_coeffs_ratio(args.odd_full_ratio)
    if args.mad:
        logging.info('Compute mean antipodal distance')
        metrics_popper.compute_mean_antipodal_distance()
        metrics_popper.save_mean_antipodal_distance(args.mad)
    if args.labels:
        logging.info('Label intra-voxel configurations')
        metrics_popper.compute_voxel_configs_labels_map()
        metrics_popper.save_fiber_config_labels(args.labels)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
