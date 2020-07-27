#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to execute full asymmetric workflow
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
                                       compare_nupeaks)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('wm_fodf',
                   help='Path to the input white matter fODF file')

    p.add_argument('a_threshold',
                   help='Path to the absolute threshold text file\n'
                        'for peaks extraction')

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

    p.add_argument('--npeaks', default=10, type=int,
                   help='Number of peaks for peak extraction')

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def get_a_threshold(filename):
    file = open(filename, 'r')
    a_threshold = file.read()
    file.close()
    return float(a_threshold)


def write_to_text_file(filename, content):
    file = open(filename, 'w')
    file.write(content)
    file.close()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    inputs = [args.wm_fodf, args.a_threshold]

    out_wm_fodf_clean = 'wm_fodf_clean.nii.gz'
    out_wm_fodf_peaks = 'wm_fodf_peaks.nii.gz'
    out_wm_fodf_nupeaks = 'wm_fodf_nupeaks.nii.gz'
    out_wm_avfodf = 'wm_avfodf.nii.gz'
    out_wm_avfodf_ofr = 'wm_avfodf_ofr.nii.gz'
    out_wm_avfodf_peaks = 'wm_avfodf_peaks.nii.gz'
    out_wm_avfodf_nupeaks = 'wm_avfodf_nupeaks.nii.gz'
    outputs = [out_wm_fodf_peaks,
               out_wm_fodf_nupeaks,
               out_wm_avfodf,
               out_wm_avfodf_ofr,
               out_wm_avfodf_peaks,
               out_wm_avfodf_nupeaks]

    # Checking args
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, outputs, check_dir_exists=True)

    # Prepare data
    sphere = get_sphere(args.sphere)
    wm_fodf = nib.nifti1.load(args.wm_fodf)

    wm_fodf_data = wm_fodf.get_fdata()
    affine = wm_fodf.affine

    a_threshold = get_a_threshold(args.a_threshold)

    FOD = AFiberOrientationDistribution(wm_fodf_data,
                                        affine,
                                        args.sh_basis,
                                        args.sh_order)

    # Execute full asym workflow
    t0 = time.perf_counter()

    logging.info('Clean FODF')
    FOD.clean_false_pos(args.epsilon)
    FOD.save_to_file(out_wm_fodf_clean)

    logging.info('Extract peaks (symmetric)')
    peaks = FOD.extract_peaks(sphere, args.npeaks, a_threshold)
    peaks.save_to_file(out_wm_fodf_peaks)

    logging.info('Compute NuPeaks on symmetric fodf')
    fodf_nupeaks = peaks.compute_nupeaks()
    nib.save(nib.Nifti1Image(fodf_nupeaks, affine),
             out_wm_fodf_nupeaks)

    logging.info('Average FODF')
    FOD.average(sphere, dot_sharpness=args.sharpness, sigma=args.sigma,
                batch_size=args.batch_size, mask=True)
    FOD.save_to_file(out_wm_avfodf)

    logging.info('Compute OFR')
    nib.save(nib.Nifti1Image(FOD.compute_odd_on_full_coeffs_ratio(), affine),
             out_wm_avfodf_ofr)

    logging.info('Extract peaks (asymmetric)')
    peaks = FOD.extract_peaks(sphere, args.npeaks, a_threshold)
    peaks.save_to_file(out_wm_avfodf_peaks)

    logging.info('Compute NuPeaks on avfodf')
    avfodf_nupeaks = peaks.compute_nupeaks()
    nib.save(nib.Nifti1Image(avfodf_nupeaks, affine),
             out_wm_avfodf_nupeaks)

    logging.info('Compare NuPeaks')
    nupeaks_compare =\
        compare_nupeaks(fodf_nupeaks, avfodf_nupeaks, args.npeaks)
    write_to_text_file('nupeaks_compare.txt', nupeaks_compare)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
