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

    p.add_argument('--in_nupeaks',
                   help='Path to the input NuPeaks file')

    p.add_argument('--in_ofr',
                   help='Path to the input OFR file')

    p.add_argument('--in_vf',
                   help='Path to input volume fractions file')

    p.add_argument('--out_ofr',
                   help='Output path of odd on full coefficients ratio file')

    p.add_argument('--out_labels',
                   help='Output path of the labeled image using ' +
                        'OFR asymmetry measure')

    p.add_argument('--out_crossings',
                   help='Output path of crossing ratios file')

    p.add_argument('--out_proportions',
                   help='Output path of the NuPeaks proportions file')

    p.add_argument('--wm_th', default=0.30, type=float,
                   help='WM threshold in volume fraction map for WM mask')

    p.add_argument(
        '--sh_order', metavar='int', default=8, type=int,
        help='SH order of the input (Default: 8)')

    add_sh_basis_args(p)
    add_overwrite_arg(p)

    return p


def get_inputs_list(args, parser):
    inputs = []
    if args.in_fodf:
        inputs.append(args.in_fodf)
    if args.in_peaks:
        inputs.append(args.in_peaks)
        if args.in_nupeaks:
            inputs.append(args.in_nupeaks)
    if args.in_ofr:
        inputs.append(args.in_ofr)
    if args.in_vf:
        inputs.append(args.in_vf)
    if not inputs:
        parser.error('No input: Please supply at least one input')
    return inputs


def get_outputs_list(args, parser):
    outputs = []
    if args.out_ofr:
        if not args.in_fodf:
            parser.error('Can\'t compute odd/full coefficients\
                          ratio without FODF file')
        outputs.append(args.out_ofr)
    if args.out_labels:
        if not args.in_peaks:
            parser.error('Can\'t produce labels without peaks file')
        outputs.append(args.out_labels)
    if args.out_crossings:
        if not args.in_peaks:
            parser.error('Can\'t compute ratio of crossing'
                         ' fibers without peaks file')
        if not args.in_vf:
            parser.error('Can\'t compute proportion of crossings without'
                         ' volume fractions')
        outputs.append(args.out_crossings)
    if args.out_proportions:
        if not args.in_peaks:
            parser.error('Can\'t compute proportions without peaks file')
        if not args.in_vf:
            parser.error('Can\'t compute proportions without volume fraction')
        outputs.append(args.out_proportions)
    if not outputs:
        parser.error('No output to be done.')
    return outputs


def write_to_file(filename, content):
    file = open(filename, 'w')
    file.write(content)
    file.close()


def load_inputs(args):
    FOD = None
    peaks = None
    ofr = None
    vf = None
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
        nupeaks =\
            nib.nifti1.load(args.in_nupeaks).get_fdata().astype(np.uint8)\
            if args.in_nupeaks else None
        peaks = APeaks(peaks_data, peaks_affine, nupeaks)
    if args.in_ofr:
        ofr = nib.nifti1.load(args.in_ofr).get_fdata()
    if args.in_vf:
        vf = nib.nifti1.load(args.in_vf).get_fdata()
    return FOD, peaks, ofr, vf


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Checking args
    inputs = get_inputs_list(args, parser)
    outputs = get_outputs_list(args, parser)
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, outputs, check_dir_exists=True)

    FOD, peaks, ofr, vf = load_inputs(args)
    metrics_popper = AFODMetricsPopper(FOD, peaks, ofr, vf)

    # Computing AVFODF metrics
    t0 = time.perf_counter()
    if args.out_ofr:
        logging.info('Compute odd/full coefficients ratio')
        metrics_popper.compute_odd_on_full_coeffs_ratio()
        metrics_popper.save_odf_on_full_coeffs_ratio(args.out_ofr)
    if args.out_labels:
        logging.info('Label intra-voxel configurations using OFR map')
        metrics_popper.compute_configs_labels_from_ofr()
        metrics_popper.save_fiber_config_labels(args.out_labels)
    if args.out_crossings:
        logging.info('Compute crossings proportion in WM '
                     'for varying threshold')
        range_max = vf[..., -1].max()
        x = np.arange(0.0, range_max, 0.01)
        y = np.array([metrics_popper.get_crossing_fibers_proportions('wm', th)
                      for th in x])
        output = 'WM mask threshold, Proportion of crossings\n'
        for i in range(x.shape[0]):
            output += '{0:.3}, {1:.4}\n'.format(x[i], y[i])
        write_to_file(args.out_crossings, output)
    if args.out_proportions:
        logging.info('Compute number of peaks proportions')
        proportions = metrics_popper.get_nupeaks_proportions('wm', args.wm_th)
        output = 'NuPeaks, '\
                 'Proportion of n-peaks FODF inside WM mask (th = {0:.3g})\n'\
                 .format(args.wm_th)
        for i in range(proportions.shape[0]):
            output += '{0}, {1}\n'.format(i, proportions[i])
        write_to_file(args.out_proportions, output)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
