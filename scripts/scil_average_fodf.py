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
from dipy.direction.peaks import reshape_peaks_for_visualization
from fury import window, actor

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_sh_basis_args)

from scilpy.reconst.asym_utils import (AFiberOrientationDistribution,
                                       APeaks)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input file')

    p.add_argument('input_type', choices={'fodf', 'peaks'},
                   help='Type of the input')

    p.add_argument('--avfod',
                   help='Output path of averaged fODF')

    p.add_argument('--asym_measure',
                   help='Output path of asymmetry measure file')

    p.add_argument('--rm_false_pos',
                   help='Output path of cleaned fODF file.')

    p.add_argument('--peaks',
                   help='Output path of peak directions file')

    p.add_argument('--labels',
                   help='Output path of the labeled image')

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
    if args.avfod:
        if args.input_type == 'peaks':
            parser.error('Can\'t compute avfod from peaks file')
        outputs.append(args.avfod)
    if args.asym_measure:
        if args.input_type == 'peaks':
            parser.error('Can\'t compute asym measuer from peaks file')
        outputs.append(args.asym_measure)
    if args.rm_false_pos:
        if args.input_type == 'peaks':
            parser.error('Can\'t clean fodf from peaks file')
        outputs.append(args.rm_false_pos)
    if args.peaks:
        if args.input_type == 'peaks':
            parser.error('Can\'t compute peaks from peaks file')
        outputs.append(args.peaks)
    if args.labels:
        if not args.peaks and not args.input_type == 'peaks':
            parser.error('Can\'t label image without peaks information')
        outputs.append(args.labels)
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

    if args.input_type == 'fodf':
        FOD = AFiberOrientationDistribution(img_data,
                                            affine,
                                            args.sh_basis,
                                            args.sh_order)
    else:
        peaks = APeaks(img_data, affine)

    # Computing neighbors average of fODFs
    t0 = time.perf_counter()
    if args.rm_false_pos:
        logging.info('Cleaning FODF')
        FOD.clean_false_pos(args.epsilon)
        FOD.save_to_file(args.rm_false_pos)
    if args.avfod:
        logging.info('Average FODF')
        FOD.average(sphere, dot_sharpness=args.sharpness,sigma=args.sigma,
                    batch_size=args.batch_size, mask=args.mask)
        FOD.save_to_file(args.avfod)
    if args.asym_measure:
        logging.info('Compute asymmetry measure')
        asym_measure = FOD.compute_asymmetry_measure()
        asym_measure_img = \
            nib.Nifti1Image(asym_measure.astype(np.float32), affine)
        asym_measure_img.to_filename(args.asym_measure)
    if args.peaks:
        logging.info('Extract peaks')
        peaks = FOD.extract_peaks(sphere, args.npeaks)
        peaks.save_to_file(args.peaks)
    if args.labels:
        logging.info('Label intra-voxel configurations')
        labels = peaks.label_configs()
        nib.save(nib.Nifti1Image(labels, affine), args.labels)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
