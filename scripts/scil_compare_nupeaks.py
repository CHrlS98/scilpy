#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create comparative table of distribution of number of
peaks inside voxels between symmetric and asymmetric FODF
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

from scilpy.reconst.asym_utils import (compare_nupeaks)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_fodf_nupeaks',
                   help='Path to the input FODF NuPeaks file')

    p.add_argument('in_avfodf_nupeaks',
                   help='Path to the input AVFODF NuPeaks file')

    p.add_argument('in_volume_frac',
                   help='Path to the input volume fraction file')

    p.add_argument('out_nupeaks_compare',
                   help='Path to the ouput NuPeaks comparative table file')

    p.add_argument('--wm_th', default=0.30, type=float,
                   help='WM threshold in volume fraction map for WM mask')

    add_overwrite_arg(p)

    return p


def get_inputs_list(args, parser):
    inputs = [args.in_fodf_nupeaks,
              args.in_avfodf_nupeaks,
              args.in_volume_frac]
    return inputs


def get_outputs_list(args, parser):
    outputs = [args.out_nupeaks_compare]
    return outputs


def write_to_file(filename, content):
    file = open(filename, 'w')
    file.write(content)
    file.close()


def load_inputs(args):
    fodf_nupeaks = nib.nifti1.load(args.in_fodf_nupeaks)\
        .get_fdata().astype(np.uint8)
    avfodf_nupeaks = nib.nifti1.load(args.in_avfodf_nupeaks)\
        .get_fdata().astype(np.uint8)
    volume_frac = nib.nifti1.load(args.in_volume_frac).get_fdata()
    return fodf_nupeaks, avfodf_nupeaks, volume_frac


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # Checking args
    inputs = get_inputs_list(args, parser)
    outputs = get_outputs_list(args, parser)
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, outputs, check_dir_exists=True)

    fodf_nupeaks, avfodf_nupeaks, volume_frac = load_inputs(args)

    t0 = time.perf_counter()
    comparative_table =\
        compare_nupeaks(fodf_nupeaks, avfodf_nupeaks, volume_frac, args.wm_th)

    output = 'FODF NuPeaks'
    for i in range(comparative_table.shape[1]):
        output += ', {0}-peaks AVFODF'.format(i)
    output += '\n'

    for i in range(comparative_table.shape[0]):
        output += '{0}'.format(i)
        for j in range(comparative_table.shape[1]):
            output += ', {0}'.format(comparative_table[i, j])
        output += '\n'
    write_to_file(args.out_nupeaks_compare, output)
    t1 = time.perf_counter()

    elapsedTime = t1 - t0
    print('Elapsed time (s): ', elapsedTime)


if __name__ == "__main__":
    main()
