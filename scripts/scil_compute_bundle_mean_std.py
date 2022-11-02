#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute mean and std for the whole bundle for each metric. This is achieved by
averaging the metrics value of all voxels occupied by the bundle.

Density weighting modifies the contribution of voxel with lower/higher
streamline count to reduce influence of spurious streamlines.
"""

import argparse
import json
import logging
import os

import nibabel as nib
import numpy as np

from scilpy.utils.filenames import split_name_with_nii
from scilpy.io.image import assert_same_resolution
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_reference_arg,
                             assert_inputs_exist)
from scilpy.utils.metrics_tools import get_bundle_metrics_mean_std


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Fiber bundle file to compute statistics on')
    p.add_argument('in_metrics', nargs='+',
                   help='Nifti file to compute statistics on. Probably some '
                        'tractometry measure(s) such as FA, MD, RD, ...')

    p.add_argument('--density_weighting', action='store_true',
                   help='If set, weight statistics by the number of '
                        'fibers passing through each voxel.')
    p.add_argument('--distance_weighting', metavar='DISTANCE_NII',
                   help='If set, weight statistics by the inverse of the '
                        'distance between a streamline and the centroid.')
    p.add_argument('--correlation_weighting', metavar='CORRELATION_NII',
                   help='If set, weight statistics by the correlation strength '
                        'between longitudinal data.')
    p.add_argument('--include_dps', action='store_true',
                   help='Save values from data_per_streamline.')
    add_reference_arg(p)
    add_json_args(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_bundle] + args.in_metrics,
                        optional=args.reference)

    assert_same_resolution(args.in_metrics)
    metrics = [nib.load(metric) for metric in args.in_metrics]

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    if args.distance_weighting:
        img = nib.load(args.distance_weighting)
        distances_map = img.get_fdata(dtype=float)
    else:
        distances_map = None

    if args.correlation_weighting:
        img = nib.load(args.correlation_weighting)
        correlation_map = img.get_fdata(dtype=float)
    else:
        correlation_map = None

    for index, metric in enumerate(metrics):
        if np.any(np.isnan(metric.get_fdata())):
            logging.warning('Metric \"{}\" contains some NaN.'.format(args.in_metrics[index]) +
                            ' Ignoring voxels with NaN.')

    bundle_stats = get_bundle_metrics_mean_std(sft.streamlines,
                                               metrics,
                                               distances_map,
                                               correlation_map,
                                               args.density_weighting)

    bundle_name, _ = os.path.splitext(os.path.basename(args.in_bundle))

    stats = {bundle_name: {}}
    for metric, (mean, std) in zip(metrics, bundle_stats):
        metric_name = split_name_with_nii(
            os.path.basename(metric.get_filename()))[0]
        stats[bundle_name][metric_name] = {
            'mean': mean,
            'std': std
        }
    if args.include_dps:
        for metric in sft.data_per_streamline.keys():
            mean = float(np.average(sft.data_per_streamline[metric]))
            std = float(np.std(sft.data_per_streamline[metric]))
            stats[bundle_name][metric] = {
                'mean': mean,
                'std': std
            }
    print(json.dumps(stats, indent=args.indent, sort_keys=args.sort_keys))


if __name__ == '__main__':
    main()
