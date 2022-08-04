#!/usr/bin/env python3
import argparse
import logging
import json
import os
from time import perf_counter
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args, add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from scilpy.reconst.ftd import ClusterForFTD


MAD_MIN_DEFAULT_TH = 40
MDF_AVG_DEFAULT_TH = 10


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_vox2tracks',
                   help='Input voxel to tracks dictionary (json).')
    p.add_argument('out_vox2ids',
                   help='Output voxel to ids dictionary (json).')

    p.add_argument('--dist_metric',
                   choices=['mdf_avg', 'mad_min'], default='mad_min',
                   help='Distance to use for clustering.\nChoices are:\n'
                        '    \'mdf_avg\': Minimum average '
                        'direct-flip (mm)\n    \'mad_min\': Minimum mean'
                        ' angular deviation (degrees)\n Default is '
                        '[%(default)s]')
    p.add_argument('--nb_points_resampling', type=int, default=6,
                   help='Number of points for streamlines resample. '
                        '[%(default)s]')
    p.add_argument('--max_nb_clusters', type=int, default=20,
                   help='Maximum number of clusters. [%(default)s]')
    p.add_argument('--max_distance', type=float,
                   help='Maximum distance between cluster elements. The unit\n'
                        'and default value depend on the choice of distance.\n'
                        '    mdf_avg: [10]\n    mad_min: [40]')
    p.add_argument('--batch_size', default=5000,
                   help='Batch size for GPU.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_json_args(p)
    add_reference_arg(p)
    return p


def _get_max_distance(args):
    if args.max_distance:
        return args.max_distance
    if args.dist_metric == 'mdf_avg':
        return MDF_AVG_DEFAULT_TH
    if args.dist_metric == 'mad_min':
        return MAD_MIN_DEFAULT_TH


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    assert_inputs_exist(parser, [args.in_tractogram, args.in_vox2tracks])
    assert_outputs_exist(parser, args, [args.out_vox2ids])

    # validate output file types
    _, ext = os.path.splitext(args.out_vox2ids)
    if ext != '.json':
        parser.error('Invalid extension for out_vox2ids: {}'
                     .format(args.out_vox2ids))

    t0 = perf_counter()
    logging.info('Loading input data...')
    # load json file
    vox2tracks = json.load(open(args.in_vox2tracks, 'r'))

    # load tractogram
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    logging.info('Loaded input data in {:.2f} seconds'
                 .format(perf_counter() - t0))

    max_distance = _get_max_distance(args)
    clusters = ClusterForFTD(sft, vox2tracks,
                             nb_points_resampling=args.nb_points_resampling,
                             max_nb_clusters=args.max_nb_clusters,
                             max_distance=max_distance,
                             dist_metric=args.dist_metric)

    # launch compute
    vox2ids = clusters.cluster_gpu(batch_size=args.batch_size)

    logging.info('Saving outputs...')
    t1 = perf_counter()

    # output json
    out_json = open(args.out_vox2ids, 'w')
    json.dump(vox2ids, out_json, indent=args.indent, sort_keys=args.sort_keys)
    logging.info('Saved outputs in {:.2f}'.format(perf_counter() - t1))

    logging.info('Total runtime: {:.2f}'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
