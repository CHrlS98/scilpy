#!/usr/bin/env python3
import argparse
import logging
import json
from time import perf_counter
import nibabel as nib
import numpy as np
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args, add_reference_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             add_overwrite_arg)
from dipy.io.streamline import save_tractogram, StatefulTractogram
from scilpy.reconst.ftd import ClusterForFTD


def _build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('in_tractogram',
                   help='Input short-tracks tractogram.')
    p.add_argument('in_json',
                   help='Input voxel to tracks dictionary.')
    p.add_argument('out_labels',
                   help='Output label map.')
    p.add_argument('out_centroids',
                   help='Output centroids tractogram.')
    p.add_argument('out_dict')

    p.add_argument('--nb_points_resampling', type=int, default=6,
                   help='Number of points for streamlines resample.')
    p.add_argument('--max_nb_clusters', type=int, default=10,
                   help='Maximum number of clusters')
    p.add_argument('--max_deviation', type=float, default=40.0,
                   help='Maximum angular deviation between cluster elements.')
    p.add_argument('--batch_size', default=1000,
                   help='Batch size for GPU.')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_json_args(p)
    add_reference_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    assert_inputs_exist(parser, [args.in_tractogram, args.in_json])
    assert_outputs_exist(parser, args, [args.out_labels,
                                        args.out_centroids,
                                        args.out_dict])

    t0 = perf_counter()
    logging.info('Loading input data...')
    # load json file
    vox2tracks = json.load(open(args.in_json, 'r'))

    # load tractogram
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    logging.info('Loaded input data in {:.2f} seconds'
                 .format(perf_counter() - t0))

    clusters = ClusterForFTD(sft, vox2tracks,
                             nb_points_resampling=args.nb_points_resampling,
                             max_nb_clusters=args.max_nb_clusters,
                             max_mean_deviation=args.max_deviation)

    # launch compute
    labels, centroids, centroids_voxel, vox2ids =\
        clusters.cluster_gpu(batch_size=args.batch_size)

    logging.info('Saving outputs...')
    t1 = perf_counter()
    nib.save(nib.Nifti1Image(labels.astype(np.float32), sft.affine),
             args.out_labels)

    # output json
    out_json = open(args.out_dict, 'w')
    json.dump(vox2ids, out_json, indent=args.indent, sort_keys=args.sort_keys)

    # data per streamline is voxel id for each centroid
    dps = {'voxel': np.asarray(centroids_voxel)}
    out_sft = StatefulTractogram.from_sft(centroids, sft,
                                          data_per_streamline=dps)
    save_tractogram(out_sft, args.out_centroids)
    logging.info('Saved outputs in {:.2f}'.format(perf_counter() - t1))

    logging.info('Total runtime: {:.2f}'.format(perf_counter() - t0))


if __name__ == '__main__':
    main()
